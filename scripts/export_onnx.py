"""Export the active classifier to ONNX (optionally int8) so the service can run without torch.

torch + a ViT-base is ~1.2GB resident, which does not fit a 512MB free instance. ONNX Runtime
with an int8-quantized ViT is roughly a fifth of that, and the *runtime* image then needs no
torch, no transformers and no facenet-pytorch at all.

Run once — on your machine or in a Docker build stage — then ship the artifacts:

    python scripts/export_onnx.py                      # HF fallback -> models/detector.onnx
    python scripts/export_onnx.py --no-quantize        # fp32, larger but bit-exact
    python scripts/export_onnx.py --checkpoint         # your trained .pth instead

Writes models/detector.onnx and models/detector.json (labels + preprocessing, so the service
never has to guess how the model was trained).
"""

import argparse
import json
import os
import sys

# torch.onnx prints status lines containing emoji; the Windows console is cp1252 and would
# otherwise kill a successful export at the print statement.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from common.config import (
    HF_FALLBACK_EXPECTS_FACE,
    HF_FALLBACK_MODEL,
    MODEL_PATH,
    NPR_CHECKPOINT,
    NPR_MODEL_PATH,
)

ONNX_PATH = os.environ.get("VERIFAI_ONNX", os.path.join("models", "detector.onnx"))
META_PATH = os.path.splitext(ONNX_PATH)[0] + ".json"


def load_hf():
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(HF_FALLBACK_MODEL)
    model = AutoModelForImageClassification.from_pretrained(HF_FALLBACK_MODEL).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

    size = getattr(processor, "size", None) or {}
    side = size.get("height") or size.get("shortest_edge") or 224
    mean = list(getattr(processor, "image_mean", [0.5, 0.5, 0.5]))
    std = list(getattr(processor, "image_std", [0.5, 0.5, 0.5]))

    class Logits(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            return self.inner(pixel_values=pixel_values).logits

    meta = {
        "source": f"hf_fallback:{HF_FALLBACK_MODEL}",
        "labels": labels,
        "imgSize": int(side),
        "mean": mean,
        "std": std,
        "expectsFace": HF_FALLBACK_EXPECTS_FACE,
        "calibrated": False,
    }
    return Logits(model), meta


def load_checkpoint():
    import torch
    import torch.nn as nn
    from torchvision import models

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    classes = ckpt["classes"]
    ctor = {"b0": models.efficientnet_b0, "b4": models.efficientnet_b4}[ckpt.get("arch", "b0")]
    model = ctor()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    meta = {
        "source": "trained_checkpoint",
        "labels": classes,
        "imgSize": int(ckpt.get("img_size", 224)),
        # ImageNet normalization — what build_transforms() used during training.
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "expectsFace": bool(ckpt.get("face_crop", False)),
        "temperature": float(ckpt.get("temperature", 1.0)),
        "calibrated": float(ckpt.get("temperature", 1.0)) != 1.0,
    }
    return model, meta


def load_npr_for_export(path):
    """NPR outputs one logit, not class probabilities — sigmoid, no label vocabulary."""
    from models.npr_model import IMG_SIZE, MEAN, STD, load_npr

    meta = {
        "source": f"npr:{os.path.basename(path)}",
        "labels": None,
        "sigmoid": True,
        "imgSize": IMG_SIZE,
        "mean": MEAN,
        "std": STD,
        "expectsFace": False,
        "calibrated": False,
    }
    return load_npr(path), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="store_true", help=f"export {MODEL_PATH} instead of the HF model")
    ap.add_argument("--npr", action="store_true",
                    help=f"export the NPR whole-image detector ({NPR_CHECKPOINT} -> {NPR_MODEL_PATH})")
    ap.add_argument("--no-quantize", action="store_true", help="keep fp32 (4x larger, bit-exact)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = NPR_MODEL_PATH if args.npr else ONNX_PATH

    import torch

    if args.npr:
        if not os.path.exists(NPR_CHECKPOINT):
            sys.exit(
                f"No NPR checkpoint at {NPR_CHECKPOINT}.\n"
                f"  Train one:  python scripts/train_npr.py --data-dir <real-vs-ai dataset>\n"
                f"  Or use the official ProGAN-trained weights:\n"
                f"    curl -L -o {NPR_CHECKPOINT} https://github.com/chuangchuangtan/"
                f"NPR-DeepfakeDetection/raw/main/model_epoch_last_3090.pth"
            )
        model, meta = load_npr_for_export(NPR_CHECKPOINT)
    elif args.checkpoint:
        if not os.path.exists(MODEL_PATH):
            sys.exit(f"No checkpoint at {MODEL_PATH}.")
        model, meta = load_checkpoint()
    else:
        model, meta = load_hf()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fp32_path = args.out if args.no_quantize else args.out.replace(".onnx", ".fp32.onnx")
    side = meta["imgSize"]

    print(f"Exporting {meta['source']} -> {fp32_path}")
    torch.onnx.export(
        model,
        torch.randn(1, 3, side, side),
        fp32_path,
        input_names=["pixel_values"],
        output_names=["logits"],
        # Occlusion saliency sends a whole batch of masked copies in one call.
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        # The dynamo exporter splits weights into a separate .onnx.data file, which then trips
        # ONNX shape inference during quantization. The TorchScript path emits one self-contained
        # file that quantizes cleanly — and one file is what we want to ship anyway.
        dynamo=False,
    )

    if not args.no_quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        print(f"Quantizing (int8 weights) -> {args.out}")
        quantize_dynamic(fp32_path, args.out, weight_type=QuantType.QUInt8)
        os.remove(fp32_path)
        meta["quantized"] = True
    else:
        meta["quantized"] = False

    meta_path = os.path.splitext(args.out)[0] + ".json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"Wrote {args.out} ({mb:.0f} MB) and {meta_path}")
    print(json.dumps(meta, indent=2))

    # Agreement check: an export that silently changed the answer is worse than no export.
    verify(args.out, meta, model, side)


def verify(onnx_path, meta, torch_model, side):
    """The exported model must agree with the torch model it came from."""
    import numpy as np
    import onnxruntime as ort
    import torch

    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, side, side)).astype(np.float32)

    def to_prob_torch(logits):
        return torch.sigmoid(logits) if meta.get("sigmoid") else torch.softmax(logits, dim=1)

    with torch.no_grad():
        ref = to_prob_torch(torch_model(torch.from_numpy(x)).float()).numpy()
    got = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).run(
        None, {"pixel_values": x}
    )[0]
    if meta.get("sigmoid"):
        got = 1.0 / (1.0 + np.exp(-got))
    else:
        got = np.exp(got - got.max(1, keepdims=True))
        got = got / got.sum(1, keepdims=True)

    delta = float(np.abs(ref - got).max())
    tol = 0.08 if meta.get("quantized") else 1e-3
    print(f"max |Δp| vs torch: {delta:.4f} (tolerance {tol})")
    if delta > tol:
        sys.exit(
            f"Export disagrees with the source model by {delta:.4f}. "
            f"Re-run with --no-quantize, or investigate before shipping this."
        )
    print("export verified")


if __name__ == "__main__":
    main()
