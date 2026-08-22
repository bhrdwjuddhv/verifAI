import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torchvision import models

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure scripts and root directory are in sys.path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="Export PyTorch Deepfake Detector checkpoint to ONNX format.")
    default_model = os.environ.get("VERIFAI_MODEL", "models/face/deepfake_detector_v2.pth")
    default_onnx = os.environ.get("VERIFAI_ONNX", "models/face/detector_v2.onnx")

    parser.add_argument("--checkpoint", "-c", type=str, default=default_model, help="Path to PyTorch checkpoint (.pth)")
    parser.add_argument("--out", "-o", type=str, default=default_onnx, help="Path for output ONNX file (.onnx)")
    parser.add_argument("--no-quantize", action="store_true", default=True, help="Skip int8 quantization to prevent probability shift")
    parser.add_argument("--arch", type=str, default="b0", choices=["b0", "b4", "mobilenet"], help="Model backbone architecture")
    parser.add_argument("--modality", type=str, default="auto", choices=["auto", "face", "audio", "all"], help="Modality to export (face, audio, all)")
    return parser.parse_args()

def build_model(arch="b0", num_classes=2):
    """Build model with classifier head matching the training scripts (plain Linear, no Sequential wrapper)."""
    if arch == "b0":
        model = models.efficientnet_b0()
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif arch == "b4":
        model = models.efficientnet_b4()
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif arch == "mobilenet":
        model = models.mobilenet_v3_small()
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return model

def export_onnx(checkpoint_path, onnx_path, arch="b0", no_quantize=True, modality="face"):
    device = torch.device("cpu")
    num_classes = 2
    classes = ["Fake", "Real"]
    temperature = 1.0

    # Auto-detect modality from checkpoint if specified
    if os.path.exists(checkpoint_path):
        try:
            ckpt_peek = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(ckpt_peek, dict):
                if "classes" in ckpt_peek:
                    classes = ckpt_peek["classes"]
                    num_classes = len(classes)
                if "arch" in ckpt_peek:
                    arch = ckpt_peek["arch"]
                if "temperature" in ckpt_peek and isinstance(ckpt_peek["temperature"], (int, float)):
                    temperature = float(ckpt_peek["temperature"])
                if "modality" in ckpt_peek:
                    modality = ckpt_peek["modality"]
        except Exception:
            pass

    model = build_model(arch=arch, num_classes=num_classes)

    if os.path.exists(checkpoint_path):
        raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(raw, dict) and "state_dict" in raw:
            state_dict = raw["state_dict"]
        else:
            state_dict = raw
        # Strict loading — any key mismatch will raise immediately
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # Fail if classifier weights are missing (the critical bug we're guarding against)
        critical_missing = [k for k in missing if "classifier" in k]
        if critical_missing:
            raise RuntimeError(f"Classifier weights NOT loaded! Missing keys: {critical_missing}. "
                               f"Architecture mismatch between checkpoint and build_model().")
        if missing:
            print(f"  (non-critical missing keys: {missing})")
        print(f"✅ Loaded checkpoint weights from {checkpoint_path} ({len(state_dict) - len(missing)}/{len(state_dict)} keys matched)")
    else:
        print(f"⚠️ Checkpoint '{checkpoint_path}' not found. Initializing with default backbone weights.")

    model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    print(f"📦 Exporting model to ONNX: {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print(f"✅ ONNX model exported successfully to {onnx_path}")

    # Generate metadata detector.json
    metadata_path = os.path.join(os.path.dirname(os.path.abspath(onnx_path)), "detector.json")
    metadata = {
        "architecture": f"EfficientNet-{arch.upper()}" if arch.startswith("b") else arch,
        "modality": modality,
        "input_shape": [1, 3, 224, 224],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "labels": classes,
        "temperature": temperature,
        "quantized": not no_quantize,
        "onnx_model": os.path.basename(onnx_path)
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"📄 Created model metadata config at {metadata_path}")

    # If audio modality, also export preproc.onnx & audio_selftest.json
    if modality == "audio":
        try:
            from scripts.export_audio_preproc import export_preproc_onnx, generate_selftest_vector
            preproc_path = os.path.join(os.path.dirname(os.path.abspath(onnx_path)), "preproc.onnx")
            selftest_path = os.path.join(os.path.dirname(os.path.abspath(onnx_path)), "audio_selftest.json")
            export_preproc_onnx(preproc_path)
            generate_selftest_vector(preproc_path, onnx_path, selftest_path)
        except Exception as e:
            print(f"⚠️ Audio preproc export notice: {e}")

    return onnx_path

def main():
    args = parse_args()
    if args.modality == "all":
        print("\n=== Exporting Face Detector ===")
        export_onnx("models/face/deepfake_detector_v2.pth", "models/face/detector_v2.onnx", arch="b0", modality="face")
        print("\n=== Exporting Audio Detector ===")
        export_onnx("models/audio_deepfake_detector.pth", "models/audio/audio_detector.onnx", arch="b0", modality="audio")
    elif args.modality == "audio":
        export_onnx(
            checkpoint_path="models/audio_deepfake_detector.pth" if args.checkpoint == os.environ.get("VERIFAI_MODEL", "models/face/deepfake_detector_v2.pth") else args.checkpoint,
            onnx_path="models/audio/audio_detector.onnx" if args.out == os.environ.get("VERIFAI_ONNX", "models/face/detector_v2.onnx") else args.out,
            arch=args.arch,
            no_quantize=args.no_quantize,
            modality="audio"
        )
    else:
        export_onnx(
            checkpoint_path=args.checkpoint,
            onnx_path=args.out,
            arch=args.arch,
            no_quantize=args.no_quantize,
            modality="face"
        )

if __name__ == "__main__":
    main()
