"""Export NPR to ONNX without PyTorch.

`export_onnx.py --npr` is the canonical path and should be used wherever torch runs. This is
the fallback for the machines where it does not — on Windows, torch's own `c10.dll` can fail
its initialisation routine (WinError 1114) with every dependency present, and no combination
of torch or Python version fixes it.

The architecture is not guessed. It is transcribed from scripts/models/npr_model.py, and the
transcription is *checked*: every tensor in the checkpoint must be consumed exactly once, so
a wrong block count or a missed downsample fails loudly here rather than producing a graph
that runs and answers wrongly.

    python scripts/export_npr_no_torch.py                 # int8, matching the deployed build
    python scripts/export_npr_no_torch.py --no-quantize   # fp32

Needs: numpy, onnx, onnxruntime. Not torch.
"""

import argparse
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.config import NPR_CHECKPOINT, NPR_MODEL_PATH
from common.torch_pickle import load_state_dict

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
BN_EPS = 1e-5  # torch.nn.BatchNorm2d default, which torchvision's ResNet does not override


class Graph:
    """Node/initializer accumulator with a name counter, so every node name is unique."""

    def __init__(self, state):
        self.nodes = []
        self.inits = []
        self.state = state
        self.used = set()
        self.n = 0

    def take(self, key):
        """Pull a tensor out of the checkpoint and record that it was used."""
        if key not in self.state:
            raise SystemExit(f"checkpoint is missing {key} — architecture mismatch")
        self.used.add(key)
        return self.state[key]

    def const(self, array, tag):
        name = f"{tag}_{self.n}"
        self.n += 1
        self.inits.append(numpy_helper.from_array(np.ascontiguousarray(array), name))
        return name

    def add(self, op, inputs, tag, **attrs):
        out = f"{tag}_{self.n}"
        self.n += 1
        self.nodes.append(helper.make_node(op, inputs, [out], name=out, **attrs))
        return out

    # -- building blocks -----------------------------------------------------------------

    def conv(self, x, prefix, stride=1, pad=0):
        w = self.take(f"{prefix}.weight")
        k = w.shape[-1]
        return self.add(
            "Conv",
            [x, self.const(w.astype(np.float32), f"{prefix}.w")],
            prefix.replace(".", "_"),
            kernel_shape=[k, k],
            strides=[stride, stride],
            pads=[pad, pad, pad, pad],
        )

    def bn(self, x, prefix):
        # num_batches_tracked exists in the checkpoint and is training bookkeeping only;
        # marking it used keeps the "everything consumed" check honest.
        tracked = f"{prefix}.num_batches_tracked"
        if tracked in self.state:
            self.used.add(tracked)
        return self.add(
            "BatchNormalization",
            [
                x,
                self.const(self.take(f"{prefix}.weight").astype(np.float32), f"{prefix}.s"),
                self.const(self.take(f"{prefix}.bias").astype(np.float32), f"{prefix}.b"),
                self.const(self.take(f"{prefix}.running_mean").astype(np.float32), f"{prefix}.m"),
                self.const(self.take(f"{prefix}.running_var").astype(np.float32), f"{prefix}.v"),
            ],
            prefix.replace(".", "_"),
            epsilon=BN_EPS,
        )

    def relu(self, x, tag):
        return self.add("Relu", [x], tag)

    def bottleneck(self, x, prefix, stride):
        """torchvision Bottleneck: 1x1 -> 3x3(stride) -> 1x1, plus the identity branch.

        The stride sits on the 3x3, not the first 1x1 — that is ResNet v1.5, which is what
        torchvision builds and therefore what these weights expect.
        """
        out = self.relu(self.bn(self.conv(x, f"{prefix}.conv1"), f"{prefix}.bn1"), f"{prefix}_r1")
        out = self.relu(
            self.bn(self.conv(out, f"{prefix}.conv2", stride=stride, pad=1), f"{prefix}.bn2"),
            f"{prefix}_r2",
        )
        out = self.bn(self.conv(out, f"{prefix}.conv3"), f"{prefix}.bn3")

        identity = x
        if f"{prefix}.downsample.0.weight" in self.state:
            identity = self.bn(
                self.conv(x, f"{prefix}.downsample.0", stride=stride), f"{prefix}.downsample.1"
            )

        return self.relu(self.add("Add", [out, identity], f"{prefix}_add"), f"{prefix}_r3")

    def repeat_interleave(self, x, axis, shape_after, tag):
        """`repeat_interleave(2, dim=axis)`, built from Concat rather than Resize.

        Resize with nearest/asymmetric would also work, but its corner semantics vary between
        runtimes and opsets. Duplicating along a new axis and folding it back is exact
        everywhere, and this operator is the whole method — an off-by-one here would silently
        change what NPR measures.
        """
        expanded = self.add("Unsqueeze", [x, self.const(np.array([axis + 1], np.int64), f"{tag}_ax")], f"{tag}_u")
        doubled = self.add("Concat", [expanded, expanded], f"{tag}_c", axis=axis + 1)
        return self.add(
            "Reshape",
            [doubled, self.const(np.array(shape_after, np.int64), f"{tag}_sh")],
            f"{tag}_r",
        )


def build(state, size=IMG_SIZE, check=False):
    g = Graph(state)
    x = "pixel_values"

    # --- NPR residual: x - upsample(downsample(x)) ------------------------------------
    # down = x[:, :, ::2, ::2]
    down = g.add(
        "Slice",
        [
            x,
            g.const(np.array([0, 0], np.int64), "npr_start"),
            g.const(np.array([size, size], np.int64), "npr_end"),
            g.const(np.array([2, 3], np.int64), "npr_axes"),
            g.const(np.array([2, 2], np.int64), "npr_step"),
        ],
        "npr_down",
    )
    half = size // 2
    up = g.repeat_interleave(down, 2, [-1, 3, size, half], "npr_h")
    up = g.repeat_interleave(up, 3, [-1, 3, size, size], "npr_w")

    residual = g.add("Sub", [x, up], "npr_sub")
    scaled = g.add("Mul", [residual, g.const(np.float32(2.0 / 3.0), "npr_scale")], "npr_mul")

    # --- truncated ResNet-50 trunk -----------------------------------------------------
    out = g.relu(g.bn(g.conv(scaled, "conv1", stride=2, pad=1), "bn1"), "stem_relu")
    out = g.add("MaxPool", [out], "stem_pool", kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])

    for block in range(3):
        out = g.bottleneck(out, f"layer1.{block}", stride=1)
    for block in range(4):
        out = g.bottleneck(out, f"layer2.{block}", stride=2 if block == 0 else 1)

    pooled = g.add("GlobalAveragePool", [out], "avgpool")
    flat = g.add("Flatten", [pooled], "flatten", axis=1)
    logits = g.add(
        "Gemm",
        [
            flat,
            g.const(g.take("fc1.weight").astype(np.float32), "fc1.w"),
            g.const(g.take("fc1.bias").astype(np.float32), "fc1.b"),
        ],
        "logits",
        transB=1,
    )

    # The strict=True equivalent. A block miscount leaves tensors behind; catch it here.
    leftover = sorted(set(state) - g.used)
    if leftover:
        raise SystemExit(
            f"{len(leftover)} checkpoint tensors were not used, so this graph is not the model "
            f"the weights came from: {leftover[:6]}{'…' if len(leftover) > 6 else ''}"
        )

    graph = helper.make_graph(
        g.nodes,
        "npr",
        [helper.make_tensor_value_info(x, TensorProto.FLOAT, ["batch", 3, size, size])],
        [helper.make_tensor_value_info(logits, TensorProto.FLOAT, ["batch", 1])],
        initializer=g.inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10  # onnxruntime 1.20 rejects the newer default

    # checker and shape inference call into onnx's C++ extension. On the machine this
    # fallback exists for, that extension crashes with an access violation — a segfault, not
    # an exception, so it cannot be caught. Off by default; the graph is validated by loading
    # it in onnxruntime-node instead, which is a stronger check anyway.
    if check:
        onnx.checker.check_model(model)
        return onnx.shape_inference.infer_shapes(model)
    return model


def verify(path, size=IMG_SIZE):
    """Structural checks that do not need torch — the numeric oracle is the live service.

    Skipped when onnxruntime cannot load. On the machine this was written for, every
    C++-heavy Python extension fails its DLL initialisation while pure-C ones (numpy) are
    fine; `scripts/verify_npr_onnx.mjs` runs the same checks through onnxruntime-node, which
    is unaffected.
    """
    try:
        import onnxruntime as ort
    except Exception as e:
        print(f"  verify     skipped — onnxruntime unavailable ({type(e).__name__})")
        print("             run: node scripts/verify_npr_onnx.mjs")
        return

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)

    batch = rng.standard_normal((2, 3, size, size)).astype(np.float32)
    out = session.run(None, {"pixel_values": batch})[0]
    if out.shape != (2, 1):
        raise SystemExit(f"expected (2, 1) logits, got {out.shape}")

    # NPR's defining property: an image that already *is* a 2x nearest up-sample has no
    # residual, so the trunk sees zeros. Two different such images must therefore produce the
    # same logit. If they do not, the residual is not being computed where it should be.
    def upsampled(seed):
        small = np.random.default_rng(seed).standard_normal((1, 3, size // 2, size // 2)).astype(np.float32)
        return np.repeat(np.repeat(small, 2, axis=2), 2, axis=3)

    a = session.run(None, {"pixel_values": upsampled(1)})[0]
    b = session.run(None, {"pixel_values": upsampled(2)})[0]
    if abs(float(a) - float(b)) > 1e-4:
        raise SystemExit(
            f"two different 2x-upsampled images gave different logits ({float(a):.6f} vs "
            f"{float(b):.6f}); the NPR residual is not zeroing them, so the operator is wrong"
        )

    # And a natural image must not land on that same degenerate value, or the residual is
    # zeroing everything rather than only what it should.
    natural = session.run(None, {"pixel_values": rng.random((1, 3, size, size)).astype(np.float32)})[0]
    if abs(float(natural) - float(a)) < 1e-6:
        raise SystemExit("a natural image produced the same logit as a flat residual")

    print(f"  structure  ok — upsampled inputs collapse to {float(a):.6f}, natural gives {float(natural):.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=NPR_CHECKPOINT)
    ap.add_argument("--out", default=NPR_MODEL_PATH)
    ap.add_argument("--no-quantize", action="store_true", help="keep fp32 (4x larger)")
    ap.add_argument("--check", action="store_true",
                    help="run onnx.checker (crashes where onnx's C++ extension is broken)")
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(
            f"No NPR checkpoint at {args.checkpoint}.\n"
            f"  curl -L -o {args.checkpoint} https://github.com/chuangchuangtan/"
            f"NPR-DeepfakeDetection/raw/main/model_epoch_last_3090.pth"
        )

    print(f"Reading {args.checkpoint} without torch")
    state = load_state_dict(args.checkpoint)
    print(f"  {len(state)} tensors")

    model = build(state, check=args.check)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fp32_path = args.out if args.no_quantize else args.out.replace(".onnx", ".fp32.onnx")
    onnx.save(model, fp32_path)
    print(f"Wrote {fp32_path} ({os.path.getsize(fp32_path) / 1048576:.1f} MB)")
    verify(fp32_path)

    quantized = False
    if not args.no_quantize:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except Exception as e:
            # fp32 is the safer failure: it is the *more* accurate artifact, and the deployed
            # service's int8 build is documented to differ from full precision "by a point or
            # two" anyway — inside the parity tolerance either way.
            print(f"  quantize   skipped — onnxruntime unavailable ({type(e).__name__}); shipping fp32")
            os.replace(fp32_path, args.out)
        else:
            print(f"Quantizing (int8 weights) -> {args.out}")
            quantize_dynamic(fp32_path, args.out, weight_type=QuantType.QUInt8)
            os.remove(fp32_path)
            quantized = True
            verify(args.out)

    meta = {
        "source": f"npr:{os.path.basename(args.checkpoint)}",
        "labels": None,
        "sigmoid": True,
        "imgSize": IMG_SIZE,
        "mean": MEAN,
        "std": STD,
        "expectsFace": False,
        "calibrated": False,
        "quantized": quantized,
        "exporter": "export_npr_no_torch.py",
    }
    meta_path = os.path.splitext(args.out)[0] + ".json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Wrote {args.out} ({os.path.getsize(args.out) / 1048576:.1f} MB) and {meta_path}")
    print("\nNumeric agreement is not provable here without torch. Check it against the")
    print("deployed service, which runs the official export:")
    print("  cd extension && npm run parity:capture && npm run parity")


if __name__ == "__main__":
    main()
