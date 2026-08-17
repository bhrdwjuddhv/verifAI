"""Settings shared by the trainer and the inference server.

These two files disagreed once — different backbone, different class count, different
checkpoint filename — and the server quietly served noise instead of failing. One
module, imported by both, so a mismatch is impossible rather than merely unlikely.
"""

import os

# Written by train_deepfake_detector.py, read by inference_server.py.
MODEL_PATH = os.environ.get("VERIFAI_MODEL", os.path.join("models", "deepfake_detector.pth"))

# Used only when no trained checkpoint exists. Verified against the Hugging Face API
# on 2026-08-15: ViTForImageClassification, 224px, id2label {0: Realism, 1: Deepfake}.
# It is a FACE deepfake classifier — it says nothing useful about a landscape.
HF_FALLBACK_MODEL = os.environ.get("VERIFAI_HF_MODEL", "prithivMLmods/Deep-Fake-Detector-v2-Model")
HF_FALLBACK_EXPECTS_FACE = os.environ.get("VERIFAI_HF_EXPECTS_FACE", "1") != "0"

# Verdict bands on P(fake), in percent.
FAKE_ABOVE = float(os.environ.get("VERIFAI_FAKE_ABOVE", "70"))
REAL_BELOW = float(os.environ.get("VERIFAI_REAL_BELOW", "30"))

# NPR (Tan et al., CVPR 2024): whole-image AI-generation detector. Complements the face
# classifier, which only sees face swaps and misses fully generated faces entirely.
NPR_CHECKPOINT = os.environ.get("VERIFAI_NPR_CHECKPOINT", os.path.join("models", "npr_detector.pth"))
NPR_MODEL_PATH = os.environ.get("NPR_MODEL_PATH", os.path.join("models", "npr_detector.onnx"))


def parse_weights(spec, defaults):
    """"face=0.6,npr=0.4" -> {"face": 0.6, "npr": 0.4}. Unknown keys are an error, not a shrug."""
    weights = dict(defaults)
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        key, _, value = part.partition("=")
        key = key.strip()
        if key not in defaults:
            raise ValueError(f"unknown fusion signal '{key}'; expected one of {sorted(defaults)}")
        weights[key] = float(value)
    if any(w < 0 for w in weights.values()):
        raise ValueError(f"negative fusion weight in {weights}")
    if sum(weights.values()) <= 0:
        raise ValueError("fusion weights sum to zero — nothing would decide the verdict")
    return weights


# How the per-detector scores combine into one P(fake). Set FUSION_WEIGHTS to retune, e.g.
# FUSION_WEIGHTS="face=0.3,npr=0.7". Weights renormalize over whichever signals actually ran.
#
# frequency defaults to 0 deliberately: it is a real measurement (share of FFT energy above
# half-Nyquist) but nobody has established which direction it points on this data — sharp real
# photos score high too. It is reported next to the verdict and given weight only once
# scripts/tune_fusion.py has measured that it helps.
FUSION_DEFAULTS = {"face": 0.5, "npr": 0.5, "frequency": 0.0}
FUSION_WEIGHTS = parse_weights(os.environ.get("FUSION_WEIGHTS", ""), FUSION_DEFAULTS)
