"""Settings shared by the trainer and the inference server.

These two files disagreed once — different backbone, different class count, different
checkpoint filename — and the server quietly served noise instead of failing. One
module, imported by both, so a mismatch is impossible rather than merely unlikely.
"""

import os


def first_existing(*candidates):
    """First path that exists, else the last one (so error messages name the canonical spot).

    Lets a trained model dropped into models/face/ take precedence over the generic fallback
    without anyone having to set an env var on the deploy host — while an explicit env var
    still wins over both.
    """
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]


# Written by train_deepfake_detector.py, read by inference_server.py. Newest first: a v2
# checkpoint in models/face/ beats v1, which beats the generic path, which beats the HF
# fallback. An explicit env var beats all of them.
MODEL_PATH = os.environ.get("VERIFAI_MODEL") or first_existing(
    os.path.join("models", "face", "deepfake_detector_v2.pth"),
    os.path.join("models", "face", "deepfake_detector.pth"),
    os.path.join("models", "deepfake_detector.pth"),
)

# The torch-free face classifier. Same precedence.
ONNX_PATH = os.environ.get("VERIFAI_ONNX") or first_existing(
    os.path.join("models", "face", "detector_v2.onnx"),
    os.path.join("models", "face", "detector.onnx"),
    os.path.join("models", "detector.onnx"),
)

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
#
# Two are installed, and the ORDER here is a measurement, not a preference:
#
#   models/npr_detector.onnx        the authors' ProGAN-trained weights. On four hand-checked
#                                   images: real 0%/0%, AI 100%/100% — decisive both ways.
#   models/images/npr_detector.onnx trained here. Same four images: 55%/53% real, 67%/61% AI.
#                                   It ranks them correctly but every score lands inside the
#                                   uncertain band (30-70), so it would turn every verdict into
#                                   "uncertain". Its own checkpoint records val AUC 0.664, which
#                                   says the same thing independently.
#
# The locally trained one is therefore reachable but second. To use it:
#   NPR_MODEL_PATH=models/images/npr_detector.onnx
# and re-measure with: python scripts/train_npr.py --eval-only --eval-dir <unseen generators>
NPR_CHECKPOINT = os.environ.get("VERIFAI_NPR_CHECKPOINT") or first_existing(
    os.path.join("models", "npr_detector.pth"),
    os.path.join("models", "images", "npr_detector.pth"),
)
NPR_MODEL_PATH = os.environ.get("NPR_MODEL_PATH") or first_existing(
    os.path.join("models", "npr_detector.onnx"),
    os.path.join("models", "images", "npr_detector.onnx"),
)


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

# Video: sampled frames, not every frame. 16 frames of two detectors is already ~30s of CPU
# on a free instance; decoding a 60s clip in full would be minutes.
VIDEO_FRAME_CAP = int(os.environ.get("VIDEO_FRAME_CAP", "16"))
VIDEO_MAX_SECONDS = float(os.environ.get("VIDEO_MAX_SECONDS", "60"))

# Voice. Absent by default; the route says so rather than guessing. Train with
# train_audio_detector.py, export with `export_onnx.py --audio`, drop the .onnx in.
AUDIO_CHECKPOINT = os.environ.get("VERIFAI_AUDIO_CHECKPOINT") or first_existing(
    os.path.join("models", "audio", "audio_deepfake_detector.pth"),
    os.path.join("models", "audio_deepfake_detector.pth"),
)
AUDIO_MODEL_PATH = os.environ.get("AUDIO_MODEL_PATH") or first_existing(
    os.path.join("models", "audio", "audio_detector.onnx"),
    os.path.join("models", "audio_detector.onnx"),
)
