"""Settings shared by the trainer and the inference server.

These two files disagreed once — different backbone, different class count, different
checkpoint filename — and the server quietly served noise instead of failing. One
module, imported by both, so a mismatch is impossible rather than merely unlikely.
"""

import os

# Every model path here is written relative to the repo root, which is correct when the
# service is started the documented way and wrong the moment anyone runs a script from
# scripts/. That failure is silent and looks identical to a genuinely missing model — it
# reports "not installed" and quietly serves less. Resolving against the repo root as a
# fallback makes cwd stop mattering, while a path that resolves from cwd still wins so an
# explicit relative override behaves as written.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve(path):
    """The given path if it exists, else the same path relative to the repo root."""
    if os.path.exists(path) or os.path.isabs(path):
        return path
    rooted = os.path.join(REPO_ROOT, path)
    return rooted if os.path.exists(rooted) else path


def env_path(name):
    """A path from the environment, resolved the same way. None when unset or empty.

    An operator who sets a relative override deserves the same cwd-independence the defaults
    get; an absolute one is left exactly as given.
    """
    value = os.environ.get(name)
    return resolve(value) if value else None


def first_existing(*candidates):
    """First path that exists, else the last one (so error messages name the canonical spot).

    Lets a trained model dropped into models/face/ take precedence over the generic fallback
    without anyone having to set an env var on the deploy host — while an explicit env var
    still wins over both.
    """
    for path in candidates:
        resolved = resolve(path)
        if os.path.exists(resolved):
            return resolved
    return resolve(candidates[-1])


# Written by train_deepfake_detector.py, read by inference_server.py. Newest first: a v2
# checkpoint in models/face/ beats v1, which beats the generic path, which beats the HF
# fallback. An explicit env var beats all of them.
MODEL_PATH = env_path("VERIFAI_MODEL") or first_existing(
    os.path.join("models", "face", "deepfake_detector_v2.pth"),
    os.path.join("models", "face", "deepfake_detector.pth"),
    os.path.join("models", "deepfake_detector.pth"),
)

# The torch-free face classifier. Same precedence.
ONNX_PATH = env_path("VERIFAI_ONNX") or first_existing(
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
#   models/npr_detector.onnx        the authors' ProGAN-trained weights (int8). The default.
#   models/images/npr_detector.onnx trained here, fp32. Records val AUC 0.664 in its own
#                                   metadata — 0.5 is a coin flip, so that is weak, and it is
#                                   the only accuracy number that exists for it.
#
# It is NOT the default, and the reason is the absence of a measurement rather than a bad one:
# nothing here has been evaluated on a labelled set of real vs generated images, so promoting
# it over published weights would be a preference dressed up as a result.
#
# What HAS been measured (2026-08-22, six probe images, identical inputs): this checkpoint was
# previously exported int8, and that export differed from fp32 by a mean of 41.9 points and a
# max of 64.7. That is not quantization noise, it is a different model — the int8 build
# collapsed into the uncertain band (4 of 6 images inside 30-70) while fp32 spans 0.2-98%.
# The int8 build has therefore been replaced. The .pth is unchanged and always was.
#
# To use the locally trained one:
#   NPR_MODEL_PATH=models/images/npr_detector.onnx
# and re-measure first with: python scripts/train_npr.py --eval-only --eval-dir <unseen generators>
NPR_CHECKPOINT = env_path("VERIFAI_NPR_CHECKPOINT") or first_existing(
    os.path.join("models", "npr_detector.pth"),
    os.path.join("models", "images", "npr_detector.pth"),
)
NPR_MODEL_PATH = env_path("NPR_MODEL_PATH") or first_existing(
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
AUDIO_CHECKPOINT = env_path("VERIFAI_AUDIO_CHECKPOINT") or first_existing(
    os.path.join("models", "audio", "audio_deepfake_detector.pth"),
    os.path.join("models", "audio_deepfake_detector.pth"),
)
AUDIO_MODEL_PATH = env_path("AUDIO_MODEL_PATH") or first_existing(
    os.path.join("models", "audio", "audio_detector.onnx"),
    os.path.join("models", "audio_detector.onnx"),
)
