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
