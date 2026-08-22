# VerifAI model service — torch-free runtime image.
#
# BUILD CONTEXT IS THE REPOSITORY ROOT:
#
#   docker build -t verifai-model .
#   docker run -p 8000:8000 verifai-model
#
# On Render: Root Directory must be EMPTY and Dockerfile Path is ./Dockerfile (both defaults).
# Root Directory is what sets the build context — pointing it at scripts/ shrinks the context
# to that folder, and every COPY below then fails with "/scripts: not found". This file lives
# at the repository root precisely so the defaults are the correct settings.
#
# That is not a style choice. The trained models live in <repo>/models, and a build context of
# scripts/ cannot reach them — Docker refuses any COPY that climbs above the context. The
# earlier version of this file used scripts/ as the context and copied `models`, which silently
# picked up scripts/models/ (a Python PACKAGE holding npr_model.py) instead. The image built
# and started cleanly while containing no trained model at all, and the service fell back to
# the third-party Hugging Face classifier — a verdict the extension then refuses to display.
#
# Nothing is exported at build time any more. Every .onnx the service loads is committed to the
# repository and simply copied in: builds drop from a ~2.5GB torch download to seconds, and the
# image is guaranteed to contain the exact weights that were measured, rather than whatever a
# fresh export produced. Re-export with scripts/export_onnx.py and commit the result.
#
# For the torch build instead (Grad-CAM rather than occlusion saliency, ~1.2GB RAM):
#   docker build --target torch -t verifai-model-torch .

# ---------------------------------------------------------------- optional torch build
FROM python:3.11-slim AS torch
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
COPY scripts/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY scripts/ ./
COPY models/ ./models/
ENV HF_HOME=/app/.hf
ENV PORT=8000
EXPOSE 8000
CMD uvicorn inference_server:create_app --factory --host 0.0.0.0 --port $PORT

# ---------------------------------------------------------------- lean runtime (default)
FROM python:3.11-slim AS lean
WORKDIR /app

# libglib: opencv-python-headless still links a couple of system libs.
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/requirements-lean.txt .
RUN pip install --no-cache-dir -r requirements-lean.txt

# Application code first, models second: the models change far less often than the code, but
# putting them last keeps a code edit from invalidating the (much larger) model layer.
COPY scripts/ ./
COPY models/ ./models/

# Fail the build rather than ship a service that quietly answers with the wrong model.
#
# Every one of these is committed to the repository, so a miss means the clone was shallow, the
# context was wrong, or .gitignore swallowed them again — all of which are far cheaper to find
# here than in a production log reading "hf_fallback".
RUN set -eu; \
    for f in models/face/detector_v2.onnx \
             models/face/detector_v2.json \
             models/npr_detector.onnx \
             models/face_detection_yunet.onnx \
             models/audio/audio_detector.onnx; do \
      [ -f "$f" ] || { echo "FATAL: $f missing from the image."; \
                       echo "Build from the REPOSITORY ROOT with an EMPTY Render root dir."; \
                       exit 1; }; \
    done; \
    echo "models present:"; du -sh models/*

# The v2 voice chain is optional — the service falls back to v1 and says so — but a half-copied
# chain is not: an .onnx without its .onnx.data sidecar fails at load with a message about a
# missing tensor, which is the hardest version of this to diagnose remotely.
RUN set -eu; \
    d=models/audio/v2/models/audio; \
    if [ -f "$d/preproc.onnx" ]; then \
      for f in preproc.onnx.data audio_detector.onnx audio_detector.onnx.data audio_selftest.json; do \
        [ -f "$d/$f" ] || { echo "FATAL: v2 audio chain is incomplete, $f is missing."; exit 1; }; \
      done; \
      echo "v2 voice chain complete"; \
    else \
      echo "no v2 voice chain in this build; the service will use v1"; \
    fi

ENV PORT=8000 \
    VERIFAI_THREADS=2 \
    VERIFAI_SALIENCY_GRID=5 \
    VIDEO_FRAME_CAP=8
EXPOSE 8000

# --verify loads every model and prints which one answered, so `docker run --rm <image> \
# python inference_server.py --verify` tells you what a deploy will actually serve.
CMD uvicorn inference_server:create_app --factory --host 0.0.0.0 --port $PORT
