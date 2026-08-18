# VerifAI model service — torch-free runtime image.
#
# Stage 1 installs torch just long enough to export an int8 ONNX model, then throws it away.
# Stage 2 ships onnxruntime + opencv and the two .onnx files: measured 205MB resident
# (peak 306MB) against 1262MB for the torch build, which is what makes a 512MB free instance
# viable. Build context is scripts/ :
#
#   docker build -t verifai-model ./scripts
#   docker run -p 8000:8000 verifai-model
#
# For the torch build instead (Grad-CAM rather than occlusion saliency, ~1.2GB RAM):
#   docker build --target torch -t verifai-model-torch ./scripts

# ---------------------------------------------------------------- stage 1: export
FROM python:3.11-slim AS builder
WORKDIR /build

# CPU-only wheels: the default PyPI torch drags in ~2.5GB of CUDA libraries this never loads.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip install --no-cache-dir transformers onnx onnxruntime onnxscript

COPY common ./common
COPY export_onnx.py .

# The exporter verifies the ONNX output against the torch model and exits non-zero if they
# disagree, so a bad quantization fails the build instead of shipping wrong scores.
RUN python export_onnx.py

ADD https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
    /build/models/face_detection_yunet.onnx

# NPR (CVPR 2024) whole-image AI-generation detector — the official ProGAN-trained weights,
# 5.8MB, which generalize to unseen generators. Without this, a fully generated face is only
# ever judged by a face-swap model that was never trained to see one.
COPY models ./models
ADD https://github.com/chuangchuangtan/NPR-DeepfakeDetection/raw/main/model_epoch_last_3090.pth \
    /build/models/npr_detector.pth
RUN python export_onnx.py --npr && rm /build/models/npr_detector.pth

# ---------------------------------------------------------------- stage 2: optional torch build (--target torch)
FROM python:3.11-slim AS torch
WORKDIR /app
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV HF_HOME=/app/.hf
RUN python -c "\
from transformers import AutoImageProcessor, AutoModelForImageClassification; \
m='prithivMLmods/Deep-Fake-Detector-v2-Model'; \
AutoImageProcessor.from_pretrained(m); AutoModelForImageClassification.from_pretrained(m)"
ENV PORT=8000
EXPOSE 8000
CMD uvicorn inference_server:create_app --factory --host 0.0.0.0 --port $PORT

# ---------------------------------------------------------------- stage 3: lean runtime (default target)
FROM python:3.11-slim AS lean
WORKDIR /app

# libGL/libglib: opencv-python-headless still links a couple of system libs.
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-lean.txt .
RUN pip install --no-cache-dir -r requirements-lean.txt

COPY . .
COPY --from=builder /build/models ./models

ENV PORT=8000 \
    VERIFAI_THREADS=2 \
    VERIFAI_SALIENCY_GRID=5 \
    VIDEO_FRAME_CAP=8
EXPOSE 8000
CMD uvicorn inference_server:create_app --factory --host 0.0.0.0 --port $PORT

