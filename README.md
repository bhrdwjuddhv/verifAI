# VerifAI — Open-Source AI Media Verification Platform

VerifAI is a deepfake verification app built with **Next.js 14 (App Router)**, **TypeScript**, **Tailwind CSS**, and **PyTorch**.

**What the detector actually does:** it finds the largest face in an uploaded **image**, crops it, and scores it with a deepfake classifier, returning `real` / `fake` / `uncertain` with a confidence and a Grad-CAM heatmap. **Its limits:** faces only, images only (video and audio are not implemented), accuracy on unseen generators is **not yet measured**, and the default classifier's confidence is uncalibrated. A "real" verdict means the detector found nothing — not that the file is authentic. See [`/model-card`](app/model-card/page.tsx) and [`flow.md`](flow.md).

---

## 🛠️ Stack Overview

| Layer | Technologies |
|---|---|
| **Frontend & App** | Next.js 14 (App Router), React 18, TypeScript, Tailwind CSS, Framer Motion, Recharts |
| **3D Visuals** | Three.js, `@react-three/fiber` |
| **State & Auth** | Zustand (`lib/store.ts`), Web Crypto HMAC-SHA256 JWT Signed Sessions |
| **Backend API** | Next.js API Routes (`app/api/scan/route.ts`, `app/api/admin/*`) |
| **ML Engine & Server** | PyTorch (EfficientNet-B0/B4, Spectrogram CNN) + FastAPI (`scripts/inference_server.py`) |
| **Explainability (XAI)** | Grad-CAM feature activation maps (`scripts/common/xai.py`) |

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
# Install Node dependencies
npm install

# Install Python ML dependencies (Python 3.10+)
pip install torch torchvision pillow fastapi uvicorn python-multipart librosa facenet-pytorch pyyaml
```

### 2. Running the model service (required)

The app has no local detector. Without a reachable model service every scan honestly reports
"Analysis unavailable" — it never falls back to a guess.

```bash
pip install -r scripts/requirements.txt
python scripts/inference_server.py --selfcheck   # maths only, no downloads
python scripts/inference_server.py --verify      # loads the model, prints which one
python scripts/inference_server.py               # serves on 0.0.0.0:8000
```

With no trained checkpoint it loads a verified Hugging Face face-deepfake classifier and
labels every response `hf_fallback:<id>`. Deployment (HF Endpoints / Modal / Render /
Railway) is covered in **[`scripts/README_inference.md`](scripts/README_inference.md)**.

### 3. Running the Web Application

```bash
cp .env.example .env    # set MODEL_SERVICE_URL (http://localhost:8000 for local dev)
npm run dev
```
Open **`http://localhost:3000`** in your browser.

- **Public Scan Route**: `http://localhost:3000/` — images only; video, audio and URL scans return 501.
- **Model Card**: `http://localhost:3000/model-card` — what is known and what is not.
- **Authenticated Admin Portal**: `http://localhost:3000/admin` (Protected, requires login).

---

## 🔐 Admin Portal (`/admin`)

The Admin Portal is a dedicated, authenticated route for team members to run direct model scans, inspect checkpoints, and launch background training runs.

### Accessing the Portal
1. Navigate to `http://localhost:3000/admin/login` (or access `/admin`).
2. Passphrase: Set `ADMIN_PASSWORD` env var (Default dev passphrase: `admin123`).

### Key Features
- **Direct Real Model Scan (`/api/admin/scan`)**: Posts media to `$MODEL_SERVICE_URL/predict` through the same client the public route uses, and reports the model's response verbatim. If the service is down it returns an explicit error — there is no heuristic fallback anywhere in the app.
- **Model Checkpoint Inspector (`/api/admin/models`)**: Lists all `.pth` and `.onnx` model files in `models/` or `public/models/`, displaying architecture, calibration temperature (T), class count, and validation metrics.
- **Background Training Runner (`/api/admin/train`)**: Spawns background training subprocesses for image or audio pipelines, records run parameters in `runs/<run_id>/run.json`, and streams real-time stdout/stderr logs (`/api/admin/train/<runId>/logs`).

---

## 🔬 Multi-Modal Training & Dataset Pipeline

### 1. Team Dataset Catalogue (`scripts/datasets.yaml`)
Candidate academic datasets (FaceForensics++, Celeb-DF v2, DFDC, Manjilkarki, WildDeepfake) are catalogued in `scripts/datasets.yaml`. Fill in your local path once access is granted:

```yaml
datasets:
  manjilkarki:
    name: "Kaggle Manjilkarki Deepfake Benchmark"
    local_path: "data/Dataset"
```

### 2. Image Pipeline (`preprocess_faces.py` & `train_deepfake_detector.py`)

Crops face regions with a 35% margin to preserve jawline/hairline forgery evidence and trains an EfficientNet backbone with near-duplicate group splitting:

```bash
# Step 1: Preprocess & crop faces
python scripts/preprocess_faces.py --dataset-config scripts/datasets.yaml --out data/faces

# Step 2: Train detector & evaluate on held-out dataset
python scripts/train_deepfake_detector.py --data-dir data/faces --epochs 12 --arch b0
```

### 3. Audio Pipeline (`preprocess_audio.py` & `train_audio_detector.py`)

Converts raw audio clips (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`) into 2D Mel-spectrogram images and trains a spectrogram CNN with Grad-CAM XAI output:

```bash
# Step 1: Convert audio files to Mel-spectrogram ImageFolder tree
python scripts/preprocess_audio.py --src data/audio_raw --out data/audio_spectrograms

# Step 2: Train audio detector & export Grad-CAM explainability heatmap
python scripts/train_audio_detector.py --data-dir data/audio_spectrograms --epochs 10
```

### 4. Running Self-Check Suite

Verify internal assertions for crop geometry, AUC rank-sum calculation, temperature calibration, and label mapping:

```bash
python scripts/inference_server.py --selfcheck
python scripts/preprocess_faces.py --selfcheck
python scripts/train_deepfake_detector.py --selfcheck
python scripts/preprocess_audio.py --selfcheck
python scripts/train_audio_detector.py --selfcheck
```

### 5. Measuring accuracy (nothing is claimed until this is run)

```bash
python scripts/train_deepfake_detector.py --eval-only --eval-dir <dataset from another source>
```

Put the resulting cross-dataset numbers in `app/model-card/page.tsx`, replacing the
`TBD — pending evaluation` rows. Do not fill them in from anywhere else.

---

## 📂 Codebase Layout

```
app/
  (admin)/admin/        Admin portal (login page, dashboard, layout with noindex)
  api/
    scan/route.ts       Public verification API (calls the model service; no fallback)
    admin/              Admin APIs (auth, direct real-model scan, models list, background train)
  model-card/page.tsx   Honest model card: what ran, what data, what is unmeasured
components/
  sections/             Landing page sections (Hero, HowItWorks, Architecture, TechStack, etc.)
  scan/                 Scan components (UploadZone, ReportPanel, ScoreRing)
lib/
  admin/auth.ts         JWT session token creation & validation
  models/model_service.ts  Single client for the model service + metadata signal
  store.ts              Global Zustand state
  verdict.ts            Score-to-verdict mapping rules (genuine / uncertain / manipulated)
scripts/
  common/               Shared modules (config.py — paths & thresholds, xai.py — Grad-CAM, calibration.py)
  requirements.txt      Model service dependencies
  README_inference.md   How to run and deploy the model service
  datasets.yaml         Team dataset paths catalogue
  preprocess_faces.py   MTCNN face-crop & dataset normalization
  train_deepfake_detector.py PyTorch image detector training script
  preprocess_audio.py   Audio-to-Mel-spectrogram converter
  train_audio_detector.py Audio spectrogram CNN detector training script
  inference_server.py   FastAPI inference engine (/predict endpoint)
```

---

## 🛡️ Security Features

- **Authenticated Admin Routes**: Protected by HttpOnly, Secure, SameSite=strict session cookies.
- **WAF & Rate Limiting**: Global bot UA blocking, 20 req/min rate limit on public APIs, separate 60 req/min limit on admin APIs.
- **SSRF Blocklist**: Blocks internal/link-local IP addresses (`127.0.0.1`, `10.x`, `192.168.x`, `169.254.169.254`, etc.) on URL scans.
- **Payload Guard**: 50 MB request payload ceiling.
- **HTTP Headers**: HSTS, Content Security Policy (CSP), `X-Frame-Options: SAMEORIGIN`, `nosniff`, and Referrer-Policy.

---

## 🔒 Environment Variables (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `ADMIN_PASSWORD` | `admin123` | Passphrase required to log into the Admin Portal |
| `JWT_SECRET` | `verifai-secret...` | Secret key used to sign admin session cookies |
| `MODEL_SERVICE_URL` | *(none — required)* | URL of the deployed model service, no trailing slash. Unset = every scan reports "Analysis unavailable" |
| `VERIFAI_MODEL` | `models/deepfake_detector.pth` | Checkpoint path, read by the model service (see `scripts/README_inference.md` for the rest) |
