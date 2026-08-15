# VerifAI model service

FastAPI service that runs a real image deepfake classifier and returns an honest verdict.
The Next.js app talks to it over `MODEL_SERVICE_URL`; it never runs a model itself.

**Images only.** Video and audio are Phase 4 — those uploads get a `415`, not a verdict.

## Two builds

| | torch build | **lean build** (default Docker target) |
|---|---|---|
| Model | fp32 via PyTorch / transformers | same weights, int8 ONNX |
| Face detector | MTCNN (facenet-pytorch) | YuNet (OpenCV, 230KB) |
| Explanation | Grad-CAM | occlusion saliency |
| **Measured RSS** | **1262MB** (peak 1268) | **205MB** (peak 306) |
| Latency, 1 image, 2 threads | ~1–2s | ~5.6s with saliency |
| Fits a 512MB instance | no | yes |

Same API, same verdict logic, same thresholds. `modelSource` says which one answered
(`onnx:...+int8` for the lean build), and the lean build adds a note that int8 shifts scores
by a point or two. Measured on the same photo: 83% fp32 vs 81% int8.

## Run it locally

```bash
pip install -r scripts/requirements.txt

python scripts/inference_server.py --selfcheck   # maths only, no model download
python scripts/inference_server.py --verify      # loads the active model, prints which one
python scripts/inference_server.py               # serves on 0.0.0.0:8000
```

`--verify` is the one to run first: it proves a real model loaded before you wire anything to it.

## Which model is active

| Condition | `modelSource` |
|---|---|
| `models/deepfake_detector.pth` exists (from `train_deepfake_detector.py`) | `trained_checkpoint` |
| `models/detector.onnx` exists (from `export_onnx.py`) | `onnx:<source>[+int8]` |
| Neither file | `hf_fallback:<id>` — default `prithivMLmods/Deep-Fake-Detector-v2-Model` |
| None of them loads | `/predict` returns **503**. It does not guess. |

The fallback is a **face** deepfake classifier (ViT, 224px, labels `Realism`/`Deepfake`). If no
face is found in the upload, the verdict is `uncertain` and the response says why — a face model
on a landscape produces a number, not evidence.

## Response

```json
{
  "verdict": "real | fake | uncertain",
  "confidence": 88,
  "modelSource": "hf_fallback:prithivMLmods/Deep-Fake-Detector-v2-Model",
  "faceDetected": true,
  "signals": { "modelScore": 88, "frequencyScore": 14 },
  "notes": ["confidence is uncalibrated — treat it as a ranking, not a probability"]
}
```

- `modelScore` — P(fake) in percent, from the model. Bands: `>70` fake, `<30` real, else uncertain.
- `frequencyScore` — share of FFT energy above half-Nyquist. A **descriptive statistic, not a
  probability**, reported alongside the verdict and never fused into it. A sharp camera photo
  scores high too.
- `confidence` — probability of the reported verdict. Calibrated only when the checkpoint carries a
  fitted temperature (`calibrated: true` on `/health`); the HF fallback is uncalibrated.
- `notes` — everything that qualifies the result. Show them.

`GET /health` reports the active model, its classes, device, thresholds and any load error.

## Environment

| Var | Default | Meaning |
|---|---|---|
| `PORT` / `HOST` | `8000` / `0.0.0.0` | bind address |
| `VERIFAI_MODEL` | `models/deepfake_detector.pth` | checkpoint path (shared with the trainer) |
| `VERIFAI_HF_MODEL` | `prithivMLmods/Deep-Fake-Detector-v2-Model` | fallback classifier |
| `VERIFAI_HF_EXPECTS_FACE` | `1` | set `0` if you swap in a whole-image AI detector |
| `VERIFAI_FAKE_ABOVE` / `VERIFAI_REAL_BELOW` | `70` / `30` | verdict bands |
| `VERIFAI_FACE_MARGIN` | `0.35` | crop margin; match the training manifest |
| `VERIFAI_CORS_ORIGINS` | `*` | comma-separated origins |

## Deploy

`scripts/Dockerfile` is multi-stage: stage 1 installs torch only long enough to export and verify
an int8 ONNX model, then throws it away. The shipped image has no torch at all.

```bash
docker build -t verifai-model ./scripts          # lean, ~205MB RSS
docker build --target torch -t verifai-torch ./scripts   # fp32 + Grad-CAM, ~1.2GB RSS
docker run -p 8000:8000 verifai-model
```

The export step fails the build if the ONNX model disagrees with the torch model it came from, so
a bad quantization cannot ship silently.

- **Render / Railway / Fly free tiers (512MB)** — the lean image fits. Give it 1 worker.
- **Google Cloud Run** — scale-to-zero, free request allowance:
  `gcloud run deploy verifai-model --source scripts --memory 1Gi --allow-unauthenticated`
- **Local + tunnel** — for a demo: run the service and expose it with ngrok or localtunnel. The
  Next.js client already sends both services' skip-interstitial headers.

Every host injects `$PORT`; the image's `CMD` already reads it.

### Tuning the lean build

| Var | Default | Effect |
|---|---|---|
| `VERIFAI_THREADS` | `2` | ONNX intra-op threads. Match your instance's vCPUs. |
| `VERIFAI_BATCH` | `8` | Forward-pass chunk. Lower = less peak RAM, slower. |
| `VERIFAI_SALIENCY_GRID` | `5` | Occlusion grid. 5 = 25 extra forwards (~4s); 7 is sharper and ~2x slower. |
| `VERIFAI_GRADCAM` | `1` | `0` disables the heatmap entirely. |

Measured on one dev core, one image, lean build:

| Setting | Time |
|---|---|
| `VERIFAI_GRADCAM=0` (no heatmap) | **0.96s** |
| `VERIFAI_SALIENCY_GRID=3` | 4.4s |
| `VERIFAI_SALIENCY_GRID=5` | 5.6s |

The heatmap is ~80% of the request. A Render **free** instance gets 0.1 CPU — roughly 10x
slower — so a scan with saliency lands near a minute there and the platform's proxy gives up
with a 502. On free tiers set `VERIFAI_GRADCAM=0`; the verdict still takes ~10s.

`/predict` is a sync endpoint on purpose, so FastAPI runs it in a threadpool and `/health`
keeps answering during a scan. As an `async def` it blocked the event loop, health checks
timed out, and the platform restarted the container mid-request.

Then set `MODEL_SERVICE_URL` in the Next.js app to the deployed URL (no trailing slash).

## Limits — read before trusting a number

- Trained/evaluated on face datasets. Non-face images get `uncertain`, by design.
- The HF fallback's accuracy on your data is **unmeasured**. Its card claims a number for its own
  test split; that is not a cross-dataset result and this repo does not repeat it.
- Cross-dataset accuracy for our own checkpoint: **TBD — pending evaluation**
  (`train_deepfake_detector.py --eval-only --eval-dir <a different dataset>`).
- Compression, resizing and screenshots degrade every one of these signals.
