# VerifAI model service

FastAPI service that runs the detectors and returns an honest verdict. The Next.js app talks to
it over `MODEL_SERVICE_URL`; it never runs a model itself.

| Endpoint | Input | Detectors |
|---|---|---|
| `POST /predict` | one image | face classifier (on the crop) + NPR (whole image) + FFT, fused |
| `POST /predict-video` | short clip | the same, per sampled frame, then averaged |
| `POST /predict-audio` | voice clip | voice model — **501 until you train one** |
| `GET /health` `GET /` | — | which models loaded, weights, thresholds |

Each detector is optional. A missing one is reported as unavailable and the rest carry on; with
none of them loaded the service returns 503 rather than a guess.

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
face is found it abstains, and NPR decides on its own — a face model on a landscape produces a
number, not evidence. With neither available the verdict is `uncertain` and the response says why.

NPR loads separately from `NPR_MODEL_PATH`, and the voice model from `AUDIO_MODEL_PATH`.

### The voice model has two paths, and one of them has to earn its place

```
v2 (preferred)  raw samples -> models/audio/v2/.../preproc.onnx -> audio_detector.onnx
v1 (fallback)   audio file  -> librosa mel-spectrogram (Python only) -> audio_detector.onnx
```

v2 exists so the browser can run the identical graph — librosa cannot, and a JavaScript
mel-spectrogram that is subtly wrong yields confident nonsense. Because the preprocessing is
part of the graph, the same chain works on both sides.

The server does not take that on trust. At startup it runs `audio_selftest.json` through
`preproc -> cnn` and compares against the probability the training pipeline recorded:

```
[AUDIO v2] self-test pass (delta 3e-06)
```

Only on a pass does v2 become the live path. On a failure — or a missing `.onnx.data` sidecar,
the usual cause — it logs why and v1 keeps serving. `/health` reports both:

```json
"audio": { "active": "v2", "selftest": "pass",
           "v2": { "selftest": { "delta": 3e-06, "tol": 0.001 } } }
```

`active` is `"v2"`, `"v1"` or `"none"`; `"none"` means `/predict-audio` returns 501. The
extension gates its on-device audio on the same fixture, so a machine that cannot reproduce the
number falls back to this service instead of scoring locally.

## Response

```json
{
  "verdict": "real | fake | uncertain",
  "confidence": 81,
  "fakeProbability": 81,
  "modelSource": "onnx:hf_fallback:…+int8 + onnx:npr:…+int8",
  "detectors": { "face": "onnx:…", "npr": "onnx:npr:…" },
  "faceDetected": true,
  "signals": { "modelScore": 81, "nprScore": 100, "frequencyScore": 1 },
  "fusion": { "weights": {"face": 0.5, "npr": 0.5, "frequency": 0.0},
              "used":    {"face": 0.5, "npr": 0.5} },
  "heatmap": "data:image/png;base64,…",
  "notes": ["confidence is uncalibrated — treat it as a ranking, not a probability"]
}
```

Video adds a `video` block (frames, duration, max frame score, peak timestamp, temporal variance,
per-frame table); audio replaces the image signals with `audioScore`.

- `fakeProbability` — the fused number the verdict is based on. Bands: `>70` fake, `<30` real,
  else uncertain.
- `modelScore` — face classifier P(fake). `null` when it abstained (no face).
- `nprScore` — NPR whole-image P(generated). `null` when the model is absent.
- `frequencyScore` — share of FFT energy above half-Nyquist. A **descriptive statistic, not a
  probability**: weight 0 by default, so it is reported and never fused. A sharp camera photo
  scores high too.
- `fusion.used` — exactly which detectors voted, and with what weight. A detector that did not
  run does not appear, and its absence does not drag the mean toward 50.
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
| `NPR_MODEL_PATH` | `models/npr_detector.onnx` | NPR whole-image AI-generation detector |
| `AUDIO_MODEL_PATH` | `models/audio/audio_detector.onnx` | v1 voice model; absent = `/predict-audio` 501s |
| `VERIFAI_AUDIO_V2_DIR` | `models/audio/v2/models/audio` | the v2 chain; used only if its self-test passes |
| `FUSION_WEIGHTS` | `face=0.5,npr=0.5,frequency=0.0` | how the detectors combine |
| `VIDEO_FRAME_CAP` / `VIDEO_MAX_SECONDS` | `16` / `60` | frames sampled per clip, length ceiling |
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

## Adding detectors

```bash
# NPR: official ProGAN weights (no training needed), or train your own first
curl -L -o models/npr_detector.pth \
  https://github.com/chuangchuangtan/NPR-DeepfakeDetection/raw/main/model_epoch_last_3090.pth
python scripts/export_onnx.py --npr        # -> models/npr_detector.onnx (1.5MB int8)

# voice: train, then export
python scripts/train_audio_detector.py --data-dir data/audio_spectrograms
python scripts/export_onnx.py --audio      # -> models/audio_detector.onnx
```

The Docker build does the NPR step for you. Every export verifies itself against the torch model
and fails the build on disagreement.

## Limits — read before trusting a number

- The face classifier only knows face swaps; NPR is what catches fully generated images. With
  NPR absent, a StyleGAN portrait is judged by a model that was never trained to see one.
- Video re-encoding weakens NPR: measured, an AI still scoring 100 scored 0 inside an mp4.
- The HF fallback's accuracy on your data is **unmeasured**. Its card claims a number for its own
  test split; that is not a cross-dataset result and this repo does not repeat it.
- Cross-dataset accuracy for our own checkpoint: **TBD — pending evaluation**
  (`train_deepfake_detector.py --eval-only --eval-dir <a different dataset>`).
- Compression, resizing and screenshots degrade every one of these signals.
