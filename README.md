<div align="center">

# VerifAI

**Multi-modal AI media verification — images, video, and voice, with explainable results.**

Next.js 14 · TypeScript · FastAPI · ONNX Runtime · PyTorch

</div>

---

VerifAI verifies whether media was generated or manipulated by AI. It runs an ensemble of
purpose-built detectors, fuses them into a single verdict, and shows exactly which detector
contributed what — including a visual heatmap of the regions that drove the result.

It covers four surfaces from one detection core:

| Surface | What it does |
|---|---|
| **Web app** | Upload an image, video, or voice clip and get a verdict with per-detector evidence |
| **Browser extension** | Verify media in place on any page, on-device, with nothing uploaded |
| **Live Guard** | Real-time synthetic-voice monitoring during browser calls |
| **Admin portal** | Direct model scans, checkpoint inspection, and managed training runs |

---

## Highlights

- **Ensemble detection.** A face-swap classifier and a whole-image generation detector (NPR,
  CVPR 2024) answer different questions and are fused by configurable weights, so a fully
  synthetic image and a swapped face are both caught.
- **Explainable by default.** Every verdict carries per-detector scores, a plain-language
  rationale, and a Grad-CAM heatmap. No result is a bare number.
- **On-device inference.** The browser extension bundles the same models the server runs and
  scores locally — media never leaves the machine unless explicitly requested.
- **Parity-verified voice.** The voice pipeline exports its own preprocessing to ONNX, so the
  browser and server execute an identical graph. A numerical self-test confirms this at runtime
  before on-device voice is enabled.
- **Provenance enforced.** Every verdict names the model that produced it, and the client
  refuses to display a score from any model that is not part of this project.
- **Deploys torch-free.** The runtime image uses ONNX Runtime at ~205MB resident; PyTorch is
  required only for training and export.

---

## Architecture

```
 Browser / Extension
         │
         ▼
 Next.js 14 (App Router)  ── /api/scan, /api/live/*, /api/admin/*
         │                    auth · rate limiting · SSRF guard · payload ceiling
         ▼
 FastAPI model service    ── ONNX Runtime
         │
         ├── Face classifier   EfficientNet / ViT, on the detected face region
         ├── NPR               CVPR 2024, whole-image generation artifacts
         ├── Voice (v2)        raw samples → preproc.onnx → CNN
         └── YuNet             face detection
```

The web app holds no detector of its own. It calls the model service over `MODEL_SERVICE_URL`
and reports what the service returns, so there is exactly one source of truth for every score.

---

## Quick start

### 1. Model service

```bash
pip install -r scripts/requirements.txt

python scripts/inference_server.py --selfcheck   # assertions only, no downloads
python scripts/inference_server.py --verify      # load models and report what is active
python scripts/inference_server.py               # serve on 0.0.0.0:8000
```

`--verify` prints the active configuration:

```
[AUDIO v2] self-test pass (delta 3e-06)
[NPR]      onnx:npr:npr_detector.pth+int8(models/npr_detector.onnx)
[FACE]     YuNet models/face_detection_yunet.onnx
[OK]       trained_checkpoint on cpu | classes=['Fake','Real'] | faceDetector=on
[VERIFY]   explanation heatmap: produced
```

### 2. Web application

```bash
npm install
cp .env.example .env          # set MODEL_SERVICE_URL=http://localhost:8000
npm run dev
```

Open **http://localhost:3000**.

| Route | Purpose |
|---|---|
| `/` | Public verification |
| `/call` | Live Guard calling |
| `/model-card` | Model provenance, training data, and evaluation status |
| `/admin` | Authenticated operations portal |

### 3. Browser extension

```bash
cd extension
npm install
npm run build          # → extension/dist
```

Load `extension/dist` at `chrome://extensions` with Developer Mode enabled.

---

## API

| Endpoint | Input | Returns |
|---|---|---|
| `POST /predict` | image | fused verdict, per-detector signals, heatmap |
| `POST /predict-video` | video clip | sampled frames, aggregated verdict, per-frame table |
| `POST /predict-audio` | voice clip | voice verdict with supporting notes |
| `POST /predict-audio-window` | 2–4s window | VAD-gated probability for live monitoring |
| `GET /health` | — | active models, thresholds, fusion weights, self-test status |

Example response:

```json
{
  "verdict": "real",
  "confidence": 78,
  "fakeProbability": 22,
  "modelSource": "trained_checkpoint + onnx:npr:npr_detector.pth+int8",
  "faceDetected": true,
  "signals": { "modelScore": 43, "nprScore": 0, "frequencyScore": 1 },
  "fusion": { "weights": { "face": 0.5, "npr": 0.5 }, "used": { "face": 0.5, "npr": 0.5 } }
}
```

Scores are renormalized over the detectors that actually ran, so a detector that did not apply
never dilutes the result.

---

## Live Guard

Real-time monitoring for calls running in a browser tab — Google Meet, Discord, Teams, Zoom.
It reports a rolling trust score, a per-window timeline, and a sustained-signal warning, with
EMA smoothing and hysteresis so a single noisy window cannot flip the reading.

**Voice runs in one of two places, and the overlay states which:**

- **On-device (parity verified)** — audio is scored in the browser and never leaves the machine.
- **Our backend** — 3-second windows are scored by your own VerifAI service.

On-device voice is enabled only after the bundled chain reproduces a known reference
probability within tolerance on that specific machine:

```
raw samples → preproc.onnx → audio_detector.onnx → P(fake)
```

Because preprocessing lives inside the ONNX graph, browser and server compute identically. The
self-test proves it at runtime rather than assuming it, and any failure routes cleanly to the
backend path.

```bash
cd extension
npm run check:audio        # fixture, bundle, and chain via onnxruntime-node
npm run check:liveguard    # provenance, call-site detection, bundle integrity
```

---

## Configuration

**Web application**

| Variable | Purpose |
|---|---|
| `MODEL_SERVICE_URL` | Model service origin, no trailing slash. Required. |
| `JWT_SECRET` | Signs admin session cookies. |
| `NEXT_PUBLIC_CALL_FUSION_WEIGHTS` | Audio/video weighting for Live Guard. |

**Model service**

| Variable | Default | Purpose |
|---|---|---|
| `VERIFAI_ONNX` | `models/face/detector_v2.onnx` | Face classifier |
| `NPR_MODEL_PATH` | `models/npr_detector.onnx` | Whole-image generation detector |
| `AUDIO_MODEL_PATH` | `models/audio/audio_detector.onnx` | Voice model |
| `VERIFAI_AUDIO_V2_DIR` | `models/audio/v2/models/audio` | Parity-verified voice chain |
| `FUSION_WEIGHTS` | `face=0.5,npr=0.5` | Detector weighting |
| `VERIFAI_FAKE_ABOVE` / `VERIFAI_REAL_BELOW` | `70` / `30` | Verdict thresholds |
| `VIDEO_FRAME_CAP` | `16` | Frames sampled per clip |

Model paths resolve relative to the repository root, so scripts behave identically from any
working directory. Full reference: **[`scripts/README_inference.md`](scripts/README_inference.md)**.

---

## Training pipelines

Datasets are catalogued in `scripts/datasets.yaml`; add your local path once access is granted.

**Face detector**

```bash
python scripts/preprocess_faces.py --dataset-config scripts/datasets.yaml --out data/faces
python scripts/train_deepfake_detector.py --data-dir data/faces --epochs 12 --arch b0
```

Faces are cropped with a 35% margin to retain jawline and hairline evidence, and split by
source group with near-duplicate detection so the validation set stays independent.

**Whole-image detector (NPR)**

```bash
curl -L -o models/npr_detector.pth \
  https://github.com/chuangchuangtan/NPR-DeepfakeDetection/raw/main/model_epoch_last_3090.pth
python scripts/export_onnx.py --npr
```

**Voice detector**

```bash
python scripts/preprocess_audio.py --src data/audio_raw --out data/audio_spectrograms
python scripts/train_audio_detector.py --data-dir data/audio_spectrograms --epochs 10
```

**Fusion weights**

```bash
python scripts/tune_fusion.py --data-dir <labelled set>
```

Step-by-step dataset guidance is in [`youhavetodo.md`](youhavetodo.md).

---

## Verification

Every module ships an executable self-check.

```bash
# Model service and pipelines
python scripts/inference_server.py --selfcheck
python scripts/preprocess_faces.py --selfcheck
python scripts/preprocess_audio.py --selfcheck
python scripts/train_deepfake_detector.py --selfcheck
python scripts/train_audio_detector.py --selfcheck
python scripts/train_npr.py --selfcheck
python scripts/tune_fusion.py --selfcheck
python scripts/eval_call_audio.py --selfcheck
python scripts/common/audio_v2.py

# Web application
node --experimental-strip-types scripts/check-risk.mts
npx tsc --noEmit && npx next build

# Extension
cd extension && npm run typecheck && npm run selfcheck \
  && npm run check:liveguard && npm run check:audio
```

**Cross-generator evaluation**

```bash
python scripts/train_deepfake_detector.py --eval-only --eval-dir <held-out dataset>
python scripts/train_npr.py --eval-only --eval-dir <unseen generators>
python scripts/eval_call_audio.py --data-dir <held-out corpus>
```

`eval_call_audio.py` measures the voice model under real call conditions — G.711 companding,
the 3.4kHz telephone band, packet loss, and line noise — which is the figure that describes
Live Guard in production.

---

## Scope and evaluation

**Supported inputs.** Images, short video clips, and voice recordings. Video is sampled rather
than decoded in full, with the frame budget set by `VIDEO_FRAME_CAP`. Live Guard monitors calls
that play in a browser tab.

**Verdict semantics.** Results are `real`, `fake`, or `uncertain`. The `uncertain` band is a
deliberate outcome for scores between the thresholds, so borderline cases are surfaced for
review rather than forced to a side.

**Evaluation status.** Benchmark figures are published on `/model-card` as each evaluation
completes, and the card distinguishes measured results from those still in progress. The
commands above reproduce every number it reports, which keeps published accuracy traceable to
a specific dataset and run — the standard the model card is built to hold.

---

## Security

- HttpOnly, Secure, SameSite=strict session cookies for all admin routes
- Rate limiting: 20 req/min public APIs, 15 req/min extension, 60 req/min admin, with bot
  user-agent filtering
- SSRF blocklist covering loopback, private, and link-local ranges
- 50 MB payload ceiling
- HSTS, Content Security Policy, `X-Frame-Options: SAMEORIGIN`, `nosniff`, Referrer-Policy
- Extension ships no remote code: all models and the ONNX Runtime WebAssembly are packaged

---

## Project layout

```
app/
  api/scan/            Public verification API
  api/live/            Live Guard frame and audio-window endpoints
  api/admin/           Auth, direct scan, model inspection, training runs
  call/                WebRTC calling with live voice monitoring
  model-card/          Model provenance and evaluation status
components/            Landing sections, scan UI, call UI
lib/
  call/risk.ts         Shared risk engine (web app and extension)
  models/              Model service client
  verdict.ts           Score-to-verdict mapping
scripts/
  inference_server.py  FastAPI service
  common/              Shared config, calibration, XAI, audio v2 chain
  train_*.py           Training pipelines
  export_onnx.py       ONNX export and quantization
extension/
  src/background/      Service worker, Live Guard session
  src/offscreen/       On-device inference and audio capture
  src/shared/          Provenance, VAD, call sites, settings
models/
  face/ images/ audio/ Trained detectors and metadata
```

---

## Documentation

| Document | Contents |
|---|---|
| [`scripts/README_inference.md`](scripts/README_inference.md) | Model service operation and deployment |
| [`extension/README.md`](extension/README.md) | Extension architecture and on-device inference |
| [`extension/STORE.md`](extension/STORE.md) | Store listing and privacy disclosures |
| [`youhavetodo.md`](youhavetodo.md) | Dataset acquisition and training walkthrough |
| [`flow.md`](flow.md) | Request routing across the stack |
