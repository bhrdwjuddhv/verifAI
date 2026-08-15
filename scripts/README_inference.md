# VerifAI model service

FastAPI service that runs a real image deepfake classifier and returns an honest verdict.
The Next.js app talks to it over `MODEL_SERVICE_URL`; it never runs a model itself.

**Images only.** Video and audio are Phase 4 — those uploads get a `415`, not a verdict.

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
| No checkpoint | `hf_fallback:<id>` — default `prithivMLmods/Deep-Fake-Detector-v2-Model` |
| Neither loads | `/predict` returns **503**. It does not guess. |

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

All four read `$PORT` and want the same command:

```
uvicorn inference_server:create_app --factory --host 0.0.0.0 --port $PORT
```

(run it from `scripts/`, or `cd scripts` in the start command).

- **Hugging Face Inference Endpoints** — closest fit if you rely on the HF fallback: the weights are
  already cached in the image. Push `scripts/` as a Space (Docker SDK) or use a custom handler.
- **Modal** — best for GPU-on-demand and cold-start-free model loading; wrap `create_app()` in an
  `@modal.asgi_app()`. Cheapest option if traffic is bursty.
- **Render / Railway** — plain container. Build `pip install -r scripts/requirements.txt`, start with
  the command above. CPU-only inference on a ViT-base is ~1–2s per image; fine for demos.

Whichever you pick: bake the model into the image or mount a volume, otherwise every cold start
re-downloads ~350MB from Hugging Face.

Then set `MODEL_SERVICE_URL` in the Next.js app to the deployed URL (no trailing slash).

## Limits — read before trusting a number

- Trained/evaluated on face datasets. Non-face images get `uncertain`, by design.
- The HF fallback's accuracy on your data is **unmeasured**. Its card claims a number for its own
  test split; that is not a cross-dataset result and this repo does not repeat it.
- Cross-dataset accuracy for our own checkpoint: **TBD — pending evaluation**
  (`train_deepfake_detector.py --eval-only --eval-dir <a different dataset>`).
- Compression, resizing and screenshots degrade every one of these signals.
