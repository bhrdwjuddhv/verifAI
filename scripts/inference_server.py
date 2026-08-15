"""VerifAI image deepfake inference service.

Images only. Video and audio are Phase 4 — those requests are rejected rather than
run through an image model and reported as a verdict.

    python scripts/inference_server.py              # serves on 0.0.0.0:$PORT (default 8000)
    python scripts/inference_server.py --selfcheck  # maths only: numpy + Pillow, no downloads
    python scripts/inference_server.py --verify     # actually load the active model, then exit

Model selection, in order:
  1. models/deepfake_detector.pth from train_deepfake_detector.py  -> "trained_checkpoint"
  2. models/detector.onnx from export_onnx.py (torch-free build)   -> "onnx:<source>[+int8]"
  3. a verified Hugging Face classifier                            -> "hf_fallback:<id>"
  4. none loads -> /predict returns 503. It never guesses.
"""

import argparse
import io
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image, UnidentifiedImageError

# scripts/ is on sys.path because this file lives there.
from common.config import (
    FAKE_ABOVE,
    HF_FALLBACK_EXPECTS_FACE,
    HF_FALLBACK_MODEL,
    MODEL_PATH,
    REAL_BELOW,
)
from common.xai import encode_overlay, gradcam_overlay

# Torch-free build artifacts (see export_onnx.py and the Dockerfile's lean stage).
ONNX_PATH = os.environ.get("VERIFAI_ONNX", os.path.join("models", "detector.onnx"))
YUNET_PATH = os.environ.get("VERIFAI_YUNET", os.path.join("models", "face_detection_yunet.onnx"))

MAX_BYTES = 50 * 1024 * 1024
FACE_MARGIN = float(os.environ.get("VERIFAI_FACE_MARGIN", "0.35"))
GRADCAM = os.environ.get("VERIFAI_GRADCAM", "1") != "0"

# Class names, from a checkpoint or from a HF config, say which side is which. Matching on
# the name beats trusting an index: ImageFolder sorts alphabetically and 'Fake' < 'Real',
# but a HF model can order its labels however it likes, and an inverted mapping reads as a
# confidently wrong verdict rather than as a bug.
FAKE_WORDS = ("fake", "deepfake", "artificial", "synthetic", "spoof", "gan", "generated")
REAL_WORDS = ("real", "realism", "human", "authentic", "genuine", "natural", "pristine")


def label_is_fake(name):
    """True / False / None when the label name commits to neither."""
    n = name.lower()
    # Real first: 'realism' contains no fake word, but 'deepfake' contains no real word
    # either, so order only matters for hybrids like 'real_vs_fake' — which is a folder
    # name, not a class, and should fall through to None rather than pick a side.
    real_hit = any(w in n for w in REAL_WORDS)
    fake_hit = any(w in n for w in FAKE_WORDS)
    if real_hit and fake_hit:
        return None
    if real_hit:
        return False
    if fake_hit:
        return True
    return None


def fake_probability(probs, labels):
    """P(fake) renormalized over the classes whose names commit to a side."""
    fake = sum(p for p, l in zip(probs, labels) if label_is_fake(l) is True)
    real = sum(p for p, l in zip(probs, labels) if label_is_fake(l) is False)
    if fake + real <= 0:
        raise RuntimeError(f"no class in {list(labels)} is identifiable as real or fake")
    return fake / (fake + real)


def fake_index(labels):
    """Index of the fake class — the one Grad-CAM explains."""
    for i, name in enumerate(labels):
        if label_is_fake(name) is True:
            return i
    raise RuntimeError(f"no fake class in {list(labels)}")


def verdict_for(fake_pct):
    """(verdict, confidence-in-that-verdict). The middle band is an answer, not a failure."""
    if fake_pct > FAKE_ABOVE:
        return "fake", fake_pct
    if fake_pct < REAL_BELOW:
        return "real", 100.0 - fake_pct
    return "uncertain", max(fake_pct, 100.0 - fake_pct)


def frequency_score(img):
    """Share of spectral energy above half-Nyquist, 0-100.

    A descriptive statistic, NOT a probability and NOT part of the verdict. Diffusion
    upsamplers and GAN checkerboarding leave energy up here — but so does a sharp camera,
    and JPEG strips it from real and fake alike. Reported so a human can weigh it.
    """
    g = np.asarray(img.convert("L").resize((256, 256), Image.BILINEAR), dtype=np.float32) / 255.0
    g = g - g.mean()
    # Hann window: without it the image border is a step edge that dumps energy into every
    # frequency, i.e. straight into the number we are trying to measure.
    w = np.hanning(256)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(g * np.outer(w, w)))) ** 2
    y, x = np.ogrid[-128:128, -128:128]
    radius = np.sqrt(x * x + y * y)
    total = float(mag.sum())
    if total <= 0:
        return 0.0
    return float(100.0 * mag[radius > 64].sum() / total)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def to_array(img, size, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """PIL image -> (1, 3, size, size) float32, normalized. numpy only — no torch."""
    arr = np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return arr.transpose(2, 0, 1)[None].astype(np.float32)


def to_tensor(img, size):
    import torch

    return torch.from_numpy(to_array(img, size)[0])


def load_checkpoint_engine():
    """The model we trained. Preprocessing settings ride along in the checkpoint."""
    import torch
    import torch.nn as nn
    from torchvision import models

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    classes = ckpt["classes"]
    size = int(ckpt.get("img_size", 224))
    temperature = float(ckpt.get("temperature", 1.0))

    ctor = {"b0": models.efficientnet_b0, "b4": models.efficientnet_b4}[ckpt.get("arch", "b0")]
    model = ctor()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    fake_i = fake_index(classes)

    def predict(img):
        t = to_tensor(img, size).unsqueeze(0).to(device)
        with torch.no_grad():
            batch = torch.cat([t, torch.flip(t, dims=[3])])  # flip TTA
            probs = torch.softmax(model(batch).float() / temperature, dim=1).mean(dim=0).tolist()
        heatmap = None
        if GRADCAM:
            heatmap = gradcam_overlay(model, t, fake_i, img.resize((size, size), Image.BILINEAR))
        return fake_probability(probs, classes), heatmap

    meta = {
        "modelSource": "trained_checkpoint",
        "classes": classes,
        "expectsFace": bool(ckpt.get("face_crop", False)),
        "calibrated": temperature != 1.0,
        "device": str(device),
        "valMetrics": ckpt.get("val_metrics"),
    }
    return predict, meta


def load_onnx_engine():
    """The lean build: ONNX Runtime, no torch, no transformers.

    torch plus a ViT-base measures ~1.2GB resident, which does not fit a 512MB instance. This
    path is the same model exported by export_onnx.py, typically int8-quantized. Grad-CAM needs
    gradients that ONNX Runtime does not provide, so explainability here is occlusion saliency:
    hide a patch, see how much the fake score drops. Slower, coarser, and it measures the
    model's actual behaviour rather than its internal activations.
    """
    import onnxruntime as ort

    with open(os.path.splitext(ONNX_PATH)[0] + ".json") as fh:
        meta_file = json.load(fh)

    labels = meta_file["labels"]
    size = int(meta_file.get("imgSize", 224))
    mean = meta_file.get("mean", IMAGENET_MEAN)
    std = meta_file.get("std", IMAGENET_STD)
    temperature = float(meta_file.get("temperature", 1.0))
    fake_i = fake_index(labels)

    options = ort.SessionOptions()
    # A ViT's attention buffers scale with batch size, and ORT's arena keeps the high-water
    # mark for the process lifetime. Both settings trade a little speed for a much lower
    # resident ceiling, which is the entire point of this build.
    options.enable_cpu_mem_arena = False
    options.intra_op_num_threads = int(os.environ.get("VERIFAI_THREADS", "2"))
    session = ort.InferenceSession(ONNX_PATH, sess_options=options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    chunk = int(os.environ.get("VERIFAI_BATCH", "8"))

    def run(batch):
        outs = [session.run(None, {input_name: batch[i:i + chunk]})[0] for i in range(0, len(batch), chunk)]
        logits = np.concatenate(outs).astype(np.float32) / temperature
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(img):
        arr = to_array(img, size, mean, std)
        probs = run(np.concatenate([arr, arr[:, :, :, ::-1]]))  # flip TTA
        heatmap = None
        if GRADCAM:
            heatmap = occlusion_overlay(run, arr, fake_i, img.resize((size, size), Image.BILINEAR))
        return fake_probability(probs.mean(axis=0).tolist(), labels), heatmap

    source = meta_file.get("source", "onnx")
    if meta_file.get("quantized"):
        source += "+int8"
    meta = {
        "modelSource": f"onnx:{source}",
        "classes": labels,
        "expectsFace": bool(meta_file.get("expectsFace", True)),
        "calibrated": bool(meta_file.get("calibrated", False)),
        "device": "cpu (onnxruntime)",
        "valMetrics": meta_file.get("valMetrics"),
        "quantized": bool(meta_file.get("quantized")),
    }
    return predict, meta


SALIENCY_GRID = int(os.environ.get("VERIFAI_SALIENCY_GRID", "5"))


def occlusion_overlay(run, arr, fake_i, base_image, grid=SALIENCY_GRID):
    """Saliency by occlusion: how far does P(fake) fall when each patch is hidden?

    One batched forward of grid² masked copies. Patches are filled with 0 — which is the
    dataset mean after normalization, i.e. "no information" rather than "black square".
    """
    base = float(run(arr)[0, fake_i])
    cells = np.repeat(arr, grid * grid, axis=0)
    side = arr.shape[-1]
    step = side / grid
    for k in range(grid * grid):
        r, c = divmod(k, grid)
        y0, y1 = int(r * step), int((r + 1) * step)
        x0, x1 = int(c * step), int((c + 1) * step)
        cells[k, :, y0:y1, x0:x1] = 0.0

    drops = base - run(cells)[:, fake_i]
    saliency = np.clip(drops.reshape(grid, grid), 0, None)
    peak = saliency.max()
    if peak <= 0:
        return None  # nothing changed the score: no honest heatmap to draw
    return encode_overlay(saliency / peak, base_image)


def load_hf_engine():
    """Explicit, labelled fallback. Not our model, and the response says so."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(HF_FALLBACK_MODEL)
    model = AutoModelForImageClassification.from_pretrained(HF_FALLBACK_MODEL).to(device).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    if all(label_is_fake(l) is None for l in labels):
        raise RuntimeError(f"{HF_FALLBACK_MODEL} labels {labels} do not say which class is fake")

    fake_i = fake_index(labels)

    def predict(img):
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits[0].float(), dim=0).tolist()
        heatmap = None
        if GRADCAM:
            pixel_values = inputs["pixel_values"]
            side = pixel_values.shape[-1]
            heatmap = gradcam_overlay(model, pixel_values, fake_i, img.resize((side, side), Image.BILINEAR))
        return fake_probability(probs, labels), heatmap

    meta = {
        "modelSource": f"hf_fallback:{HF_FALLBACK_MODEL}",
        "classes": labels,
        "expectsFace": HF_FALLBACK_EXPECTS_FACE,
        "calibrated": False,  # nobody temperature-scaled this on our data
        "device": str(device),
        "valMetrics": None,
    }
    return predict, meta


def load_engine():
    if os.path.exists(MODEL_PATH):
        print(f"[MODEL] checkpoint {MODEL_PATH}")
        return load_checkpoint_engine()
    if os.path.exists(ONNX_PATH):
        print(f"[MODEL] ONNX {ONNX_PATH} (torch-free build)")
        return load_onnx_engine()
    print(f"[MODEL] no checkpoint at {MODEL_PATH} -> Hugging Face fallback {HF_FALLBACK_MODEL}")
    return load_hf_engine()


class YuNetDetector:
    """OpenCV's YuNet, wrapped to look like MTCNN so crop_face() works unchanged.

    A 340KB ONNX model against facenet-pytorch's torch dependency. This is what makes the
    lean image possible — the classifier is not the only thing that was dragging torch in.
    """

    def __init__(self, model_path):
        import cv2

        self.cv2 = cv2
        self.net = cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_threshold=0.7)

    def detect(self, img):
        arr = np.asarray(img.convert("RGB"))[:, :, ::-1]  # PIL RGB -> OpenCV BGR
        self.net.setInputSize((arr.shape[1], arr.shape[0]))
        _, faces = self.net.detect(arr)
        if faces is None or len(faces) == 0:
            return None, None
        # YuNet rows are [x, y, w, h, ...landmarks..., score]; largest face first.
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0][:4]
        return [[x, y, x + w, y + h]], [float(faces[0][-1])]


def load_face_detector():
    """MTCNN when torch is present, YuNet when it is not, None when neither is."""
    try:
        from preprocess_faces import load_detector

        return load_detector(None)
    except BaseException as e:  # load_detector calls sys.exit when facenet-pytorch is absent
        print(f"[INFO] MTCNN unavailable ({str(e).splitlines()[0]})")

    if os.path.exists(YUNET_PATH):
        try:
            print(f"[FACE] YuNet {YUNET_PATH}")
            return YuNetDetector(YUNET_PATH)
        except Exception as e:
            print(f"[WARN] YuNet failed to load ({e})")

    print("[WARN] no face detector available")
    return None


PREDICT, META, LOAD_ERROR, DETECTOR = None, {}, None, None


def startup():
    global PREDICT, META, LOAD_ERROR, DETECTOR
    try:
        PREDICT, META = load_engine()
    except Exception as e:
        LOAD_ERROR = f"{type(e).__name__}: {e}"
        print(f"[ERROR] no model loaded — /predict will return 503. {LOAD_ERROR}")
        return
    DETECTOR = load_face_detector()
    print(f"[OK] {META['modelSource']} on {META['device']} | classes={META['classes']} "
          f"| expectsFace={META['expectsFace']} | faceDetector={'on' if DETECTOR else 'off'}")


def analyze(image):
    """The whole verdict, as a dict. Kept out of the endpoint so it stays testable."""
    from preprocess_faces import crop_face

    notes = []
    face_detected, crop = None, None
    if DETECTOR is not None:
        try:
            crop = crop_face(DETECTOR, image, FACE_MARGIN)
            face_detected = crop is not None
        except Exception as e:
            notes.append(f"face detection failed: {e}")

    frequency = round(frequency_score(image))

    if META["expectsFace"] and face_detected is False:
        # A face-deepfake model on a landscape produces a number, not evidence.
        return {
            "verdict": "uncertain",
            "confidence": 0,
            "modelSource": META["modelSource"],
            "faceDetected": False,
            "signals": {"modelScore": None, "frequencyScore": frequency},
            "heatmap": None,
            "notes": notes + ["no face detected; the active model only applies to faces"],
        }

    target = crop if (META["expectsFace"] and crop is not None) else image
    if META["expectsFace"] and DETECTOR is None:
        notes.append("face detector unavailable; ran the full frame through a face model")

    fake_prob, heatmap = PREDICT(target)
    # Band the number we are going to SHOW. Deciding on 70.4 and displaying 70 next to a
    # rule that says "above 70 is fake" is a contradiction the reader can see.
    fake_pct = float(round(100.0 * fake_prob))
    verdict, confidence = verdict_for(fake_pct)
    if not META["calibrated"]:
        notes.append("confidence is uncalibrated — treat it as a ranking, not a probability")
    if heatmap is None and GRADCAM:
        notes.append("no explanation heatmap available for this model")
    if META.get("quantized"):
        notes.append("int8-quantized build; scores can differ from the full-precision model by a point or two")

    return {
        "verdict": verdict,
        "confidence": round(confidence),
        "modelSource": META["modelSource"],
        "faceDetected": face_detected,
        "signals": {"modelScore": int(fake_pct), "frequencyScore": frequency},
        "heatmap": heatmap,
        "notes": notes,
    }


def create_app():
    """App factory. Deploy with: uvicorn inference_server:create_app --factory --host 0.0.0.0"""
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware

    if PREDICT is None and LOAD_ERROR is None:
        startup()

    app = FastAPI(title="VerifAI Deepfake Inference", version="3.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("VERIFAI_CORS_ORIGINS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    # Platforms probe "/" by default (Render logs it as HEAD / -> 404 and calls the service
    # unhealthy), so the root is the same handler rather than a 404 waiting to cause a restart.
    @app.get("/")
    def health():
        return {"status": "ok" if PREDICT else "no_model", "error": LOAD_ERROR,
                "faceDetector": DETECTOR is not None,
                "thresholds": {"fakeAbove": FAKE_ABOVE, "realBelow": REAL_BELOW}, **META}

    # Deliberately `def`, not `async def`: inference is CPU-bound and would otherwise block
    # the event loop for its whole duration, starving the platform's health checks until it
    # restarts the container mid-request. FastAPI runs sync endpoints in a threadpool.
    @app.post("/predict")
    def predict_image(file: UploadFile = File(...)):
        if PREDICT is None:
            raise HTTPException(status_code=503, detail=f"No model loaded: {LOAD_ERROR}")

        contents = file.file.read()
        if len(contents) > MAX_BYTES:
            raise HTTPException(status_code=413, detail="File exceeds 50MB limit.")
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            # TODO(Phase 4): video (frame sampling) and audio (mel-spectrogram + AASIST).
            raise HTTPException(status_code=415, detail="Images only in this version.")

        try:
            return analyze(image)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return app


def selfcheck():
    """numpy + Pillow only. Covers the label mapping, the bands and the FFT statistic."""
    assert label_is_fake("Fake") is True
    assert label_is_fake("Deepfake") is True
    assert label_is_fake("artificial") is True
    assert label_is_fake("Real") is False
    assert label_is_fake("Realism") is False, "the HF fallback's real class"
    assert label_is_fake("human") is False
    assert label_is_fake("class_0") is None
    assert label_is_fake("real_vs_fake") is None, "ambiguous names must not pick a side"

    # Label order must not change the answer — an inverted mapping is the bug this prevents.
    assert abs(fake_probability([0.2, 0.8], ["Realism", "Deepfake"]) - 0.8) < 1e-9
    assert abs(fake_probability([0.8, 0.2], ["Fake", "Real"]) - 0.8) < 1e-9
    assert abs(fake_probability([0.5, 0.25, 0.25], ["junk", "Fake", "Real"]) - 0.5) < 1e-9
    try:
        fake_probability([1.0], ["class_0"])
        raise AssertionError("unmappable labels must raise, not default to a verdict")
    except RuntimeError:
        pass

    assert verdict_for(90.0)[0] == "fake"
    assert verdict_for(5.0) == ("real", 95.0)
    assert verdict_for(50.0)[0] == "uncertain"
    assert verdict_for(70.0)[0] == "uncertain", "boundary is inclusive of uncertainty"
    assert verdict_for(30.0)[0] == "uncertain"

    # A smooth gradient is nearly all low frequency; per-pixel noise is not.
    ramp = Image.fromarray(np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1)))
    rng = np.random.default_rng(0)
    noise = Image.fromarray(rng.integers(0, 256, (256, 256), dtype=np.uint8))
    flat = Image.new("L", (256, 256), 128)
    assert frequency_score(ramp) < 5, frequency_score(ramp)
    assert frequency_score(noise) > 40, frequency_score(noise)
    assert frequency_score(flat) == 0.0, "a constant image has no energy to apportion"

    assert fake_index(["Realism", "Deepfake"]) == 1
    assert fake_index(["Fake", "Real"]) == 0

    # Occlusion saliency, with a stub model that only reacts to the top-left corner: the
    # overlay must appear there, and a model that ignores every patch must yield no heatmap
    # rather than a picture of nothing.
    probe = np.zeros((1, 3, 32, 32), dtype=np.float32)

    def corner_run(batch):
        # P(fake) is high unless the top-left cell is blanked.
        lit = batch[:, 0, 0:6, 0:6].reshape(len(batch), -1).any(axis=1)
        return np.stack([1.0 - lit * 0.9, 0.1 + lit * 0.9], axis=1).astype(np.float32)

    probe[:, :, 0:6, 0:6] = 1.0
    url = occlusion_overlay(corner_run, probe, 1, Image.new("RGB", (32, 32)), grid=4)
    assert url and url.startswith("data:image/png;base64,")
    flat = occlusion_overlay(lambda b: np.tile([[0.5, 0.5]], (len(b), 1)).astype(np.float32),
                             probe, 1, Image.new("RGB", (32, 32)), grid=4)
    assert flat is None, "a model whose score never moves has nothing to explain"

    from common.xai import selfcheck as xai_selfcheck

    xai_selfcheck()
    print("selfcheck passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true", help="run assertions and exit")
    ap.add_argument("--verify", action="store_true", help="load the active model and exit")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return

    if args.verify:
        startup()
        if PREDICT is None:
            sys.exit(f"FAILED to load a model: {LOAD_ERROR}")
        probe = Image.new("RGB", (256, 256), (127, 127, 127))
        prob, heatmap = PREDICT(probe)
        print(f"[VERIFY] {META['modelSource']} ran: P(fake)={100 * prob:.1f}% on a grey square")
        print(f"[VERIFY] explanation heatmap: {'produced' if heatmap else 'unavailable'}")
        return

    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    print(f"[START] VerifAI inference -> http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
