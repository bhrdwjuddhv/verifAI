"""VerifAI deepfake inference service.

    POST /predict         one image  -> fused verdict + per-detector signals + heatmap
    POST /predict-video   short clip -> sampled frames, aggregated, fused
    POST /predict-audio   voice clip -> mel-spectrogram -> voice model
    POST /predict-audio-window  2-4s window -> VAD gate -> voice model (live calls)

Detectors, fused by FUSION_WEIGHTS over whichever ones actually ran:
    face   face-swap classifier, on the cropped face
    npr    NPR (CVPR 2024) whole-image AI-generation detector, on the full frame

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
import tempfile
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image, UnidentifiedImageError

# scripts/ is on sys.path because this file lives there.
from common.config import (
    AUDIO_MODEL_PATH,
    FAKE_ABOVE,
    ONNX_PATH,
    FUSION_WEIGHTS,
    VIDEO_FRAME_CAP,
    VIDEO_MAX_SECONDS,
    HF_FALLBACK_EXPECTS_FACE,
    HF_FALLBACK_MODEL,
    MODEL_PATH,
    NPR_MODEL_PATH,
    REAL_BELOW,
)
from common.xai import encode_overlay, gradcam_overlay
from common import audio_v2

# Torch-free build artifacts (see export_onnx.py and the Dockerfile's lean stage). ONNX_PATH
# comes from common.config so the exporter and the server resolve the same file.
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


def fuse(scores, weights=None):
    """Weighted mean of the signals that actually ran. Returns (fused_pct, used) or (None, {}).

    Renormalizing over what is present is the whole point: no face means no face score, and a
    missing signal must not drag the average toward 50 as if it had voted "don't know".
    """
    weights = FUSION_WEIGHTS if weights is None else weights
    # Weight 0 means "reported, not trusted" — it must not appear as a voter.
    used = {k: weights.get(k, 0.0) for k, v in scores.items() if v is not None and weights.get(k, 0.0) > 0}
    total = sum(used.values())
    if not used or total <= 0:
        return None, {}
    fused = sum(scores[k] * w for k, w in used.items()) / total
    return float(round(fused)), used


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


def to_array(img, size, mean=IMAGENET_MEAN, std=IMAGENET_STD, crop=False):
    """PIL image -> (1, 3, size, size) float32, normalized. numpy only — no torch.

    crop=True centre-crops at native resolution instead of resizing. NPR measures the
    generator's up-sampling artifact, and resizing lays our own resampling over exactly the
    signal being read — so it only scales up when the image is smaller than the crop.
    """
    if crop:
        w, h = img.size
        if w < size or h < size:
            s = size / min(w, h)
            img = img.resize((max(size, int(w * s)), max(size, int(h * s))), Image.BICUBIC)
            w, h = img.size
        left, top = (w - size) // 2, (h - size) // 2
        img = img.crop((left, top, left + size, top + size))
    else:
        img = img.resize((size, size), Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0
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

    def predict(img, explain=None):
        t = to_tensor(img, size).unsqueeze(0).to(device)
        with torch.no_grad():
            batch = torch.cat([t, torch.flip(t, dims=[3])])  # flip TTA
            probs = torch.softmax(model(batch).float() / temperature, dim=1).mean(dim=0).tolist()
        heatmap = None
        if GRADCAM if explain is None else explain:
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


def load_onnx_engine(model_path=None):
    """The lean build: ONNX Runtime, no torch, no transformers.

    torch plus a ViT-base measures ~1.2GB resident, which does not fit a 512MB instance. This
    path is the same model exported by export_onnx.py, typically int8-quantized. Grad-CAM needs
    gradients that ONNX Runtime does not provide, so explainability here is occlusion saliency:
    hide a patch, see how much the fake score drops. Slower, coarser, and it measures the
    model's actual behaviour rather than its internal activations.
    """
    import onnxruntime as ort

    model_path = model_path or ONNX_PATH
    with open(os.path.splitext(model_path)[0] + ".json") as fh:
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
    session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    chunk = int(os.environ.get("VERIFAI_BATCH", "8"))

    def run(batch):
        outs = [session.run(None, {input_name: batch[i:i + chunk]})[0] for i in range(0, len(batch), chunk)]
        logits = np.concatenate(outs).astype(np.float32) / temperature
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(img, explain=None):
        arr = to_array(img, size, mean, std)
        probs = run(np.concatenate([arr, arr[:, :, :, ::-1]]))  # flip TTA
        heatmap = None
        if GRADCAM if explain is None else explain:
            heatmap = occlusion_overlay(run, arr, fake_i, img.resize((size, size), Image.BILINEAR))
        return fake_probability(probs.mean(axis=0).tolist(), labels), heatmap

    source = meta_file.get("source", "onnx")
    if meta_file.get("quantized"):
        source += "+int8"
    source += f"({os.path.basename(model_path)})"
    meta = {
        "modelSource": f"onnx:{source}",
        "classes": labels,
        "expectsFace": bool(meta_file.get("expectsFace", True)),
        "calibrated": bool(meta_file.get("calibrated", False)),
        "device": "cpu (onnxruntime)",
        "valMetrics": meta_file.get("valMetrics"),
        "valSplit": meta_file.get("valSplit"),
        "imgSize": size,
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

    def predict(img, explain=None):
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits[0].float(), dim=0).tolist()
        heatmap = None
        if GRADCAM if explain is None else explain:
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


def load_npr_engine():
    """NPR: whole-image AI-generation detector, ONNX only. Returns (predict, meta).

    Separate from the face classifier on purpose — they answer different questions. The face
    model asks "was this face swapped", NPR asks "did a generator's decoder make these pixels",
    which is the question a fully generated StyleGAN face fails the first test on.
    """
    if not os.path.exists(NPR_MODEL_PATH):
        return None, {"available": False, "reason": f"no NPR model at {NPR_MODEL_PATH}"}
    try:
        import onnxruntime as ort
    except ImportError as e:
        return None, {"available": False, "reason": f"onnxruntime missing ({e})"}

    try:
        with open(os.path.splitext(NPR_MODEL_PATH)[0] + ".json") as fh:
            meta_file = json.load(fh)

        size = int(meta_file.get("imgSize", 224))
        mean = meta_file.get("mean", IMAGENET_MEAN)
        std = meta_file.get("std", IMAGENET_STD)

        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.intra_op_num_threads = int(os.environ.get("VERIFAI_THREADS", "2"))
        session = ort.InferenceSession(NPR_MODEL_PATH, sess_options=options,
                                       providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        def predict(img):
            arr = to_array(img, size, mean, std, crop=True)
            logit = float(session.run(None, {input_name: arr})[0].reshape(-1)[0])
            return 1.0 / (1.0 + np.exp(-logit))  # official convention: 1 = fake

        source = meta_file.get("source", "npr")
        if meta_file.get("quantized"):
            source += "+int8"
        # Name the file: two NPR builds ship with an identical `source` field, and a verdict
        # that cannot say which one answered is a verdict you cannot audit.
        source += f"({os.path.normpath(NPR_MODEL_PATH).replace(os.sep, chr(47))})"
        return predict, {"available": True, "modelSource": f"onnx:{source}",
                         "calibrated": bool(meta_file.get("calibrated"))}
    except Exception as e:
        return None, {"available": False, "reason": f"{type(e).__name__}: {e}"}


def load_engine():
    """Try each source in order and FALL THROUGH on failure.

    Committing to the first candidate whose file merely exists was a bug: dropping a .pth next
    to the ONNX made the torch-free build pick the .pth, fail on `import torch`, and report no
    face model at all — with a perfectly good ONNX sitting right there. Existence is not
    loadability, so each attempt has to be allowed to fail.
    """
    attempts = []
    if os.path.exists(MODEL_PATH):
        attempts.append((f"checkpoint {MODEL_PATH}", load_checkpoint_engine))
    if os.path.exists(ONNX_PATH):
        attempts.append((f"ONNX {ONNX_PATH} (torch-free)", load_onnx_engine))
    attempts.append((f"Hugging Face fallback {HF_FALLBACK_MODEL}", load_hf_engine))

    errors = []
    for label, loader in attempts:
        print(f"[MODEL] trying {label}")
        try:
            return loader()
        except Exception as e:
            errors.append(f"{label}: {type(e).__name__}: {e}")
            print(f"[MODEL]  -> unavailable ({type(e).__name__}: {e})")

    raise RuntimeError("no face model could be loaded — " + " | ".join(errors))


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
NPR_PREDICT, NPR_META = None, {"available": False, "reason": "not loaded"}
AUDIO_PREDICT, AUDIO_META = None, {"available": False, "reason": "not loaded"}
# v2 runs raw samples through preproc.onnx; it is used only when its self-test passes.
AUDIO_V2_PREDICT, AUDIO_V2_META = None, {"available": False, "reason": "not loaded"}


def startup():
    global PREDICT, META, LOAD_ERROR, DETECTOR, NPR_PREDICT, NPR_META, AUDIO_PREDICT, AUDIO_META
    global AUDIO_V2_PREDICT, AUDIO_V2_META
    AUDIO_V2_PREDICT, AUDIO_V2_META = load_audio_v2_engine()
    status = AUDIO_V2_META.get("selftest", {}).get("status", "unavailable")
    print(f"[AUDIO v2] self-test {status}"
          + (f" — {AUDIO_V2_META['reason']}" if not AUDIO_V2_PREDICT else
             f" (delta {AUDIO_V2_META['selftest'].get('delta')})"))

    AUDIO_PREDICT, AUDIO_META = load_audio_engine()
    print(f"[AUDIO] {AUDIO_META.get('modelSource') if AUDIO_PREDICT else 'unavailable — ' + AUDIO_META['reason']}")
    NPR_PREDICT, NPR_META = load_npr_engine()
    print(f"[NPR] {NPR_META.get('modelSource') if NPR_PREDICT else 'unavailable — ' + NPR_META['reason']}")
    try:
        PREDICT, META = load_engine()
    except Exception as e:
        LOAD_ERROR = f"{type(e).__name__}: {e}"
        print(f"[ERROR] no face model loaded. {LOAD_ERROR}")
        if NPR_PREDICT is None:
            print("[ERROR] no detector at all — /predict will return 503.")
        return
    DETECTOR = load_face_detector()
    print(f"[OK] {META['modelSource']} on {META['device']} | classes={META['classes']} "
          f"| expectsFace={META['expectsFace']} | faceDetector={'on' if DETECTOR else 'off'}")


def analyze(image, explain=None):
    """The whole verdict, as a dict. Kept out of the endpoint so it stays testable.

    explain=None keeps whatever GRADCAM is set to, which is what every existing caller gets.
    explain=False skips the heatmap: the browser extension asks for that on feed scans, where
    nothing displays one and it costs 26 forward passes per image.
    """
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

    # NPR reads the whole frame and needs no face, so it runs on everything.
    npr_pct = None
    if NPR_PREDICT is not None:
        try:
            npr_pct = round(100.0 * NPR_PREDICT(image))
        except Exception as e:
            notes.append(f"NPR detector failed: {e}")

    # Face classifier: only where it applies. A face model on a landscape produces a number,
    # not evidence, so it abstains rather than voting.
    face_pct, heatmap = None, None
    if PREDICT is not None and not (META["expectsFace"] and face_detected is False):
        target = crop if (META["expectsFace"] and crop is not None) else image
        if META["expectsFace"] and DETECTOR is None:
            notes.append("face detector unavailable; ran the full frame through a face model")
        try:
            fake_prob, heatmap = PREDICT(target, explain=explain)
            # Band the number we are going to SHOW. Deciding on 70.4 and displaying 70 next to
            # a rule that says "above 70 is fake" is a contradiction the reader can see.
            face_pct = float(round(100.0 * fake_prob))
        except Exception as e:
            notes.append(f"face classifier failed: {e}")
    elif PREDICT is None:
        notes.append(f"face classifier unavailable ({LOAD_ERROR})")
    else:
        notes.append("no face detected — the face classifier does not apply, so it did not vote")

    fused, used = fuse({"face": face_pct, "npr": npr_pct, "frequency": float(frequency)})

    if fused is None:
        return {
            "verdict": "uncertain",
            "confidence": 0,
            "fakeProbability": None,
            "modelSource": NPR_META.get("modelSource") or META.get("modelSource"),
            "detectors": {"face": META.get("modelSource") if PREDICT else None,
                          "npr": NPR_META.get("modelSource") if NPR_PREDICT else None},
            "faceDetected": face_detected,
            "signals": {"modelScore": face_pct, "nprScore": npr_pct, "frequencyScore": frequency},
            "fusion": {"weights": FUSION_WEIGHTS, "used": {}},
            "heatmap": None,
            "notes": notes + ["no detector applied to this file, so there is no verdict"],
        }

    verdict, confidence = verdict_for(fused)
    if not META.get("calibrated", False):
        notes.append("confidence is uncalibrated — treat it as a ranking, not a probability")
    if heatmap is None and (GRADCAM if explain is None else explain) and face_pct is not None:
        notes.append("no explanation heatmap available for this model")
    if META.get("quantized"):
        notes.append("int8-quantized build; scores can differ from the full-precision model by a point or two")
    if npr_pct is None:
        notes.append(f"NPR whole-image detector unavailable ({NPR_META.get('reason')}) — a fully "
                     "generated image may not be caught by the face model alone")
    if len(used) > 1 and face_pct is not None and npr_pct is not None and abs(face_pct - npr_pct) > 50:
        notes.append(f"the detectors disagree sharply (face {face_pct:.0f}% vs NPR {npr_pct:.0f}%); "
                     "the fused score sits between them, which is why this may read as uncertain")

    return {
        "verdict": verdict,
        "confidence": round(confidence),
        # The fused number the verdict is actually based on.
        "fakeProbability": int(fused),
        "modelSource": " + ".join(
            s for s in (META.get("modelSource") if face_pct is not None else None,
                        NPR_META.get("modelSource") if npr_pct is not None else None) if s
        ),
        "detectors": {"face": META.get("modelSource") if PREDICT else None,
                      "npr": NPR_META.get("modelSource") if NPR_PREDICT else None},
        "faceDetected": face_detected,
        "signals": {"modelScore": face_pct, "nprScore": npr_pct, "frequencyScore": frequency},
        "fusion": {"weights": FUSION_WEIGHTS, "used": used},
        "heatmap": heatmap,
        "notes": notes,
    }


def frame_signals(image, explain=False):
    """Per-frame detector scores. Shared by the image and video paths so they cannot drift."""
    from preprocess_faces import crop_face

    crop, face_detected = None, None
    if DETECTOR is not None:
        try:
            crop = crop_face(DETECTOR, image, FACE_MARGIN)
            face_detected = crop is not None
        except Exception:
            face_detected = None

    face_pct, heatmap = None, None
    if PREDICT is not None and not (META["expectsFace"] and face_detected is False):
        target = crop if (META["expectsFace"] and crop is not None) else image
        fake_prob, heatmap = PREDICT(target, explain=explain)
        face_pct = float(round(100.0 * fake_prob))

    npr_pct = None
    if NPR_PREDICT is not None:
        npr_pct = float(round(100.0 * NPR_PREDICT(image)))

    return {
        "face": face_pct,
        "npr": npr_pct,
        "frequency": float(round(frequency_score(image))),
        "faceDetected": face_detected,
        "heatmap": heatmap,
    }


def sample_frames(path, cap=None):
    """Evenly spaced frames as PIL images, with their timestamps.

    Even spacing, not the first N: a manipulated clip is usually manipulated in the middle,
    and the first second of anything is often a title card.
    """
    import cv2

    want = cap or VIDEO_FRAME_CAP
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError("could not decode this video")

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    duration = total / fps if total else 0.0
    if duration > VIDEO_MAX_SECONDS:
        capture.release()
        raise ValueError(f"video is {duration:.0f}s long; the limit is {VIDEO_MAX_SECONDS:.0f}s")

    frames = []
    if total > 0:
        indices = sorted({int(i * (total - 1) / max(1, want - 1)) for i in range(min(want, total))})
        for idx in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = capture.read()
            if ok:
                frames.append((idx / fps, Image.fromarray(frame[:, :, ::-1])))
    else:
        # Streams with no frame count: read sequentially until the cap.
        while len(frames) < want:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append((len(frames) / fps, Image.fromarray(frame[:, :, ::-1])))
    capture.release()

    if not frames:
        raise ValueError("no readable frames in this video")
    return frames, duration


def mean_of(values):
    present = [v for v in values if v is not None]
    return float(round(sum(present) / len(present))) if present else None


def analyze_video(path):
    """Sample frames, score each with the image detectors, aggregate, fuse.

    No new model: a deepfaked video is a sequence of deepfaked images. The one video-specific
    number is temporal variance — a swapped face flickers between frames in a way a filmed one
    does not — and it is reported, not fused, because nobody has calibrated it here.
    """
    frames, duration = sample_frames(path)
    per_frame = []
    for timestamp, image in frames:
        sig = frame_signals(image, explain=False)
        fused_frame, _ = fuse({k: sig[k] for k in ("face", "npr", "frequency")})
        per_frame.append({
            "t": round(timestamp, 2),
            "face": sig["face"],
            "npr": sig["npr"],
            "frequency": sig["frequency"],
            "faceDetected": sig["faceDetected"],
            "fused": fused_frame,
        })

    means = {k: mean_of([f[k] for f in per_frame]) for k in ("face", "npr", "frequency")}
    fused, used = fuse(means)
    fused_frames = [f["fused"] for f in per_frame if f["fused"] is not None]
    faces_found = sum(1 for f in per_frame if f["faceDetected"])

    detectors = {"face": META.get("modelSource") if PREDICT else None,
                 "npr": NPR_META.get("modelSource") if NPR_PREDICT else None}
    notes = [f"sampled {len(per_frame)} frames across {duration:.1f}s"]
    if faces_found == 0:
        notes.append("no face found in any sampled frame — the face classifier did not vote")
    elif faces_found < len(per_frame):
        notes.append(f"a face was found in {faces_found} of {len(per_frame)} sampled frames")

    video = {"frames": len(per_frame), "durationSeconds": round(duration, 2),
             "perFrame": per_frame}

    if fused is None:
        return {
            "verdict": "uncertain", "confidence": 0, "fakeProbability": None,
            "modelSource": NPR_META.get("modelSource") or META.get("modelSource"),
            "detectors": detectors,
            "faceDetected": faces_found > 0,
            "signals": {"modelScore": means["face"], "nprScore": means["npr"],
                        "frequencyScore": means["frequency"]},
            "fusion": {"weights": FUSION_WEIGHTS, "used": {}},
            "video": video,
            "heatmap": None,
            "notes": notes + ["no detector applied to these frames, so there is no verdict"],
        }

    # Flicker: spread of the per-frame verdicts. A partly manipulated clip is inconsistent —
    # worth a human's attention, but reported, never fused.
    variance = None
    if len(fused_frames) > 1:
        mean_f = sum(fused_frames) / len(fused_frames)
        variance = round(sum((v - mean_f) ** 2 for v in fused_frames) / len(fused_frames), 1)

    verdict, confidence = verdict_for(fused)
    peak = max(per_frame, key=lambda f: f["fused"] if f["fused"] is not None else -1)
    if variance is not None and variance > 400:
        notes.append(f"per-frame scores vary a lot (variance {variance:.0f}) — the clip may be "
                     "partly manipulated, or the frames may simply differ in quality")
    if not META.get("calibrated", False):
        notes.append("confidence is uncalibrated — treat it as a ranking, not a probability")
    notes.append("no heatmap for video: it would be one per frame, and per-frame explanation "
                 "costs more than the verdict itself")
    # Measured, not assumed: an AI image NPR scored 100 as a still scored 0 after being
    # re-encoded into an mp4. Every video is re-encoded, so this caveat is permanent.
    notes.append("video re-encoding weakens the up-sampling artifact the AI-generation "
                 "detector reads — treat its score on video as weaker evidence than on a still")

    video.update({
        "meanFakeProbability": int(fused),
        "maxFakeProbability": int(max(fused_frames)) if fused_frames else None,
        "temporalVariance": variance,
        "peakFrameSeconds": peak["t"] if fused_frames else None,
    })

    return {
        "verdict": verdict,
        "confidence": round(confidence),
        "fakeProbability": int(fused),
        "modelSource": " + ".join(x for x in (
            META.get("modelSource") if means["face"] is not None else None,
            NPR_META.get("modelSource") if means["npr"] is not None else None) if x),
        "detectors": detectors,
        "faceDetected": faces_found > 0,
        "signals": {"modelScore": means["face"], "nprScore": means["npr"],
                    "frequencyScore": means["frequency"]},
        "fusion": {"weights": FUSION_WEIGHTS, "used": used},
        "video": video,
        "heatmap": None,
        "notes": notes,
    }


def load_audio_v2_engine():
    """The v2 chain: raw samples -> preproc.onnx -> CNN, and ONLY if the self-test passes.

    The self-test is the whole point. It proves this build reproduces the training pipeline's
    number on a fixed input, which is what makes the same chain safe to run in a browser. A
    chain that will not load, or that answers differently, is reported and NOT used — the v1
    path keeps serving instead. There is no third behaviour where a verdict gets invented.
    """
    result = audio_v2.run_selftest(int(os.environ.get("VERIFAI_THREADS", "2")))
    if result.get("status") != "pass":
        return None, {"available": False, "reason": result.get("reason", "self-test did not pass"),
                      "selftest": result}

    preproc, cnn, meta, reason = audio_v2.load_chain(int(os.environ.get("VERIFAI_THREADS", "2")))
    if reason:
        return None, {"available": False, "reason": reason, "selftest": result}

    def predict(image_or_samples, explain=None):
        raise RuntimeError("v2 takes raw samples via predict_samples, not a spectrogram image")

    def predict_samples(samples):
        probs = audio_v2.probabilities(preproc, cnn, samples)
        return float(probs[0, 0])  # index 0 = Fake

    return predict_samples, {
        "available": True,
        "modelSource": "onnx:audio_v2(preproc.onnx -> audio_detector.onnx)",
        "calibrated": False,
        "selftest": result,
        "imgSize": 224,
    }


def load_audio_engine():
    """Voice-clone detector. Same ONNX classifier machinery, fed a mel-spectrogram."""
    if not os.path.exists(AUDIO_MODEL_PATH):
        return None, {"available": False, "reason": f"no audio model at {AUDIO_MODEL_PATH}"}
    try:
        predict, meta = load_onnx_engine(AUDIO_MODEL_PATH)
        meta["available"] = True
        return predict, meta
    except Exception as e:
        return None, {"available": False, "reason": f"{type(e).__name__}: {e}"}


def analyze_audio(path):
    """Audio -> mel-spectrogram -> the voice model. Same bands, same honesty as images.

    The spectrogram comes from preprocess_audio.generate_spectrogram_image — the exact
    function the training set was built with. Reimplementing the mel maths here would be a
    second chance to get n_mels, hop length or the dB reference subtly different, and the
    model would degrade with nothing in the logs to explain it.
    """
    from preprocess_audio import generate_spectrogram_image

    image = generate_spectrogram_image(path, AUDIO_META.get("imgSize", 224))
    if image is None:
        raise ValueError("could not decode this audio, or it was empty")

    fake_prob, _ = AUDIO_PREDICT(image, explain=False)
    fake_pct = float(round(100.0 * fake_prob))
    verdict, confidence = verdict_for(fake_pct)

    notes = ["verdict is from the voice model alone — the image detectors do not apply to audio"]
    if not AUDIO_META.get("calibrated", False):
        notes.append("confidence is uncalibrated — treat it as a ranking, not a probability")
    if AUDIO_META.get("quantized"):
        notes.append("int8-quantized build; scores can differ from full precision by a point or two")

    return {
        "verdict": verdict,
        "confidence": round(confidence),
        "fakeProbability": int(fake_pct),
        "modelSource": AUDIO_META["modelSource"],
        "detectors": {"face": None, "npr": None, "audio": AUDIO_META["modelSource"]},
        "faceDetected": None,
        "signals": {"modelScore": None, "nprScore": None, "frequencyScore": None,
                    "audioScore": int(fake_pct)},
        "fusion": {"weights": {"audio": 1.0}, "used": {"audio": 1.0}},
        "heatmap": None,
        "notes": notes,
    }


# --- live-call windows -------------------------------------------------------------------
# A call is mostly silence, breathing and line noise. Scoring those produces confident
# nonsense, so a window has to look like speech before the model is allowed an opinion.
VAD_MIN_DBFS = float(os.environ.get("VERIFAI_VAD_MIN_DBFS", "-45"))
VAD_MAX_FLATNESS = float(os.environ.get("VERIFAI_VAD_MAX_FLATNESS", "0.45"))


def speech_metrics(samples, sr=16000):
    """(is_speech, {rmsDbfs, spectralFlatness}).

    Two cheap gates, no model:
      * loudness — silence and room tone sit far below speech,
      * spectral flatness — geometric/arithmetic mean of the power spectrum. Voiced speech is
        harmonic, so flatness is low; hiss, fans and codec comfort-noise approach 1.

    A sustained pure tone also passes, which is a known and acceptable false accept: it is a
    gate on "is there something worth scoring", not a speech classifier.
    """
    y = np.asarray(samples, dtype=np.float32).reshape(-1)
    if y.size < 256:
        return False, {"rmsDbfs": None, "spectralFlatness": None, "reason": "window too short"}

    rms = float(np.sqrt(np.mean(y * y)))
    dbfs = 20.0 * float(np.log10(rms + 1e-12))
    if dbfs < VAD_MIN_DBFS:
        return False, {"rmsDbfs": round(dbfs, 1), "spectralFlatness": None,
                       "reason": f"too quiet ({dbfs:.0f} dBFS < {VAD_MIN_DBFS:.0f})"}

    window = np.hanning(len(y))
    power = np.abs(np.fft.rfft(y * window)) ** 2
    power = power[1:]                      # drop DC; it says nothing about voicing
    power = power[power > 0]
    if power.size == 0:
        return False, {"rmsDbfs": round(dbfs, 1), "spectralFlatness": None, "reason": "no spectrum"}

    flatness = float(np.exp(np.mean(np.log(power))) / np.mean(power))
    if flatness > VAD_MAX_FLATNESS:
        return False, {"rmsDbfs": round(dbfs, 1), "spectralFlatness": round(flatness, 3),
                       "reason": f"noise-like (flatness {flatness:.2f} > {VAD_MAX_FLATNESS:.2f})"}

    return True, {"rmsDbfs": round(dbfs, 1), "spectralFlatness": round(flatness, 3), "reason": None}


def analyze_audio_window(samples, sr=16000):
    """Score one short window. The live path: no file, no verdict bands, just the number.

    Bands are deliberately absent here — a single 3-second window is not a verdict about a
    call. Phase 4 aggregates windows over time and applies hysteresis; this returns evidence.
    """
    from preprocess_audio import spectrogram_from_samples

    seconds = round(len(samples) / float(sr), 2) if sr else 0.0
    is_speech, vad = speech_metrics(samples, sr)
    base = {
        "speechDetected": is_speech,
        "windowSeconds": seconds,
        "vad": vad,
        "modelSource": (AUDIO_V2_META.get("modelSource") if AUDIO_V2_PREDICT
                        else AUDIO_META.get("modelSource")),
    }
    if not is_speech:
        # Not scored on purpose. Returning a probability here is how a live guard cries wolf
        # at a silent room.
        return {**base, "fakeProbability": None,
                "notes": [f"not scored: {vad['reason']}"]}

    if AUDIO_V2_PREDICT is not None:
        # Verified chain: the same graph the browser runs, so server and on-device agree.
        fake_prob = AUDIO_V2_PREDICT(samples)
        notes = ["v2 chain (preproc.onnx -> CNN), self-test passed"]
        if not AUDIO_V2_META.get("calibrated", False):
            notes.append("uncalibrated — treat as a ranking, not a probability")
        return {**base, "modelSource": AUDIO_V2_META["modelSource"],
                "fakeProbability": int(round(100.0 * fake_prob)), "notes": notes}

    image = spectrogram_from_samples(samples, sr, AUDIO_META.get("imgSize", 224))
    if image is None:
        return {**base, "fakeProbability": None, "notes": ["could not build a spectrogram"]}

    fake_prob, _ = AUDIO_PREDICT(image, explain=False)
    notes = []
    if not AUDIO_META.get("calibrated", False):
        notes.append("uncalibrated — treat as a ranking, not a probability")
    if AUDIO_META.get("valSplit") and "source" not in str(AUDIO_META.get("valSplit")):
        notes.append("this model's validation split was not source-held-out; its reported "
                     "accuracy is optimistic until evaluated on an untouched corpus")
    return {**base, "fakeProbability": int(round(100.0 * fake_prob)), "notes": notes}


def create_app():
    """App factory. Deploy with: uvicorn inference_server:create_app --factory --host 0.0.0.0"""
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
        return {"status": "ok" if (PREDICT or NPR_PREDICT) else "no_model", "error": LOAD_ERROR,
                "faceDetector": DETECTOR is not None, "npr": NPR_META,
                "audio": {**AUDIO_META, "v2": AUDIO_V2_META,
                          "selftest": AUDIO_V2_META.get("selftest", {}).get("status", "unavailable"),
                          "active": ("v2" if AUDIO_V2_PREDICT else "v1" if AUDIO_PREDICT else "none")},
                "fusionWeights": FUSION_WEIGHTS,
                # The extension mirrors these locally; /api/extension/manifest relays them so a
                # retune here cannot leave installed clients scoring against stale settings.
                "saliencyGrid": SALIENCY_GRID,
                "thresholds": {"fakeAbove": FAKE_ABOVE, "realBelow": REAL_BELOW}, **META}

    # Deliberately `def`, not `async def`: inference is CPU-bound and would otherwise block
    # the event loop for its whole duration, starving the platform's health checks until it
    # restarts the container mid-request. FastAPI runs sync endpoints in a threadpool.
    @app.post("/predict")
    def predict_image(file: UploadFile = File(...), explain: Optional[bool] = None):
        if PREDICT is None and NPR_PREDICT is None:
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
            return analyze(image, explain)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    @app.post("/predict-video")
    def predict_video(file: UploadFile = File(...)):
        if PREDICT is None and NPR_PREDICT is None:
            raise HTTPException(status_code=503, detail=f"No model loaded: {LOAD_ERROR}")

        contents = file.file.read()
        if len(contents) > MAX_BYTES:
            raise HTTPException(status_code=413, detail="File exceeds 50MB limit.")

        # OpenCV decodes from a path, not a buffer. Deleted in the finally, always.
        suffix = os.path.splitext(file.filename or "")[1][:8] or ".mp4"
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(contents)
            return analyze_video(temp_path)
        except ValueError as e:
            # Undecodable, empty or over the length limit — the caller's problem, stated plainly.
            raise HTTPException(status_code=415, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    # Guard note: /predict-audio-window checks AUDIO_PREDICT, and v2 alone is enough to serve.
    @app.post("/predict-audio-window")
    def predict_audio_window(
        file: UploadFile = File(...),
        sample_rate: int = Form(16000),
        encoding: str = Form("wav"),
    ):
        """One short window (2-4s) for a live call. WAV or raw 16-bit PCM.

        WAV and PCM only, deliberately: decoding Opus/WebM would mean ffmpeg in the runtime
        image, and the browser can encode WAV from an AudioContext in a few lines.
        """
        if AUDIO_PREDICT is None and AUDIO_V2_PREDICT is None:
            raise HTTPException(
                status_code=501,
                detail=f"Voice detection is not available ({AUDIO_META.get('reason')}; "
                       f"v2: {AUDIO_V2_META.get('reason')}).",
            )

        raw = file.file.read()
        if len(raw) > 8 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Window exceeds 8MB — send 2-4 seconds.")

        if encoding.lower() in ("pcm", "pcm16", "s16le"):
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            sr = int(sample_rate)
        else:
            from preprocess_audio import read_audio

            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(raw)
                samples = read_audio(temp_path, 16000, 10.0)
                sr = 16000
            finally:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            if samples is None:
                raise HTTPException(status_code=415,
                                    detail="Could not decode this window. Send WAV or raw PCM16.")

        try:
            return analyze_audio_window(samples, sr)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    @app.post("/predict-audio")
    def predict_audio(file: UploadFile = File(...)):
        if AUDIO_PREDICT is None:
            # No model means no verdict. Saying so is the whole point.
            raise HTTPException(
                status_code=501,
                detail=f"Voice detection is not available yet ({AUDIO_META.get('reason')}). "
                       f"Train a model and drop the ONNX in — see youhavetodo.md.",
            )

        contents = file.file.read()
        if len(contents) > MAX_BYTES:
            raise HTTPException(status_code=413, detail="File exceeds 50MB limit.")

        suffix = os.path.splitext(file.filename or "")[1][:8] or ".wav"
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(contents)
            return analyze_audio(temp_path)
        except ValueError as e:
            raise HTTPException(status_code=415, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

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

    # Fusion: only present signals vote, and a missing one must not pull the mean toward 50.
    w = {"face": 0.5, "npr": 0.5, "frequency": 0.0}
    assert fuse({"face": 80.0, "npr": 20.0}, w)[0] == 50.0
    assert fuse({"face": 90.0, "npr": None}, w)[0] == 90.0, "absent signal must not dilute"
    assert fuse({"face": None, "npr": 100.0}, w)[0] == 100.0
    assert fuse({"face": 60.0, "npr": 40.0, "frequency": 99.0}, w)[0] == 50.0, "zero weight = no vote"
    assert "frequency" not in fuse({"face": 60.0, "frequency": 99.0}, w)[1], "0-weight must not read as a voter"
    assert fuse({"face": None, "npr": None}, w) == (None, {})
    assert fuse({"frequency": 99.0}, w) == (None, {}), "only zero-weighted signals = no verdict"
    # Weights are relative, not absolute: doubling both changes nothing.
    assert fuse({"face": 80.0, "npr": 20.0}, {"face": 1.0, "npr": 1.0})[0] == 50.0
    assert fuse({"face": 80.0, "npr": 20.0}, {"face": 3.0, "npr": 1.0})[0] == 65.0

    from common.config import parse_weights

    assert parse_weights("face=0.3,npr=0.7", w) == {"face": 0.3, "npr": 0.7, "frequency": 0.0}
    assert parse_weights("", w) == w
    for bad in ("bogus=1", "face=-1", "face=0,npr=0,frequency=0"):
        try:
            parse_weights(bad, w)
            raise AssertionError(f"{bad!r} must be rejected, not silently accepted")
        except ValueError:
            pass

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

    # Video aggregation: means ignore absent signals, variance measures flicker.
    assert mean_of([10.0, 20.0, None]) == 15.0, "absent frames must not count as zero"
    assert mean_of([None, None]) is None
    assert mean_of([]) is None

    frames_steady = [50.0] * 8
    frames_flicker = [0.0, 100.0] * 4
    def variance(values):
        m = sum(values) / len(values)
        return sum((v - m) ** 2 for v in values) / len(values)
    assert variance(frames_steady) == 0.0
    assert variance(frames_flicker) == 2500.0, "alternating frames must read as high variance"

    # VAD: the gate that stops a live guard scoring silence.
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    silence = np.zeros(sr, dtype=np.float32)
    quiet_room = (np.random.default_rng(0).standard_normal(sr) * 0.0005).astype(np.float32)
    hiss = (np.random.default_rng(1).standard_normal(sr) * 0.2).astype(np.float32)
    voiced = sum(np.sin(2 * np.pi * f * t) / (i + 1)
                 for i, f in enumerate((150, 300, 450, 600))).astype(np.float32) * 0.3

    assert speech_metrics(silence, sr)[0] is False, "silence must not be scored"
    assert speech_metrics(quiet_room, sr)[0] is False, "room tone must not be scored"
    ok, m = speech_metrics(hiss, sr)
    assert ok is False and "flatness" in m["reason"], f"white noise must be rejected: {m}"
    ok, m = speech_metrics(voiced, sr)
    assert ok is True, f"harmonic speech-like audio must pass the gate: {m}"
    assert speech_metrics(np.zeros(10), sr)[0] is False, "a 10-sample window is not speech"
    assert m["rmsDbfs"] < 0, "dBFS is negative for anything below full scale"

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
