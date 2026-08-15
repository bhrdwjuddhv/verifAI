"""Build a clean training folder from one or more raw deepfake datasets.

Two jobs, one pass, because both are "walk the raw data and write a canonical folder":

  1. Crop to the detected face (MTCNN) with a generous margin. Full frames spend most
     of their pixels on background; the forgery evidence is the jaw seam, the eye
     region and the mouth boundary. Worth 5-15 points of real-world accuracy.
  2. Merge datasets whose folder layouts disagree into one ImageFolder tree, recording
     provenance in manifest.json so training can group-split and report per-source.

Everything is re-encoded to JPEG q95. That is deliberate: when a dataset stores reals
as JPEG and fakes as PNG, a model will happily learn the container instead of the
forgery. Normalizing the format removes that shortcut.

    python scripts/preprocess_faces.py --src <kaggle_dir> [--src <dir2>] --out data/faces

Face cropping needs facenet-pytorch (pip install facenet-pytorch); pass --no-crop to
merge only.
"""

import argparse
import json
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from PIL import Image, UnidentifiedImageError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Every leaf directory in these datasets says which class it is in its own name:
#   manjilkarki  Dataset/Train/{Fake,Real}
#   140k         real_vs_fake/real-vs-fake/{train,valid,test}/{real,fake}
#   ciplab       real_and_fake_face/{training_real,training_fake}
# Check 'fake' first — "real_and_fake_face" contains both words.
LABEL_RULES = [("fake", "Fake"), ("real", "Real")]


def canonical_label(dirname, overrides):
    name = dirname.lower()
    if name in overrides:
        return overrides[name]
    for needle, label in LABEL_RULES:
        if needle in name:
            return label
    return None


def find_image_dirs(root):
    """Yields (label, dirpath, [filenames]) for every leaf dir holding images."""
    for dirpath, _, filenames in os.walk(root):
        images = [f for f in filenames if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
        if images:
            yield os.path.basename(dirpath), dirpath, images


def load_detector(device):
    try:
        from facenet_pytorch import MTCNN
    except ImportError:
        sys.exit(
            "❌ facenet-pytorch not installed.\n"
            "   pip install facenet-pytorch   (or rerun with --no-crop to merge only)"
        )
    import torch
    dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Loading MTCNN face detector on {dev}...")
    return MTCNN(keep_all=False, select_largest=True, device=dev, post_process=False)


def crop_face(detector, img, margin):
    """Returns a square crop around the largest face, or None if no face is found.

    The margin matters more than it looks: face-swap blending seams sit on the jawline
    and hairline, so a tight crop cuts away the single most discriminative region.
    """
    boxes, probs = detector.detect(img)
    if boxes is None or len(boxes) == 0:
        return None
    if probs is not None and probs[0] is not None and probs[0] < 0.90:
        return None

    x1, y1, x2, y2 = boxes[0]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (1 + margin) / 2

    # Clamp to the image, keeping the box square so the later resize never stretches.
    half = min(half, cx, cy, img.width - cx, img.height - cy)
    if half < 24:
        return None
    return img.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half)))


def selfcheck():
    """Exercises the crop geometry with a stubbed detector — MTCNN itself is library code,
    but the square/margin/clamp maths is mine and is easy to get subtly wrong."""

    class Stub:
        def __init__(self, box, prob=0.99):
            self.box, self.prob = box, prob

        def detect(self, img):
            return [self.box], [self.prob]

    img = Image.new("RGB", (400, 300))

    # Centred face: box expands by the margin and comes out square.
    out = crop_face(Stub([180, 130, 220, 170]), img, 0.35)
    assert out.width == out.height, f"not square: {out.size}"
    assert 52 <= out.width <= 56, f"40px face +35% margin should be ~54px, got {out.width}"

    # Face near the edge: the box must clamp inside the image and stay square.
    out = crop_face(Stub([5, 5, 45, 45]), img, 1.0)
    assert out.width == out.height and out.width > 0
    assert out.crop((0, 0, 1, 1)) is not None  # crop stayed in bounds, no PIL error

    # Low-confidence and no-detection both mean "no usable face".
    assert crop_face(Stub([180, 130, 220, 170], prob=0.5), img, 0.35) is None
    assert crop_face(Stub([180, 140, 190, 150]), img, 0.0) is None, "tiny face must be rejected"

    class Empty:
        def detect(self, img):
            return None, None

    assert crop_face(Empty(), img, 0.35) is None

    # Label rules must cover all three dataset layouts. 'fake' wins in ambiguous names.
    assert canonical_label("Fake", {}) == "Fake"
    assert canonical_label("training_real", {}) == "Real"
    assert canonical_label("real-vs-fake", {}) == "Fake"
    assert canonical_label("images", {}) is None
    assert canonical_label("Images", {"images": "Real"}) == "Real"
    print("✅ selfcheck passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", help="raw dataset root (repeatable)")
    ap.add_argument("--dataset-config", default="scripts/datasets.yaml", help="YAML file listing configured dataset roots")
    ap.add_argument("--out", default="data/faces")
    ap.add_argument("--size", type=int, default=256, help="output edge; leave room above the 224 train crop")
    ap.add_argument("--margin", type=float, default=0.35, help="box expansion, fraction of face size")
    ap.add_argument("--no-crop", action="store_true", help="merge and normalize only, no face detection")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="stop after N images per source (smoke test)")
    ap.add_argument("--label-map", default=None, help='JSON like {"training_real":"Real"}')
    ap.add_argument("--selfcheck", action="store_true", help="run internal assertions and exit")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return

    # Auto-resolve dataset sources from datasets.yaml if --src is not passed
    if not args.src:
        if os.path.exists(args.dataset_config):
            cfg_path = args.dataset_config
            try:
                import yaml
                with open(cfg_path, "r", encoding="utf-8") as fh:
                    cfg = yaml.safe_load(fh)
                    configured_paths = [
                        ds["local_path"] for ds in cfg.get("datasets", {}).values()
                        if ds.get("local_path") and os.path.exists(ds.get("local_path"))
                    ]
                    if configured_paths:
                        args.src = configured_paths
                        print(f"📖 Loaded {len(configured_paths)} dataset paths from {cfg_path}: {configured_paths}")
            except Exception as e:
                print(f"⚠️ Note reading {cfg_path}: {e}")

    if not args.src:
        ap.error("--src is required or configure valid local_path in scripts/datasets.yaml")

    overrides = json.loads(args.label_map) if args.label_map else {}
    overrides = {k.lower(): v for k, v in overrides.items()}
    detector = None if args.no_crop else load_detector(args.device)

    for label in ("Real", "Fake"):
        os.makedirs(os.path.join(args.out, label), exist_ok=True)

    counts = {"Real": 0, "Fake": 0}
    sources, skipped_dirs = {}, []
    no_face = unreadable = 0

    for src in args.src:
        if not os.path.isdir(src):
            sys.exit(f"❌ Not a directory: {src}")
        tag = os.path.basename(os.path.normpath(src))[:24]
        written_here = 0
        print(f"\n📂 {src}  (tag: {tag})")

        for dirname, dirpath, filenames in find_image_dirs(src):
            label = canonical_label(dirname, overrides)
            if label is None:
                skipped_dirs.append(dirpath)
                continue
            print(f"  → {dirname} [{label}] {len(filenames)} files")

            for fname in sorted(filenames):
                if args.limit and written_here >= args.limit:
                    break
                try:
                    with Image.open(os.path.join(dirpath, fname)) as raw:
                        img = raw.convert("RGB")
                except (UnidentifiedImageError, OSError):
                    unreadable += 1
                    continue

                if detector is not None:
                    img = crop_face(detector, img, args.margin)
                    if img is None:
                        no_face += 1
                        continue

                img = img.resize((args.size, args.size), Image.LANCZOS)
                stem = os.path.splitext(fname)[0]
                # Tag-prefixed so merged sources never collide and provenance survives.
                out_name = f"{tag}__{dirname}__{stem}.jpg"
                img.save(os.path.join(args.out, label, out_name), "JPEG", quality=95)

                counts[label] += 1
                written_here += 1
                if written_here % 500 == 0:
                    print(f"     ...{written_here} written")

        sources[tag] = written_here
        print(f"  ✅ {written_here} images from {tag}")

    if skipped_dirs:
        print(f"\n⚠️  {len(skipped_dirs)} directories had no 'real'/'fake' in their name and were skipped:")
        for d in skipped_dirs[:8]:
            print(f"     {d}")
        print("     Use --label-map to classify them explicitly.")

    manifest = {
        "face_crop": not args.no_crop,
        "size": args.size,
        "margin": args.margin,
        "counts": counts,
        "sources": sources,
        "dropped": {"no_face": no_face, "unreadable": unreadable},
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n📊 Real {counts['Real']} | Fake {counts['Fake']}")
    if no_face:
        pct = 100 * no_face / max(1, no_face + sum(counts.values()))
        print(f"⚠️  {no_face} images dropped with no detectable face ({pct:.1f}%).")
        if pct > 25:
            print("    That is high — check the dataset isn't full-scene rather than portrait.")
    if unreadable:
        print(f"⚠️  {unreadable} unreadable files skipped.")
    if min(counts.values()) == 0:
        sys.exit("❌ One class is empty. Check the source layout or pass --label-map.")

    print(f"💾 Wrote manifest → {os.path.join(args.out, 'manifest.json')}")
    print(f"\nNext:  python scripts/train_deepfake_detector.py --data-dir {args.out}")


if __name__ == "__main__":
    main()
