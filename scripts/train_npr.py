"""Fine-tune NPR (Tan et al., CVPR 2024) on a real-vs-AI-generated image set.

    python scripts/train_npr.py --data-dir data/ai_images --epochs 5
    python scripts/train_npr.py --eval-only --eval-dir data/unseen_generators

Dataset layout is ImageFolder with two classes whose names contain "real" and "fake"/"ai" —
same convention as preprocess_faces.py. Paths come from --data-dir, VERIFAI_NPR_DATA, or
scripts/datasets.yaml; nothing is hardcoded.

You may not need this at all: the official ProGAN-trained checkpoint already generalizes
across generators, and is what `export_onnx.py --npr` picks up.

    curl -L -o models/npr_detector.pth \\
      https://github.com/chuangchuangtan/NPR-DeepfakeDetection/raw/main/model_epoch_last_3090.pth

Train only when you have a labelled set the official weights do badly on.
"""

import argparse
import io
import json
import os
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common.config import NPR_CHECKPOINT
from common.datasets import TransformSubset
from models.npr_model import IMG_SIZE, MEAN, STD, NPRDetector, load_npr
from common.metrics import binary_auc
from train_deepfake_detector import RandomJpeg, split_indices

FAKE_WORDS = ("fake", "ai", "synthetic", "generated", "gan", "diffusion")
REAL_WORDS = ("real", "authentic", "genuine", "natural", "nature")


class RandomBlur:
    """Mild blur. Social platforms resample; a detector that only knows pristine files is a
    detector for pristine files."""

    def __init__(self, p=0.2, max_radius=1.0):
        self.p, self.max_radius = p, max_radius

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(random.uniform(0.1, self.max_radius)))


class RandomResample:
    """Downscale-then-upscale.

    Off by default, and that is deliberate: NPR's whole signal IS the up-sampling artifact,
    so this augmentation stamps a synthetic one onto real images and teaches the model that
    real images have it too. Enable with --resample-aug only if your reals are themselves
    resized web images, where the model has to cope with it regardless.
    """

    def __init__(self, p=0.3, min_scale=0.5):
        self.p, self.min_scale = p, min_scale

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        s = random.uniform(self.min_scale, 1.0)
        small = img.resize((max(32, int(w * s)), max(32, int(h * s))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


def fake_class_index(classes):
    """Which ImageFolder class is the fake one. Guessing this wrong inverts every score."""
    fake = [i for i, c in enumerate(classes) if any(w in c.lower() for w in FAKE_WORDS)]
    real = [i for i, c in enumerate(classes) if any(w in c.lower() for w in REAL_WORDS)]
    if len(classes) != 2 or not fake or not real or fake[0] == real[0]:
        sys.exit(f"Need exactly two classes, one real-ish and one fake-ish. Got {classes}.")
    return fake[0]


def build_transforms(resample_aug):
    """Crop at native resolution — no resize.

    Resizing would apply our own up-sampling on top of the generator's, which is the exact
    signal NPR reads. The official code crops to 224 too (cropSize=224).
    """
    norm = transforms.Normalize(MEAN, STD)
    train = [RandomJpeg(), RandomBlur()]
    if resample_aug:
        train.append(RandomResample())
    train += [
        transforms.RandomCrop(IMG_SIZE, pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm,
    ]
    val = [transforms.CenterCrop(IMG_SIZE), transforms.ToTensor(), norm]
    return transforms.Compose(train), transforms.Compose(val)


def loaders(data_dir, batch_size, workers, val_frac, resample_aug, seed=42):
    train_tf, val_tf = build_transforms(resample_aug)

    # ONE walk of the tree. Building ImageFolder twice (once per transform) stats every file
    # twice — ~10 minutes each on a 279k-file symlinked Kaggle dataset, before epoch one.
    print(f"Indexing {data_dir} once...")
    base = datasets.ImageFolder(data_dir)
    classes = base.classes
    fake_i = fake_class_index(classes)

    train_idx, val_idx = split_indices(base.targets, len(classes), val_frac, seed)
    counts = [base.targets.count(i) for i in range(len(classes))]
    print(f"classes {classes} (fake = '{classes[fake_i]}') | counts {counts} | "
          f"train {len(train_idx)} | val {len(val_idx)}")
    if min(counts) == 0:
        sys.exit(f"Class {classes[counts.index(0)]} has no images.")

    pin = torch.cuda.is_available()
    return (
        DataLoader(TransformSubset(base, train_idx, train_tf), batch_size=batch_size, shuffle=True,
                   num_workers=workers, pin_memory=pin, drop_last=len(train_idx) > batch_size,
                   persistent_workers=workers > 0),
        DataLoader(TransformSubset(base, val_idx, val_tf), batch_size=batch_size, shuffle=False,
                   num_workers=workers, pin_memory=pin, persistent_workers=workers > 0),
        classes, fake_i, counts,
    )


@torch.no_grad()
def evaluate(model, loader, device, fake_i):
    """AUC first: accuracy at a fixed 0.5 threshold hides a model that ranks well but is
    mis-centred, and the threshold is ours to move later."""
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        logits = model(x.to(device)).float().flatten()
        scores += torch.sigmoid(logits).cpu().tolist()
        labels += [1 if t == fake_i else 0 for t in y.tolist()]

    auc = binary_auc(scores, labels)
    preds = [1 if s > 0.5 else 0 for s in scores]
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    n_pos, n_neg = sum(labels), len(labels) - sum(labels)
    return {
        "auc": auc,
        "acc": sum(1 for p, l in zip(preds, labels) if p == l) / max(1, len(labels)),
        "recall_fake": tp / n_pos if n_pos else float("nan"),
        "recall_real": tn / n_neg if n_neg else float("nan"),
        "n": len(labels),
    }


def resolve_data_dir(arg):
    if arg:
        return arg
    env = os.environ.get("VERIFAI_NPR_DATA")
    if env:
        return env
    cfg = "scripts/datasets.yaml"
    if os.path.exists(cfg):
        try:
            import yaml

            with open(cfg, encoding="utf-8") as fh:
                for name, ds in (yaml.safe_load(fh) or {}).get("datasets", {}).items():
                    path = ds.get("local_path")
                    if ds.get("kind") == "ai_images" and path and os.path.isdir(path):
                        print(f"Using '{name}' from {cfg}: {path}")
                        return path
        except Exception as e:
            print(f"Note reading {cfg}: {e}")
    sys.exit("No dataset. Pass --data-dir, set VERIFAI_NPR_DATA, or add a "
             "kind: ai_images entry with a local_path to scripts/datasets.yaml.")


def selfcheck():
    from models.npr_model import selfcheck as model_selfcheck

    model_selfcheck()

    assert fake_class_index(["real", "fake"]) == 1
    assert fake_class_index(["1_fake", "0_real"]) == 0, "order comes from the folder, not luck"
    assert fake_class_index(["nature", "ai_generated"]) == 1
    for bad in (["a", "b"], ["real", "authentic"], ["real"]):
        try:
            fake_class_index(bad)
            raise AssertionError(f"{bad} must be rejected, not guessed")
        except SystemExit:
            pass

    # Augmentations must return a same-size PIL image, not silently drop to something else.
    img = Image.new("RGB", (64, 48), (120, 30, 200))
    for aug in (RandomJpeg(p=1.0), RandomBlur(p=1.0), RandomResample(p=1.0)):
        out = aug(img)
        assert out.size == img.size, f"{type(aug).__name__} changed size to {out.size}"

    print("train_npr selfcheck passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="ImageFolder root: one real class, one fake class")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--resample-aug", action="store_true", help="see RandomResample — usually a bad idea for NPR")
    ap.add_argument("--init", default=None, help="start from an existing NPR checkpoint (e.g. the official one)")
    ap.add_argument("--out", default=NPR_CHECKPOINT)
    ap.add_argument("--eval-dir", default=None, help="held-out set from DIFFERENT generators")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return

    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.eval_only:
        if not args.eval_dir:
            sys.exit("--eval-only needs --eval-dir.")
        if not os.path.exists(args.out):
            sys.exit(f"No checkpoint at {args.out}.")
        model = load_npr(args.out, device)
        _, val_tf = build_transforms(False)
        ds = datasets.ImageFolder(args.eval_dir, val_tf)
        fake_i = fake_class_index(ds.classes)
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
        print(f"\nCross-generator evaluation — {args.eval_dir}")
        print(json.dumps(evaluate(model, loader, device, fake_i), indent=2))
        print("\nPut these numbers in app/model-card/page.tsx. Nothing else goes there.")
        return

    data_dir = resolve_data_dir(args.data_dir)
    train_loader, val_loader, classes, fake_i, counts = loaders(
        data_dir, args.batch_size, args.workers, args.val_frac, args.resample_aug
    )

    model = load_npr(args.init, device) if args.init else NPRDetector().to(device)
    # pos_weight, not resampling: it costs one scalar instead of a sampler and a shuffled epoch.
    pos_weight = torch.tensor([counts[1 - fake_i] / max(1, counts[fake_i])], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_auc, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        total, seen = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            target = (y == fake_i).float().unsqueeze(1).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), target)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
            seen += x.size(0)

        m = evaluate(model, val_loader, device, fake_i)
        print(f"epoch {epoch + 1}/{args.epochs}  loss {total / seen:.4f}  "
              f"AUC {m['auc']:.4f}  acc {m['acc']:.4f}  "
              f"recall fake {m['recall_fake']:.3f} / real {m['recall_real']:.3f}")
        if m["auc"] > best_auc:
            best_auc = m["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  new best AUC {best_auc:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "imgSize": IMG_SIZE, "mean": MEAN, "std": STD,
                "classes": classes, "fake_index": fake_i, "val_auc": best_auc}, args.out)
    print(f"Saved {args.out} (best val AUC {best_auc:.4f})")

    if args.eval_dir:
        _, val_tf = build_transforms(False)
        ds = datasets.ImageFolder(args.eval_dir, val_tf)
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
        print(f"\nCross-generator evaluation — {args.eval_dir}")
        print(json.dumps(evaluate(model, loader, device, fake_class_index(ds.classes)), indent=2))

    print(f"\nNext: python scripts/export_onnx.py --npr")


if __name__ == "__main__":
    main()
