import argparse
import io
import json
import os
import random
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Multi-Dataset Kaggle Deepfake Benchmarks:
# 1. manjilkarki/deepfake-and-real-images
# 2. xhlulu/140k-real-and-fake-faces
# 3. ciplab/real-and-fake-face-detection
DATASETS = [
    "manjilkarki/deepfake-and-real-images",
    "xhlulu/140k-real-and-fake-faces",
    "ciplab/real-and-fake-face-detection",
]

# Shared with inference_server.py so the two can never drift apart again.
from common.config import MODEL_PATH as MODEL_SAVE_PATH
from common.calibration import fit_temperature
from common.datasets import TransformSubset
from common.metrics import binary_auc

ONNX_SAVE_PATH = "public/models/deepfake_detector.onnx"

BACKBONES = {
    "b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 224),
    "b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT, 380),
}


def download_all_datasets():
    """Downloads the Deepfake & Real image datasets via kagglehub."""
    print("📥 Downloading Deepfake Benchmarks...")
    downloaded_paths = []
    try:
        import kagglehub
        for handle in DATASETS:
            print(f"  -> Downloading {handle}...")
            path = kagglehub.dataset_download(handle)
            downloaded_paths.append(path)
            print(f"  ✅ Saved: {path}")
        # ponytail: only the first dataset is used. The three have incompatible folder
        # layouts and label vocabularies; merging them needs a manifest step, not a loop.
        print(f"⚠️  Training on {DATASETS[0]} only — see README for merging the rest.")
        return downloaded_paths[0]
    except Exception as e:
        print(f"⚠️ kagglehub download note: ({e}). Checking local 'data/Dataset' folder...")
        local_path = os.path.abspath("data/Dataset")
        if os.path.exists(local_path):
            print(f"✅ Found local dataset at: {local_path}")
            return local_path
        print("❌ Local dataset directory not found. Place benchmark folders in data/Dataset.")
        sys.exit(1)


class RandomJpeg:
    """Re-encodes at a random JPEG quality.

    The single most important augmentation here: uploads arrive re-compressed by
    WhatsApp/Twitter/screenshots, and a model trained only on pristine files learns
    the dataset's compression signature instead of the forgery.
    """

    def __init__(self, p=0.7, quality_range=(35, 95)):
        self.p = p
        self.quality_range = quality_range

    def __call__(self, img):
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=random.randint(*self.quality_range))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomDownscale:
    """Downscale-then-upscale. Simulates low-res sources without rotating pixels."""

    def __init__(self, p=0.3, min_scale=0.4):
        self.p = p
        self.min_scale = min_scale

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        s = random.uniform(self.min_scale, 1.0)
        small = img.resize((max(32, int(w * s)), max(32, int(h * s))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


def build_transforms(img_size):
    """Forensic-safe augmentation.

    Deliberately absent: rotation, colour jitter, blur. Deepfake evidence is
    high-frequency (GAN checkerboard, JPEG grid, PRNU sensor noise) and resampling
    filters destroy exactly that signal. Flips and crops are geometry-preserving;
    compression augmentation attacks the shortcut the model would otherwise learn.
    """
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        RandomDownscale(),
        RandomJpeg(),
        transforms.ToTensor(),
        norm,
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    ])
    val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm,
    ])
    return train, val


def dhash(path, size=8):
    """64-bit difference hash: 1 bit per horizontal gradient sign.

    Robust to resize, re-compression and small brightness shifts, which is exactly the
    kind of near-duplication that leaks between splits.
    """
    with Image.open(path) as raw:
        small = raw.convert("L").resize((size + 1, size), Image.BILINEAR)
    px = small.tobytes()  # mode "L" is one byte per pixel, row-major
    bits = 0
    for r in range(size):
        row = px[r * (size + 1):(r + 1) * (size + 1)]
        for c in range(size):
            bits = (bits << 1) | (1 if row[c] < row[c + 1] else 0)
    return bits


def group_near_duplicates(paths, cache_path=None):
    """Union-find over LSH bands. Returns a group id per path.

    Naive pairwise Hamming distance is O(n²) — 1e10 comparisons on 100k images. Instead
    split each 64-bit hash into four 16-bit bands and union anything sharing a band:
    linear time, and any pair within ~3 bits almost certainly collides in some band.

    This is a stand-in for identity grouping. These Kaggle sets ship no identity labels,
    and the leak that actually matters here is near-duplicate frames and fake/original
    pairs straddling the split — which is what this catches.
    """
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as fh:
            cache = json.load(fh)

    print(f"🔎 Hashing {len(paths)} images for near-duplicate grouping...")
    hashes, missed = [], 0
    for i, p in enumerate(paths):
        key = os.path.basename(p)
        if key in cache:
            hashes.append(cache[key])
        else:
            try:
                h = dhash(p)
            except Exception:
                h = -(i + 1)  # unreadable: unique negative id, groups with nothing
                missed += 1
            cache[key] = h
            hashes.append(h)
        if (i + 1) % 5000 == 0:
            print(f"   ...{i + 1}/{len(paths)}")

    if cache_path:
        with open(cache_path, "w") as fh:
            json.dump(cache, fh)

    parent = list(range(len(paths)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for band in range(4):
        buckets = {}
        shift = band * 16
        for i, h in enumerate(hashes):
            if h < 0:
                continue
            buckets.setdefault((band, (h >> shift) & 0xFFFF), []).append(i)
        for members in buckets.values():
            for m in members[1:]:
                union(members[0], m)

    groups = [find(i) for i in range(len(paths))]
    n_groups = len(set(groups))
    print(f"   {len(paths)} images → {n_groups} groups "
          f"({len(paths) - n_groups} near-duplicates absorbed"
          f"{f', {missed} unreadable' if missed else ''})")
    return groups


def split_indices(targets, n_classes, val_frac, seed, groups=None):
    """Stratified holdout. With groups, whole groups move together.

    Group-aware splitting costs a little class balance — a group can span classes when a
    fake and its source original hash alike — so we fill val group-by-group and stop on
    the total, then verify no class was starved.
    """
    rng = random.Random(seed)
    if groups is None:
        by_class = {}
        for idx, label in enumerate(targets):
            by_class.setdefault(label, []).append(idx)
        train_idx, val_idx = [], []
        for idxs in by_class.values():
            rng.shuffle(idxs)
            cut = max(1, int(len(idxs) * val_frac))
            val_idx += idxs[:cut]
            train_idx += idxs[cut:]
        return train_idx, val_idx

    members = {}
    for idx, g in enumerate(groups):
        members.setdefault(g, []).append(idx)
    order = sorted(members)
    rng.shuffle(order)

    target_n = int(len(targets) * val_frac)
    val_idx, seen = [], 0
    for g in order:
        if seen >= target_n:
            break
        val_idx += members[g]
        seen += len(members[g])
    val_set = set(val_idx)
    train_idx = [i for i in range(len(targets)) if i not in val_set]

    for c in range(n_classes):
        in_val = sum(1 for i in val_idx if targets[i] == c)
        in_train = sum(1 for i in train_idx if targets[i] == c)
        if in_val == 0 or in_train == 0:
            sys.exit(f"❌ Group split starved class {c} (train={in_train}, val={in_val}). "
                     f"Try --split-by image or a larger --val-frac.")
    return train_idx, val_idx


def build_eval_loader(eval_dir, classes, img_size, batch_size, workers):
    """Loads a held-out dataset and remaps its labels onto the training class order.

    ImageFolder indexes alphabetically per directory, so 'real'/'fake' in one dataset and
    'Real'/'Fake' in another can silently produce inverted labels — and an inverted AUC
    reads as a catastrophically bad model rather than a bug. Match by name, not position.
    """
    _, val_tf = build_transforms(img_size)
    ds = datasets.ImageFolder(eval_dir, val_tf)
    lookup = {c.lower(): i for i, c in enumerate(classes)}
    missing = [c for c in ds.classes if c.lower() not in lookup]
    if missing:
        sys.exit(f"❌ {eval_dir} has classes {missing} absent from the trained model {classes}.")

    remap = {i: lookup[c.lower()] for i, c in enumerate(ds.classes)}
    if any(k != v for k, v in remap.items()):
        print(f"↔️  Remapping eval labels {ds.classes} → training order {classes}")
    ds.samples = [(p, remap[t]) for p, t in ds.samples]
    ds.targets = [t for _, t in ds.samples]
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)


def get_data_loaders(data_dir, img_size, batch_size, workers, val_frac=0.15, seed=42,
                     split_by="group"):
    """Returns loaders with a guaranteed-disjoint train/val split.

    The original code fell back to validating on the training directory, which made
    every reported accuracy a memorization score. If the dataset ships no Validation
    folder we now carve a stratified holdout instead.
    """
    train_tf, val_tf = build_transforms(img_size)

    nested = os.path.join(data_dir, "Dataset")
    root = nested if os.path.isdir(nested) else data_dir
    train_dir = os.path.join(root, "Train")
    val_dir = os.path.join(root, "Validation")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        train_ds = datasets.ImageFolder(train_dir, train_tf)
        val_ds = datasets.ImageFolder(val_dir, val_tf)
        classes = train_ds.classes
        assert val_ds.classes == classes, f"class mismatch: {val_ds.classes} vs {classes}"
        train_targets = train_ds.targets
        print("ℹ️  Using the dataset's own Train/Validation folders. If they share source "
              "images, that split leaks — rerun with --data-dir pointing at Train/ to "
              "force a group-aware holdout.")
    else:
        src = train_dir if os.path.isdir(train_dir) else root
        print(f"ℹ️  No Validation folder — carving a {val_frac:.0%} holdout from {src} "
              f"(split-by: {split_by})")
        # One index, two transforms: building ImageFolder per-transform stats every file
        # twice, which on a large tree is minutes of waiting for an index already in memory.
        print(f"ℹ️  Indexing {src} once...")
        base = datasets.ImageFolder(src)
        base_train = base
        classes = base.classes

        groups = None
        if split_by == "group":
            groups = group_near_duplicates(
                [p for p, _ in base.samples],
                cache_path=os.path.join(src, ".dhash_cache.json"),
            )

        train_idx, val_idx = split_indices(
            base.targets, len(classes), val_frac, seed, groups
        )
        train_ds = TransformSubset(base, train_idx, train_tf)
        val_ds = TransformSubset(base, val_idx, val_tf)
        train_targets = [base.targets[i] for i in train_idx]

    counts = [train_targets.count(i) for i in range(len(classes))]
    print(f"📊 Classes {classes} | train counts {counts} | val size {len(val_ds)}")
    if min(counts) == 0:
        sys.exit(f"❌ Class {classes[counts.index(0)]} has no training images.")

    pin = torch.cuda.is_available()
    drop_train_last = len(train_ds) > batch_size
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=workers, pin_memory=pin, drop_last=drop_train_last),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=workers, pin_memory=pin),
    }
    return loaders, classes, counts


def build_model(arch, num_classes):
    ctor, weights, _ = BACKBONES[arch]
    print(f"⚙️ Initializing pre-trained EfficientNet-{arch.upper()} ({num_classes} classes)...")
    model = ctor(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model, trainable):
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = trainable


@torch.no_grad()
def evaluate(model, loader, device, classes, real_idx):
    """Returns (logits, labels, metrics). Accuracy alone hides class collapse."""
    model.eval()
    all_logits, all_labels = [], []
    for inputs, labels in loader:
        all_logits.append(model(inputs.to(device)).float().cpu())
        all_labels.append(labels)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    preds = logits.argmax(1)
    n = len(classes)
    confusion = [[0] * n for _ in range(n)]
    for t, p in zip(labels.tolist(), preds.tolist()):
        confusion[t][p] += 1

    recalls = {}
    for i, name in enumerate(classes):
        total = sum(confusion[i])
        recalls[name] = confusion[i][i] / total if total else float("nan")

    probs = torch.softmax(logits, dim=1)
    auc = binary_auc(probs[:, real_idx].tolist(), [int(l == real_idx) for l in labels.tolist()])

    present = [r for r in recalls.values() if r == r]  # skip classes absent from the split
    metrics = {
        "acc": (preds == labels).float().mean().item(),
        "balanced_acc": sum(present) / len(present) if present else float("nan"),
        "auc_real_vs_rest": auc,
        "recall_per_class": recalls,
        "confusion": confusion,
    }
    return logits, labels, metrics


def train_model(model, loaders, device, classes, real_idx, epochs, lr, freeze_epochs, counts,
                out_path=MODEL_SAVE_PATH):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    total = sum(counts)
    weights = torch.tensor([total / (len(counts) * c) for c in counts], device=device)
    print(f"⚖️  Class weights: {[round(w, 3) for w in weights.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps_per_epoch = max(1, len(loaders["train"]))
    # OneCycle divides the run into warm-up and anneal phases; with only a handful of steps a
    # phase can come out zero-length and the scheduler divides by zero. A short run (smoke test,
    # a small fine-tune) gets a flat LR instead of a crash.
    total_steps = epochs * steps_per_epoch
    if total_steps < 10:
        scheduler = None
        print(f"ℹ️  Only {total_steps} optimizer steps — using a constant LR instead of OneCycle.")
    else:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch, pct_start=0.25
        )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_metric, best_state, start = -1.0, None, time.time()
    print(f"🚀 Training on {device} for {epochs} epochs (head-only for the first {freeze_epochs})...")

    for epoch in range(epochs):
        # Warm the randomly-initialized head before disturbing pretrained features.
        set_backbone_trainable(model, epoch >= freeze_epochs)
        model.train()
        running_loss, seen = 0.0, 0

        for inputs, labels in loaders["train"]:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(inputs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            running_loss += loss.item() * inputs.size(0)
            seen += inputs.size(0)

        _, _, m = evaluate(model, loaders["val"], device, classes, real_idx)
        print(f"\nEpoch {epoch + 1}/{epochs}  train_loss {running_loss / seen:.4f}")
        print(f"  val acc {m['acc']:.4f} | balanced {m['balanced_acc']:.4f} | AUC {m['auc_real_vs_rest']:.4f}")
        print(f"  recall {({k: round(v, 3) for k, v in m['recall_per_class'].items()})}")

        # Select on balanced accuracy: plain accuracy rewards predicting the majority class.
        if m["balanced_acc"] > best_metric:
            best_metric = m["balanced_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ⭐ new best (balanced acc {best_metric:.4f})")

    elapsed = time.time() - start
    print(f"\n🎉 Done in {elapsed // 60:.0f}m {elapsed % 60:.0f}s. Best balanced acc: {best_metric:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    return model


def report_cross_dataset(cross, in_dist, eval_dir, classes):
    """The number that actually predicts field performance.

    In-distribution accuracy on a Kaggle split says how well the model memorized one
    generator's fingerprint. What matters is the drop when the generator changes, and
    it is normally large. Print both so the gap is impossible to miss.
    """
    print("\n" + "=" * 62)
    print(f"🔬 CROSS-DATASET EVALUATION — {eval_dir}")
    print("=" * 62)
    print(f"  balanced acc  {cross['balanced_acc']:.4f}")
    print(f"  AUC           {cross['auc_real_vs_rest']:.4f}")
    print(f"  recall        {({k: round(v, 3) for k, v in cross['recall_per_class'].items()})}")
    print(f"  confusion     (rows=true {classes})")
    for name, row in zip(classes, cross["confusion"]):
        print(f"    {name:<12} {row}")

    if in_dist:
        drop = in_dist["balanced_acc"] - cross["balanced_acc"]
        print(f"\n  in-distribution {in_dist['balanced_acc']:.4f} → unseen {cross['balanced_acc']:.4f} "
              f"(drop {drop:+.4f})")
        if cross["auc_real_vs_rest"] < 0.5:
            print("  ⚠️  AUC below 0.5 — the model is anti-correlated on this set. Check the "
                  "class-name mapping before concluding anything.")
        elif drop > 0.20:
            print("  ⚠️  Large drop: the model learned this dataset's generator, not forgery in "
                  "general. Add generator diversity before trusting the in-distribution score.")
    print("=" * 62)


def export_to_onnx(model, device, img_size):
    """Optional. The FastAPI server runs the .pth directly, so a failure here is a
    warning, not a lost training run — the checkpoint is already on disk."""
    print(f"📦 Exporting to ONNX at: {ONNX_SAVE_PATH}...")
    os.makedirs(os.path.dirname(ONNX_SAVE_PATH), exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    try:
        torch.onnx.export(
            model, dummy, ONNX_SAVE_PATH,
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print("✅ ONNX export complete.")
    except ModuleNotFoundError as e:
        print(f"⚠️  ONNX export skipped ({e}). torch>=2.9 needs: pip install onnxscript onnx")
    except Exception as e:
        print(f"⚠️  ONNX export failed: {e}")


def selfcheck():
    """Runnable check for the non-obvious maths: AUC, calibration, split disjointness."""
    assert abs(binary_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) - 0.75) < 1e-9
    assert binary_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert binary_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert abs(binary_auc([0.5, 0.5], [0, 1]) - 0.5) < 1e-9, "ties must average to 0.5"

    # Overconfident logits must be cooled (T > 1) towards the true error rate.
    torch.manual_seed(0)
    labels = torch.randint(0, 2, (512,))
    logits = torch.zeros(512, 2)
    for i, l in enumerate(labels):
        correct = i % 4 != 0                      # 75% accurate...
        logits[i, l if correct else 1 - l] = 8.0  # ...but claims ~100% confidence
    assert fit_temperature(logits, labels) > 1.5, "calibration must temper overconfidence"

    # Confusion matrix rows must sum to the true class support.
    class Fake(torch.utils.data.Dataset):
        def __len__(self): return 40
        def __getitem__(self, i): return torch.zeros(3, 8, 8), i % 2

    net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))
    _, _, m = evaluate(net, DataLoader(Fake(), batch_size=8), torch.device("cpu"), ["fake", "real"], 1)
    assert [sum(r) for r in m["confusion"]] == [20, 20]
    assert set(m["recall_per_class"]) == {"fake", "real"}

    # Per-image split: disjoint, and every class is represented on both sides.
    targets = [0] * 100 + [1] * 100
    tr, va = split_indices(targets, 2, 0.2, seed=1)
    assert not (set(tr) & set(va)) and len(tr) + len(va) == 200
    assert sorted(targets[i] for i in va).count(0) == 20

    # Group split: a group must never straddle the split — that is the whole point.
    groups = [i // 5 for i in range(200)]  # 40 groups of 5
    tr, va = split_indices(targets, 2, 0.25, seed=1, groups=groups)
    assert not (set(tr) & set(va)) and len(tr) + len(va) == 200
    val_groups = {groups[i] for i in va}
    assert not (val_groups & {groups[i] for i in tr}), "group leaked across the split"

    # dHash grouping must pull a re-compressed near-duplicate in with its original,
    # and must keep a visually different image out.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        # Low-frequency content, like a real photo. Per-pixel noise would be destroyed by
        # JPEG and would test nothing that resembles the actual data.
        rng = random.Random(7)
        seed_img = Image.new("RGB", (16, 16))
        seed_img.putdata([tuple(rng.randrange(256) for _ in range(3)) for _ in range(16 * 16)])
        base = seed_img.resize((128, 128), Image.BILINEAR)

        paths = []
        for name, img in [
            ("a.png", base),
            ("a_jpeg.jpg", base),                                        # same image, lossy
            ("a_small.png", base.resize((48, 48)).resize((128, 128))),   # same image, resized
            ("b.png", base.transpose(Image.ROTATE_90)),                  # different content
        ]:
            p = os.path.join(tmp, name)
            if name.endswith(".jpg"):
                img.save(p, "JPEG", quality=40)
            else:
                img.save(p, "PNG")
            paths.append(p)
        g = group_near_duplicates(paths)
        assert g[0] == g[1] == g[2], f"near-duplicates must group: {g}"
        assert g[3] != g[0], f"distinct image must not group: {g}"

    print("✅ selfcheck passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=BACKBONES, default="b0")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--freeze-epochs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--data-dir", default=None, help="skip kagglehub and use this folder")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--split-by", choices=("group", "image"), default="group",
                    help="group: keep near-duplicates on one side of the split (default)")
    ap.add_argument("--eval-dir", default=None,
                    help="held-out dataset from a DIFFERENT source; the honest accuracy number")
    ap.add_argument("--eval-only", action="store_true",
                    help="load the saved checkpoint and just run --eval-dir")
    ap.add_argument("--out", default=MODEL_SAVE_PATH,
                    help="where to write the checkpoint. Defaults to the path the service reads, "
                         "which is also --init's default — pass a new path when fine-tuning so a "
                         "bad run cannot destroy the model you started from")
    ap.add_argument("--init", default=None,
                    help="fine-tune FROM this checkpoint instead of ImageNet weights "
                         "(e.g. models/face/deepfake_detector.pth)")
    ap.add_argument("--selfcheck", action="store_true", help="run internal assertions and exit")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return

    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.eval_only:
        if not args.eval_dir:
            sys.exit("❌ --eval-only needs --eval-dir.")
        if not os.path.exists(args.out):
            sys.exit(f"❌ No checkpoint at {args.out}.")
        ckpt = torch.load(args.out, map_location=device, weights_only=True)
        classes, real_idx, img_size = ckpt["classes"], ckpt["real_idx"], ckpt["img_size"]
        model = build_model(ckpt["arch"], len(classes)).to(device)
        model.load_state_dict(ckpt["state_dict"])
        loader = build_eval_loader(args.eval_dir, classes, img_size, args.batch_size, args.workers)
        _, _, m = evaluate(model, loader, device, classes, real_idx)
        report_cross_dataset(m, ckpt.get("val_metrics"), args.eval_dir, classes)
        return

    img_size = BACKBONES[args.arch][2]
    data_dir = args.data_dir or download_all_datasets()

    # preprocess_faces.py leaves a manifest; carry its settings into the checkpoint so
    # the server applies the SAME preprocessing. Train on crops and infer on full frames
    # and accuracy quietly collapses with nothing in the logs to explain it.
    manifest_path = os.path.join(data_dir, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        print(f"📋 manifest: face_crop={manifest.get('face_crop')} sources={manifest.get('sources')}")

    loaders, classes, counts = get_data_loaders(
        data_dir, img_size, args.batch_size, args.workers,
        val_frac=args.val_frac, split_by=args.split_by,
    )

    real_candidates = [i for i, c in enumerate(classes) if c.lower() in ("real", "authentic", "genuine")]
    if not real_candidates:
        sys.exit(f"❌ No 'real' class found in {classes}. Rename the authentic-image folder to 'Real'.")
    real_idx = real_candidates[0]
    print(f"🎯 Trust score will be P(class '{classes[real_idx]}') at index {real_idx}")

    model = build_model(args.arch, len(classes)).to(device)
    if args.init:
        prior = torch.load(args.init, map_location=device, weights_only=True)
        if prior.get("arch", args.arch) != args.arch:
            sys.exit(f"❌ {args.init} is arch '{prior.get('arch')}' but --arch is '{args.arch}'.")
        if list(prior.get("classes", classes)) != list(classes):
            sys.exit(f"❌ {args.init} was trained on classes {prior.get('classes')}, this data has "
                     f"{classes}. Fine-tuning across different label sets silently inverts scores.")
        model.load_state_dict(prior["state_dict"])
        print(f"↻ Fine-tuning from {args.init} "
              f"(its val balanced acc was {prior.get('val_metrics', {}).get('balanced_acc', float('nan')):.4f})")

    model = train_model(model, loaders, device, classes, real_idx,
                        args.epochs, args.lr, args.freeze_epochs, counts, args.out)

    logits, labels, metrics = evaluate(model, loaders["val"], device, classes, real_idx)
    temperature = fit_temperature(logits, labels)

    # Everything inference needs travels with the weights — no more arch/index guessing.
    torch.save({
        "state_dict": model.state_dict(),
        "arch": args.arch,
        "classes": classes,
        "real_idx": real_idx,
        "img_size": img_size,
        "temperature": temperature,
        "face_crop": bool(manifest.get("face_crop", False)),
        "face_margin": manifest.get("margin", 0.35),
        "val_metrics": metrics,
    }, args.out)
    print(f"💾 Saved checkpoint → {args.out}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion"}, indent=2))

    if args.eval_dir:
        loader = build_eval_loader(args.eval_dir, classes, img_size, args.batch_size, args.workers)
        _, _, cross = evaluate(model, loader, device, classes, real_idx)
        report_cross_dataset(cross, metrics, args.eval_dir, classes)

    export_to_onnx(model, device, img_size)


if __name__ == "__main__":
    main()
