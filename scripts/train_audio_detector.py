"""Audio deepfake detector: EfficientNet/MobileNet over mel-spectrogram images.

    python scripts/train_audio_detector.py --data-dir data/audio_spectrograms --epochs 15
    python scripts/train_audio_detector.py --selfcheck

Spectrograms come from preprocess_audio.py, which names every file
``{source_tag}__{folder}__{original_stem}.jpg`` — that prefix is what lets this script split
by SOURCE rather than at random. Reading the same speaker, or the same corpus split, in both
train and validation is how a voice detector reports 0.97 and then fails on real uploads.
"""

import argparse
import json
import os
import random
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.calibration import fit_temperature
from common.config import AUDIO_CHECKPOINT as MODEL_SAVE_PATH
from common.datasets import TransformSubset
from common.metrics import binary_auc
from common.xai import gradcam_overlay
from train_deepfake_detector import split_indices

BACKBONES = {
    "b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 224),
    "mobilenet": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT, 224),
}


def build_audio_transforms(img_size: int):
    """No horizontal flip.

    A mel-spectrogram's x-axis is time: flipping it plays the audio backwards, which is not a
    label-preserving augmentation — it is a clip that could never occur. The previous version
    flipped, and worse, applied the *training* transform to the validation set too, so every
    reported validation number carried random augmentation noise.
    """
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm,
        # Time/frequency masking (SpecAugment-style): the standard, label-preserving
        # augmentation for spectrograms — occlude a band, keep the meaning.
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm,
    ])
    return train_tf, val_tf


def source_key(path):
    """Group id from preprocess_audio's filename convention.

    ``audio_raw__fake_cloned__for_train_fake_0001.jpg`` -> ``audio_raw|for_train``

    The class folder is deliberately dropped: including it would make every group belong to
    exactly one class, and a group-aware split would then hand an entire class to validation.
    The stem's leading tokens are what actually identify the corpus and its split.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("__")
    tag = parts[0] if parts else ""
    rest = parts[-1] if len(parts) > 1 else ""
    tokens = [t for t in rest.split("_") if t and not t.isdigit()][:2]
    return f"{tag}|{'_'.join(tokens)}" if tokens else tag


def build_groups(paths, targets, n_classes):
    """Source groups, but only if they can actually carry a split.

    Returns (groups, reason). A group scheme that leaves any class with fewer than two groups
    cannot produce a split that holds out a source without holding out a class, so it is
    rejected and the caller falls back to a stratified split — with that fact printed, not
    hidden.
    """
    groups = [source_key(p) for p in paths]
    per_class = {c: set() for c in range(n_classes)}
    for g, t in zip(groups, targets):
        per_class[t].add(g)

    distinct = sorted({g for g in groups})
    if len(distinct) < 2 or any(len(v) < 2 for v in per_class.values()):
        return None, (f"only {len(distinct)} source group(s) "
                      f"({', '.join(distinct[:6])}) — not enough to split by source")

    index = {g: i for i, g in enumerate(distinct)}
    return [index[g] for g in groups], f"{len(distinct)} source groups: {', '.join(distinct[:8])}"


@torch.no_grad()
def evaluate(model, loader, device, classes, real_idx):
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

    present = [r for r in recalls.values() if r == r]
    metrics = {
        "acc": (preds == labels).float().mean().item(),
        "balanced_acc": sum(present) / len(present) if present else float("nan"),
        "auc_real_vs_rest": auc,
        "recall_per_class": recalls,
        "confusion": confusion,
    }
    return logits, labels, metrics


def build_model(arch, num_classes):
    ctor, weights, _ = BACKBONES[arch]
    model = ctor(weights=weights)
    if isinstance(model.classifier, nn.Sequential):
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feat, num_classes)
    else:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def selfcheck():
    """The maths that decides labels, splits and weighting — the parts that fail silently."""
    assert abs(binary_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) - 0.75) < 1e-9

    # Source keys must ignore the class folder, or every group belongs to one class.
    assert source_key("audio_raw__fake_cloned__for_train_fake_0001.jpg") == "audio_raw|for_train"
    assert source_key("audio_raw__real_speech__for_train_real_0002.jpg") == "audio_raw|for_train"
    assert (source_key("audio_raw__real_speech__asvspoof_dev_x.jpg")
            != source_key("audio_raw__real_speech__for_train_x.jpg"))

    # A corpus present in both classes can carry a split; one group per class cannot.
    paths_ok = ([f"t__real__for_train_{i}.jpg" for i in range(4)]
                + [f"t__fake__for_train_{i}.jpg" for i in range(4)]
                + [f"t__real__asv_dev_{i}.jpg" for i in range(4)]
                + [f"t__fake__asv_dev_{i}.jpg" for i in range(4)])
    targets_ok = [0] * 4 + [1] * 4 + [0] * 4 + [1] * 4
    groups, reason = build_groups(paths_ok, targets_ok, 2)
    assert groups is not None and "2 source groups" in reason, reason

    # The degenerate case the recommendation missed: one folder per class, one group each.
    paths_bad = [f"t__real__x_{i}.jpg" for i in range(4)] + [f"t__fake__x_{i}.jpg" for i in range(4)]
    groups_bad, reason_bad = build_groups(paths_bad, [0] * 4 + [1] * 4, 2)
    assert groups_bad is None and "not enough" in reason_bad, reason_bad

    # Group split must not leak a source across the boundary.
    tr, va = split_indices(targets_ok, 2, 0.25, seed=1, groups=groups)
    assert not (set(tr) & set(va))
    assert not ({groups[i] for i in va} & {groups[i] for i in tr}), "source leaked across the split"

    # Stratified fallback keeps both classes on both sides.
    tr, va = split_indices([0] * 50 + [1] * 50, 2, 0.2, seed=1)
    assert sorted(([0] * 50 + [1] * 50)[i] for i in va).count(0) == 10

    # Class weights must favour the minority class, not the majority.
    counts = [37469, 77421]                       # the real imbalance in this dataset
    total = sum(counts)
    w = [total / (len(counts) * c) for c in counts]
    assert w[0] > 1 > w[1], w
    assert abs(w[0] * counts[0] - w[1] * counts[1]) < 1e-6, "weighted mass must balance"

    print("✅ audio training selfcheck passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=BACKBONES, default="b0")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--data-dir", default="data/audio_spectrograms")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--split-by", choices=("source", "stratified"), default="source",
                    help="source: hold out whole corpora (falls back to stratified if it cannot)")
    ap.add_argument("--patience", type=int, default=4, help="early-stop after N epochs without gain")
    ap.add_argument("--out", default=MODEL_SAVE_PATH)
    ap.add_argument("--out-dir", default="runs/latest_audio_run")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return

    torch.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("⚠️ No CUDA GPU — training on CPU will be slow.")

    if not os.path.exists(args.data_dir):
        sys.exit(f"❌ '{args.data_dir}' not found. Run scripts/preprocess_audio.py first.")

    train_tf, val_tf = build_audio_transforms(224)
    # One index, two transforms. Each split still gets its own augmentation — the previous
    # version gave validation the *training* transform, which is the bug that mattered — but
    # the tree is only walked once.
    print(f"Indexing {args.data_dir} once...")
    base = datasets.ImageFolder(args.data_dir)
    classes = base.classes
    targets = base.targets
    paths = [p for p, _ in base.samples]

    real_candidates = [i for i, c in enumerate(classes) if c.lower() in ("real", "authentic", "genuine")]
    if not real_candidates:
        sys.exit(f"❌ No 'real' class in {classes}. Defaulting to index 0 would invert every score.")
    real_idx = real_candidates[0]

    groups, reason = (None, "stratified split requested")
    if args.split_by == "source":
        groups, reason = build_groups(paths, targets, len(classes))
    print(f"🔀 Split: {reason}")
    if groups is None and args.split_by == "source":
        print("   Falling back to a stratified split. Validation may therefore share a corpus "
              "with training, which inflates the number — test on a held-out corpus before "
              "trusting it.")

    train_idx, val_idx = split_indices(targets, len(classes), args.val_frac, 42, groups)
    train_ds = TransformSubset(base, train_idx, train_tf)
    val_ds = TransformSubset(base, val_idx, val_tf)

    counts = [sum(1 for i in train_idx if targets[i] == c) for c in range(len(classes))]
    print(f"📊 Classes {classes} (real='{classes[real_idx]}') | train {counts} | val {len(val_idx)}")
    if min(counts) == 0:
        sys.exit(f"❌ Class {classes[counts.index(0)]} has no training samples.")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin,
                              drop_last=len(train_idx) > args.batch_size,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin,
                            persistent_workers=args.workers > 0)

    model = build_model(args.arch, len(classes)).to(device)

    # Class weights: 37k real vs 77k fake trains a model that guesses "fake" and looks accurate.
    total = sum(counts)
    weights = torch.tensor([total / (len(counts) * c) for c in counts],
                           dtype=torch.float32, device=device)
    print(f"⚖️  Class weights: {[round(w, 3) for w in weights.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_metric, best_state, stale = -1.0, None, 0
    print(f"🚀 Training {args.arch} on {device} for up to {args.epochs} epochs "
          f"(early stop after {args.patience} without improvement)...")

    for epoch in range(args.epochs):
        model.train()
        running_loss, seen = 0.0, 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(inputs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
            seen += inputs.size(0)

        _, _, m = evaluate(model, val_loader, device, classes, real_idx)
        scheduler.step(m["balanced_acc"])
        print(f"Epoch {epoch + 1}/{args.epochs} | loss {running_loss / max(1, seen):.4f} | "
              f"acc {m['acc']:.4f} | balanced {m['balanced_acc']:.4f} | AUC {m['auc_real_vs_rest']:.4f} | "
              f"recall {({k: round(v, 3) for k, v in m['recall_per_class'].items()})}")

        # Select on balanced accuracy: plain accuracy rewards predicting the majority class.
        if m["balanced_acc"] > best_metric:
            best_metric, stale = m["balanced_acc"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ⭐ new best (balanced {best_metric:.4f})")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"⏹️  Early stop at epoch {epoch + 1}; best balanced {best_metric:.4f}.")
                break

    if best_state is None:
        sys.exit("❌ No epoch completed — nothing to save.")
    print("↩️  Restoring best epoch (not the last one).")
    model.load_state_dict(best_state)

    logits, labels, metrics = evaluate(model, val_loader, device, classes, real_idx)
    temperature = fit_temperature(logits, labels)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "arch": args.arch,
        "classes": classes,
        "real_idx": real_idx,
        "modality": "audio",
        "img_size": 224,
        "temperature": temperature,
        "val_metrics": metrics,
        "class_counts": dict(zip(classes, counts)),
        "split": reason,
    }, args.out)

    print("\nFINAL VALIDATION METRICS")
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion"}, indent=2))
    print(f"confusion (rows=true {classes}): {metrics['confusion']}")
    print(f"temperature {temperature:.4f}")
    print(f"💾 Saved checkpoint → {args.out}")
    print("\nNext: python scripts/export_onnx.py --audio")
    if groups is None:
        print("⚠️  This number came from a split that may share a corpus with training. "
              "Evaluate on an untouched corpus (FoR testing, ASVspoof eval) before quoting it.")

    try:
        sample_input, sample_label = val_ds[0]
        xai_out = os.path.join(args.out_dir, "xai_explanation.jpg")
        overlay = gradcam_overlay(model, sample_input.unsqueeze(0).to(device), sample_label,
                                  transforms.ToPILImage()(sample_input))
        if overlay:
            import base64

            os.makedirs(os.path.dirname(xai_out) or ".", exist_ok=True)
            with open(xai_out, "wb") as fh:
                fh.write(base64.b64decode(overlay.split(",", 1)[1]))
            print(f"Saved Grad-CAM sample -> {xai_out}")
        else:
            print("Grad-CAM unavailable for this model — skipped.")
    except Exception as e:
        print(f"⚠️ XAI export note: {e}")


if __name__ == "__main__":
    main()
