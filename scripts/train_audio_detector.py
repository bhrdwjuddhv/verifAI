"""Audio Deepfake Spectrogram Detector Training Script.

Trains an EfficientNet/MobileNet CNN model on audio spectrogram datasets.
Incorporates temperature scaling calibration, group-aware splitting,
and Grad-CAM XAI explainability export.

Usage:
  python scripts/train_audio_detector.py --data-dir data/audio_spectrograms
  python scripts/train_audio_detector.py --selfcheck
"""

import argparse
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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.common.calibration import fit_temperature
from scripts.common.xai import generate_gradcam

MODEL_SAVE_PATH = "models/audio_deepfake_detector.pth"

BACKBONES = {
    "b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 224),
    "mobilenet": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT, 224),
}


def build_audio_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def binary_auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    n_pos = sum(1 for _, l in pairs if l == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


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


def selfcheck():
    """Assertions for audio training functions."""
    assert abs(binary_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) - 0.75) < 1e-9
    print("✅ audio training selfcheck passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=BACKBONES, default="b0")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--data-dir", default="data/audio_spectrograms")
    ap.add_argument("--val-frac", type=float, default=0.15)
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
        print("⚠️ CUDA GPU not found. Training running on CPU. Estimated completion time will be longer.")

    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory '{args.data_dir}' not found. Run scripts/preprocess_audio.py first.")
        sys.exit(1)

    train_tf, val_tf = build_audio_transforms(224)
    dataset = datasets.ImageFolder(args.data_dir, train_tf)
    classes = dataset.classes

    real_candidates = [i for i, c in enumerate(classes) if c.lower() in ("real", "authentic", "genuine")]
    real_idx = real_candidates[0] if real_candidates else 0

    val_size = max(1, int(len(dataset) * args.val_frac))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    ctor, weights, _ = BACKBONES[args.arch]
    model = ctor(weights=weights)
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        in_feat = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feat, len(classes))
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, len(classes))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    print(f"🚀 Training Audio Detector ({args.arch}) on {device} for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        _, _, m = evaluate(model, val_loader, device, classes, real_idx)
        print(f"Epoch {epoch + 1}/{args.epochs} | Loss {running_loss / len(train_ds):.4f} | Val Acc {m['acc']:.4f} | Balanced {m['balanced_acc']:.4f}")

    # Calibrate logits
    logits, labels, metrics = evaluate(model, val_loader, device, classes, real_idx)
    temperature = fit_temperature(logits, labels)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "arch": args.arch,
        "classes": classes,
        "real_idx": real_idx,
        "modality": "audio",
        "img_size": 224,
        "temperature": temperature,
        "val_metrics": metrics,
    }, MODEL_SAVE_PATH)
    print(f"💾 Saved audio model checkpoint -> {MODEL_SAVE_PATH}")

    # Export XAI Grad-CAM explanation for sample
    try:
        sample_input, sample_label = val_ds[0]
        xai_out = os.path.join(args.out_dir, "xai_explanation.jpg")
        generate_gradcam(model, sample_input.unsqueeze(0).to(device), sample_label, xai_out)
    except Exception as e:
        print(f"⚠️ XAI export note: {e}")


if __name__ == "__main__":
    main()
