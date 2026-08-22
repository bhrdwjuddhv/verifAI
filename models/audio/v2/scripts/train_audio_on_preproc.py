"""Retrain Audio Detector on AudioMelPreproc tensor output (not PIL spectrograms).

This ensures the CNN learns from the exact same tensor representation
that preproc.onnx produces at inference time, eliminating the PIL JPEG
quantization / resize mismatch that causes near-random ONNX predictions.
"""

import os
import sys
import random
import glob
import time
import json
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.export_audio_preproc import AudioMelPreproc, SR
from scripts.common.calibration import fit_temperature

MODEL_SAVE_PATH = "models/audio_deepfake_detector.pth"


class AudioWavDataset(Dataset):
    """Dataset that loads raw WAV files and produces mel-spectrogram tensors
    using the exact AudioMelPreproc module (matching preproc.onnx)."""

    def __init__(self, real_dir: str, fake_dir: str, augment: bool = False):
        self.preproc = AudioMelPreproc(include_imagenet_norm=True).eval()
        self.augment = augment
        self.samples = []  # (wav_path, label)  label: 0=Fake, 1=Real

        for wav in sorted(glob.glob(os.path.join(real_dir, "*.wav"))):
            self.samples.append((wav, 1))
        for wav in sorted(glob.glob(os.path.join(fake_dir, "*.wav"))):
            self.samples.append((wav, 0))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, label = self.samples[idx]
        y, _ = librosa.load(wav_path, sr=SR, mono=True)
        y3 = y[:SR * 3] if len(y) >= SR * 3 else np.pad(y, (0, SR * 3 - len(y)))

        with torch.no_grad():
            tensor = self.preproc(torch.tensor(y3[None, :], dtype=torch.float32))  # [1,3,224,224]

        img = tensor.squeeze(0)  # [3, 224, 224]

        # SpecAugment-style masking during training
        if self.augment:
            _, H, W = img.shape
            for _ in range(2):
                f_len = random.randint(0, min(20, H // 4))
                f0 = random.randint(0, H - f_len)
                img[:, f0:f0 + f_len, :] = 0.0

                t_len = random.randint(0, min(25, W // 4))
                t0 = random.randint(0, W - t_len)
                img[:, :, t0:t0 + t_len] = 0.0

        return img, label


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction='none',
                                         label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


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


def main():
    from torchvision import models

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    real_dir = "data/audio_realistic_dataset/Real"
    fake_dir = "data/audio_realistic_dataset/Fake"

    if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
        print("Dataset not found. Run scripts/generate_rich_audio_dataset.py first.")
        return

    # Build datasets from raw WAV files using AudioMelPreproc
    full_ds = AudioWavDataset(real_dir, fake_dir, augment=False)
    n_total = len(full_ds)
    n_val = max(2, int(n_total * 0.2))
    n_train = n_total - n_val
    train_ds_base, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])

    # Create augmented training dataset
    train_ds_aug = AudioWavDataset(real_dir, fake_dir, augment=True)
    # Use the same indices as train_ds_base
    train_ds = torch.utils.data.Subset(train_ds_aug, train_ds_base.indices)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    classes = ["Fake", "Real"]
    real_idx = 1

    # Build model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_feat = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feat, 2)
    model = model.to(device)

    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    classifier_params = [p for n, p in model.named_parameters() if "classifier" in n]
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": 2e-4},
        {"params": classifier_params, "lr": 8e-4}
    ], weight_decay=1e-2)

    epochs = 10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_state = None

    print(f"\nTraining on {n_train} samples, validating on {n_val} samples ({n_total} total)")
    print(f"Training for {epochs} epochs with Focal Loss, SpecAugment, and AudioMelPreproc tensors\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                val_logits.append(model(inputs.to(device)).cpu())
                val_labels.append(labels)

        logits = torch.cat(val_logits)
        labels_t = torch.cat(val_labels)
        val_preds = logits.argmax(1)
        val_acc = (val_preds == labels_t).float().mean().item()

        probs = torch.softmax(logits, dim=1)
        auc = binary_auc(probs[:, real_idx].tolist(), [int(l == real_idx) for l in labels_t.tolist()])

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - t0
        print(f"Epoch {epoch:2d}/{epochs} [{elapsed:4.1f}s] | "
              f"Train Loss: {running_loss/max(1,total):.4f} Acc: {correct/max(1,total)*100:5.1f}% | "
              f"Val Acc: {val_acc*100:5.1f}% | AUC: {auc:.4f}")

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    # Calibrate
    model.eval()
    val_logits, val_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            val_logits.append(model(inputs.to(device)).cpu())
            val_labels.append(labels)
    logits = torch.cat(val_logits)
    labels_t = torch.cat(val_labels)
    temperature = fit_temperature(logits, labels_t)

    # Save
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "arch": "b0",
        "classes": classes,
        "real_idx": real_idx,
        "modality": "audio",
        "img_size": 224,
        "temperature": temperature,
    }, MODEL_SAVE_PATH)
    print(f"\nSaved best model (Val Acc: {best_val_acc*100:.1f}%) -> {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
