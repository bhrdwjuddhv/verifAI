"""Train Audio Deepfake Detector on real-world ASVspoof / enhanced data.

Uses AudioMelPreproc tensors directly (matching preproc.onnx) to ensure
perfect parity between training and ONNX inference.

Enhanced training with:
- Mixup augmentation
- Stronger SpecAugment  
- Warmup + Cosine schedule
- Class-balanced sampling
- More epochs for better convergence
"""

import os
import sys
import random
import glob
import time
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.export_audio_preproc import AudioMelPreproc, SR
from scripts.common.calibration import fit_temperature

MODEL_SAVE_PATH = "models/audio_deepfake_detector.pth"


class RealWorldAudioDataset(Dataset):
    """Dataset that loads WAV files from multiple data directories."""

    def __init__(self, data_dirs: list, augment: bool = False):
        """
        Args:
            data_dirs: list of (real_dir, fake_dir) tuples
            augment: apply SpecAugment + noise augmentation
        """
        self.preproc = AudioMelPreproc(include_imagenet_norm=True).eval()
        self.augment = augment
        self.samples = []

        for real_dir, fake_dir in data_dirs:
            if os.path.isdir(real_dir):
                for wav in sorted(glob.glob(os.path.join(real_dir, "*.wav"))):
                    self.samples.append((wav, 1))  # 1 = Real
            if os.path.isdir(fake_dir):
                for wav in sorted(glob.glob(os.path.join(fake_dir, "*.wav"))):
                    self.samples.append((wav, 0))  # 0 = Fake

        random.shuffle(self.samples)
        print(f"  Loaded {len(self.samples)} samples "
              f"({sum(1 for _, l in self.samples if l == 1)} real, "
              f"{sum(1 for _, l in self.samples if l == 0)} fake)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, label = self.samples[idx]
        try:
            y, _ = librosa.load(wav_path, sr=SR, mono=True)
        except Exception:
            y = np.zeros(SR * 3, dtype=np.float32)

        y3 = y[:SR * 3] if len(y) >= SR * 3 else np.pad(y, (0, SR * 3 - len(y)))

        # Random time shift augmentation
        if self.augment and random.random() > 0.5:
            shift = random.randint(-SR // 4, SR // 4)
            y3 = np.roll(y3, shift)

        # Random gain augmentation
        if self.augment and random.random() > 0.5:
            gain = random.uniform(0.7, 1.3)
            y3 = np.clip(y3 * gain, -1.0, 1.0)

        # Add background noise
        if self.augment and random.random() > 0.4:
            snr = random.uniform(15, 35)  # dB
            noise = np.random.randn(len(y3)).astype(np.float32)
            signal_power = np.mean(y3 ** 2) + 1e-10
            noise_power = signal_power / (10 ** (snr / 10))
            y3 = y3 + noise * np.sqrt(noise_power)

        with torch.no_grad():
            tensor = self.preproc(torch.tensor(y3[None, :], dtype=torch.float32))

        img = tensor.squeeze(0)

        # SpecAugment masking
        if self.augment:
            _, H, W = img.shape
            # Frequency masking (2 bands)
            for _ in range(2):
                f_len = random.randint(0, min(27, H // 5))
                f0 = random.randint(0, H - f_len)
                img[:, f0:f0 + f_len, :] = 0.0
            # Time masking (2 bands)
            for _ in range(2):
                t_len = random.randint(0, min(30, W // 5))
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

    # Collect all available data directories
    data_dirs = []
    possible_pairs = [
        ("data/asvspoof_realworld/Real", "data/asvspoof_realworld/Fake"),
        ("data/audio_realistic_dataset/Real", "data/audio_realistic_dataset/Fake"),
    ]
    for real_dir, fake_dir in possible_pairs:
        if os.path.isdir(real_dir) or os.path.isdir(fake_dir):
            data_dirs.append((real_dir, fake_dir))

    if not data_dirs:
        print("No data directories found! Run download_real_audio_data.py first.")
        return

    print("Building dataset from raw WAV files using AudioMelPreproc...")
    full_ds = RealWorldAudioDataset(data_dirs, augment=False)
    n_total = len(full_ds)

    if n_total < 10:
        print(f"Only {n_total} samples found. Need at least 10.")
        return

    # 80/20 train/val split
    n_val = max(4, int(n_total * 0.2))
    n_train = n_total - n_val
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_total))

    # Augmented training dataset
    train_ds_aug = RealWorldAudioDataset(data_dirs, augment=True)
    train_ds = torch.utils.data.Subset(train_ds_aug, train_indices)
    val_ds = torch.utils.data.Subset(full_ds, val_indices)

    # Class-balanced sampling
    train_labels = [full_ds.samples[i][1] for i in train_indices]
    n_real = sum(train_labels)
    n_fake = len(train_labels) - n_real
    weights = [1.0 / n_real if l == 1 else 1.0 / n_fake for l in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    # Model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_feat = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feat, 2)
    model = model.to(device)

    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)

    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    classifier_params = [p for n, p in model.named_parameters() if "classifier" in n]

    epochs = 5
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": 1e-4},
        {"params": classifier_params, "lr": 5e-4}
    ], weight_decay=1e-2)

    # Warmup + Cosine
    warmup_epochs = 2
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc = 0.0
    best_auc = 0.0
    best_state = None
    patience = 5
    no_improve = 0

    print(f"\nTraining: {n_train} train / {n_val} val ({n_total} total)")
    print(f"Epochs: {epochs}, Warmup: {warmup_epochs}, Patience: {patience}")
    print(f"Augmentations: SpecAugment, TimeShift, Gain, BackgroundNoise\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Mixup augmentation (50% of batches)
            if random.random() > 0.5 and inputs.size(0) > 1:
                lam = np.random.beta(0.2, 0.2)
                idx = torch.randperm(inputs.size(0)).to(device)
                mixed = lam * inputs + (1 - lam) * inputs[idx]
                loss = lam * criterion(model(mixed), labels) + (1 - lam) * criterion(model(mixed), labels[idx])
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                preds = model(inputs).argmax(1) if random.random() > 0.5 else outputs.argmax(1) if 'outputs' in dir() else model(inputs).argmax(1)
            running_loss += loss.item() * inputs.size(0)
            correct += (model(inputs).argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_logits_list, val_labels_list = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                val_logits_list.append(model(inputs.to(device)).cpu())
                val_labels_list.append(labels)

        logits = torch.cat(val_logits_list)
        labels_t = torch.cat(val_labels_list)
        val_preds = logits.argmax(1)
        val_acc = (val_preds == labels_t).float().mean().item()

        probs = torch.softmax(logits, dim=1)
        auc = binary_auc(probs[:, 1].tolist(), [int(l == 1) for l in labels_t.tolist()])

        # Per-class accuracy
        real_mask = labels_t == 1
        fake_mask = labels_t == 0
        real_acc = (val_preds[real_mask] == labels_t[real_mask]).float().mean().item() if real_mask.sum() > 0 else 0
        fake_acc = (val_preds[fake_mask] == labels_t[fake_mask]).float().mean().item() if fake_mask.sum() > 0 else 0

        improved = ""
        if val_acc > best_val_acc or (val_acc == best_val_acc and auc > best_auc):
            best_val_acc = val_acc
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            improved = " *BEST*"
        else:
            no_improve += 1

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:2d}/{epochs} [{elapsed:5.1f}s] LR={lr_now:.2e} | "
              f"Loss: {running_loss/max(1,total):.4f} TrAcc: {correct/max(1,total)*100:5.1f}% | "
              f"Val: {val_acc*100:5.1f}% (R:{real_acc*100:.0f}% F:{fake_acc*100:.0f}%) AUC:{auc:.4f}{improved}")

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    # Calibrate
    model.eval()
    val_logits_list, val_labels_list = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            val_logits_list.append(model(inputs.to(device)).cpu())
            val_labels_list.append(labels)
    logits = torch.cat(val_logits_list)
    labels_t = torch.cat(val_labels_list)
    temperature = fit_temperature(logits, labels_t)

    # Save
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "arch": "b0",
        "classes": ["Fake", "Real"],
        "real_idx": 1,
        "modality": "audio",
        "img_size": 224,
        "temperature": temperature,
    }, MODEL_SAVE_PATH)
    print(f"\nSaved best model (Val Acc: {best_val_acc*100:.1f}%, AUC: {best_auc:.4f}) -> {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
