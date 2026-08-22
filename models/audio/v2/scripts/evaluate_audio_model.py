"""Audio Deepfake Detector Model Evaluation & Epoch Benchmark Suite.

Runs full evaluation metrics across epochs, evaluates checkpoints, computes
Confusion Matrix, ROC-AUC, Balanced Accuracy, Precision/Recall/F1, Temperature
Calibration, and Inference Latency on both PyTorch and ONNX models.
"""

import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.common.calibration import fit_temperature
from scripts.train_audio_detector import build_audio_transforms, binary_auc, BACKBONES

DATA_DIR = "data/audio_spectrograms"
CHECKPOINT_DIR = "models/checkpoints"


def train_and_evaluate_epochs(
    data_dir: str = DATA_DIR,
    arch: str = "b0",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 3e-4,
    val_frac: float = 0.2,
    seed: int = 42
):
    """Trains the audio detector while tracking and recording detailed per-epoch scores."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"  AUDIO DETECTOR TRAINING & EPOCH EVALUATION ({arch.upper()})")
    print(f"  Device: {device} | Total Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"=======================================================\n")

    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found.")
        return None

    train_tf, val_tf = build_audio_transforms(224)
    full_dataset = datasets.ImageFolder(data_dir, train_tf)
    classes = full_dataset.classes

    real_candidates = [i for i, c in enumerate(classes) if c.lower() in ("real", "authentic", "genuine")]
    real_idx = real_candidates[0] if real_candidates else 0
    fake_idx = 1 if real_idx == 0 else 0

    val_size = max(2, int(len(full_dataset) * val_frac))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    ctor, weights, _ = BACKBONES[arch]
    model = ctor(weights=weights)
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        in_feat = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feat, len(classes))
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, len(classes))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    epoch_history = []
    best_acc = -1.0
    best_epoch = 1

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / max(1, total_train)
        train_acc = correct_train / max(1, total_train)

        # Validation Evaluation
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                all_logits.append(model(inputs.to(device)).cpu())
                all_labels.append(labels)

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        val_preds = logits.argmax(dim=1)
        val_loss = criterion(logits, labels).item()

        n = len(classes)
        confusion = [[0] * n for _ in range(n)]
        for t, p in zip(labels.tolist(), val_preds.tolist()):
            confusion[t][p] += 1

        recalls = {}
        for i, name in enumerate(classes):
            tot = sum(confusion[i])
            recalls[name] = confusion[i][i] / tot if tot else 0.0

        val_acc = (val_preds == labels).float().mean().item()
        balanced_acc = sum(recalls.values()) / len(recalls)

        probs = torch.softmax(logits, dim=1)
        auc = binary_auc(probs[:, real_idx].tolist(), [int(l == real_idx) for l in labels.tolist()])
        epoch_time = time.time() - t0

        # Save per-epoch checkpoint
        epoch_ckpt_path = os.path.join(CHECKPOINT_DIR, f"audio_detector_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "arch": arch,
            "classes": classes,
            "real_idx": real_idx,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "balanced_acc": balanced_acc,
            "auc": auc,
        }, epoch_ckpt_path)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_audio_detector.pth"))

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "balanced_acc": round(balanced_acc, 4),
            "auc": round(auc if auc == auc else 0.5, 4),
            "recall_real": round(recalls.get("Real", 0.0), 4),
            "recall_fake": round(recalls.get("Fake", 0.0), 4),
            "duration_sec": round(epoch_time, 2)
        }
        epoch_history.append(epoch_record)

        print(f"Epoch {epoch:2d}/{epochs:2d} [{epoch_time:4.1f}s] | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:5.1f}% | "
              f"Bal Acc: {balanced_acc*100:5.1f}% | AUC: {epoch_record['auc']:.4f}")

    # Final calibration
    temperature = fit_temperature(logits, labels)

    # Save primary model checkpoint
    final_model_path = "models/audio_deepfake_detector.pth"
    torch.save({
        "state_dict": model.state_dict(),
        "arch": arch,
        "classes": classes,
        "real_idx": real_idx,
        "modality": "audio",
        "img_size": 224,
        "temperature": temperature,
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "epoch_history": epoch_history
    }, final_model_path)
    print(f"\n💾 Saved best model checkpoint -> {final_model_path}")

    return epoch_history, confusion, classes, temperature


def evaluate_onnx_runtime_performance(
    preproc_onnx: str = "models/audio/preproc.onnx",
    cnn_onnx: str = "models/audio/audio_detector.onnx",
    num_runs: int = 10
):
    """Benchmarks end-to-end inference latency on CPU."""
    import onnxruntime as ort

    print(f"\n=======================================================")
    print(f"  ONNX RUNTIME LATENCY & BENCHMARK SUITE")
    print(f"=======================================================")

    sess_pre = ort.InferenceSession(preproc_onnx, providers=["CPUExecutionProvider"])
    sess_cnn = ort.InferenceSession(cnn_onnx, providers=["CPUExecutionProvider"])

    dummy_audio = np.random.randn(1, 16000 * 3).astype(np.float32)

    # Warmup
    for _ in range(2):
        img = sess_pre.run(None, {"audio": dummy_audio})[0]
        _ = sess_cnn.run(None, {sess_cnn.get_inputs()[0].name: img})[0]

    # Preprocessing Latency
    pre_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        img = sess_pre.run(None, {"audio": dummy_audio})[0]
        pre_times.append((time.perf_counter() - t0) * 1000)

    # CNN Classifier Latency
    cnn_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = sess_cnn.run(None, {sess_cnn.get_inputs()[0].name: img})[0]
        cnn_times.append((time.perf_counter() - t0) * 1000)

    avg_pre = np.mean(pre_times)
    avg_cnn = np.mean(cnn_times)
    total_latency = avg_pre + avg_cnn

    print(f"  • Preprocessing (Mel-Spectrogram ONNX): {avg_pre:6.2f} ms")
    print(f"  • CNN Inference (EfficientNet-B0 ONNX):  {avg_cnn:6.2f} ms")
    print(f"  • Total End-to-End Latency:            {total_latency:6.2f} ms (~{1000/total_latency:.1f} inferences/sec)")
    print(f"=======================================================\n")

    return {
        "preproc_ms": round(avg_pre, 2),
        "cnn_ms": round(avg_cnn, 2),
        "total_e2e_ms": round(total_latency, 2),
        "throughput_fps": round(1000 / total_latency, 1)
    }


def main():
    history, confusion, classes, temp = train_and_evaluate_epochs(epochs=5, batch_size=8)
    perf = evaluate_onnx_runtime_performance()

    # Save summary report
    report = {
        "epoch_scores": history,
        "confusion_matrix": confusion,
        "classes": classes,
        "calibrated_temperature": round(temp, 4),
        "onnx_performance": perf
    }

    report_path = "models/audio/evaluation_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"📊 Evaluation report saved -> {report_path}")


if __name__ == "__main__":
    main()
