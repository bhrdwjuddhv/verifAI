"""Evaluate the full ONNX pipeline on all audio datasets."""
import sys, os, glob
import numpy as np
import librosa
import onnxruntime as ort

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SR = 16000
DUR = 3

sess_pre = ort.InferenceSession('models/audio/preproc.onnx', providers=['CPUExecutionProvider'])
sess_cnn = ort.InferenceSession('models/audio/audio_detector.onnx', providers=['CPUExecutionProvider'])

def evaluate_clip(wav_path):
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    y3 = y[:SR*DUR] if len(y) >= SR*DUR else np.pad(y, (0, SR*DUR - len(y)))
    img = sess_pre.run(None, {'audio': y3[None, :].astype(np.float32)})[0]
    out = sess_cnn.run(None, {sess_cnn.get_inputs()[0].name: img})[0]
    exp = np.exp(out - np.max(out))
    probs = exp / np.sum(exp)
    return float(probs[0, 0]), float(probs[0, 1])  # (Fake_prob, Real_prob)

dirs_to_eval = [
    ("Rich Acoustic Dataset", "data/audio_realistic_dataset/Real", "data/audio_realistic_dataset/Fake"),
    ("Multi-Vocoder RealWorld Dataset", "data/asvspoof_realworld/Real", "data/asvspoof_realworld/Fake"),
]

total_correct_all = 0
total_samples_all = 0

print('=' * 65)
print('  ONNX PIPELINE EVALUATION: preproc.onnx -> audio_detector.onnx')
print('=' * 65)

for title, real_dir, fake_dir in dirs_to_eval:
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        continue
    
    real_wavs = sorted(glob.glob(os.path.join(real_dir, "*.wav")))
    fake_wavs = sorted(glob.glob(os.path.join(fake_dir, "*.wav")))
    
    cr = sum(1 for w in real_wavs if evaluate_clip(w)[1] > 0.5)
    cf = sum(1 for w in fake_wavs if evaluate_clip(w)[0] > 0.5)
    tr = len(real_wavs)
    tf = len(fake_wavs)
    
    print(f"\n--- {title} ({tr+tf} samples) ---")
    print(f"  Real Audio Accuracy:  {cr:4d}/{tr:4d}  ({cr/max(1,tr)*100:5.1f}%)")
    print(f"  Fake Audio Accuracy:  {cf:4d}/{tf:4d}  ({cf/max(1,tf)*100:5.1f}%)")
    print(f"  Dataset Total:        {cr+cf:4d}/{tr+tf:4d}  ({(cr+cf)/max(1,tr+tf)*100:5.1f}%)")
    
    total_correct_all += (cr + cf)
    total_samples_all += (tr + tf)

print('\n' + '=' * 65)
print(f'  OVERALL ONNX BENCHMARK: {total_correct_all}/{total_samples_all} ({total_correct_all/max(1,total_samples_all)*100:5.1f}%)')
print('=' * 65)
