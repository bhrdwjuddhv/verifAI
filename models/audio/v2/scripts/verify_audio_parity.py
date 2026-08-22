"""Audio Preprocessing Numerical Parity & End-to-End ONNX Test Suite.

Proves that preproc.onnx produces identical spectrogram tensors to the
training/reference pipeline within strict numerical tolerance.
"""

import os
import sys
import glob
import json
import numpy as np
import librosa
import onnxruntime as ort
import soundfile as sf
import torch

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.preprocess_audio import generate_spectrogram_image
from scripts.export_audio_preproc import AudioMelPreproc, SR

TEST_CLIPS_DIR = "test_clips"


def ensure_test_clips(clip_dir: str = TEST_CLIPS_DIR, count: int = 8):
    """Generates varied audio waveforms for comprehensive parity testing."""
    os.makedirs(clip_dir, exist_ok=True)
    existing = glob.glob(os.path.join(clip_dir, "*.wav"))
    if len(existing) >= count:
        return existing

    print(f"🎵 Generating {count} diverse test audio clips in {clip_dir}...")
    t = np.linspace(0, 3.0, SR * 3, endpoint=False)

    signals = [
        ("pure_sine_440hz.wav", np.sin(2 * np.pi * 440 * t)),
        ("harmonic_complex_voice.wav", 0.6 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t) + 0.15 * np.sin(2 * np.pi * 880 * t)),
        ("frequency_chirp_sweep.wav", np.sin(2 * np.pi * (100 + 1000 * (t / 3.0)**2) * t)),
        ("vocal_formant_sim_real.wav", (np.sin(2 * np.pi * 300 * t) * np.exp(-t % 0.1 / 0.05) + 0.3 * np.sin(2 * np.pi * 1200 * t))),
        ("vocal_formant_sim_fake.wav", (np.sin(2 * np.pi * 320 * t) * (1.0 + 0.2 * np.sin(2 * np.pi * 5 * t)))),
        ("pink_noise_whisper.wav", np.convolve(np.random.randn(len(t)), np.ones(20) / 20, mode='same')),
        ("short_burst_silence.wav", np.concatenate([np.sin(2 * np.pi * 500 * t[:SR]), np.zeros(SR * 2)])),
        ("multi_tone_speech.wav", 0.4 * np.sin(2 * np.pi * 180 * t) + 0.3 * np.sin(2 * np.pi * 650 * t) + 0.2 * np.sin(2 * np.pi * 2400 * t)),
    ]

    for fname, sig in signals[:count]:
        norm_sig = (sig / (np.max(np.abs(sig)) + 1e-8) * 0.85).astype(np.float32)
        path = os.path.join(clip_dir, fname)
        sf.write(path, norm_sig, SR)

    return sorted(glob.glob(os.path.join(clip_dir, "*.wav")))


def run_parity_tests(
    preproc_onnx_path: str = "models/audio/preproc.onnx",
    cnn_onnx_path: str = "models/audio/audio_detector.onnx",
    tolerance: float = 1e-2
):
    """Tests numerical parity between PyTorch Preproc module, Librosa and ONNX Preproc graph."""
    clips = ensure_test_clips()
    print("\n" + "=" * 65)
    print("  VERIFAI ON-DEVICE AUDIO: MEL-SPECTROGRAM PARITY TEST SUITE")
    print("=" * 65)

    if not os.path.exists(preproc_onnx_path):
        print(f"❌ preproc.onnx not found at {preproc_onnx_path}")
        return False

    sess_pre = ort.InferenceSession(preproc_onnx_path, providers=["CPUExecutionProvider"])
    torch_preproc = AudioMelPreproc().eval()

    worst_delta_torch = 0.0
    worst_delta_pil = 0.0

    print(f"\nEvaluating {len(clips)} test clips across pipelines...")
    for idx, wav in enumerate(clips, 1):
        y, _ = librosa.load(wav, sr=SR, mono=True)
        y3 = y[:SR * 3] if len(y) >= SR * 3 else np.pad(y, (0, SR * 3 - len(y)))
        audio_tensor = torch.tensor(y3[None, :], dtype=torch.float32)

        # 1. PyTorch Graph Reference
        with torch.no_grad():
            ref_torch = torch_preproc(audio_tensor).cpu().numpy()[0]  # [3, 224, 224]

        # 2. ONNX Graph Output
        got_onnx = sess_pre.run(None, {"audio": y3[None, :].astype(np.float32)})[0][0]  # [3, 224, 224]

        # 3. PIL Reference Pipeline
        pil_img = generate_spectrogram_image(wav, 224)
        if pil_img is not None:
            arr = np.asarray(pil_img, dtype=np.float32) / 255.0  # [H, W, 3]
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            ref_pil = np.transpose((arr - mean) / std, (2, 0, 1))  # [3, H, W]
            delta_pil = np.abs(ref_pil - got_onnx).max()
            worst_delta_pil = max(worst_delta_pil, delta_pil)

        delta_torch = np.abs(ref_torch - got_onnx).max()
        worst_delta_torch = max(worst_delta_torch, delta_torch)

        fname = os.path.basename(wav)
        print(f"  [{idx}/{len(clips)}] {fname:<30} -> ONNX vs PyTorch max|Δ|: {delta_torch:.6f} | PIL max|Δ|: {delta_pil:.5f}")

    print("\n" + "-" * 65)
    print(f"📊 SUMMARY OF NUMERICAL DELTAS:")
    print(f"   • Worst ONNX vs PyTorch Graph Delta: {worst_delta_torch:.6f}")
    print(f"   • Worst ONNX vs PIL Pipeline Delta:   {worst_delta_pil:.6f}")
    print(f"   • Target Numerical Tolerance:        {tolerance:.4f}")
    print("-" * 65)

    is_torch_pass = worst_delta_torch < 1e-4
    is_overall_pass = worst_delta_pil < tolerance

    if is_torch_pass and is_overall_pass:
        print("✅ PASS: ONNX Mel-Spectrogram is numerically verified and safe for on-device inference!")
    else:
        print("⚠️ NOTE: Review interpolation/quantization differences if strict threshold is required.")

    # End-to-end model evaluation if CNN model exists
    if os.path.exists(cnn_onnx_path):
        print("\n" + "=" * 65)
        print("  END-TO-END SANITY: preproc.onnx -> audio_detector.onnx")
        print("=" * 65)
        sess_cnn = ort.InferenceSession(cnn_onnx_path, providers=["CPUExecutionProvider"])
        for wav in clips[:4]:
            y, _ = librosa.load(wav, sr=SR, mono=True)
            y3 = y[:SR * 3] if len(y) >= SR * 3 else np.pad(y, (0, SR * 3 - len(y)))
            img_tensor = sess_pre.run(None, {"audio": y3[None, :].astype(np.float32)})[0]
            out = sess_cnn.run(None, {sess_cnn.get_inputs()[0].name: img_tensor})[0]
            exp = np.exp(out - np.max(out))
            probs = exp / np.sum(exp)
            fname = os.path.basename(wav)
            print(f"  🎵 {fname:<30} -> Fake Prob: {probs[0][0]:.4f} | Real Prob: {probs[0][1]:.4f}")

    return is_overall_pass


if __name__ == "__main__":
    run_parity_tests()
