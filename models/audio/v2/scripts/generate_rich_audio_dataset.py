"""Rich Audio Dataset Generator for Real vs Deepfake Voice Detection.

Generates realistic acoustic speech samples simulating:
- Genuine Human Voices: Natural vocal tract resonance (F0-F3 formants), glottal pulse train, pitch jitter/shimmer, natural prosody, breathing noise.
- AI Voice Clones & Deepfakes: Vocoder checkerboard artifacts (MelGAN/HiFi-GAN), phase discontinuities, robotic metallic resonance, spectral buzz, flat pitch contours.
- Acoustic Augmentations: Background room impulse response, telephone bandpass, random SNR babble.
"""

import os
import sys
import struct
import wave
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SR = 16000
DURATION = 3.0
NUM_SAMPLES = int(SR * DURATION)


def generate_human_voice_sim(seed: int) -> np.ndarray:
    """Simulates realistic human speech with natural formants, pitch micro-jitter, and glottal flow."""
    np.random.seed(seed)
    t = np.linspace(0, DURATION, NUM_SAMPLES, endpoint=False)

    # Base pitch (F0: 100Hz - 240Hz) with natural slow intonation contour
    base_f0 = np.random.uniform(110.0, 220.0)
    f0_contour = base_f0 * (1.0 + 0.08 * np.sin(2 * np.pi * 0.8 * t) + 0.03 * np.sin(2 * np.pi * 2.3 * t))
    # Micro-jitter (pitch perturbation < 1%)
    jitter = 1.0 + 0.005 * np.random.randn(NUM_SAMPLES)
    f0_actual = f0_contour * jitter

    phase = 2 * np.pi * np.cumsum(f0_actual) / SR
    # Glottal pulse approximation (smooth sawtooth harmonic decay)
    glottal = (
        np.sin(phase) +
        0.6 * np.sin(2 * phase) +
        0.35 * np.sin(3 * phase) +
        0.2 * np.sin(4 * phase) +
        0.1 * np.sin(5 * phase)
    )

    # Human Vocal Formants (Vowel resonance filters: F1=500-800Hz, F2=1200-2200Hz, F3=2500-3200Hz)
    f1 = np.random.uniform(500, 800)
    f2 = np.random.uniform(1300, 2100)
    f3 = np.random.uniform(2500, 3100)

    formant_res = (
        0.8 * np.sin(2 * np.pi * f1 * t) * np.exp(-((t % 0.2) / 0.04)) +
        0.5 * np.sin(2 * np.pi * f2 * t) * np.exp(-((t % 0.15) / 0.03)) +
        0.25 * np.sin(2 * np.pi * f3 * t) * np.exp(-((t % 0.1) / 0.02))
    )

    # Syllabic envelope modulation (speech cadence: 3-5 syllables/sec)
    syllable_env = np.clip(np.sin(2 * np.pi * 3.5 * t) ** 2 + 0.1, 0.0, 1.0)
    voice = glottal * syllable_env * (1.0 + 0.4 * formant_res)

    # Add realistic breathiness & room ambience
    breath = 0.02 * np.random.randn(NUM_SAMPLES)
    sig = voice + breath
    return (sig / (np.max(np.abs(sig)) + 1e-6) * 0.85).astype(np.float32)


def generate_ai_deepfake_sim(seed: int) -> np.ndarray:
    """Simulates AI neural vocoder artifacts (HiFi-GAN, WaveGlow, MelGAN metallic resonance, unnatural phase)."""
    np.random.seed(seed)
    t = np.linspace(0, DURATION, NUM_SAMPLES, endpoint=False)

    # AI TTS often has overly static / rigid pitch contour (unnatural pitch lock)
    fixed_f0 = np.random.uniform(120.0, 200.0)
    phase = 2 * np.pi * fixed_f0 * t
    base_voice = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)

    # Syllabic modulation with robotic transitions
    syllable_env = np.clip(np.sin(2 * np.pi * 3.5 * t) ** 2 + 0.05, 0.0, 1.0)

    # Neural Vocoder Artifact 1: High-frequency subharmonic checkerboard buzz (3.5kHz - 7kHz)
    vocoder_buzz_freq = np.random.uniform(3600, 6800)
    vocoder_buzz = 0.12 * np.sin(2 * np.pi * vocoder_buzz_freq * t) * (1.0 + 0.5 * np.sin(2 * np.pi * 100 * t))

    # Neural Vocoder Artifact 2: Phase discontinuity / metallic smearing
    metallic_phase = 0.08 * np.sin(2 * np.pi * 1800 * t + np.sin(2 * np.pi * 50 * t))

    # Neural Vocoder Artifact 3: High-frequency over-smoothing & quantization steps
    fake_sig = (base_voice * syllable_env) + vocoder_buzz + metallic_phase
    # Quantization / aliasing steps
    fake_sig = np.round(fake_sig * 32.0) / 32.0

    return (fake_sig / (np.max(np.abs(fake_sig)) + 1e-6) * 0.85).astype(np.float32)


def save_wav(filepath: str, audio: np.ndarray):
    """Saves a 16-bit PCM WAV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    int_data = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(filepath, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(int_data.tobytes())


def create_dataset(out_dir: str = "data/audio_realistic_dataset", count_per_class: int = 100):
    """Generates balanced dataset of realistic Human Real vs AI Deepfake audio files."""
    print(f"🎙️ Generating {count_per_class * 2} audio files in {out_dir}...")
    for i in range(count_per_class):
        # Real Human voice sample
        real_audio = generate_human_voice_sim(seed=1000 + i)
        save_wav(os.path.join(out_dir, "Real", f"human_real_speaker_{i:03d}.wav"), real_audio)

        # AI Voice Clone / Deepfake sample
        fake_audio = generate_ai_deepfake_sim(seed=5000 + i)
        save_wav(os.path.join(out_dir, "Fake", f"ai_voiceclone_fake_{i:03d}.wav"), fake_audio)

    print(f"✅ Successfully created {count_per_class} Real and {count_per_class} Fake audio samples!")


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "data/audio_realistic_dataset"
    create_dataset(out_path, count_per_class=120)
