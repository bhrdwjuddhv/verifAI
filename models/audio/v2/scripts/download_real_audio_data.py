"""Download ASVspoof 2019 LA subset and prepare real-world training data.

Uses HuggingFace datasets to stream/download real bonafide + spoofed speech 
for training a robust deepfake audio detector.
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

# Output directories
REAL_DIR = "data/asvspoof_realworld/Real"
FAKE_DIR = "data/asvspoof_realworld/Fake"

# How many samples per class to download
MAX_PER_CLASS = 500


def save_wav(path: str, audio: np.ndarray, sr: int = SR):
    """Save a mono float32 array as 16-bit WAV."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def download_asvspoof():
    """Download ASVspoof 2019 LA train+dev data from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system(f"{sys.executable} -m pip install datasets soundfile --quiet")
        from datasets import load_dataset

    os.makedirs(REAL_DIR, exist_ok=True)
    os.makedirs(FAKE_DIR, exist_ok=True)

    print(f"Downloading ASVspoof 2019 LA from HuggingFace (streaming mode)...")
    print(f"Target: {MAX_PER_CLASS} real + {MAX_PER_CLASS} fake samples\n")

    real_count = 0
    fake_count = 0

    # Try multiple dataset sources
    dataset_names = [
        "Bisher/ASVspoof_2019_LA",
        "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA",
    ]

    ds = None
    for ds_name in dataset_names:
        try:
            print(f"  Trying {ds_name}...")
            ds = load_dataset(ds_name, streaming=True, trust_remote_code=True)
            # Check what splits are available
            print(f"  Available splits: {list(ds.keys())}")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if ds is None:
        print("\nCould not load from HuggingFace. Falling back to generating enhanced synthetic data...")
        return generate_enhanced_synthetic()

    # Process available splits
    for split_name in ds.keys():
        if real_count >= MAX_PER_CLASS and fake_count >= MAX_PER_CLASS:
            break

        print(f"\n  Processing split: {split_name}")
        try:
            for sample in ds[split_name]:
                if real_count >= MAX_PER_CLASS and fake_count >= MAX_PER_CLASS:
                    break

                # Extract audio
                audio_data = sample.get("audio", None)
                if audio_data is None:
                    continue

                if isinstance(audio_data, dict):
                    waveform = np.array(audio_data["array"], dtype=np.float32)
                    sample_rate = audio_data.get("sampling_rate", SR)
                else:
                    continue

                # Resample if needed
                if sample_rate != SR:
                    try:
                        import librosa
                        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=SR)
                    except:
                        # Simple decimation
                        ratio = SR / sample_rate
                        indices = np.round(np.arange(0, len(waveform), 1/ratio)).astype(int)
                        indices = indices[indices < len(waveform)]
                        waveform = waveform[indices]

                # Pad or trim to 3 seconds
                if len(waveform) >= NUM_SAMPLES:
                    waveform = waveform[:NUM_SAMPLES]
                else:
                    waveform = np.pad(waveform, (0, NUM_SAMPLES - len(waveform)))

                # Normalize
                peak = np.max(np.abs(waveform))
                if peak > 0:
                    waveform = waveform / peak * 0.9

                # Determine label
                label = sample.get("label", sample.get("is_bonafide", sample.get("key", "")))
                
                # Handle different label formats
                is_real = False
                if isinstance(label, str):
                    is_real = label.lower() in ("bonafide", "real", "genuine", "human")
                elif isinstance(label, (int, float)):
                    # Convention: 1 = bonafide, 0 = spoof (varies by dataset)
                    is_real = (label == 1)
                elif isinstance(label, bool):
                    is_real = label

                if is_real and real_count < MAX_PER_CLASS:
                    path = os.path.join(REAL_DIR, f"bonafide_{real_count:04d}.wav")
                    save_wav(path, waveform)
                    real_count += 1
                    if real_count % 50 == 0:
                        print(f"    Real: {real_count}/{MAX_PER_CLASS}")
                elif not is_real and fake_count < MAX_PER_CLASS:
                    path = os.path.join(FAKE_DIR, f"spoof_{fake_count:04d}.wav")
                    save_wav(path, waveform)
                    fake_count += 1
                    if fake_count % 50 == 0:
                        print(f"    Fake: {fake_count}/{MAX_PER_CLASS}")

        except Exception as e:
            print(f"    Error processing split {split_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nDownloaded: {real_count} real + {fake_count} fake samples")

    if real_count < 50 or fake_count < 50:
        print("Not enough samples from HuggingFace. Supplementing with enhanced synthetic data...")
        supplement_with_synthetic(MAX_PER_CLASS - real_count, MAX_PER_CLASS - fake_count, real_count, fake_count)

    return real_count, fake_count


def generate_enhanced_synthetic():
    """Fallback: Generate much more diverse synthetic data if HuggingFace download fails."""
    print("\nGenerating enhanced synthetic dataset (diverse speakers, conditions)...")
    
    real_count = 0
    fake_count = 0
    
    for i in range(MAX_PER_CLASS):
        np.random.seed(i + 10000)
        wav = _gen_diverse_real(i)
        save_wav(os.path.join(REAL_DIR, f"synth_real_{i:04d}.wav"), wav)
        real_count += 1
        
        np.random.seed(i + 20000)
        wav = _gen_diverse_fake(i)
        save_wav(os.path.join(FAKE_DIR, f"synth_fake_{i:04d}.wav"), wav)
        fake_count += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{MAX_PER_CLASS} per class")

    print(f"Generated: {real_count} real + {fake_count} fake samples")
    return real_count, fake_count


def supplement_with_synthetic(n_real_needed, n_fake_needed, start_real, start_fake):
    """Add synthetic samples to supplement incomplete download."""
    for i in range(n_real_needed):
        np.random.seed(i + 30000)
        wav = _gen_diverse_real(i + 30000)
        save_wav(os.path.join(REAL_DIR, f"synth_real_{start_real + i:04d}.wav"), wav)
    
    for i in range(n_fake_needed):
        np.random.seed(i + 40000)
        wav = _gen_diverse_fake(i + 40000)
        save_wav(os.path.join(FAKE_DIR, f"synth_fake_{start_fake + i:04d}.wav"), wav)

    print(f"  Supplemented with {n_real_needed} real + {n_fake_needed} fake synthetic samples")


def _gen_diverse_real(seed):
    """Generate diverse real-like speech with wide parameter variation."""
    np.random.seed(seed)
    t = np.linspace(0, DURATION, NUM_SAMPLES, endpoint=False)
    
    # Very diverse F0 range (children to deep male voices)
    f0_base = np.random.uniform(85, 300)
    # Natural prosody with varied contour shape
    contour_type = np.random.choice(['falling', 'rising', 'flat', 'question', 'wave'])
    if contour_type == 'falling':
        f0_mod = 1.0 + 0.15 * np.exp(-t / DURATION)
    elif contour_type == 'rising':
        f0_mod = 1.0 + 0.12 * (t / DURATION)
    elif contour_type == 'question':
        f0_mod = 1.0 + 0.2 * (t / DURATION) ** 2
    elif contour_type == 'wave':
        f0_mod = 1.0 + 0.08 * np.sin(2 * np.pi * np.random.uniform(0.5, 2.0) * t)
    else:
        f0_mod = np.ones_like(t)
    
    # Micro-jitter (real speech characteristic: 0.5-2%)
    jitter = 1.0 + np.random.uniform(0.003, 0.015) * np.random.randn(NUM_SAMPLES)
    f0 = f0_base * f0_mod * jitter
    
    phase = 2 * np.pi * np.cumsum(f0) / SR
    
    # Harmonic series with natural rolloff
    n_harmonics = np.random.randint(4, 8)
    signal = np.zeros(NUM_SAMPLES)
    for h in range(1, n_harmonics + 1):
        amp = 1.0 / (h ** np.random.uniform(0.8, 1.5))
        # Shimmer (amplitude perturbation)
        shimmer = 1.0 + np.random.uniform(0.01, 0.04) * np.random.randn(NUM_SAMPLES)
        signal += amp * shimmer * np.sin(h * phase)
    
    # Formant resonances (varied vowel space)
    vowel_type = np.random.randint(0, 5)
    formant_sets = [
        (300, 870, 2240),    # /u/
        (730, 1090, 2440),   # /a/
        (270, 2290, 3010),   # /i/
        (530, 1840, 2480),   # /e/
        (660, 1720, 2410),   # /o/
    ]
    f1, f2, f3 = formant_sets[vowel_type]
    # Add formant transitions
    f1_t = f1 + 100 * np.sin(2 * np.pi * 3.0 * t)
    f2_t = f2 + 200 * np.sin(2 * np.pi * 2.5 * t)
    
    formant = (
        0.6 * np.sin(2 * np.pi * f1_t * t / SR * SR) * np.exp(-((t % 0.15) / 0.035)) +
        0.35 * np.sin(2 * np.pi * f2_t * t / SR * SR) * np.exp(-((t % 0.12) / 0.025)) +
        0.15 * np.sin(2 * np.pi * f3 * t) * np.exp(-((t % 0.1) / 0.02))
    )
    
    # Speech-like amplitude envelope
    syllable_rate = np.random.uniform(2.5, 5.5)
    env = np.clip(np.sin(2 * np.pi * syllable_rate * t) ** 2 + 0.08, 0.0, 1.0)
    # Add pauses
    n_pauses = np.random.randint(1, 4)
    for _ in range(n_pauses):
        pause_start = np.random.randint(0, NUM_SAMPLES - SR // 2)
        pause_len = np.random.randint(SR // 8, SR // 3)
        fade = np.linspace(1, 0, min(pause_len, SR // 8))
        env[pause_start:pause_start + len(fade)] *= fade
        env[pause_start + len(fade):pause_start + pause_len] *= 0.02
    
    voice = signal * env * (1.0 + 0.3 * formant)
    
    # Natural noise floor (breathing, room tone)
    noise_level = np.random.uniform(0.01, 0.04)
    noise = noise_level * np.random.randn(NUM_SAMPLES)
    
    # Occasional breath sounds
    if np.random.random() > 0.3:
        for _ in range(np.random.randint(1, 3)):
            pos = np.random.randint(0, NUM_SAMPLES - SR // 4)
            breath_len = np.random.randint(SR // 8, SR // 3)
            breath = 0.03 * np.random.randn(breath_len) * np.hanning(breath_len)
            voice[pos:pos + breath_len] += breath[:min(breath_len, NUM_SAMPLES - pos)]
    
    sig = voice + noise
    
    # Random room acoustics simulation
    if np.random.random() > 0.4:
        # Simple reverb via convolution with decay
        reverb_len = np.random.randint(SR // 10, SR // 3)
        impulse = np.random.randn(reverb_len) * np.exp(-np.arange(reverb_len) / (reverb_len * 0.3))
        impulse[0] = 1.0
        impulse /= np.sum(np.abs(impulse))
        sig = np.convolve(sig, impulse, mode='same')
    
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig = sig / peak * np.random.uniform(0.7, 0.95)
    
    return sig.astype(np.float32)


def _gen_diverse_fake(seed):
    """Generate diverse AI-like speech with various vocoder artifact signatures."""
    np.random.seed(seed)
    t = np.linspace(0, DURATION, NUM_SAMPLES, endpoint=False)
    
    vocoder_type = np.random.choice(['melgan', 'hifigan', 'waveglow', 'griffin_lim', 'world'])
    
    f0_base = np.random.uniform(100, 250)
    
    if vocoder_type == 'melgan':
        # MelGAN: checkerboard artifacts, metallic buzz
        phase = 2 * np.pi * f0_base * t
        base = np.sin(phase) + 0.5 * np.sin(2 * phase)
        # Checkerboard pattern in mel-space
        artifact_freq = np.random.uniform(60, 120)
        checkerboard = 0.15 * np.sin(2 * np.pi * artifact_freq * t) * np.sin(2 * np.pi * 500 * t)
        # Metallic buzz from aliased harmonics
        buzz = 0.08 * np.sin(2 * np.pi * 3750 * t) + 0.05 * np.sin(2 * np.pi * 4200 * t)
        signal = base + checkerboard + buzz
        
    elif vocoder_type == 'hifigan':
        # HiFi-GAN: phase discontinuities at frame boundaries
        hop_size = 256
        n_frames = NUM_SAMPLES // hop_size
        phase = np.zeros(NUM_SAMPLES)
        current_phase = 0
        for fr in range(n_frames):
            start = fr * hop_size
            end = min(start + hop_size, NUM_SAMPLES)
            # Slight phase jump at each frame boundary
            if np.random.random() > 0.7:
                current_phase += np.random.uniform(-0.3, 0.3)
            t_frame = np.arange(end - start) / SR
            phase[start:end] = 2 * np.pi * f0_base * t_frame + current_phase
            current_phase += 2 * np.pi * f0_base * (end - start) / SR
        signal = np.sin(phase) + 0.4 * np.sin(2 * phase)
        # Subtle periodic noise at hop boundaries
        for fr in range(0, n_frames, 3):
            pos = fr * hop_size
            if pos + 32 < NUM_SAMPLES:
                signal[pos:pos + 32] += 0.05 * np.random.randn(32)
        
    elif vocoder_type == 'waveglow':
        # WaveGlow: slightly over-smooth, lack of fine temporal detail
        phase = 2 * np.pi * f0_base * t
        signal = np.sin(phase) + 0.45 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)
        # Over-smoothing via strong lowpass
        kernel_size = np.random.randint(5, 15)
        kernel = np.ones(kernel_size) / kernel_size
        signal = np.convolve(signal, kernel, mode='same')
        # Almost no noise (too clean compared to real speech)
        signal += 0.002 * np.random.randn(NUM_SAMPLES)
        
    elif vocoder_type == 'griffin_lim':
        # Griffin-Lim: characteristic "underwater" / phasy quality
        phase = 2 * np.pi * f0_base * t
        signal = np.sin(phase) + 0.5 * np.sin(2 * phase)
        # Random phase shifts create "underwater" effect
        n_components = 10
        for k in range(n_components):
            freq = f0_base * (k + 4)
            if freq > SR / 2:
                break
            random_phase = np.random.uniform(0, 2 * np.pi)
            signal += (0.1 / (k + 1)) * np.sin(2 * np.pi * freq * t + random_phase)
        
    else:  # world vocoder
        # WORLD: aperiodic noise mismodeling, band-limited artifacts
        phase = 2 * np.pi * np.cumsum(np.ones(NUM_SAMPLES) * f0_base) / SR
        signal = np.sin(phase) + 0.3 * np.sin(2 * phase)
        # Unnatural aperiodic component
        ap_noise = 0.12 * np.random.randn(NUM_SAMPLES)
        # Band-limit the noise unnaturally
        kernel = np.hanning(64)
        kernel /= kernel.sum()
        ap_noise = np.convolve(ap_noise, kernel, mode='same')
        signal += ap_noise
    
    # Common AI artifacts:
    
    # 1. Unnaturally flat pitch (no micro-jitter)
    # Already the case - no jitter added (unlike real speech)
    
    # 2. Robotic syllabic timing
    syllable_rate = np.random.uniform(3.0, 4.5)
    env = np.clip(np.sin(2 * np.pi * syllable_rate * t) ** 2 + 0.05, 0.0, 1.0)
    # Very regular, no natural pauses
    signal *= env
    
    # 3. Spectral artifacts: energy at unusual frequencies
    if np.random.random() > 0.3:
        artifact_f = np.random.uniform(5000, 7500)
        signal += np.random.uniform(0.02, 0.06) * np.sin(2 * np.pi * artifact_f * t)
    
    # 4. Quantization noise (from mel-to-waveform conversion)
    if np.random.random() > 0.4:
        n_levels = np.random.choice([128, 256, 512])
        signal = np.round(signal * n_levels) / n_levels
    
    # 5. Very low noise floor (too clean)
    noise_level = np.random.uniform(0.001, 0.008)  # Much less than real speech
    signal += noise_level * np.random.randn(NUM_SAMPLES)
    
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * np.random.uniform(0.75, 0.95)
    
    return signal.astype(np.float32)


if __name__ == "__main__":
    real_n, fake_n = download_asvspoof()
    print(f"\nDataset ready: {real_n} real + {fake_n} fake in data/asvspoof_realworld/")
