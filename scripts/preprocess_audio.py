"""Audio Deepfake Spectrogram Preprocessing Pipeline.

Converts audio files (.wav, .mp3, .flac, .ogg, .m4a) into Mel-spectrogram images
and constructs a standardized ImageFolder dataset structure (Real/ and Fake/).
Saves manifest.json detailing audio sample counts, sample rate, and provenance.

Usage:
  python scripts/preprocess_audio.py --src data/audio_raw --out data/audio_spectrograms
  python scripts/preprocess_audio.py --selfcheck
"""

import argparse
import json
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from PIL import Image

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
LABEL_RULES = [("fake", "Fake"), ("spoof", "Fake"), ("cloned", "Fake"), ("real", "Real"), ("authentic", "Real"), ("genuine", "Real")]


def canonical_label(dirname: str, overrides: dict = None) -> str:
    overrides = overrides or {}
    name = dirname.lower()
    if name in overrides:
        return overrides[name]
    for needle, label in LABEL_RULES:
        if needle in name:
            return label
    return None


SAMPLE_RATE = 16000
CLIP_SECONDS = 3.0


def spectrogram_from_samples(y, sr: int = SAMPLE_RATE, out_size: int = 224):
    """Mono float samples -> the same mel-spectrogram image the training set was built from.

    Split out of generate_spectrogram_image so the live-call window path and the training path
    run identical maths. A second implementation for streaming would be a second chance to get
    n_mels, hop length or the dB reference subtly wrong, and the model would degrade with
    nothing in the logs to explain it.
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return None

    try:
        import librosa

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=out_size)
        S_dB = librosa.power_to_db(S, ref=np.max)
    except Exception:
        # librosa-free fallback: plain STFT magnitude. Different features from the mel path, so
        # a model trained with librosa must be served with librosa — the service says which.
        n_fft, stride = 256, 128
        frames = max(0, (len(y) - n_fft) // stride + 1)
        if frames < 2:
            return None
        spec = [np.abs(np.fft.rfft(y[i * stride:i * stride + n_fft]))[: out_size // 2]
                for i in range(min(frames, out_size))]
        S_dB = np.array(spec).T

    if S_dB.size == 0:
        return None
    norm = ((S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-6) * 255).astype(np.uint8)
    return Image.fromarray(norm).convert("RGB").resize((out_size, out_size), Image.LANCZOS)


def read_audio(filepath: str, sr: int = SAMPLE_RATE, seconds: float = CLIP_SECONDS):
    """Decode to mono float32 at `sr`. Returns None when the file is not decodable audio."""
    try:
        import librosa

        y, _ = librosa.load(filepath, sr=sr, duration=seconds)
        return y if len(y) else None
    except Exception:
        pass

    try:
        import wave

        with wave.open(filepath, "rb") as wf:
            if wf.getsampwidth() != 2:
                return None
            frames = wf.readframes(min(wf.getnframes(), int(wf.getframerate() * seconds)))
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if wf.getnchannels() > 1:
                data = data.reshape(-1, wf.getnchannels()).mean(axis=1)
            if data.size == 0:
                return None
            if wf.getframerate() != sr:
                # Linear resample. Crude, but the alternative is refusing every non-16k WAV.
                idx = np.linspace(0, len(data) - 1, int(len(data) * sr / wf.getframerate()))
                data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)
            return data
    except Exception:
        return None


def generate_spectrogram_image(filepath: str, out_size: int = 224):
    """Converts an audio file into a 2D mel-spectrogram RGB image, or None if undecodable.

    Returning None matters: this used to fall back to a random-noise spectrogram, which the
    model then scored with total confidence, and which seeded the training set with garbage.
    """
    y = read_audio(filepath, SAMPLE_RATE, CLIP_SECONDS)
    if y is None:
        return None
    return spectrogram_from_samples(y, SAMPLE_RATE, out_size)


def selfcheck():
    """Runs internal assertions for audio spectrogram geometry and label resolution."""
    assert canonical_label("fake_voice", {}) == "Fake"
    assert canonical_label("spoof_audio", {}) == "Fake"
    assert canonical_label("real_speech", {}) == "Real"
    assert canonical_label("unknown_folder", {}) is None
    assert canonical_label("unknown_folder", {"unknown_folder": "Real"}) == "Real"

    # An undecodable file must produce NO features. This used to return a random-noise
    # spectrogram, which the model then scored with total confidence — the single most
    # dangerous line in the audio path.
    import math
    import struct
    import tempfile
    import wave

    with tempfile.TemporaryDirectory() as tmp:
        junk = os.path.join(tmp, "not_audio.wav")
        with open(junk, "wb") as fh:
            fh.write(b"this is not audio at all")
        assert generate_spectrogram_image(junk) is None, "undecodable audio must return None"

        empty = os.path.join(tmp, "empty.wav")
        with wave.open(empty, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
        assert generate_spectrogram_image(empty) is None, "an empty clip has no features"

        tone = os.path.join(tmp, "tone.wav")
        with wave.open(tone, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"".join(
                struct.pack("<h", int(9000 * math.sin(2 * math.pi * 440 * t / 16000)))
                for t in range(16000)
            ))
        img = generate_spectrogram_image(tone, 224)
        assert img is not None and img.size == (224, 224) and img.mode == "RGB", (
            f"a real PCM wav must yield a 224x224 RGB spectrogram, got {img}")

    # The file path and the samples path must produce the same image, or live windows and
    # training data stop being the same kind of thing.
    with tempfile.TemporaryDirectory() as tmp:
        tone = os.path.join(tmp, "parity.wav")
        with wave.open(tone, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(
                struct.pack("<h", int(9000 * math.sin(2 * math.pi * 440 * t / SAMPLE_RATE)))
                for t in range(SAMPLE_RATE)
            ))
        samples = read_audio(tone)
        assert samples is not None and len(samples) == SAMPLE_RATE
        from_file = generate_spectrogram_image(tone, 224)
        from_samples = spectrogram_from_samples(samples, SAMPLE_RATE, 224)
        assert np.array_equal(np.asarray(from_file), np.asarray(from_samples)), \
            "file and sample paths must produce identical spectrograms"

    assert spectrogram_from_samples(np.zeros(0)) is None, "no samples, no features"

    print("✅ audio preprocessing selfcheck passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", help="raw audio dataset directory (repeatable)")
    ap.add_argument("--out", default="data/audio_spectrograms")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--limit", type=int, default=0, help="limit samples per source")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return

    if not args.src:
        ap.error("--src directory is required")

    for label in ("Real", "Fake"):
        os.makedirs(os.path.join(args.out, label), exist_ok=True)

    counts = {"Real": 0, "Fake": 0}
    sources = {}

    for src in args.src:
        if not os.path.isdir(src):
            print(f"⚠️ Warning: Directory not found: {src}")
            continue

        tag = os.path.basename(os.path.normpath(src))[:24]
        written_here = 0

        for dirpath, _, filenames in os.walk(src):
            dirname = os.path.basename(dirpath)
            label = canonical_label(dirname)
            if label is None:
                continue

            audio_files = [f for f in filenames if os.path.splitext(f)[1].lower() in AUDIO_EXTS]
            for fname in sorted(audio_files):
                if args.limit and written_here >= args.limit:
                    break

                file_path = os.path.join(dirpath, fname)
                spec_img = generate_spectrogram_image(file_path, args.size)
                if spec_img is None:
                    continue

                stem = os.path.splitext(fname)[0]
                out_name = f"{tag}__{dirname}__{stem}.jpg"
                spec_img.save(os.path.join(args.out, label, out_name), "JPEG", quality=95)

                counts[label] += 1
                written_here += 1

        sources[tag] = written_here
        print(f"  ✅ {written_here} spectrograms written from {tag}")

    manifest = {
        "modality": "audio",
        "spectrogram_size": args.size,
        "sample_rate": 16000,
        "counts": counts,
        "sources": sources,
    }

    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n📊 Audio Spectrograms: Real {counts['Real']} | Fake {counts['Fake']}")
    print(f"💾 Manifest written -> {manifest_path}")


if __name__ == "__main__":
    main()
