"""Add slices from genuine LibriSpeech recordings into Real dataset."""
import os, librosa, soundfile as sf, numpy as np

os.makedirs('data/asvspoof_realworld/Real', exist_ok=True)

for name in ['libri1', 'brahms', 'choice', 'fishin']:
    try:
        p = librosa.ex(name)
        y, sr = librosa.load(p, sr=16000, mono=True)
        # Slice into 3s clips
        num_clips = int(len(y) // (16000 * 3))
        for i in range(max(1, num_clips)):
            clip = y[i*48000:(i+1)*48000]
            if len(clip) < 48000:
                clip = np.pad(clip, (0, 48000 - len(clip)))
            out_p = f"data/asvspoof_realworld/Real/genuine_{name}_{i}.wav"
            sf.write(out_p, clip, 16000)
            print(f"Added {out_p}")
    except Exception as e:
        print(f"Notice for {name}: {e}")
