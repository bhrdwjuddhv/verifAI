import os
import numpy as np
from PIL import Image
import wave
import struct

os.makedirs("data/raw_smoke_test/Real", exist_ok=True)
os.makedirs("data/raw_smoke_test/Fake", exist_ok=True)
os.makedirs("data/audio_smoke_test/Real", exist_ok=True)
os.makedirs("data/audio_smoke_test/Fake", exist_ok=True)

# Generate 10 Real images and 10 Fake images
for i in range(10):
    img_real = Image.fromarray(np.uint8(np.random.randint(0, 255, (256, 256, 3))))
    img_real.save(f"data/raw_smoke_test/Real/img_real_{i}.jpg")

    img_fake = Image.fromarray(np.uint8(np.random.randint(0, 255, (256, 256, 3))))
    img_fake.save(f"data/raw_smoke_test/Fake/img_fake_{i}.jpg")

# Generate 10 Real audio WAV files and 10 Fake audio WAV files
def create_wav(filepath):
    with wave.open(filepath, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # 1 second of audio
        data = [int(1000 * np.sin(2 * np.pi * 440 * t / 16000)) for t in range(16000)]
        packed = struct.pack(f'<{len(data)}h', *data)
        wf.writeframes(packed)

for i in range(10):
    create_wav(f"data/audio_smoke_test/Real/audio_real_{i}.wav")
    create_wav(f"data/audio_smoke_test/Fake/audio_fake_{i}.wav")

print("Created smoke test datasets in data/raw_smoke_test and data/audio_smoke_test")
