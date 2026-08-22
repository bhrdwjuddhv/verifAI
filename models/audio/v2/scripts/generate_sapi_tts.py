"""Generate real Windows SAPI TTS samples with diverse sentences, rates, and voices into Fake training data."""
import os, sys, subprocess

sentences = [
    "This is an automated voice test to verify artificial intelligence deepfake detection systems.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Artificial intelligence audio generation has improved dramatically over recent years.",
    "Please verify your credentials before proceeding with the transaction.",
    "Today's weather forecast predicts heavy rain and scattered thunderstorms in the evening.",
    "Welcome to the customer support helpline. Your call is important to us.",
    "The financial markets experienced significant volatility following the announcement.",
    "Deepfake audio detection utilizes mel-spectrogram analysis and convolutional neural networks.",
    "Good morning everyone, today we will discuss neural vocoder architectures and pitch synthesis.",
    "In recent tests, text to speech models can mimic human prosody with high fidelity."
]

os.makedirs('data/asvspoof_realworld/Fake', exist_ok=True)
count = 0

for idx, text in enumerate(sentences):
    for rate in [-3, -1, 0, 1, 3]:
        out_wav = f"data/asvspoof_realworld/Fake/sapi_tts_{count:03d}.wav".replace('\\', '/')
        escaped_text = text.replace("'", "''")
        ps_cmd = f"Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = {rate}; $s.SetOutputToWaveFile('{out_wav}'); $s.Speak('{escaped_text}'); $s.Dispose()"
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=True, capture_output=True)
            count += 1
        except Exception as e:
            pass

print(f"Generated {count} real SAPI TTS audio clips in data/asvspoof_realworld/Fake/")
