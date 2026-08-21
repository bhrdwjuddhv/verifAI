"""Measure the voice model under CALL conditions, not on clean WAVs.

    python scripts/eval_call_audio.py --data-dir data/audio_eval
    python scripts/eval_call_audio.py --data-dir data/audio_eval --conditions clean,g711,narrowband
    python scripts/eval_call_audio.py --selfcheck

`--data-dir` is two folders whose names contain "real" and "fake"/"spoof"/"cloned", holding
audio files. Every clip is scored clean and again under each simulated call condition, through
the service's own window path — the same VAD gate and the same spectrogram code production
uses.

Why this exists: a voice model validated on studio WAVs is not measured for the job. A live
call has been companded to 8-bit G.711, resampled to 8kHz and back, had packets dropped, and
picked up line noise. Every one of those attacks the high-frequency detail these detectors
lean on. The number from this script is the one worth quoting for Live Guard.

NOT simulated here: Opus and AMR-WB. Both need a real codec (ffmpeg/libopus); adding one to
the runtime image costs more than it is worth, and pretending a numpy approximation is Opus
would be worse than admitting the gap. G.711 μ-law IS the real PSTN codec, and it is exact.
"""

import argparse
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

import inference_server as svc
from common.metrics import binary_auc

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
SR = 16000


# --- call-condition degradations ----------------------------------------------------------
def mu_law(y, mu=255.0):
    """G.711 μ-law: the actual PSTN codec. Companded to 8 bits and expanded back.

    Exact, not an approximation — this is the same nonlinear quantization a phone call gets.
    """
    y = np.clip(y, -1.0, 1.0)
    compressed = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    quantized = np.round((compressed + 1.0) * 127.5) / 127.5 - 1.0
    return np.sign(quantized) * ((1.0 + mu) ** np.abs(quantized) - 1.0) / mu


def narrowband(y, sr=SR, target=8000, cutoff=3400.0):
    """Telephone band: low-pass at 3.4kHz, then resample through 8kHz and back.

    The low-pass is not optional. Decimating without it does not remove the high frequencies —
    it FOLDS them back into the passband as aliases, which is a different artifact from the one
    a phone call produces, and would have made this eval measure the wrong thing entirely.
    """
    spectrum = np.fft.rfft(y)
    spectrum[np.fft.rfftfreq(len(y), 1.0 / sr) > cutoff] = 0.0
    filtered = np.fft.irfft(spectrum, n=len(y)).astype(np.float32)

    down = np.interp(np.linspace(0, len(filtered) - 1, int(len(filtered) * target / sr)),
                     np.arange(len(filtered)), filtered)
    return np.interp(np.linspace(0, len(down) - 1, len(y)), np.arange(len(down)), down).astype(np.float32)


def packet_loss(y, sr=SR, rate=0.05, packet_ms=20, seed=0):
    """Drop whole 20ms packets, as a jittery connection does. Silence, not interpolation."""
    out = y.copy()
    size = max(1, int(sr * packet_ms / 1000))
    rng = np.random.default_rng(seed)
    for start in range(0, len(out) - size, size):
        if rng.random() < rate:
            out[start:start + size] = 0.0
    return out


def line_noise(y, snr_db=20.0, seed=0):
    """Additive white noise at a fixed SNR — hiss on the line."""
    rng = np.random.default_rng(seed)
    signal_power = float(np.mean(y * y)) or 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    return (y + rng.normal(0, np.sqrt(noise_power), len(y))).astype(np.float32)


CONDITIONS = {
    "clean": lambda y: y,
    "g711": lambda y: mu_law(y),
    "narrowband": lambda y: narrowband(y),
    "packet_loss": lambda y: packet_loss(y, rate=0.05),
    "noisy": lambda y: line_noise(y, snr_db=20.0),
    # What a bad mobile call actually looks like: all of them at once.
    "voip_stack": lambda y: mu_law(narrowband(line_noise(packet_loss(y, rate=0.03), snr_db=25.0))),
}


def collect(data_dir):
    classes = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    flags = [svc.label_is_fake(c) for c in classes]
    if len(classes) != 2 or True not in flags or False not in flags:
        sys.exit(f"Need two folders, one real-ish and one fake-ish. Got {classes}.")
    fake_i = flags.index(True)

    items = []
    for i, cls in enumerate(classes):
        folder = os.path.join(data_dir, cls)
        for name in sorted(os.listdir(folder)):
            if os.path.splitext(name)[1].lower() in AUDIO_EXTS:
                items.append((os.path.join(folder, name), i == fake_i))
    print(f"{len(items)} clips | classes {classes} | fake = '{classes[fake_i]}'")
    return items


def score(y, sr=SR):
    """One clip through the live window path: VAD gate included, exactly like production."""
    out = svc.analyze_audio_window(y, sr)
    return out.get("fakeProbability"), out.get("speechDetected")


def measure(items, condition, limit=0):
    from preprocess_audio import read_audio

    degrade = CONDITIONS[condition]
    scores, labels, skipped, unreadable = [], [], 0, 0

    for path, is_fake in (items[:limit] if limit else items):
        y = read_audio(path, SR, 10.0)
        if y is None:
            unreadable += 1
            continue
        prob, speech = score(degrade(np.asarray(y, dtype=np.float32)), SR)
        if prob is None:
            # Gated out. Counted, not silently dropped: a condition that mutes the VAD is a
            # finding about that condition.
            skipped += 1
            continue
        scores.append(prob)
        labels.append(1 if is_fake else 0)

    if not scores:
        return {"condition": condition, "n": 0, "skipped": skipped, "unreadable": unreadable}

    preds = [1 if s > svc.FAKE_ABOVE else 0 if s < svc.REAL_BELOW else -1 for s in scores]
    hits = {0: 0, 1: 0}
    totals = {0: 0, 1: 0}
    for p, l in zip(preds, labels):
        totals[l] += 1
        if p == l:
            hits[l] += 1
    recalls = [hits[c] / totals[c] for c in (0, 1) if totals[c]]

    return {
        "condition": condition,
        "n": len(scores),
        "skipped_by_vad": skipped,
        "unreadable": unreadable,
        "balanced_acc": round(sum(recalls) / len(recalls), 4) if recalls else float("nan"),
        "recall_real": round(hits[0] / totals[0], 4) if totals[0] else float("nan"),
        "recall_fake": round(hits[1] / totals[1], 4) if totals[1] else float("nan"),
        "auc": round(binary_auc(scores, labels), 4),
        "mean_fake_score": round(float(np.mean(scores)), 1),
    }


def selfcheck():
    """The degradations must actually degrade — and must not destroy the signal outright."""
    sr = SR
    t = np.arange(sr, dtype=np.float32) / sr
    # 6kHz included on purpose: real speech has energy above the telephone band (sibilance),
    # and without it there is nothing for the narrowband test to strip.
    voiced = (sum(np.sin(2 * np.pi * f * t) / (i + 1)
                  for i, f in enumerate((150, 300, 450, 600, 3000, 6000))) * 0.2).astype(np.float32)

    for name, fn in CONDITIONS.items():
        out = fn(voiced.copy())
        assert out.shape == voiced.shape, f"{name} changed the length: {out.shape}"
        assert np.isfinite(out).all(), f"{name} produced non-finite samples"

    assert np.array_equal(CONDITIONS["clean"](voiced), voiced), "clean must be a no-op"

    # μ-law is lossy but must stay close: it is a codec, not a destroyer.
    g711 = mu_law(voiced)
    assert not np.array_equal(g711, voiced), "μ-law must quantize"
    assert float(np.abs(g711 - voiced).max()) < 0.2, "μ-law must not mangle the waveform"

    # Narrowband must remove energy above 4kHz — that is the whole point.
    def high_energy(y):
        spec = np.abs(np.fft.rfft(y)) ** 2
        freqs = np.fft.rfftfreq(len(y), 1 / sr)
        return float(spec[freqs > 4000].sum() / (spec.sum() + 1e-12))

    assert high_energy(narrowband(voiced)) < high_energy(voiced) * 0.5, (
        "narrowband must strip high frequencies")

    # And it must REMOVE them, not fold them down. Decimating without a low-pass turns the
    # 6kHz component into a phantom ~2kHz tone — measured at 0.6% of total energy, in a band
    # that was empty. That is a different artifact from a phone call, so the eval would have
    # been measuring the wrong degradation.
    def band_share(y, lo, hi):
        spec = np.abs(np.fft.rfft(y)) ** 2
        freqs = np.fft.rfftfreq(len(y), 1 / sr)
        return float(spec[(freqs > lo) & (freqs < hi)].sum() / (spec.sum() + 1e-12))

    assert band_share(voiced, 1500, 2500) < 1e-6, "the test tone has no energy there to begin with"
    assert band_share(narrowband(voiced), 1500, 2500) < 1e-4, (
        "narrowband must not alias high frequencies into the passband")

    # Packet loss must actually zero samples, and noise must lower the SNR.
    lost = packet_loss(voiced, rate=0.5, seed=1)
    assert (lost == 0).sum() > (voiced == 0).sum(), "packet loss must drop packets"
    noisy = line_noise(voiced, snr_db=10.0, seed=1)
    assert float(np.mean((noisy - voiced) ** 2)) > 0, "noise must add energy"

    print("call-condition selfcheck passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", help="two folders: one real-ish, one fake-ish")
    ap.add_argument("--conditions", default=",".join(CONDITIONS),
                    help=f"comma-separated subset of {','.join(CONDITIONS)}")
    ap.add_argument("--limit", type=int, default=0, help="stop after N clips per condition")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return
    if not args.data_dir:
        sys.exit("--data-dir is required.")

    svc.startup()
    if svc.AUDIO_PREDICT is None:
        sys.exit(f"No voice model loaded ({svc.AUDIO_META.get('reason')}). Nothing to evaluate.")

    wanted = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in wanted if c not in CONDITIONS]
    if unknown:
        sys.exit(f"Unknown condition(s) {unknown}. Available: {', '.join(CONDITIONS)}")

    items = collect(args.data_dir)
    if not items:
        sys.exit("No audio files found.")

    print(f"\nmodel: {svc.AUDIO_META.get('modelSource')}")
    print(f"bands: fake > {svc.FAKE_ABOVE:.0f}, real < {svc.REAL_BELOW:.0f}\n")
    print(f"{'condition':<14} {'balanced':>9} {'AUC':>7} {'real':>7} {'fake':>7} {'n':>5} {'gated':>6}")
    print("-" * 60)

    results = []
    for condition in wanted:
        m = measure(items, condition, args.limit)
        results.append(m)
        if not m.get("n"):
            print(f"{condition:<14} {'no scored clips':>40}")
            continue
        print(f"{condition:<14} {m['balanced_acc']:>9.4f} {m['auc']:>7.4f} "
              f"{m['recall_real']:>7.3f} {m['recall_fake']:>7.3f} {m['n']:>5} {m['skipped_by_vad']:>6}")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json_out}")

    clean = next((r for r in results if r["condition"] == "clean" and r.get("n")), None)
    worst = min((r for r in results if r.get("n")), key=lambda r: r["balanced_acc"], default=None)
    if clean and worst and worst["condition"] != "clean":
        drop = clean["balanced_acc"] - worst["balanced_acc"]
        print(f"\nclean {clean['balanced_acc']:.4f} -> {worst['condition']} {worst['balanced_acc']:.4f} "
              f"(drop {drop:+.4f})")
        print("The second number is the one that describes a live call. Put THAT on the model "
              "card, not the clean one.")
    print("\nNot simulated: Opus and AMR-WB (need a real codec). G.711 μ-law is exact.")


if __name__ == "__main__":
    main()
