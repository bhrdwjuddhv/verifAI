"""The v2 audio chain: raw samples -> preproc.onnx -> CNN, gated by a numerical self-test.

Why a self-test gates this at all: the browser cannot run librosa, so on-device audio was
previously impossible to trust — a JS mel-spectrogram that is subtly wrong produces confident
nonsense with nothing in the logs. v2 solves it by exporting the preprocessing itself to ONNX,
so server and browser run the same graph. The self-test is what proves that claim at runtime
instead of assuming it: one fixed input, one expected probability, one tolerance.

Nothing here falls back to a guess. If the chain will not load, or the self-test does not
reproduce `expected_prob` within `tol`, v2 is reported unavailable and the caller keeps using
whatever it was using before.
"""

import json
import os

import numpy as np

try:
    from .config import resolve          # imported as common.audio_v2, the normal path
except ImportError:                      # run directly: python scripts/common/audio_v2.py
    from config import resolve

# resolve() so this works from any working directory. Without it, running a script from
# scripts/ reports "v2 audio not installed" — indistinguishable from actually not having it,
# and the service would quietly serve v1 instead.
V2_DIR = resolve(os.environ.get("VERIFAI_AUDIO_V2_DIR")
                 or os.path.join("models", "audio", "v2", "models", "audio"))
V2_PREPROC = os.path.join(V2_DIR, "preproc.onnx")
V2_CNN = os.path.join(V2_DIR, "audio_detector.onnx")
V2_SELFTEST = os.path.join(V2_DIR, "audio_selftest.json")

SR = 16000


def reconstruct_selftest_audio(meta):
    """Rebuild the full self-test vector, verified against what the file does carry.

    The shipped JSON stores only the first 1600 of 48000 samples (see the generator in
    v2/scripts/export_audio_preproc.py: `y3[:1600]`), while `expected_prob` was computed on
    all 48000. Feeding the stored 1600 into the chain therefore CANNOT reproduce it.

    The vector is a documented deterministic chirp, so it is regenerated here and then checked
    two ways — the stored prefix must match sample-for-sample, and the sum of absolute values
    must match `full_audio_checksum`. If either fails, this returns None and the gate stays
    shut: a self-test run on the wrong input proves nothing.
    """
    stored = np.asarray(meta.get("audio") or [], dtype=np.float32)
    total = int(meta.get("num_samples") or 0)
    if stored.size == 0 or total <= 0:
        return None, "self-test JSON carries no audio vector"

    if stored.size == total:
        return stored, None  # whole vector present: nothing to reconstruct

    seconds = float(meta.get("duration_sec") or (total / SR))
    rate = int(meta.get("sample_rate") or SR)
    t = np.linspace(0, seconds, total, endpoint=False)
    rebuilt = (0.6 * np.sin(2 * np.pi * 440 * t)
               + 0.3 * np.sin(2 * np.pi * 880 * t)
               + 0.1 * np.sin(2 * np.pi * 1320 * t)).astype(np.float32)

    if rate != SR:
        return None, f"self-test sample rate {rate} is not {SR}"
    if not np.allclose(rebuilt[: stored.size], stored, atol=1e-5):
        return None, ("self-test vector is truncated and the reconstruction does not match its "
                      "stored prefix — regenerate the JSON with the full vector")

    checksum = meta.get("full_audio_checksum")
    if checksum is not None and abs(float(np.sum(np.abs(rebuilt))) - float(checksum)) > 1e-2:
        return None, ("self-test vector is truncated and the reconstruction does not match "
                      "full_audio_checksum — regenerate the JSON with the full vector")

    return rebuilt, None


def load_chain(threads=2):
    """(preproc, cnn, meta, None) or (None, None, None, reason). Loads nothing on failure."""
    for path in (V2_PREPROC, V2_CNN, V2_SELFTEST):
        if not os.path.exists(path):
            return None, None, None, f"v2 audio not installed: {path} is missing"

    try:
        import onnxruntime as ort
    except ImportError as e:
        return None, None, None, f"onnxruntime unavailable ({e})"

    try:
        with open(V2_SELFTEST) as fh:
            meta = json.load(fh)

        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.intra_op_num_threads = threads
        preproc = ort.InferenceSession(V2_PREPROC, sess_options=options, providers=["CPUExecutionProvider"])
        cnn = ort.InferenceSession(V2_CNN, sess_options=options, providers=["CPUExecutionProvider"])
        return preproc, cnn, meta, None
    except Exception as e:
        # The common case, and worth naming precisely: an ONNX exported with external data
        # needs its .onnx.data sidecar next to it, and a drop that omits one loads nowhere.
        reason = f"{type(e).__name__}: {str(e).splitlines()[0]}"
        if "External data" in reason or "external" in reason.lower():
            reason += " — the .onnx.data sidecar was not shipped alongside the .onnx"
        return None, None, None, reason


def probabilities(preproc, cnn, samples):
    """raw samples -> preproc.onnx -> CNN -> softmax probabilities."""
    audio = np.asarray(samples, dtype=np.float32).reshape(1, -1)
    image = preproc.run(None, {preproc.get_inputs()[0].name: audio})[0]
    logits = cnn.run(None, {cnn.get_inputs()[0].name: image})[0].astype(np.float32)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def run_selftest(threads=2):
    """The gate. Returns a dict that /health can show verbatim."""
    preproc, cnn, meta, reason = load_chain(threads)
    if reason:
        return {"status": "unavailable", "reason": reason}

    samples, why = reconstruct_selftest_audio(meta)
    if samples is None:
        return {"status": "unavailable", "reason": why}

    try:
        probs = probabilities(preproc, cnn, samples)
    except Exception as e:
        return {"status": "fail", "reason": f"chain raised {type(e).__name__}: {e}"}

    expected = float(meta.get("expected_prob", 0.5))
    tol = float(meta.get("tol", 1e-3))
    observed = float(probs[0, 0])  # index 0 = Fake, per the v2 detector.json labels
    delta = abs(observed - expected)

    return {
        "status": "pass" if delta <= tol else "fail",
        "observed": round(observed, 5),
        "expected": expected,
        "tol": tol,
        "delta": round(delta, 6),
        "pipeline": meta.get("pipeline"),
        "reason": None if delta <= tol else f"observed {observed:.5f} vs expected {expected:.5f} (tol {tol})",
    }


def selfcheck():
    """Exercises the gate's logic without the graphs — which is most of what can go wrong."""
    real = json.load(open(V2_SELFTEST)) if os.path.exists(V2_SELFTEST) else {
        "sample_rate": SR, "duration_sec": 3.0, "num_samples": 48000,
        "audio": [], "full_audio_checksum": 0.0,
    }

    if real.get("audio"):
        samples, why = reconstruct_selftest_audio(real)
        assert samples is not None, f"the shipped self-test must reconstruct: {why}"
        assert len(samples) == real["num_samples"]
        assert abs(float(np.sum(np.abs(samples))) - real["full_audio_checksum"]) < 1e-2

    # A truncated vector whose prefix does NOT match must be refused, not "reconstructed".
    bad = dict(real)
    bad["audio"] = [0.0] * 100
    bad["num_samples"] = 48000
    samples, why = reconstruct_selftest_audio(bad)
    assert samples is None and "prefix" in (why or ""), f"a wrong prefix must fail closed: {why}"

    # A vector already complete is passed through untouched.
    whole = {"sample_rate": SR, "duration_sec": 0.01, "num_samples": 4, "audio": [0.1, 0.2, 0.3, 0.4]}
    samples, why = reconstruct_selftest_audio(whole)
    assert why is None and len(samples) == 4

    # No vector at all: unavailable, never a silent pass.
    samples, why = reconstruct_selftest_audio({"num_samples": 0, "audio": []})
    assert samples is None and why

    result = run_selftest()
    assert result["status"] in {"pass", "fail", "unavailable"}
    assert result["status"] != "pass" or result["delta"] <= result["tol"]
    print(f"audio_v2 selfcheck passed (chain status: {result['status']} — {result.get('reason') or 'ok'})")


if __name__ == "__main__":
    selfcheck()
