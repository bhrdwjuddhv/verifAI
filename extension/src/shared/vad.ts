/**
 * The voice-activity gate, ported from the service's `speech_metrics`.
 *
 * A call is mostly silence, breathing and line noise. Scoring those produces confident
 * nonsense, so a window has to look like speech before the model is allowed an opinion. The
 * on-device path needs the same gate as the server, or the two would disagree about which
 * windows count — which looks like a model disagreement and is not one.
 *
 * The one deliberate difference: numpy transforms all 48000 samples with a mixed-radix FFT,
 * and shipping one of those to the browser is not worth it. Flatness is computed over the
 * largest power-of-two prefix instead (32768 = 2.05s of a 3s window). Measured against numpy
 * on signals spanning silence, tones, voiced speech, white and band-limited noise, the largest
 * divergence was 0.0047 — against a threshold of 0.45, and no signal changed its verdict.
 * scripts/check-audio-v2.mjs re-asserts that on fixtures both languages can regenerate.
 */

/** Mirrors VERIFAI_VAD_MIN_DBFS / VERIFAI_VAD_MAX_FLATNESS in scripts/inference_server.py. */
export const VAD_MIN_DBFS = -45;
export const VAD_MAX_FLATNESS = 0.45;

export interface VadResult {
  isSpeech: boolean;
  rmsDbfs: number | null;
  spectralFlatness: number | null;
  /** Why this window was not scored, or null when it was. Shown verbatim, never smoothed. */
  reason: string | null;
}

/** In-place iterative radix-2 Cooley–Tukey. `re`/`im` must be a power of two long. */
function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ar = re[i + k];
        const ai = im[i + k];
        const br = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const bi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ar + br;
        im[i + k] = ai + bi;
        re[i + k + len / 2] = ar - br;
        im[i + k + len / 2] = ai - bi;
        const nr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = nr;
      }
    }
  }
}

/** Spectral flatness: geometric over arithmetic mean of the power spectrum, DC dropped. */
export function spectralFlatness(samples: Float32Array): number | null {
  // Largest power of two that fits. `Math.log2` would round 32768 to 32768 but is fragile at
  // the boundary for other lengths; the shift loop cannot be off by one.
  let n = 1;
  while (n * 2 <= samples.length) n *= 2;
  if (n < 256) return null;

  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    // np.hanning(n) is the symmetric window: denominator n-1, endpoints exactly zero.
    re[i] = samples[i] * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)));
  }
  fft(re, im);

  // Bins 1..n/2 — the rfft range minus DC, which says nothing about voicing.
  let logSum = 0;
  let sum = 0;
  let count = 0;
  for (let k = 1; k <= n / 2; k++) {
    const power = re[k] * re[k] + im[k] * im[k];
    if (power <= 0) continue;
    logSum += Math.log(power);
    sum += power;
    count++;
  }
  if (count === 0) return null;
  return Math.exp(logSum / count) / (sum / count);
}

/**
 * Two cheap gates, no model: loudness, then spectral flatness. Voiced speech is harmonic so
 * flatness is low; hiss, fans and codec comfort-noise approach 1. A sustained pure tone also
 * passes — a known and acceptable false accept, since this gates "is there anything worth
 * scoring", not "is this speech".
 */
export function speechMetrics(samples: Float32Array): VadResult {
  if (samples.length < 256) {
    return { isSpeech: false, rmsDbfs: null, spectralFlatness: null, reason: 'window too short' };
  }

  let sumSquares = 0;
  for (let i = 0; i < samples.length; i++) sumSquares += samples[i] * samples[i];
  const dbfs = 20 * Math.log10(Math.sqrt(sumSquares / samples.length) + 1e-12);
  const rounded = Math.round(dbfs * 10) / 10;

  if (dbfs < VAD_MIN_DBFS) {
    return {
      isSpeech: false,
      rmsDbfs: rounded,
      spectralFlatness: null,
      reason: `too quiet (${dbfs.toFixed(0)} dBFS < ${VAD_MIN_DBFS})`,
    };
  }

  const flatness = spectralFlatness(samples);
  if (flatness === null) {
    return { isSpeech: false, rmsDbfs: rounded, spectralFlatness: null, reason: 'no spectrum' };
  }
  const flat = Math.round(flatness * 1000) / 1000;

  if (flatness > VAD_MAX_FLATNESS) {
    return {
      isSpeech: false,
      rmsDbfs: rounded,
      spectralFlatness: flat,
      reason: `noise-like (flatness ${flatness.toFixed(2)} > ${VAD_MAX_FLATNESS})`,
    };
  }
  return { isSpeech: true, rmsDbfs: rounded, spectralFlatness: flat, reason: null };
}
