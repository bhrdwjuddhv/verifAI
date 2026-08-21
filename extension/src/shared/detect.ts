/**
 * The scoring maths, ported from the Python service so the two cannot drift.
 *
 * Every function here has a named counterpart in scripts/inference_server.py or
 * scripts/common/config.py, and the port is deliberately literal — including the parts that
 * look like they could be simplified. `fake_probability` renormalising over only the classes
 * whose names commit to a side, and `fuse` renormalising over only the detectors that ran,
 * are both load-bearing: shortcut either and a missing detector starts voting "50".
 *
 * Pure: no DOM, no chrome.*, no ORT. That is what lets the parity harness run this exact
 * source under Node.
 */

export const IMAGENET_MEAN = [0.485, 0.456, 0.406];
export const IMAGENET_STD = [0.229, 0.224, 0.225];

/** common/config.py — verdict bands on P(fake), in percent. */
export const FAKE_ABOVE = 70;
export const REAL_BELOW = 30;

/** common/config.py FUSION_DEFAULTS. frequency is measured and reported, but does not vote. */
export const FUSION_DEFAULTS: Record<string, number> = { face: 0.5, npr: 0.5, frequency: 0 };

const FAKE_WORDS = ['fake', 'deepfake', 'artificial', 'synthetic', 'spoof', 'gan', 'generated'];
const REAL_WORDS = ['real', 'realism', 'human', 'authentic', 'genuine', 'natural', 'pristine'];

/**
 * True / false / null when the label name commits to neither.
 *
 * Matching on the name beats trusting an index: ImageFolder sorts alphabetically so
 * 'Fake' < 'Real', but a Hugging Face model can order its labels however it likes, and an
 * inverted mapping reads as a confidently wrong verdict rather than as a bug.
 */
export function labelIsFake(name: string): boolean | null {
  const n = name.toLowerCase();
  const realHit = REAL_WORDS.some((w) => n.includes(w));
  const fakeHit = FAKE_WORDS.some((w) => n.includes(w));
  if (realHit && fakeHit) return null; // 'real_vs_fake' is a folder name, not a class
  if (realHit) return false;
  if (fakeHit) return true;
  return null;
}

export function fakeIndex(labels: string[]): number {
  const i = labels.findIndex((l) => labelIsFake(l) === true);
  if (i < 0) throw new Error(`no fake class in ${JSON.stringify(labels)}`);
  return i;
}

/** P(fake) renormalised over the classes whose names commit to a side. */
export function fakeProbability(probs: number[], labels: string[]): number {
  let fake = 0;
  let real = 0;
  labels.forEach((label, i) => {
    const side = labelIsFake(label);
    if (side === true) fake += probs[i];
    else if (side === false) real += probs[i];
  });
  if (fake + real <= 0) {
    throw new Error(`no class in ${JSON.stringify(labels)} is identifiable as real or fake`);
  }
  return fake / (fake + real);
}

/** Temperature is applied to the logits before the softmax, exactly as the service does. */
export function softmax(logits: ArrayLike<number>, temperature = 1): number[] {
  const scaled: number[] = [];
  for (let i = 0; i < logits.length; i++) scaled.push(logits[i] / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export interface FusionResult {
  /** Rounded percent, or null when nothing voted. */
  fused: number | null;
  used: Record<string, number>;
}

/**
 * Weighted mean of the signals that actually ran.
 *
 * Weight 0 means "reported, not trusted" and must not appear as a voter — which is why the
 * filter tests the weight as well as the value.
 */
export function fuse(
  scores: Record<string, number | null | undefined>,
  weights: Record<string, number> = FUSION_DEFAULTS
): FusionResult {
  const used: Record<string, number> = {};
  for (const [key, value] of Object.entries(scores)) {
    const weight = weights[key] ?? 0;
    if (value !== null && value !== undefined && weight > 0) used[key] = weight;
  }
  const total = Object.values(used).reduce((a, b) => a + b, 0);
  if (!total) return { fused: null, used: {} };

  let sum = 0;
  for (const [key, weight] of Object.entries(used)) sum += (scores[key] as number) * weight;
  return { fused: Math.round(sum / total), used };
}

export type Verdict = 'real' | 'fake' | 'uncertain';

/** (verdict, confidence in that verdict). The middle band is an answer, not a failure. */
export function verdictFor(fakePct: number): { verdict: Verdict; confidence: number } {
  if (fakePct > FAKE_ABOVE) return { verdict: 'fake', confidence: fakePct };
  if (fakePct < REAL_BELOW) return { verdict: 'real', confidence: 100 - fakePct };
  return { verdict: 'uncertain', confidence: Math.max(fakePct, 100 - fakePct) };
}

// ---------------------------------------------------------------------------------------
// Frequency score

/**
 * Share of spectral energy above half-Nyquist, 0-100.
 *
 * A descriptive statistic, NOT a probability and NOT part of the verdict — diffusion
 * upsamplers leave energy up here, but so does a sharp camera. Reported because the app
 * reports it, and a blank where the website shows a number reads as a bug.
 *
 * Expects the 256x256 grayscale the service builds: convert("L") first, then resize.
 */
export function frequencyScore(gray256: Float32Array): number {
  const N = 256;
  const mean = gray256.reduce((a, b) => a + b, 0) / gray256.length;

  // Hann window. Without it the image border is a step edge that dumps energy into every
  // frequency — i.e. straight into the number being measured.
  const hann = new Float32Array(N);
  for (let i = 0; i < N; i++) hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1));

  const re = new Float64Array(N * N);
  const im = new Float64Array(N * N);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      re[y * N + x] = (gray256[y * N + x] - mean) * hann[y] * hann[x];
    }
  }

  fft2(re, im, N);

  let total = 0;
  let high = 0;
  for (let u = 0; u < N; u++) {
    // fftshift: bin u carries frequency u - N/2 once shifted, and the mask is on radius.
    const fy = ((u + N / 2) % N) - N / 2;
    for (let v = 0; v < N; v++) {
      const fx = ((v + N / 2) % N) - N / 2;
      const i = u * N + v;
      const power = re[i] * re[i] + im[i] * im[i];
      total += power;
      if (Math.hypot(fx, fy) > N / 4) high += power;
    }
  }
  return total <= 0 ? 0 : Math.round((100 * high) / total);
}

/** In-place 2D FFT by rows then columns. N must be a power of two. */
function fft2(re: Float64Array, im: Float64Array, N: number): void {
  const rowRe = new Float64Array(N);
  const rowIm = new Float64Array(N);

  for (let y = 0; y < N; y++) {
    rowRe.set(re.subarray(y * N, y * N + N));
    rowIm.set(im.subarray(y * N, y * N + N));
    fft(rowRe, rowIm);
    re.set(rowRe, y * N);
    im.set(rowIm, y * N);
  }

  for (let x = 0; x < N; x++) {
    for (let y = 0; y < N; y++) {
      rowRe[y] = re[y * N + x];
      rowIm[y] = im[y * N + x];
    }
    fft(rowRe, rowIm);
    for (let y = 0; y < N; y++) {
      re[y * N + x] = rowRe[y];
      im[y * N + x] = rowIm[y];
    }
  }
}

/** Iterative radix-2 Cooley-Tukey, in place. */
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
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const uRe = re[i + k];
        const uIm = im[i + k];
        const vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
        const vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
        re[i + k] = uRe + vRe;
        im[i + k] = uIm + vIm;
        re[i + k + len / 2] = uRe - vRe;
        im[i + k + len / 2] = uIm - vIm;
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }
}
