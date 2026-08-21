/**
 * Live-call risk engine: fuse the per-window signals, smooth them over time, and only move the
 * displayed band when the evidence has actually persisted.
 *
 * Why smoothing and hysteresis rather than showing each window: a single 3-second window is a
 * noisy measurement. Codec dropouts, a cough, or one bad frame will spike it. A guard that
 * flips between "fine" and "synthetic" every few seconds is one the user learns to ignore,
 * which is worse than not having it.
 *
 * Pure functions with no browser APIs, so `scripts/check-risk.mts` can exercise them.
 */

export type RiskBand = 'low' | 'uncertain' | 'suspicious' | 'high' | 'idle';

export interface CallWindowSignals {
  /** Voice model P(synthetic), 0-100. Null when the window held no speech. */
  audio: number | null;
  /** Fused face+NPR P(fake) for the nearest sampled frame, 0-100. Null on an audio-only call. */
  video: number | null;
}

/**
 * The `call` fusion profile. Same rule as the image pipeline: renormalize over whichever
 * signals actually arrived, so a missing one does not drag the mean toward 50.
 *
 * 0.5/0.5 is a starting point, not a measurement — exactly like the image defaults before
 * tune_fusion.py measured them. Override with NEXT_PUBLIC_CALL_FUSION_WEIGHTS="audio=0.7,video=0.3".
 */
export const CALL_FUSION_DEFAULTS: Record<string, number> = { audio: 0.5, video: 0.5 };

export function parseCallWeights(spec?: string): Record<string, number> {
  const weights = { ...CALL_FUSION_DEFAULTS };
  for (const part of (spec ?? '').split(',').map((p) => p.trim()).filter(Boolean)) {
    const [key, value] = part.split('=');
    if (!(key in weights)) continue; // unknown key: ignore rather than crash a live call
    const parsed = Number(value);
    if (Number.isFinite(parsed) && parsed >= 0) weights[key] = parsed;
  }
  return Object.values(weights).some((w) => w > 0) ? weights : { ...CALL_FUSION_DEFAULTS };
}

/** Weighted mean over the signals that ran. Returns null when nothing did. */
export function fuseWindow(
  signals: CallWindowSignals,
  weights: Record<string, number> = CALL_FUSION_DEFAULTS
): { risk: number | null; used: string[] } {
  const present = Object.entries(signals).filter(
    ([key, value]) => value !== null && (weights[key] ?? 0) > 0
  ) as [string, number][];

  const total = present.reduce((sum, [key]) => sum + weights[key], 0);
  if (!present.length || total <= 0) return { risk: null, used: [] };

  const risk = present.reduce((sum, [key, value]) => sum + value * weights[key], 0) / total;
  return { risk: Math.round(risk), used: present.map(([key]) => key) };
}

export const BAND_THRESHOLDS = { suspicious: 30, high: 70 } as const;

/** Band for a single smoothed value. Mirrors the file pipeline's >70 / <30 split. */
export function bandFor(risk: number): Exclude<RiskBand, 'idle'> {
  if (risk > BAND_THRESHOLDS.high) return 'high';
  if (risk > BAND_THRESHOLDS.suspicious) return 'suspicious';
  if (risk > 15) return 'uncertain';
  return 'low';
}

export interface RiskEngineOptions {
  /** EMA weight for the newest window. 0.4 ≈ a 5-window memory. */
  alpha?: number;
  /** Windows that must agree before the displayed band moves. */
  hysteresis?: number;
  weights?: Record<string, number>;
}

export interface RiskState {
  /** Smoothed P(synthetic) 0-100, or null before the first scored window. */
  smoothed: number | null;
  /** Trust score shown to the user: 100 - smoothed. */
  trust: number | null;
  /** The band actually displayed — changes only after `hysteresis` agreeing windows. */
  band: RiskBand;
  /** The band the latest smoothed value would imply, before hysteresis. */
  pendingBand: RiskBand;
  pendingCount: number;
  scored: number;
  skipped: number;
  /** Consecutive scored windows in the `high` band — what the alert counts. */
  consecutiveHigh: number;
}

export function initialState(): RiskState {
  return {
    smoothed: null,
    trust: null,
    band: 'idle',
    pendingBand: 'idle',
    pendingCount: 0,
    scored: 0,
    skipped: 0,
    consecutiveHigh: 0,
  };
}

/**
 * Fold one window into the state.
 *
 * A window with no usable signal counts as skipped and changes nothing — silence is not
 * evidence of safety, so the band holds rather than drifting toward "low".
 */
export function pushWindow(
  state: RiskState,
  signals: CallWindowSignals,
  options: RiskEngineOptions = {}
): RiskState {
  const alpha = options.alpha ?? 0.4;
  const hysteresis = options.hysteresis ?? 3;
  const { risk } = fuseWindow(signals, options.weights ?? CALL_FUSION_DEFAULTS);

  if (risk === null) {
    return { ...state, skipped: state.skipped + 1 };
  }

  const smoothed = state.smoothed === null ? risk : alpha * risk + (1 - alpha) * state.smoothed;
  const rounded = Math.round(smoothed);
  const implied = bandFor(rounded);

  let { band, pendingBand, pendingCount } = state;
  if (implied === band) {
    // Already there: reset any half-finished move.
    pendingBand = implied;
    pendingCount = 0;
  } else if (implied === pendingBand) {
    pendingCount += 1;
    // The first window that disagrees starts the count at 1, so it takes `hysteresis` of them.
    if (pendingCount >= hysteresis) {
      band = implied;
      pendingCount = 0;
    }
  } else {
    pendingBand = implied;
    pendingCount = 1;
  }

  // `band === 'idle'` only before the first scored window; adopt the first reading directly so
  // the meter is not stuck on "Listening…" for three windows.
  if (state.band === 'idle') {
    band = implied;
    pendingBand = implied;
    pendingCount = 0;
  }

  return {
    smoothed: rounded,
    trust: 100 - rounded,
    band,
    pendingBand,
    pendingCount,
    scored: state.scored + 1,
    skipped: state.skipped,
    consecutiveHigh: implied === 'high' ? state.consecutiveHigh + 1 : 0,
  };
}

/**
 * Voice timbre drift — a cheap check for the voice changing mid-call.
 *
 * NOT speaker recognition and not a deepfake score: it compares the average shape of the
 * spectrum against the call's running reference. A real speaker moving closer to the mic will
 * move this too, so it is reported as a prompt to look, never as a verdict.
 */
export function timbreDistance(reference: number[], current: number[]): number | null {
  if (!reference.length || reference.length !== current.length) return null;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < reference.length; i++) {
    dot += reference[i] * current[i];
    na += reference[i] * reference[i];
    nb += current[i] * current[i];
  }
  if (na === 0 || nb === 0) return null;
  return Math.round((1 - dot / (Math.sqrt(na) * Math.sqrt(nb))) * 1000) / 1000;
}
