/**
 * Live Guard shared types, and the risk engine.
 *
 * The engine is imported from the website's `lib/call/risk.ts` rather than reimplemented: the
 * extension and the website must band identically, and `scripts/check-risk.mts` already
 * asserts the awkward parts (a spike must not flip the band, silence must not read as safety).
 * A second copy here would be a second set of thresholds to keep in step.
 */

export {
  bandFor,
  fuseWindow,
  initialState,
  parseCallWeights,
  pushWindow,
  timbreDistance,
  CALL_FUSION_DEFAULTS,
  type RiskBand,
  type RiskState,
  type CallWindowSignals,
  // explicit .ts so both vite and a bare 'node --experimental-strip-types' resolve it
} from '../../../lib/call/risk.ts';

/** Where a Live Guard verdict came from. Never a placeholder — see `assertOurModel`. */
export type GuardSource = 'server' | 'device';

export interface GuardWindow {
  /** Seconds since monitoring started. */
  t: number;
  audio: number | null;
  video: number | null;
  fused: number | null;
  speech: boolean;
  /** Names the model(s) that produced this window, e.g. "onnx:trained_audio_checkpoint(...)". */
  modelSource: string | null;
  note?: string;
}

/**
 * Where the voice score for this session is coming from, and why.
 *
 * `verified` is the only state in which audio stays on the machine, and it is only reachable
 * by the parity self-test passing at runtime. Every other state routes to the backend — the
 * `reason` is shown rather than hidden, because "we quietly stopped doing the private thing"
 * is exactly the kind of change a user should not have to infer.
 */
export interface AudioMode {
  onDevice: boolean;
  state: 'verified' | 'backend' | 'off';
  reason: string | null;
  selftest?: { status: string; delta?: number; tol?: number; ep?: string } | null;
}

export interface GuardStatus {
  active: boolean;
  tabId: number | null;
  siteLabel: string | null;
  source: GuardSource;
  /** Whether voice ran here or on the backend, and what decided that. */
  audio: AudioMode;
  /** 0-100 trust (100 - smoothed risk), null before the first scored window. */
  trust: number | null;
  band: string;
  scored: number;
  skipped: number;
  consecutiveHigh: number;
  modelSource: string | null;
  windows: GuardWindow[];
  /** Why monitoring is degraded or stopped: shown verbatim, never smoothed over. */
  note: string | null;
}

/**
 * Our models, by the `modelSource` strings the service and the bundled JSONs actually use.
 *
 * A verdict is only shown if it names one of these. The point is not paranoia: it is that a
 * silent fallback to some other model — a stale bundle, the HF fallback, a third-party API —
 * would produce numbers a user reads as VerifAI's. Failing loudly is the honest outcome.
 */
const OUR_MODEL_MARKERS = ['trained_checkpoint', 'trained_audio_checkpoint', 'npr:', 'audio_v2'];

export function isOurModel(modelSource: string | null | undefined): boolean {
  if (!modelSource) return false;
  return OUR_MODEL_MARKERS.some((marker) => modelSource.includes(marker));
}

/** Returns null when the source is ours; otherwise the reason to show instead of a score. */
export function rejectForeignModel(modelSource: string | null | undefined): string | null {
  if (isOurModel(modelSource)) return null;
  if (!modelSource) return 'The detector did not say which model produced this — not shown.';
  if (modelSource.includes('hf_fallback')) {
    return `Refusing to show a verdict: the service answered with the third-party fallback (${modelSource}), not this project's trained model.`;
  }
  return `Refusing to show a verdict from an unrecognized model (${modelSource}).`;
}

export const WINDOW_SECONDS = 3;
export const FRAME_EVERY_MS = 6000;
export const HYSTERESIS = 3;
export const ALPHA = 0.4;
