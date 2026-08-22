/**
 * Live Guard: monitor a browser call, one 3-second window at a time.
 *
 * The service worker owns the session because it is the only context that survives the popup
 * closing. It never touches audio itself — the offscreen document captures and windows it,
 * this file scores the windows and keeps the risk state.
 *
 * Every verdict is checked against `rejectForeignModel` before it is shown. A score from the
 * Hugging Face fallback, a stale bundle or anything unrecognised is discarded with the reason
 * displayed, because a number a user reads as VerifAI's must have come from VerifAI's models.
 */

import { getSettings } from '../shared/settings';
import { isInCall, siteForUrl } from '../shared/callsites';
// Chrome: creates the offscreen document if it is gone. Firefox: a no-op, because its
// background page already is the document. Neither caller here needs to know which.
import { audioSelftest, ensureOffscreen, type AudioSelftestResult } from '#host';
import {
  ALPHA,
  FRAME_EVERY_MS,
  HYSTERESIS,
  WINDOW_SECONDS,
  initialState,
  pushWindow,
  rejectForeignModel,
  type AudioMode,
  type GuardStatus,
  type GuardWindow,
  type RiskState,
} from '../shared/liveguard';

const MAX_WINDOWS = 200;

const AUDIO_OFF: AudioMode = {
  onDevice: false,
  state: 'off',
  reason: 'On-device voice is switched off; windows go to the backend.',
  selftest: null,
};

interface Session {
  tabId: number;
  siteLabel: string;
  startedAt: number;
  risk: RiskState;
  windows: GuardWindow[];
  latestVideo: number | null;
  modelSource: string | null;
  note: string | null;
  audio: AudioMode;
  frameTimer: ReturnType<typeof setInterval> | null;
}

let session: Session | null = null;

export function guardStatus(): GuardStatus {
  if (!session) {
    return {
      active: false, tabId: null, siteLabel: null, source: 'server', audio: AUDIO_OFF, trust: null,
      band: 'idle', scored: 0, skipped: 0, consecutiveHigh: 0, modelSource: null,
      windows: [], note: null,
    };
  }
  return {
    active: true,
    tabId: session.tabId,
    siteLabel: session.siteLabel,
    source: session.audio.onDevice ? 'device' : 'server',
    audio: session.audio,
    trust: session.risk.trust,
    band: session.risk.band,
    scored: session.risk.scored,
    skipped: session.risk.skipped,
    consecutiveHigh: session.risk.consecutiveHigh,
    modelSource: session.modelSource,
    windows: session.windows.slice(-60),
    note: session.note,
  };
}

async function broadcast() {
  const status = guardStatus();
  if (session) {
    await chrome.tabs.sendMessage(session.tabId, { type: 'guard:status', status }).catch(() => undefined);
  }
  chrome.runtime.sendMessage({ type: 'guard:status', status }).catch(() => undefined);
}

/** POST one window to the project's own backend. No third-party services, ever. */
async function scoreWindow(wav: ArrayBuffer): Promise<void> {
  if (!session) return;
  const { serverUrl } = await getSettings();

  const form = new FormData();
  form.append('file', new Blob([wav], { type: 'audio/wav' }), 'window.wav');

  let res: Response;
  try {
    res = await fetch(`${serverUrl}/api/live/audio-window`, {
      method: 'POST',
      body: form,
      signal: AbortSignal.timeout(20_000),
    });
  } catch (err) {
    session.note = `Detector unreachable: ${(err as Error)?.message ?? err}. The reading is held, not reset.`;
    await broadcast();
    return;
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    session.note = body.detail || body.error || `Detector returned HTTP ${res.status}.`;
    await broadcast();
    return;
  }

  await recordWindow(await res.json());
}

/**
 * Fold one window result into the risk state, whichever path produced it.
 *
 * Both paths answer in the same shape on purpose, so the provenance check, the VAD handling
 * and the smoothing happen once. A second copy of this for the on-device path would be a
 * second place for "unscored" to quietly become "safe".
 */
async function recordWindow(out: any): Promise<void> {
  if (!session) return;
  const t = (Date.now() - session.startedAt) / 1000;

  const foreign = rejectForeignModel(out.modelSource);
  if (foreign) {
    // Fail honestly rather than fall back to something else.
    session.note = foreign;
    session.windows.push({ t, audio: null, video: null, fused: null, speech: false, modelSource: out.modelSource ?? null, note: 'rejected: not our model' });
    await broadcast();
    return;
  }

  session.note = null;
  session.modelSource = out.modelSource;

  if (!out.speechDetected || out.fakeProbability === null) {
    session.risk = pushWindow(session.risk, { audio: null, video: null }, { alpha: ALPHA, hysteresis: HYSTERESIS });
    session.windows.push({ t, audio: null, video: null, fused: null, speech: false, modelSource: out.modelSource, note: out.vad?.reason ?? out.notes?.[0] });
  } else {
    const audio: number = out.fakeProbability;
    session.risk = pushWindow(session.risk, { audio, video: session.latestVideo }, { alpha: ALPHA, hysteresis: HYSTERESIS });
    session.windows.push({
      t, audio, video: session.latestVideo, fused: session.risk.smoothed, speech: true,
      modelSource: out.modelSource,
    });
  }

  if (session.windows.length > MAX_WINDOWS) session.windows.splice(0, session.windows.length - MAX_WINDOWS);
  await broadcast();
}

/**
 * Sample one frame of the call.
 *
 * `captureVisibleTab` rather than reading the <video> element: a DRM-protected or
 * cross-origin video taints a canvas and throws, while the tab capture is a screenshot of what
 * is actually on screen. A black or blank frame is reported as such — never scored as "real".
 */
async function scoreFrame(): Promise<void> {
  if (!session) return;
  let dataUrl: string;
  try {
    dataUrl = await chrome.tabs.captureVisibleTab({ format: 'jpeg', quality: 80 });
  } catch {
    return; // tab not focused, or permission not granted for this tab right now
  }

  const blob = await (await fetch(dataUrl)).blob();
  const { serverUrl } = await getSettings();
  const form = new FormData();
  form.append('file', blob, 'frame.jpg');

  try {
    const res = await fetch(`${serverUrl}/api/live/frame`, {
      method: 'POST',
      body: form,
      signal: AbortSignal.timeout(30_000),
    });
    if (!res.ok) return;
    const out = await res.json();
    if (rejectForeignModel(out.modelSource)) return;
    session.latestVideo = out.fakeProbability ?? null;
    if (out.modelSource) session.modelSource = out.modelSource;
  } catch {
    /* a frame that fails to score simply does not vote */
  }
}

/**
 * Decide, once per session, whether voice runs on this machine.
 *
 * The setting is only a request. What actually opens the gate is the offscreen document
 * reproducing `expected_prob` from `audio_selftest.json` through `preproc.onnx -> CNN` within
 * `tol`, here, on this GPU or this wasm build. That is the whole safety argument for running
 * audio off-server at all: without it, a subtly different spectrogram would produce confident
 * nonsense that looks exactly like a real verdict.
 *
 * Every failure path lands on the backend. None of them lands on a made-up number.
 */
async function armAudio(): Promise<AudioMode> {
  const { onDeviceAudio } = await getSettings();
  if (!onDeviceAudio) return AUDIO_OFF;

  const result: AudioSelftestResult = await audioSelftest().catch((err: Error) => ({
    status: 'unavailable' as const,
    reason: err.message,
  }));

  const selftest = {
    status: result?.status ?? 'unavailable',
    delta: result?.delta,
    tol: result?.tol,
    ep: result?.ep,
  };

  if (result?.status === 'pass') {
    return {
      onDevice: true,
      state: 'verified',
      reason: null,
      selftest,
    };
  }
  return {
    onDevice: false,
    state: 'backend',
    reason:
      `On-device voice is switched on but its parity self-test did not pass ` +
      `(${result?.reason ?? 'no reason given'}), so windows go to the backend instead.`,
    selftest,
  };
}

export async function startGuard(tabId: number): Promise<{ ok: boolean; reason?: string }> {
  if (session) await stopGuard();

  const tab = await chrome.tabs.get(tabId).catch(() => null);
  const site = siteForUrl(tab?.url);
  if (!site) return { ok: false, reason: 'This tab is not a supported call platform.' };
  if (!isInCall(tab?.url)) {
    return { ok: false, reason: `Open a call on ${site.label} first — this looks like a lobby or home page.` };
  }

  const granted = await chrome.permissions.contains({ origins: site.origins }).catch(() => false);
  if (!granted) return { ok: false, reason: `needs-permission:${site.origins[0]}` };

  // getMediaStreamId is callback-style in @types/chrome; promisified here rather than left
  // as a floating callback the caller cannot await.
  let streamId: string;
  try {
    streamId = await new Promise<string>((resolve, reject) => {
      chrome.tabCapture.getMediaStreamId({ targetTabId: tabId }, (id) => {
        const failure = chrome.runtime.lastError;
        if (failure || !id) reject(new Error(failure?.message ?? 'no stream id returned'));
        else resolve(id);
      });
    });
  } catch (err) {
    return { ok: false, reason: `Could not capture this tab: ${(err as Error)?.message ?? err}` };
  }

  // Chrome evicts the offscreen document; without this, the first window of a call after an
  // idle period would have nowhere to be captured.
  await ensureOffscreen();

  // Decided before capture starts and fixed for the session: switching paths mid-call would
  // put two differently-calibrated score sources into one smoothed trust number.
  const audio = await armAudio();

  session = {
    tabId,
    siteLabel: site.label,
    startedAt: Date.now(),
    risk: initialState(),
    windows: [],
    latestVideo: null,
    modelSource: null,
    note: audio.state === 'backend' ? audio.reason : null,
    audio,
    frameTimer: null,
  };

  const started = await chrome.runtime.sendMessage({
    type: 'offscreen:live-start',
    streamId,
    windowSeconds: WINDOW_SECONDS,
    onDevice: audio.onDevice,
  }).catch((err: Error) => ({ ok: false, reason: err.message }));

  if (!started?.ok) {
    session = null;
    return { ok: false, reason: started?.reason ?? 'The offscreen document did not start capture.' };
  }

  session.frameTimer = setInterval(() => void scoreFrame(), FRAME_EVERY_MS);
  await broadcast();
  return { ok: true };
}

export async function stopGuard(): Promise<void> {
  if (session?.frameTimer) clearInterval(session.frameTimer);
  session = null;
  await chrome.runtime.sendMessage({ type: 'offscreen:live-stop' }).catch(() => undefined);
  await broadcast();
}

/** Called by the offscreen document for every completed window, on the backend route. */
export async function onLiveWindow(wav: ArrayBuffer): Promise<void> {
  if (!session) return;
  await scoreWindow(wav);
}

/**
 * Called by the offscreen document for every window it scored itself.
 *
 * A window the chain could not score demotes the whole session to the backend rather than
 * being dropped. Silently losing coverage is the failure a live guard cannot afford: a user
 * watching a steady trust number has no way to tell it has stopped updating.
 */
export async function onLiveScore(out: any): Promise<void> {
  if (!session) return;

  if (out?.onDeviceFailed) {
    session.audio = {
      onDevice: false,
      state: 'backend',
      reason: `On-device voice stopped mid-call (${out.notes?.[0] ?? 'unknown error'}); ` +
        'switched to the backend for the rest of this call.',
      selftest: session.audio.selftest,
    };
    await chrome.runtime
      .sendMessage({ type: 'offscreen:live-route', onDevice: false })
      .catch(() => undefined);
    session.note = session.audio.reason;
    await broadcast();
    return;
  }

  await recordWindow(out);
}

export function guardTabId(): number | null {
  return session?.tabId ?? null;
}
