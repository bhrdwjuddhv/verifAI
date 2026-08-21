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
import {
  ALPHA,
  FRAME_EVERY_MS,
  HYSTERESIS,
  WINDOW_SECONDS,
  initialState,
  pushWindow,
  rejectForeignModel,
  type GuardStatus,
  type GuardWindow,
  type RiskState,
} from '../shared/liveguard';

const MAX_WINDOWS = 200;

interface Session {
  tabId: number;
  siteLabel: string;
  startedAt: number;
  risk: RiskState;
  windows: GuardWindow[];
  latestVideo: number | null;
  modelSource: string | null;
  note: string | null;
  frameTimer: ReturnType<typeof setInterval> | null;
}

let session: Session | null = null;

export function guardStatus(): GuardStatus {
  if (!session) {
    return {
      active: false, tabId: null, siteLabel: null, source: 'server', trust: null,
      band: 'idle', scored: 0, skipped: 0, consecutiveHigh: 0, modelSource: null,
      windows: [], note: null,
    };
  }
  return {
    active: true,
    tabId: session.tabId,
    siteLabel: session.siteLabel,
    source: 'server',
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

  const out = await res.json();
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
    session.windows.push({ t, audio: null, video: null, fused: null, speech: false, modelSource: out.modelSource, note: out.vad?.reason });
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

  session = {
    tabId,
    siteLabel: site.label,
    startedAt: Date.now(),
    risk: initialState(),
    windows: [],
    latestVideo: null,
    modelSource: null,
    note: null,
    frameTimer: null,
  };

  const started = await chrome.runtime.sendMessage({
    type: 'offscreen:live-start',
    streamId,
    windowSeconds: WINDOW_SECONDS,
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

/** Called by the offscreen document for every completed window. */
export async function onLiveWindow(wav: ArrayBuffer): Promise<void> {
  if (!session) return;
  await scoreWindow(wav);
}

export function guardTabId(): number | null {
  return session?.tabId ?? null;
}
