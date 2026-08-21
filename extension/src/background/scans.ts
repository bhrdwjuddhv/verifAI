/**
 * The scan lifecycle, and the only place scan state lives.
 *
 * State is mirrored into chrome.storage.session so a result survives the service worker
 * being killed — which happens routinely, and would otherwise blank the popup between
 * starting a scan and reading it.
 *
 * Two paths end up here. Deep scan uploads the bytes and needs consent plus host access;
 * on-device scan needs neither, because nothing leaves the machine. Keeping that distinction
 * in one place is what stops "on-device" from quietly becoming an upload.
 */

import type { ScanState } from '../shared/protocol';
import { broadcast, sendToTab } from '../shared/protocol';
import { getSettings, hasConsented, originPattern } from '../shared/settings';
import { getCached, putCached, sha256Hex } from './cache';
import { postScan } from './api';
import { base64ToBytes, resolveMedia, type MediaBytes } from './media';
import { backOff, cancel, enqueue } from './queue';
import { paintAction } from './action';
import { scoreOnDevice } from '#host';
import { checkDrift, driftNote } from './drift';
import { wasmAllowed } from '../shared/wasm';

declare const __VERSION__: string;

const MAX_HISTORY = 12;

let states: ScanState[] = [];
let hydrated = false;

async function hydrate(): Promise<void> {
  if (hydrated) return;
  const stored = await chrome.storage.session.get('scans');
  states = Array.isArray(stored.scans) ? (stored.scans as ScanState[]) : [];
  hydrated = true;
}

async function persist(): Promise<void> {
  states = states.slice(0, MAX_HISTORY);
  await chrome.storage.session.set({ scans: states });
}

export async function listScans(): Promise<ScanState[]> {
  await hydrate();
  return states;
}

export async function dismiss(id: string): Promise<void> {
  await hydrate();
  cancel(id);
  states = states.filter((s) => s.id !== id);
  await persist();
}

async function put(state: ScanState): Promise<ScanState> {
  await hydrate();
  states = [state, ...states.filter((s) => s.id !== state.id)];
  await persist();
  await broadcast({ type: 'scan:update', state });
  await paintAction(state);
  if (state.tabId !== undefined) await paintBadge(state);
  return state;
}

async function patch(id: string, changes: Partial<ScanState>): Promise<ScanState | null> {
  await hydrate();
  const current = states.find((s) => s.id === id);
  if (!current) return null;
  return put({ ...current, ...changes });
}

/**
 * The badge is drawn by a content script injected on demand — there is no declared content
 * script, so the extension holds no site access until the moment a user asks for a scan.
 */
async function paintBadge(state: ScanState): Promise<void> {
  const tabId = state.tabId!;
  if (!(await sendToTab(tabId, { type: 'content:ping' }))) {
    try {
      await chrome.scripting.executeScript({ target: { tabId }, files: ['content.js'] });
    } catch {
      return; // no access to this tab (chrome:// page, or activeTab already expired)
    }
  }
  await sendToTab(tabId, { type: 'content:badge', state });
}

export interface StartOptions {
  mediaUrl: string;
  kind: 'image' | 'video';
  pageUrl?: string;
  tabId?: number;
  frameId?: number;
  /** On-device only: also compute the occlusion heatmap. */
  explain?: boolean;
  /**
   * Bytes the caller already holds — a video frame grabbed in the page. Skips the fetch
   * entirely, which is what makes frame scanning need no CDN permission.
   */
  inline?: { b64: string; mime: string; filename: string };
  /** Started by auto-scan. Subject to the pending cap below. */
  auto?: boolean;
}

/**
 * How many auto scans may be in flight at once.
 *
 * A feed produces candidates faster than anything can score them, and an unbounded queue
 * would still be working through what you scrolled past a minute ago. Dropping is correct:
 * the element is re-offered when it next enters the viewport.
 */
const MAX_AUTO_PENDING = 6;

export async function startScan(options: StartOptions): Promise<ScanState | null> {
  const id = crypto.randomUUID();
  const settings = await getSettings();
  const onDevice = settings.mode === 'device';

  if (options.auto) {
    await hydrate();
    const busy = states.filter(
      (s) => s.auto && (s.phase === 'queued' || s.phase === 'reading' || s.phase === 'scanning')
    ).length;
    if (busy >= MAX_AUTO_PENDING) return null;
  }

  const state = await put({
    id,
    createdAt: Date.now(),
    mediaUrl: options.mediaUrl,
    pageUrl: options.pageUrl,
    kind: options.kind,
    source: onDevice ? 'device' : 'server',
    phase: 'queued',
    tabId: options.tabId,
    auto: options.auto,
  });

  if (onDevice) {
    // Video needs frame sampling, which is Phase 4. Saying so beats scoring the first bytes
    // of an MP4 as if they were a picture.
    if (options.kind === 'video') {
      return (await patch(id, {
        phase: 'error',
        errorKind: 'not-implemented',
        error: 'On-device video scanning is not built yet. Switch to deep scan for video.',
      }))!;
    }
    // No consent gate and no host permission for the server: nothing is uploaded.
    enqueue({
      id,
      costsToken: false,
      onPosition: (position) => void patch(id, { phase: 'queued', queuePosition: position }),
      run: () => runScan(id, options, settings.serverUrl, true),
    });
    return state;
  }

  if (!hasConsented(settings)) {
    await chrome.tabs.create({ url: chrome.runtime.getURL('onboarding.html') });
    return (await patch(id, { phase: 'needs-consent' }))!;
  }

  const serverOrigin = originPattern(settings.serverUrl);
  if (!serverOrigin) {
    return (await patch(id, {
      phase: 'error',
      errorKind: 'internal',
      error: `"${settings.serverUrl}" is not a valid server URL. Fix it in options.`,
    }))!;
  }
  if (!(await chrome.permissions.contains({ origins: [serverOrigin] }))) {
    return (await patch(id, { phase: 'needs-permission', needsOrigin: serverOrigin }))!;
  }

  enqueue({
    id,
    costsToken: true,
    onPosition: (position) => void patch(id, { phase: 'queued', queuePosition: position }),
    run: () => runScan(id, options, settings.serverUrl, false),
  });

  return state;
}

export async function retry(id: string): Promise<void> {
  await rerun(id, false);
}

/**
 * Re-scan with the heatmap on.
 *
 * The bytes are fetched again rather than kept in memory: the service worker is evicted
 * between the scan and the click often enough that holding them would be a cache that is
 * usually empty, and re-reading costs a fraction of the 26 forward passes that follow.
 */
export async function explain(id: string): Promise<void> {
  await rerun(id, true);
}

async function rerun(id: string, withHeatmap: boolean): Promise<void> {
  await hydrate();
  const state = states.find((s) => s.id === id);
  if (!state) return;
  await startScan({
    mediaUrl: state.mediaUrl,
    kind: state.kind,
    pageUrl: state.pageUrl,
    tabId: state.tabId,
    explain: withHeatmap,
  });
  await dismiss(id);
}

async function runScan(
  id: string,
  options: StartOptions,
  serverUrl: string,
  onDevice: boolean
): Promise<void> {
  await patch(id, { phase: 'reading', queuePosition: undefined });

  // A frame captured in the page needs no fetch and no permission — it is already here.
  const media = options.inline
    ? {
        ok: true as const,
        media: {
          bytes: base64ToBytes(options.inline.b64),
          mime: options.inline.mime,
          filename: options.inline.filename,
        },
      }
    : await resolveMedia(options.mediaUrl, options.kind, options.tabId, options.frameId);

  if (!media.ok) {
    if (media.kind === 'permission') {
      await patch(id, { phase: 'needs-permission', needsOrigin: media.origin });
    } else {
      await patch(id, { phase: 'error', errorKind: 'network', error: media.message });
    }
    return;
  }

  // Namespaced by source: the two paths can have different detectors loaded, so a device
  // verdict must never be served for a deep-scan request or the other way round. The
  // explained variant is its own entry, or asking for a heatmap would return the cached
  // result that has none.
  const suffix = options.explain ? ':x' : '';
  const hash = `${onDevice ? 'd' : 's'}:${await sha256Hex(media.media.bytes)}${suffix}`;
  const cached = await getCached(hash);
  if (cached) {
    await patch(id, {
      phase: 'done',
      cached: true,
      bytes: media.media.bytes.byteLength,
      result: cached,
    });
    return;
  }

  await patch(id, { phase: 'scanning', bytes: media.media.bytes.byteLength });

  if (onDevice) await runOnDevice(id, hash, media.media, options.explain === true);
  else await runOnServer(id, hash, media.media, serverUrl);
}

async function runOnDevice(
  id: string,
  hash: string,
  media: MediaBytes,
  withHeatmap: boolean
): Promise<void> {
  // Checked before the attempt: when the CSP forbids WebAssembly every model fails to load,
  // and ONNX Runtime reports that as "no available backend found" — which reads as a missing
  // model rather than a manifest that was never reloaded.
  const wasm = wasmAllowed();
  if (!wasm.ok) {
    await patch(id, {
      phase: 'error',
      errorKind: 'internal',
      error: `On-device scanning needs WebAssembly, and it is ${wasm.reason}`,
    });
    return;
  }

  try {
    const { result, ms } = await scoreOnDevice({
      b64: bytesToBase64(media.bytes),
      filename: media.filename,
      mime: media.mime,
      explain: withHeatmap,
    });

    // Cached with the note, so a stale-settings warning survives a cache hit.
    const note = driftNote(await checkDrift());
    if (note) {
      result.notes = [...result.notes, note];
      result.reasons = [...result.reasons, note];
    }

    await putCached(hash, result);
    await patch(id, { phase: 'done', cached: false, result, ms });
  } catch (err) {
    // 'internal', not 'unavailable': nothing was uploaded and no service was contacted, so
    // telling the user their model service is down would point them at the wrong problem.
    await patch(id, {
      phase: 'error',
      errorKind: 'internal',
      error: `On-device scan failed: ${(err as Error)?.message ?? err}`,
    });
  }
}

async function runOnServer(
  id: string,
  hash: string,
  media: MediaBytes,
  serverUrl: string
): Promise<void> {
  let outcome = await postScan(serverUrl, media.bytes, media.filename, media.mime, __VERSION__);

  // One retry for a 429, because the queue's own budget is deliberately below the server's
  // and a single burst from another tab should not lose the scan.
  if (!outcome.ok && outcome.kind === 'rate-limited') {
    const wait = outcome.retryAfterSeconds ?? 60;
    backOff(wait);
    await patch(id, { phase: 'queued', error: `Rate-limited; retrying in ${wait}s.` });
    await new Promise((resolve) => setTimeout(resolve, wait * 1000));
    await patch(id, { phase: 'scanning', error: undefined });
    outcome = await postScan(serverUrl, media.bytes, media.filename, media.mime, __VERSION__);
  }

  if (!outcome.ok) {
    await patch(id, { phase: 'error', errorKind: outcome.kind, error: outcome.message });
    return;
  }

  await putCached(hash, outcome.result);
  await patch(id, { phase: 'done', cached: false, result: outcome.result });
}

/** Chunked so a multi-megabyte image does not blow the argument limit of fromCharCode. */
function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}
