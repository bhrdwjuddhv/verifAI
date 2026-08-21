/**
 * Every message that crosses a context boundary, in one discriminated union.
 *
 * Two things about Chrome messaging drive the design here:
 *   - it JSON-serialises, so nothing in a payload may be an ArrayBuffer, Blob or Map;
 *   - the service worker is the single source of truth, because it is the only context that
 *     survives a popup closing mid-scan.
 */

import type { ScanResult, FailureKind } from './scan-types';

export type ScanPhase =
  | 'queued'
  | 'reading' // pulling the bytes out of the page or the network
  | 'scanning' // in flight to the model
  | 'done'
  | 'error'
  | 'needs-consent'
  | 'needs-permission';

export interface ScanState {
  id: string;
  createdAt: number;
  /** The media we were asked about. Shown truncated; never sent anywhere but the model. */
  mediaUrl: string;
  pageUrl?: string;
  kind: 'image' | 'video';
  source: 'server' | 'device';
  phase: ScanPhase;
  /** Places ahead of this scan in the queue, while phase === 'queued'. */
  queuePosition?: number;
  /** Set with phase 'needs-permission': the match pattern the user has to grant. */
  needsOrigin?: string;
  /** Set once the bytes are known, so the UI can show what is being scanned. */
  bytes?: number;
  cached?: boolean;
  /** Started by auto-scan rather than by a right-click. */
  auto?: boolean;
  /**
   * True when the server could do better than this result — a video that only got one frame
   * locally. The popup turns it into a button rather than making the user know to switch.
   */
  offerServer?: boolean;
  /** Wall-clock milliseconds of on-device inference. Absent for deep scans. */
  ms?: number;
  result?: ScanResult;
  error?: string;
  errorKind?: FailureKind;
  tabId?: number;
}

/** Live Guard messages. The worker owns the session; the overlay and popup only render it. */
export type GuardRequest =
  | { type: 'guard:start'; tabId: number }
  | { type: 'guard:stop' }
  | { type: 'guard:get-status' };

/** UI (popup / options / onboarding) -> service worker. */
export type UiRequest =
  | { type: 'ui:get-state' }
  /**
   * Also the "I just granted the missing permission" signal. The grant itself has to happen
   * in the popup: chrome.permissions.request needs a user gesture, and a message handler in
   * the service worker does not inherit one.
   */
  | { type: 'ui:retry'; id: string }
  | { type: 'ui:dismiss'; id: string }
  /**
   * Re-runs an on-device scan with the occlusion heatmap. Separate from 'ui:retry' because
   * it is a deliberate, expensive action, not a recovery.
   */
  | { type: 'ui:explain'; id: string }
  /** Re-run one scan against the server, whatever the default mode is. */
  | { type: 'ui:rescan-server'; id: string }
  | { type: 'ui:clear-cache' }
  | { type: 'ui:cache-stats' }
  /** Runs the offscreen capability probe — spikes S1 and S2 in PLAN.md, on this machine. */
  | { type: 'ui:probe' }
  /** Re-reads the server's manifest and reports how far this build has drifted from it. */
  | { type: 'ui:drift' }
  /** Scan the video playing in the active tab — the entry point YouTube's own menu blocks. */
  | { type: 'ui:scan-tab' }
  /** Why auto-scan is or is not running in the active tab. */
  | { type: 'ui:auto-status' }
  /** The service worker's build stamp, for detecting a worker left behind by a reload. */
  | { type: 'ui:build' };

export type UiResponse =
  | { type: 'state'; scans: ScanState[] }
  | { type: 'cache-stats'; entries: number; heatmaps: number; approxBytes: number }
  | { type: 'ok' }
  | { type: 'error'; message: string };

/** Service worker -> anyone listening (popup, injected content script). */
export type Broadcast = { type: 'scan:update'; state: ScanState };

/** Service worker -> injected content script. */
export type ContentRequest =
  | { type: 'content:badge'; state: ScanState }
  | { type: 'content:ping' }
  /** Turns feed watching on or off in this tab. `reason` is shown when it is off. */
  | { type: 'content:auto'; enabled: boolean; reason?: string }
  /** Grab one frame from whatever is playing, now. The manual path for Shorts and Reels. */
  | { type: 'content:capture' };

/**
 * Content script -> service worker.
 *
 * Frames arrive as base64 because chrome messaging is JSON; images arrive as a URL because
 * the worker is the only context that can fetch cross-origin without the page's CORS.
 */
export type ContentEvent =
  | { type: 'auto:hello' }
  | { type: 'auto:image'; url: string }
  | { type: 'auto:frame'; url: string; b64: string; mime: string; dedupeKey: string; t: number };

export type AnyMessage = UiRequest | Broadcast | ContentRequest | ContentEvent;

/**
 * sendMessage rejects when nothing is listening — a closed popup, a tab without the content
 * script. That is the normal case, not an error worth surfacing.
 */
export async function broadcast(message: Broadcast): Promise<void> {
  try {
    await chrome.runtime.sendMessage(message);
  } catch {
    /* no receiver */
  }
}

export async function sendToTab(tabId: number, message: ContentRequest): Promise<boolean> {
  try {
    await chrome.tabs.sendMessage(tabId, message);
    return true;
  } catch {
    return false;
  }
}
