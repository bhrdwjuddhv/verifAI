/**
 * The toolbar badge, per tab.
 *
 * Called directly from the scan store rather than off the broadcast: a service worker does
 * not receive its own runtime.sendMessage, so a listener here would never fire.
 */

import type { ScanState } from '../shared/protocol';
import { NEUTRAL } from '../shared/format';

export async function paintAction(state: ScanState): Promise<void> {
  const tabId = state.tabId;
  if (tabId === undefined) return;

  try {
    if (state.phase === 'done' && state.result) {
      // The fused P(fake) — the number the verdict is actually based on.
      const fake = state.result.score === null ? null : Math.round(100 - state.result.score);
      await chrome.action.setBadgeText({ tabId, text: fake === null ? '?' : String(fake) });
      await chrome.action.setBadgeBackgroundColor({
        tabId,
        color: state.result.verdict.ringColor || NEUTRAL,
      });
      return;
    }

    const needsAttention =
      state.phase === 'error' || state.phase === 'needs-permission' || state.phase === 'needs-consent';
    await chrome.action.setBadgeText({ tabId, text: needsAttention ? '!' : '…' });
    await chrome.action.setBadgeBackgroundColor({ tabId, color: NEUTRAL });
  } catch {
    // The tab closed mid-scan. Nothing to paint, nothing to report.
  }
}
