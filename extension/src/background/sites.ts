/**
 * Site access for auto-scan, and the content script that goes with it.
 *
 * Nothing is registered at install time. A content script is registered only for origins the
 * user has explicitly granted, and unregistered the moment the grant is revoked or auto-scan
 * is switched off — so the extension's site access always matches what the options page says
 * it is, with no stale registration surviving in the background.
 */

import { getSettings } from '../shared/settings';
import { AUTO_SITES, matchesHost, type AutoSite } from '../shared/sites';
import { CALL_ORIGINS } from '../shared/callsites';

export { AUTO_SITES };
export type { AutoSite };

const SCRIPT_ID = 'verifai-auto';

export async function grantedSites(): Promise<AutoSite[]> {
  const granted: AutoSite[] = [];
  for (const site of AUTO_SITES) {
    if (await chrome.permissions.contains({ origins: site.origins })) granted.push(site);
  }
  return granted;
}

/**
 * Brings the registered content script in line with the settings and the grants.
 *
 * Registration is all-or-nothing per call rather than incremental: computing the diff would
 * be a second source of truth about what is registered, and the two would drift.
 */
/**
 * Call origins the user has granted.
 *
 * Separate from auto-scan on purpose: Live Guard needs the content script to draw its overlay,
 * but it has nothing to do with scanning feeds and must not be switched off with them. Tying
 * the two together is what left Live Guard unreachable — the overlay could never load on a
 * call, because the only content script registration was gated behind `autoScan`.
 */
export async function grantedCallOrigins(): Promise<string[]> {
  const granted: string[] = [];
  for (const origin of CALL_ORIGINS) {
    if (await chrome.permissions.contains({ origins: [origin] })) granted.push(origin);
  }
  return granted;
}

export async function syncContentScripts(): Promise<void> {
  const settings = await getSettings();
  const sites = await grantedSites();
  const feeds = settings.autoScan ? sites.flatMap((site) => site.origins) : [];
  // A granted call origin always gets the script, whatever auto-scan is set to.
  const matches = [...new Set([...feeds, ...(await grantedCallOrigins())])];

  const existing = await chrome.scripting.getRegisteredContentScripts({ ids: [SCRIPT_ID] }).catch(() => []);

  if (!matches.length) {
    if (existing.length) await chrome.scripting.unregisterContentScripts({ ids: [SCRIPT_ID] });
    return;
  }

  const definition: chrome.scripting.RegisteredContentScript = {
    id: SCRIPT_ID,
    js: ['content.js'],
    matches,
    runAt: 'document_idle',
    // The feed is in the top document; scanning every ad iframe as well would be noise.
    allFrames: false,
    persistAcrossSessions: true,
  };

  if (existing.length) await chrome.scripting.updateContentScripts([definition]);
  else await chrome.scripting.registerContentScripts([definition]);

  await adoptOpenTabs(matches, feeds);
}

/**
 * Registration only affects future navigations.
 *
 * Without this, switching auto-scan on while sitting on a Shorts tab does nothing at all
 * until the page is reloaded — which reads as "the feature is broken", because from the
 * outside it is indistinguishable.
 */
async function adoptOpenTabs(matches: string[], feeds: string[]): Promise<void> {
  let tabs: chrome.tabs.Tab[] = [];
  try {
    tabs = await chrome.tabs.query({ url: matches });
  } catch {
    return;
  }

  const isFeed = async (url?: string) => {
    if (!url || !feeds.length) return false;
    try {
      return (await chrome.tabs.query({ url: feeds })).some((t) => t.url === url);
    } catch {
      return false;
    }
  };

  for (const tab of tabs) {
    if (tab.id === undefined) continue;
    try {
      await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ['content.js'] });
      // A call tab gets the script for the Live Guard overlay, but must NOT start watching a
      // feed: it would sample frames of the call itself, which the worker then rejects because
      // the origin is not an auto-scan site. Work done to be thrown away, on a video call.
      await chrome.tabs.sendMessage(tab.id, { type: 'content:auto', enabled: await isFeed(tab.url) });
    } catch {
      // A tab that is still loading, or discarded. It will pick the script up on its own.
    }
  }
}

/** Whether auto-scan should actually run in a given tab. */
export async function autoAllowed(url: string | undefined): Promise<{ ok: boolean; reason?: string }> {
  const settings = await getSettings();
  if (!settings.autoScan) return { ok: false, reason: 'auto-scan is off' };
  // Auto-scan is on-device by construction: it never passes a `source`, and a scan without one
  // runs locally. There is no setting that can turn a feed into an upload.
  if (!url) return { ok: false, reason: 'no page URL' };

  for (const site of await grantedSites()) {
    if (await chrome.permissions.contains({ origins: site.origins })) {
      const host = new URL(url).hostname;
      if (site.origins.some((pattern) => matchesHost(pattern, host))) return { ok: true };
    }
  }
  return { ok: false, reason: 'this site is not granted' };
}
