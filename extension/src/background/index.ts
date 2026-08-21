/**
 * Service worker: menus, message routing, toolbar badge. No inference, no DOM.
 *
 * Everything here must survive being killed at any moment — the worker is evicted after ~30s
 * idle — so nothing is kept in a module variable that is not also written to storage.
 */

import type { ContentEvent, UiRequest, UiResponse } from '../shared/protocol';
import { sendToTab } from '../shared/protocol';
import { cacheStats, clearCache } from './cache';
import { dismiss, explain, listScans, rescanOnServer, retry, startScan } from './scans';
import { probeCapabilities } from '#host';
import { autoAllowed, syncContentScripts } from './sites';
import { REFRESH_ALARM, REFRESH_MINUTES, checkDrift, refreshManifest } from './drift';

declare const __BUILD__: string;

const MENU_ID = 'verifai-scan';

chrome.runtime.onInstalled.addListener(async (details) => {
  await installMenus();
  await syncContentScripts();
  await scheduleManifestRefresh();
  if (details.reason === 'install') {
    await chrome.tabs.create({ url: chrome.runtime.getURL('onboarding.html') });
  }
});

// Site access can be granted or revoked from the options page or from Chrome's own UI, and
// the registered content script has to follow either one. Registering only what is currently
// granted is what keeps "what the options page says" and "what the extension can reach" the
// same sentence.
chrome.permissions.onAdded.addListener(() => void syncContentScripts());
chrome.permissions.onRemoved.addListener(() => void syncContentScripts());

chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== 'sync') return;
  if (!('autoScan' in changes) && !('mode' in changes)) return;
  void syncContentScripts();
  void retellTabs();
});

/** Tells every open tab whether auto-scan applies to it now, without waiting for a reload. */
async function retellTabs(): Promise<void> {
  const tabs = await chrome.tabs.query({});
  for (const tab of tabs) {
    if (tab.id === undefined) continue;
    const verdict = await autoAllowed(tab.url);
    await sendToTab(tab.id, { type: 'content:auto', enabled: verdict.ok, reason: verdict.reason });
  }
}

// Menus live in the browser profile, not the worker, but re-registering on startup keeps a
// reloaded unpacked build from ending up with none.
chrome.runtime.onStartup.addListener(() => {
  void installMenus();
  void syncContentScripts();
  void scheduleManifestRefresh();
});

/**
 * A daily read of /api/extension/manifest.
 *
 * On-device mode scores with a bundled copy of the server's thresholds and fusion weights.
 * Without this, retuning them server-side leaves every installed extension blending by the
 * old numbers with nothing on screen to say so.
 */
async function scheduleManifestRefresh(): Promise<void> {
  await chrome.alarms.create(REFRESH_ALARM, { periodInMinutes: REFRESH_MINUTES, when: Date.now() + 5000 });
}

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === REFRESH_ALARM) void refreshManifest();
});

async function installMenus(): Promise<void> {
  await chrome.contextMenus.removeAll();
  chrome.contextMenus.create({
    id: MENU_ID,
    title: 'Verify with VerifAI',
    contexts: ['image', 'video'],
  });
}

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== MENU_ID || !info.srcUrl) return;
  await startScan({
    mediaUrl: info.srcUrl,
    kind: info.mediaType === 'video' ? 'video' : 'image',
    pageUrl: info.pageUrl,
    tabId: tab?.id,
    frameId: info.frameId,
  });
});

chrome.runtime.onMessage.addListener((message: UiRequest | ContentEvent, sender, sendResponse) => {
  if (message?.type?.startsWith('auto:')) {
    handleAuto(message as ContentEvent, sender)
      .then(sendResponse)
      .catch(() => sendResponse({ ok: false }));
    return true;
  }
  handle(message as UiRequest)
    .then(sendResponse)
    .catch((err) => sendResponse({ type: 'error', message: String(err?.message ?? err) }));
  return true; // keeps the channel open for the async reply
});

/**
 * Auto-scan traffic from a content script.
 *
 * Every branch re-checks `autoAllowed`, which is not redundant: a content script registered
 * for a granted site keeps running after auto-scan is switched off or the grant is revoked,
 * and the rule has to hold at the moment of the scan rather than at the moment of
 * registration.
 */
/**
 * Scans the video playing in the active tab.
 *
 * The context menu cannot reach a YouTube player — the page overrides right-click with its own
 * menu, so Chrome never offers ours. Clicking the toolbar icon grants activeTab, which is
 * enough to ask the page for a frame directly.
 */
async function scanActiveTab(): Promise<UiResponse> {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return { type: 'error', message: 'No active tab.' };

  try {
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ['content.js'] });
  } catch {
    return { type: 'error', message: 'This page does not allow extensions to read it.' };
  }

  let frame: { b64?: string; mime?: string; t?: number; sourceUrl?: string; error?: string } | undefined;
  try {
    frame = await chrome.tabs.sendMessage(tab.id, { type: 'content:capture' });
  } catch {
    return { type: 'error', message: 'The page did not answer. Try reloading it.' };
  }

  if (!frame || frame.error || !frame.b64) {
    return { type: 'error', message: frame?.error ?? 'Nothing playable was found on this page.' };
  }

  await startScan({
    mediaUrl: frame.sourceUrl || tab.url || 'video',
    kind: 'image', // a frame is scored as a still
    pageUrl: tab.url,
    tabId: tab.id,
    inline: { b64: frame.b64, mime: frame.mime ?? 'image/png', filename: `frame-${frame.t ?? 0}s.png` },
  });
  return { type: 'ok' };
}

async function handleAuto(message: ContentEvent, sender: chrome.runtime.MessageSender): Promise<unknown> {
  const tabId = sender.tab?.id;
  const verdict = await autoAllowed(sender.tab?.url);

  if (message.type === 'auto:hello') return { enabled: verdict.ok, reason: verdict.reason };
  if (!verdict.ok || tabId === undefined) return { ok: false, reason: verdict.reason };

  if (message.type === 'auto:image') {
    await startScan({ mediaUrl: message.url, kind: 'image', tabId, auto: true });
    return { ok: true };
  }

  if (message.type === 'auto:frame') {
    await startScan({
      mediaUrl: message.url,
      kind: 'image', // a frame is scored as a still, not handed to the video route
      tabId,
      auto: true,
      inline: { b64: message.b64, mime: message.mime, filename: `frame-${message.t}s.png` },
    });
    return { ok: true };
  }

  return { ok: false };
}

async function handle(message: UiRequest): Promise<UiResponse> {
  switch (message.type) {
    case 'ui:get-state':
      return { type: 'state', scans: await listScans() };
    case 'ui:retry':
      await retry(message.id);
      return { type: 'ok' };
    case 'ui:explain':
      await explain(message.id);
      return { type: 'ok' };
    case 'ui:rescan-server':
      await rescanOnServer(message.id);
      return { type: 'ok' };
    case 'ui:dismiss':
      await dismiss(message.id);
      return { type: 'ok' };
    case 'ui:clear-cache':
      await clearCache();
      return { type: 'ok' };
    case 'ui:cache-stats': {
      const stats = await cacheStats();
      return { type: 'cache-stats', ...stats };
    }
    case 'ui:drift': {
      await refreshManifest();
      return (await checkDrift()) as unknown as UiResponse;
    }
    case 'ui:scan-tab':
      return await scanActiveTab();
    case 'ui:auto-status': {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      const verdict = await autoAllowed(tab?.url);
      return { type: 'ok', ...verdict, url: tab?.url } as unknown as UiResponse;
    }
    case 'ui:build':
      return { type: 'ok', build: __BUILD__ } as unknown as UiResponse;
    case 'ui:probe':
      // Returned as-is rather than wrapped: the options page renders the raw capability
      // record, and inventing a schema for it here would only add a place to drift.
      return (await probeCapabilities()) as unknown as UiResponse;
    default:
      // Naming the type matters: an unrecognised message almost always means the worker is
      // older than the page that sent it, and "unknown request" alone says none of that.
      return {
        type: 'error',
        message: `this build of the background worker does not handle "${(message as { type?: string }).type}" — reload the extension at chrome://extensions`,
      };
  }
}
