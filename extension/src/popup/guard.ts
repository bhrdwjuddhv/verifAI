/**
 * The Live Guard panel: the only way a user can start call monitoring.
 *
 * The engine behind it — capture, windowing, VAD, the v2 chain, the parity gate, provenance,
 * hysteresis — already existed and was tested. What did not exist was any way to reach it: no
 * button anywhere sent `guard:start`, so a fully working pipeline was unreachable from the UI.
 * This file is that ignition.
 *
 * Two things must happen here rather than in the service worker:
 *   * `chrome.permissions.request` needs a user gesture, and a popup click is one. The worker
 *     has no gesture and the request would be rejected outright.
 *   * The status poll only runs while the popup is open, so a closed popup costs nothing.
 */

import { siteForUrl, isInCall } from '../shared/callsites';
import type { GuardStatus } from '../shared/liveguard';
import { append, clear, h } from '../ui/dom';

const panel = document.getElementById('guard')!;

let timer: ReturnType<typeof setInterval> | null = null;
let activeTab: chrome.tabs.Tab | null = null;

function status(): Promise<GuardStatus | null> {
  return chrome.runtime.sendMessage({ type: 'guard:get-status' }).catch(() => null);
}

/** Trust is 100 - smoothed risk, so the colour follows the same bands as a scan verdict. */
function trustColour(trust: number | null): string {
  if (trust === null) return 'var(--text-faint)';
  if (trust < 30) return '#EF4444';
  if (trust < 70) return '#F59E0B';
  return '#10B981';
}

function windowColour(audio: number | null, speech: boolean): string {
  if (!speech || audio === null) return 'var(--line)';
  return audio > 70 ? '#EF4444' : audio > 30 ? '#F59E0B' : '#10B981';
}

async function start(tabId: number, origin: string): Promise<void> {
  const result = await chrome.runtime.sendMessage({ type: 'guard:start', tabId }).catch(() => null);

  // The worker cannot ask for a permission; it reports what it needs and the popup asks,
  // because only a click carries the user gesture Chrome requires.
  if (result?.reason?.startsWith('needs-permission:')) {
    const wanted = result.reason.slice('needs-permission:'.length) || origin;
    const granted = await chrome.permissions.request({ origins: [wanted] }).catch(() => false);
    if (!granted) {
      return paint(`Monitoring needs access to ${new URL(wanted.replace('*', 'x')).hostname}. Nothing was started.`);
    }
    // The content script that draws the on-page overlay is registered from the grant, and
    // registration only affects future loads — so the open call tab needs it injected now.
    await chrome.runtime.sendMessage({ type: 'guard:sync-sites' }).catch(() => undefined);
    const retry = await chrome.runtime.sendMessage({ type: 'guard:start', tabId }).catch(() => null);
    return paint(retry?.ok ? undefined : retry?.reason);
  }

  paint(result?.ok ? undefined : result?.reason);
}

async function stop(): Promise<void> {
  await chrome.runtime.sendMessage({ type: 'guard:stop' }).catch(() => undefined);
  paint();
}

export async function paint(note?: string): Promise<void> {
  const state = await status();
  const url = activeTab?.url;
  const site = siteForUrl(url);

  // Not a call platform: the panel stays hidden rather than explaining itself on every page.
  if (!site && !state?.active) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  clear(panel);

  const running = state?.active === true;
  const head = h('div', { class: 'guard-head' }, h('h2', { text: 'Live Guard' }));
  if (running) {
    head.append(
      h('span', {
        class: 'trust',
        style: `color:${trustColour(state!.trust)}`,
        text: state!.trust === null ? '—' : `${state!.trust}`,
      })
    );
  }
  panel.append(head);

  panel.append(
    h('div', {
      class: 'site',
      text: running
        ? `${state!.siteLabel} · ${state!.band}`
        : `${site?.label ?? 'Call'} detected on this tab`,
    })
  );

  if (running) {
    // One cell per window. A sustained signal reads as a run of colour, which a single
    // number cannot show.
    const timeline = h('div', { class: 'timeline' });
    for (const w of state!.windows.slice(-40)) {
      timeline.append(h('i', { style: `background:${windowColour(w.audio, w.speech)}`, title: w.note ?? `${w.t.toFixed(0)}s` }));
    }
    if (state!.windows.length) panel.append(timeline);

    panel.append(
      h('div', { class: 'stats' },
        h('span', { text: `${state!.scored} scored` }),
        state!.skipped ? h('span', { text: `${state!.skipped} silent` }) : null,
        h('span', {
          text: state!.audio?.state === 'verified' ? 'voice: on-device' : 'voice: backend',
        })
      )
    );

    if (state!.modelSource) {
      panel.append(h('div', { class: 'site mono', text: state!.modelSource }));
    }
  }

  panel.append(
    h('button', {
      class: running ? '' : 'primary',
      text: running ? 'Stop monitoring' : 'Start monitoring',
      onclick: () => {
        if (running) return void stop();
        if (activeTab?.id !== undefined && site) void start(activeTab.id, site.origins[0]);
      },
      disabled: !running && (!site || !isInCall(url)),
    })
  );

  // Every reason a start would fail, said before the click rather than after it.
  const why =
    note ??
    state?.note ??
    (running
      ? state!.audio?.state === 'verified'
        ? 'Voice is scored on this machine; no call audio is uploaded.'
        : 'Voice windows go to your VerifAI backend. Nothing is sent anywhere else.'
      : site && !isInCall(url)
        ? `Open a call on ${site.label} first — this looks like a lobby or home page.`
        : 'Monitors the voice in this call. Nothing is uploaded unless your settings say so.');

  append(panel, h('p', { class: note || state?.note ? 'why warn' : 'why', text: why }));
}

export async function initGuard(): Promise<void> {
  [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
  await paint();

  // Poll only while open. The worker broadcasts too, but a popup opened mid-call has to catch
  // up on state that was already pushed before it existed.
  timer ??= setInterval(() => void paint(), 1000);
  addEventListener('unload', () => {
    if (timer) clearInterval(timer);
    timer = null;
  });
}

chrome.runtime.onMessage.addListener((message: { type?: string }) => {
  if (message?.type === 'guard:status') void paint();
  return undefined;
});
