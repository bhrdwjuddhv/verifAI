/**
 * The popup: what the newest scan found, and what to do when it could not run.
 *
 * The service worker owns scan state; this page reads it on open and then follows the
 * `scan:update` broadcast. It never starts work of its own, so closing the popup mid-scan
 * loses nothing.
 */

import '../ui/ui.css';
import './popup.css';

import type { Broadcast, ScanState, UiRequest, UiResponse } from '../shared/protocol';
import type { ScanResult } from '../shared/scan-types';
import { detectorRows, fakePercent, fmtBytes, shortUrl, NEUTRAL } from '../shared/format';
import {
  CONSENT_VERSION,
  DEFAULT_SERVER_URL,
  getSettings,
  originPattern,
  setSettings,
} from '../shared/settings';
import { append, clear, h } from '../ui/dom';

const list = document.getElementById('scans')!;
const footer = document.getElementById('footer')!;
const tabActions = document.getElementById('tab-actions')!;

declare const __BUILD__: string;

let scans: ScanState[] = [];
let serverUrl = DEFAULT_SERVER_URL;

async function ask(request: UiRequest): Promise<UiResponse> {
  return (await chrome.runtime.sendMessage(request)) as UiResponse;
}

chrome.runtime.onMessage.addListener((message: Broadcast) => {
  if (message?.type !== 'scan:update') return;
  scans = [message.state, ...scans.filter((s) => s.id !== message.state.id)];
  paint();
});

/**
 * Catches the commonest confusion in unpacked development: Chrome keeps a service worker from
 * an earlier build while serving the new popup, so features the UI offers simply are not
 * there. The stamp is baked into every bundle of a build, so a mismatch is conclusive.
 */
async function checkForStaleWorker(): Promise<void> {
  const reply = (await ask({ type: 'ui:build' })) as unknown as { type?: string; build?: string };
  if (reply?.build === __BUILD__) return;

  // A worker that does not recognise the message at all is certainly older than this popup.
  // Anything else that simply has no stamp cannot be judged — the preview harness, for one —
  // and a warning there would be noise rather than a signal.
  if (reply?.type !== 'error' && !reply?.build) return;

  const banner = h(
    'div',
    { class: 'card stale' },
    h('b', { text: 'Reload the extension' }),
    h('p', {
      class: 'muted',
      text: 'The background worker is running an older build than this popup, so newer actions will fail. Open chrome://extensions and press reload on VerifAI.',
    })
  );
  list.prepend(banner);
}

async function boot(): Promise<void> {
  const response = await ask({ type: 'ui:get-state' });
  if (response.type === 'state') scans = response.scans;
  paint();
  await paintFooter();
  await paintTabActions();
  await checkForStaleWorker();
}

/**
 * The entry point for video.
 *
 * A right-click never reaches a YouTube player — the page replaces Chrome's menu with its own,
 * so our item is not there to click. Clicking the toolbar icon grants activeTab, which is
 * enough to ask the page for the current frame. The same button covers Reels, and any other
 * player that swallows the context menu.
 *
 * Below it, the honest reason auto-scan is or is not watching this tab. Silence was the worst
 * possible answer to "why is nothing happening".
 */
async function paintTabActions(): Promise<void> {
  clear(tabActions);

  const scanButton = h('button', {
    text: 'Scan the video playing here',
    onclick: async () => {
      scanButton.disabled = true;
      scanButton.textContent = 'Grabbing a frame…';
      const result = await ask({ type: 'ui:scan-tab' });
      if (result.type === 'error') {
        scanButton.disabled = false;
        scanButton.textContent = 'Scan the video playing here';
        clear(tabActions);
        tabActions.append(scanButton, h('span', { class: 'why', text: result.message }));
        return;
      }
      window.close(); // the badge lands on the page; the popup has nothing left to show
    },
  }) as HTMLButtonElement;

  tabActions.append(scanButton);

  const status = (await ask({ type: 'ui:auto-status' })) as unknown as {
    ok?: boolean;
    reason?: string;
    url?: string;
  };

  if (status?.ok) {
    tabActions.append(h('span', { class: 'why', text: 'Auto-scan is watching this page.' }));
    return;
  }

  const why = h('span', { class: 'why' });
  why.append(document.createTextNode(`Auto-scan is not running here: ${status?.reason ?? 'unavailable'}. `));
  why.append(
    h('a', {
      text: 'Open options',
      role: 'button',
      tabindex: '0',
      onclick: () => chrome.tabs.create({ url: chrome.runtime.getURL('options.html') }),
    })
  );
  tabActions.append(why);
}

function paint(): void {
  clear(list);
  if (!scans.length) {
    list.append(emptyState());
    return;
  }
  const [newest, ...rest] = scans;
  list.append(detailCard(newest));
  if (rest.length) {
    list.append(h('h2', { class: 'section-title', text: 'Earlier' }));
    for (const state of rest) list.append(compactRow(state));
  }
}

/**
 * A status line, not a control.
 *
 * This was a mode switch, which put the same binary question in front of the user on every
 * single popup open — the exact decision the redesign exists to stop asking. On-device is what
 * the extension does; a deep scan is offered per scan where it is the better answer. All the
 * footer owes anyone is a plain statement of what just happened, and where a file would go if
 * they ever said yes to one.
 */
async function paintFooter(): Promise<void> {
  const settings = await getSettings();
  serverUrl = settings.serverUrl;
  clear(footer);

  footer.append(
    h('span', { text: 'Scored on this device — nothing is uploaded' }),
    h('span', { class: 'mono src muted', title: settings.serverUrl, text: hostOf(settings.serverUrl) })
  );
}

function emptyState(): HTMLElement {
  return h(
    'div',
    { class: 'card empty' },
    h('p', { text: 'Right-click any image or video and choose "Verify with VerifAI".' }),
    h('p', {
      class: 'muted',
      text: 'A verdict says what the detectors found. "Real" means nothing was detected — not that the file is authentic.',
    }),
    h('p', {
      class: 'muted',
      text: 'The switch below decides whether the file leaves this machine.',
    })
  );
}

// ---------------------------------------------------------------------------------------

function detailCard(state: ScanState): HTMLElement {
  const card = h('div', { class: 'card scan' });
  card.append(header(state));

  switch (state.phase) {
    case 'done':
      card.append(...resultBody(state, state.result!));
      break;
    case 'needs-consent':
      card.append(...consentBody(state));
      break;
    case 'needs-permission':
      card.append(...permissionBody(state));
      break;
    case 'error':
      append(
        card,
        h('p', { class: 'error-text', text: state.error ?? 'The scan failed.' }),
        state.errorKind === 'unavailable' && state.source === 'server'
          ? h('p', { class: 'muted', text: 'The app answered, but its model service did not. No verdict is better than a guessed one.' })
          : null,
        state.source === 'device' && state.phase === 'error' && !state.offerServer
          ? h('p', { class: 'muted', text: 'Nothing was uploaded.' })
          : null,
        state.offerServer ? serverOffer(state, 'Scan every frame on the server') : null,
        h('div', { class: 'actions' },
          h('button', { text: 'Try again', onclick: () => void ask({ type: 'ui:retry', id: state.id }) })
        )
      );
      break;
    default:
      card.append(progressBody(state));
  }

  return card;
}

function header(state: ScanState): HTMLElement {
  const colour = state.result?.verdict.ringColor ?? NEUTRAL;
  const fake = state.result ? fakePercent(state.result) : null;

  return h(
    'div',
    { class: 'scan-head' },
    h(
      'div',
      { class: 'row' },
      h('span', { class: 'dot', style: `background:${colour}` }),
      h('h2', { text: state.result?.verdict.label ?? phaseLabel(state) })
    ),
    fake === null
      ? null
      : h('div', { class: 'score mono', style: `color:${colour}` }, `${Math.round(fake)}%`),
    h('button', {
      class: 'ghost close',
      text: '×',
      title: 'Dismiss',
      onclick: async () => {
        await ask({ type: 'ui:dismiss', id: state.id });
        scans = scans.filter((s) => s.id !== state.id);
        paint();
      },
    })
  );
}

function phaseLabel(state: ScanState): string {
  switch (state.phase) {
    case 'queued':
      return state.queuePosition ? `Queued · ${state.queuePosition} ahead` : 'Queued';
    case 'reading':
      return 'Reading the file';
    case 'scanning':
      return 'Scanning';
    case 'needs-consent':
      return 'Consent needed';
    case 'needs-permission':
      return 'Permission needed';
    case 'error':
      return 'Could not scan';
    default:
      return 'Scan';
  }
}

function progressBody(state: ScanState): HTMLElement {
  return h(
    'div',
    { class: 'stack gap' },
    h('p', { class: 'soft mono src', text: shortUrl(state.mediaUrl) }),
    h('div', { class: 'bar' }, h('i', { class: 'indeterminate', style: `background:${NEUTRAL}` })),
    state.bytes ? h('p', { class: 'muted', text: `${fmtBytes(state.bytes)} read from the page` }) : null
  );
}

/**
 * The upload confirmation, asked at the moment of upload.
 *
 * It used to live on the setup screen, which meant agreeing to something abstract before
 * scanning anything. Here it names the destination, and one click covers both the consent and
 * the host permission — which has to be requested from a page with a live gesture anyway.
 */
function consentBody(state: ScanState): HTMLElement[] {
  const host = hostOf(serverUrl);
  return [
    h('p', {
      class: 'soft',
      text: `A deep scan uploads this file to ${host}, analyses it there and returns the verdict. Nothing else is sent — no page address, no history, no identifiers.`,
    }),
    h('p', { class: 'muted', text: 'On-device scanning uploads nothing, and stays the default.' }),
    h(
      'div',
      { class: 'actions' },
      h('button', {
        class: 'primary',
        text: `Upload and scan`,
        onclick: async () => {
          const origin = originPattern(serverUrl);
          if (origin && !(await chrome.permissions.request({ origins: [origin] }))) return;
          await setSettings({ consentVersion: CONSENT_VERSION });
          await ask({ type: 'ui:rescan-server', id: state.id });
        },
      }),
      h('button', {
        text: 'Not now',
        onclick: async () => {
          await ask({ type: 'ui:dismiss', id: state.id });
          scans = scans.filter((s) => s.id !== state.id);
          paint();
        },
      })
    ),
  ];
}

/** Offered where the server genuinely does something on-device cannot. */
function serverOffer(state: ScanState, label: string): HTMLElement {
  return h(
    'div',
    { class: 'actions' },
    h('button', {
      text: label,
      title: `Uploads this file to ${hostOf(serverUrl)}.`,
      onclick: async (event: Event) => {
        (event.currentTarget as HTMLButtonElement).disabled = true;
        await ask({ type: 'ui:rescan-server', id: state.id });
      },
    })
  );
}

function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

function permissionBody(state: ScanState): HTMLElement[] {
  const origin = state.needsOrigin ?? '';
  const host = origin.replace(/^\w+:\/\//, '').replace(/\/\*$/, '');
  return [
    h('p', {
      class: 'soft',
      text: `VerifAI needs one-time access to ${host} to read this file. Nothing is installed on that site — the permission only lets the extension fetch it.`,
    }),
    h(
      'div',
      { class: 'actions' },
      h('button', {
        class: 'primary',
        text: `Allow ${host}`,
        // permissions.request has to run in a page with a user gesture; the service worker
        // cannot inherit this click, which is why the grant happens here and not there.
        onclick: async () => {
          const granted = await chrome.permissions.request({ origins: [origin] });
          if (granted) await ask({ type: 'ui:retry', id: state.id });
        },
      })
    ),
  ];
}

function resultBody(state: ScanState, result: ScanResult): HTMLElement[] {
  const nodes: HTMLElement[] = [
    h('p', { class: 'summary', text: result.verdict.laymanSummary }),
  ];

  const rows = h('div', { class: 'detectors' });
  for (const row of detectorRows(result)) {
    const value = row.value === null ? null : Math.round(row.value);
    rows.append(
      h(
        'div',
        { class: 'detector', title: row.hint },
        h('span', { class: 'name', text: row.label }),
        h(
          'span',
          { class: `value mono ${row.value === null ? 'muted' : ''}` },
          value === null ? 'did not apply' : `${value}%`
        ),
        h(
          'div',
          { class: 'bar' },
          h('i', {
            style: `width:${value ?? 0}%;background:${row.weight > 0 ? result.verdict.ringColor : NEUTRAL};opacity:${row.weight > 0 ? 1 : 0.45}`,
          })
        ),
        h('span', {
          class: 'weight mono muted',
          text: row.value === null ? '—' : row.weight > 0 ? `w ${row.weight}` : 'no weight',
        })
      )
    );
  }
  nodes.push(rows);

  if (result.heatmap) {
    nodes.push(
      h('img', { class: 'heatmap', src: result.heatmap, alt: 'Where the model looked' }),
      h('p', {
        class: 'muted',
        text: `${state.source === 'device' ? 'Occlusion saliency' : 'Grad-CAM'} over ${
          result.signals.faceDetected ? 'the cropped face — not the whole photo' : 'the region the model scored'
        }.`,
      })
    );
  }

  if (state.source === 'device' && !result.heatmap && result.signals.modelScore !== null) {
    nodes.push(
      h(
        'div',
        { class: 'actions' },
        h('button', {
          text: 'Explain this score',
          title: 'Hides each region in turn and measures the drop in P(fake). 26 forward passes.',
          onclick: (event: Event) => {
            const button = event.currentTarget as HTMLButtonElement;
            button.disabled = true;
            button.textContent = 'Measuring 25 regions…';
            void ask({ type: 'ui:explain', id: state.id });
          },
        })
      )
    );
  }

  if (state.source === 'device') {
    nodes.push(serverOffer(state, 'Second opinion from the server'));
  }


  // The API appends the model's notes to `reasons` as well as sending them separately, so
  // showing both lists whole prints every caveat twice. Measurements here, caveats below.
  const measurements = result.reasons.filter((reason) => !result.notes.includes(reason));
  nodes.push(disclosure('Why this verdict', measurements));
  if (result.notes.length) nodes.push(disclosure('Caveats from the model', result.notes));

  nodes.push(
    h(
      'div',
      { class: 'meta' },
      h('span', { class: 'tag', text: state.cached ? 'cached' : 'fresh' }),
      h('span', {
        class: 'tag',
        title: state.source === 'device' ? 'Scored here; nothing was uploaded.' : 'Uploaded to your VerifAI server.',
        text: state.source === 'device' ? 'on-device' : 'deep scan',
      }),
      state.ms ? h('span', { class: 'mono muted', text: `${state.ms}ms` }) : null,
      h('span', { class: 'mono muted', text: result.modelSource })
    )
  );

  return nodes;
}

function disclosure(title: string, items: string[]): HTMLElement {
  const body = h('ul', { class: 'reasons' }, ...items.map((text) => h('li', { text })));
  body.hidden = true;
  const toggle = h('button', {
    class: 'ghost disclosure',
    text: `${title} (${items.length})`,
    onclick: () => {
      body.hidden = !body.hidden;
      toggle.classList.toggle('open', !body.hidden);
    },
  });
  return h('div', { class: 'stack' }, toggle, body);
}

function compactRow(state: ScanState): HTMLElement {
  const colour = state.result?.verdict.ringColor ?? NEUTRAL;
  const fake = state.result ? fakePercent(state.result) : null;
  return h(
    'div',
    { class: 'compact' },
    h('span', { class: 'dot', style: `background:${colour}` }),
    h('span', { class: 'compact-label', text: state.result?.verdict.shortLabel ?? phaseLabel(state) }),
    h('span', { class: 'mono muted src', text: shortUrl(state.mediaUrl, 28) }),
    fake === null ? null : h('span', { class: 'mono', text: `${Math.round(fake)}%` })
  );
}

void boot();
