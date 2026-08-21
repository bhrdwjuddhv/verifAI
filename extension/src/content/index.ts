/**
 * The in-page badge. DOM only — this script never scores anything.
 *
 * Injected on demand by the service worker (there is no declared content script, so the
 * extension holds no site access until a scan is asked for), which means it can be injected
 * twice into the same page. Everything below is written to be idempotent.
 *
 * Styling lives inside a shadow root: Instagram and YouTube both ship aggressive global CSS,
 * and a badge that inherits `img { width: 100% }` is a badge that covers the post.
 */

import { renderGuard } from './liveguard';
import type { ContentRequest, ScanState } from '../shared/protocol';
import { detectorRows, fakePercent, NEUTRAL } from '../shared/format';
import { startAuto, stopAuto } from './auto';
import { captureVisible } from './frames';
import { onTeardown, send } from './runtime';

const FLAG = '__verifaiContentReady';

declare global {
  interface Window {
    __verifaiContentReady?: boolean;
  }
}

if (!window[FLAG]) {
  window[FLAG] = true;
  chrome.runtime.onMessage.addListener((message: ContentRequest, _sender, sendResponse) => {
    if (message?.type === 'content:ping') {
      sendResponse({ pong: true });
      return false;
    }
    if (message?.type === 'content:badge') {
      render(message.state);
      sendResponse({ ok: true });
      return false;
    }
    if (message?.type === 'content:capture') {
      void captureVisible().then(sendResponse);
      return true; // async reply
    }
    if (message?.type === 'content:auto') {
      if (message.enabled) startAuto();
      else stopAuto();
      sendResponse({ ok: true });
      return false;
    }
    return false;
  });

  // Registered content scripts start on every load of a granted site, so this side asks
  // rather than assumes: settings and permissions can both have changed since registration.
  void send<{ enabled?: boolean }>({ type: 'auto:hello' }).then((reply) => {
    if (reply?.enabled) startAuto();
  });

  // A badge left behind by a reloaded extension can never update again, and describes a scan
  // whose result is no longer reachable.
  onTeardown(() => detachAll());
}

// ---------------------------------------------------------------------------------------
// Finding what was scanned

/**
 * Matched by URL rather than by anything the page provides. `info.srcUrl` from the context
 * menu is the resolved `currentSrc`, which is what an <img srcset> actually loaded — so this
 * survives responsive images, and does not depend on a class name that changes next week.
 */
function findMedia(url: string): HTMLElement | null {
  const candidates = [...document.querySelectorAll<HTMLImageElement | HTMLVideoElement>('img, video')];
  const matches = candidates.filter((el) => el.currentSrc === url || el.getAttribute('src') === url);
  if (!matches.length) return null;

  // Same image can appear as a thumbnail and as the opened post; badge the biggest one.
  return matches.sort((a, b) => area(b) - area(a))[0];
}

function area(el: Element): number {
  const r = el.getBoundingClientRect();
  return r.width * r.height;
}

// ---------------------------------------------------------------------------------------
// Rendering

interface Mounted {
  host: HTMLDivElement;
  root: ShadowRoot;
  target: HTMLElement | null;
  detach: () => void;
}

/**
 * One badge per media element — not one per scan.
 *
 * Keying by scan id was wrong in a way that only shows up on a feed: auto-scan samples four
 * frames per clip, so four badges appeared per Short, and once the recycled player swapped
 * its source `findMedia` stopped matching and every one of them parked in the same corner.
 * Scrolling built a pile.
 *
 * Now a scan updates the badge belonging to its element, and an auto scan with nothing left
 * to point at is dropped rather than parked: the user never asked for it, and a corner badge
 * describing a video they have already scrolled past is worse than no badge at all.
 */
const byElement = new WeakMap<HTMLElement, Mounted>();

/** Manual scans whose element could not be found. They were asked for, so they still show. */
const orphans = new Map<string, Mounted>();

/** Everything currently on screen, in creation order, for pruning and the cap. */
const live = new Set<Mounted>();

/** A feed can outrun any reasonable reading pace; past this the oldest is dropped. */
const MAX_BADGES = 4;

function render(state: ScanState): void {
  const target = findMedia(state.mediaUrl);

  if (!target) {
    if (state.auto) return;
    const view = orphans.get(state.id) ?? mount(null, state.id);
    orphans.set(state.id, view);
    paint(view, state);
    reposition(view);
    return;
  }

  const view = byElement.get(target) ?? mount(target, state.id);
  byElement.set(target, view);
  view.target = target;
  paint(view, state);
  reposition(view);
  enforceCap();
}

/** Drops the badge on a given element — used when a recycled player loads a different clip. */
export function clearFor(element: HTMLElement): void {
  byElement.get(element)?.detach();
}

/** Takes every badge down. Used when the extension is reloaded away from under this script. */
function detachAll(): void {
  for (const view of [...live]) view.detach();
}

function enforceCap(): void {
  while (live.size > MAX_BADGES) {
    const oldest = live.values().next().value as Mounted | undefined;
    if (!oldest) return;
    oldest.detach();
  }
}

function mount(target: HTMLElement | null, id: string): Mounted {
  const host = document.createElement('div');
  host.style.cssText = 'all: initial; position: fixed; z-index: 2147483647; pointer-events: none;';
  const root = host.attachShadow({ mode: 'closed' });
  applyStyles(root);
  document.documentElement.appendChild(host);

  const onScroll = () => reposition(view);
  addEventListener('scroll', onScroll, { passive: true, capture: true });
  addEventListener('resize', onScroll, { passive: true });

  const view: Mounted = {
    host,
    root,
    target,
    detach: () => {
      removeEventListener('scroll', onScroll, { capture: true });
      removeEventListener('resize', onScroll);
      host.remove();
      live.delete(view);
      orphans.delete(id);
      if (view.target) byElement.delete(view.target);
    },
  };

  live.add(view);
  return view;
}

function reposition(view: Mounted): void {
  if (!view.target) {
    // A manual scan we could not anchor. Park it in the corner rather than dropping a result
    // somebody explicitly asked for.
    view.host.style.top = '16px';
    view.host.style.right = '16px';
    view.host.style.left = 'auto';
    return;
  }

  // The element was removed from the page — a feed recycling its DOM. There is nothing left
  // for this badge to describe.
  if (!view.target.isConnected) {
    view.detach();
    return;
  }

  const rect = view.target.getBoundingClientRect();
  const offscreen = rect.bottom < 0 || rect.top > innerHeight || rect.width === 0;
  view.host.style.visibility = offscreen ? 'hidden' : 'visible';
  view.host.style.top = `${Math.max(8, rect.top + 8)}px`;
  view.host.style.left = `${Math.max(8, rect.left + 8)}px`;
  view.host.style.right = 'auto';
}

function paint(view: Mounted, state: ScanState): void {
  view.root.querySelector('.card')?.remove();

  const card = document.createElement('div');
  card.className = 'card';
  card.appendChild(pill(view, state));

  const detail = details(state);
  if (detail) {
    const wrap = document.createElement('div');
    wrap.className = 'detail';
    wrap.hidden = true;
    wrap.appendChild(detail);
    card.appendChild(wrap);
    card.querySelector('.pill')!.addEventListener('click', () => {
      wrap.hidden = !wrap.hidden;
    });
  }

  view.root.appendChild(card);
}

function pill(view: Mounted, state: ScanState): HTMLElement {
  const el = document.createElement('button');
  el.className = 'pill';
  el.type = 'button';

  const colour = state.result?.verdict.ringColor || NEUTRAL;
  const dot = document.createElement('span');
  dot.className = 'dot';
  dot.style.background = colour;
  if (state.phase !== 'done' && state.phase !== 'error') dot.classList.add('pulsing');

  const label = document.createElement('span');
  label.className = 'label';
  label.textContent = headline(state);

  const close = document.createElement('span');
  close.className = 'close';
  close.textContent = '×';
  close.setAttribute('role', 'button');
  close.setAttribute('aria-label', 'Dismiss');
  close.addEventListener('click', (event) => {
    event.stopPropagation();
    view.detach();
  });

  el.append(dot, label, close);
  return el;
}

function headline(state: ScanState): string {
  switch (state.phase) {
    case 'queued':
      return state.queuePosition ? `VerifAI · queued (${state.queuePosition} ahead)` : 'VerifAI · queued';
    case 'reading':
      return 'VerifAI · reading image';
    case 'scanning':
      return 'VerifAI · scanning';
    case 'needs-consent':
      return 'VerifAI · needs your consent';
    case 'needs-permission':
      return 'VerifAI · needs permission';
    case 'error':
      return `VerifAI · ${state.error ?? 'failed'}`;
    case 'done': {
      const result = state.result!;
      const fake = fakePercent(result);
      const suffix = state.cached ? ' · cached' : '';
      return fake === null
        ? `${result.verdict.label} · no score${suffix}`
        : `${result.verdict.label} · ${Math.round(fake)}% likely AI${suffix}`;
    }
  }
}

function details(state: ScanState): HTMLElement | null {
  if (state.phase !== 'done' || !state.result) return null;
  const result = state.result;
  const wrap = document.createElement('div');

  const summary = document.createElement('p');
  summary.className = 'summary';
  summary.textContent = result.verdict.laymanSummary;
  wrap.appendChild(summary);

  for (const row of detectorRows(result)) {
    const line = document.createElement('div');
    line.className = 'row';

    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = row.label;

    const value = document.createElement('span');
    value.className = 'value';
    value.textContent =
      row.value === null
        ? 'did not apply'
        : `${Math.round(row.value)}%${row.weight > 0 ? '' : ' · no weight'}`;
    if (row.value === null) value.classList.add('muted');

    line.append(name, value);
    wrap.appendChild(line);
  }

  if (result.heatmap) {
    const img = document.createElement('img');
    img.className = 'heatmap';
    img.src = result.heatmap;
    img.alt = 'Where the model looked';
    wrap.appendChild(img);

    const caption = document.createElement('p');
    caption.className = 'caption';
    caption.textContent = result.signals.faceDetected
      ? 'Grad-CAM over the cropped face, not the whole photo.'
      : 'Grad-CAM over the analysed region.';
    wrap.appendChild(caption);
  }

  const source = document.createElement('p');
  source.className = 'caption';
  source.textContent = `Model: ${result.modelSource}`;
  wrap.appendChild(source);

  return wrap;
}

/**
 * adoptedStyleSheets rather than a <style> element: pages with a strict `style-src` CSP —
 * GitHub and Instagram among them — block injected style tags, and the badge would render
 * as unstyled text on top of the post.
 */
function applyStyles(root: ShadowRoot): void {
  try {
    const sheet = new CSSStyleSheet();
    sheet.replaceSync(CSS_TEXT);
    root.adoptedStyleSheets = [sheet];
  } catch {
    const style = document.createElement('style');
    style.textContent = CSS_TEXT;
    root.appendChild(style);
  }
}

const CSS_TEXT = `
    :host { all: initial; }
    .card {
      pointer-events: auto;
      font: 12px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
      max-width: 280px;
      color: #E6E8EF;
    }
    .pill {
      display: flex; align-items: center; gap: 7px;
      background: rgba(10, 12, 18, .92);
      border: 1px solid rgba(255, 255, 255, .14);
      border-radius: 999px;
      padding: 5px 8px 5px 9px;
      color: inherit; font: inherit; cursor: pointer;
      box-shadow: 0 4px 16px rgba(0, 0, 0, .45);
      backdrop-filter: blur(6px);
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
    .pulsing { animation: pulse 1.4s ease-in-out infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1 } 50% { opacity: .35 } }
    @media (prefers-reduced-motion: reduce) { .pulsing { animation: none } }
    .label { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 220px; }
    .close { opacity: .5; padding: 0 2px; font-size: 14px; line-height: 1; }
    .close:hover { opacity: 1; }
    .detail {
      margin-top: 6px;
      background: rgba(10, 12, 18, .96);
      border: 1px solid rgba(255, 255, 255, .14);
      border-radius: 10px;
      padding: 10px 11px;
      box-shadow: 0 8px 28px rgba(0, 0, 0, .5);
    }
    .summary { margin: 0 0 8px; }
    .row { display: flex; justify-content: space-between; gap: 10px; padding: 3px 0; }
    .name { color: #A8AEC2; }
    .value { font-variant-numeric: tabular-nums; }
    .muted { color: #767D93; }
    .heatmap { display: block; width: 100%; border-radius: 6px; margin: 8px 0 4px; }
    .caption { margin: 4px 0 0; color: #767D93; font-size: 11px; }
`;


// Live Guard overlay: the worker pushes status, this only draws it.
chrome.runtime.onMessage.addListener((message: any) => {
  if (message?.type === 'guard:status') renderGuard(message.status);
  return undefined;
});

// A reload mid-call should not leave the user believing monitoring stopped when it has not.
chrome.runtime.sendMessage({ type: 'guard:get-status' })
  .then((status) => { if (status?.active) renderGuard(status); })
  .catch(() => undefined);
