/**
 * The on-page Live Guard overlay.
 *
 * Deliberately small, draggable-free and corner-pinned: it sits over someone's call, so it
 * must never cover a participant tile or steal a click. Everything is in a shadow root so the
 * host page's CSS cannot restyle it into something misleading.
 */

import type { GuardStatus } from '../shared/liveguard';

const BAND_COLORS: Record<string, string> = {
  low: '#10B981',
  uncertain: '#94A3B8',
  suspicious: '#F59E0B',
  high: '#EF4444',
  idle: '#5C6370',
};

const BAND_LABELS: Record<string, string> = {
  low: 'No synthetic signal',
  uncertain: 'Uncertain',
  suspicious: 'Possible synthetic voice',
  high: 'Strong synthetic signal',
  idle: 'Listening…',
};

let host: HTMLElement | null = null;
let root: ShadowRoot | null = null;

function ensureOverlay(): ShadowRoot {
  if (root) return root;

  host = document.createElement('div');
  host.id = 'verifai-live-guard';
  host.style.cssText = 'position:fixed;top:16px;right:16px;z-index:2147483647;';
  root = host.attachShadow({ mode: 'closed' });

  root.innerHTML = `
    <style>
      :host { all: initial; }
      .card {
        font: 12px/1.45 system-ui, -apple-system, Segoe UI, sans-serif;
        width: 232px; padding: 12px; border-radius: 14px;
        background: rgba(12,14,18,.94); color: #E7E9EE;
        border: 1px solid rgba(255,255,255,.14);
        box-shadow: 0 8px 28px rgba(0,0,0,.45); backdrop-filter: blur(8px);
      }
      .row { display:flex; align-items:center; gap:7px; }
      .dot { width:8px; height:8px; border-radius:50%; flex:none; }
      .rec { animation: pulse 1.6s ease-in-out infinite; background:#EF4444; }
      @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:.35 } }
      .title { font-weight:600; font-size:11px; letter-spacing:.04em; text-transform:uppercase; color:#9AA1AC; }
      .score { font-size:26px; font-weight:800; font-variant-numeric:tabular-nums; margin:6px 0 2px; }
      .band { font-size:11px; font-weight:600; }
      .bar { height:5px; border-radius:3px; background:rgba(255,255,255,.1); overflow:hidden; margin:8px 0 6px; }
      .fill { height:100%; transition:width .5s ease; }
      .timeline { display:flex; align-items:flex-end; gap:2px; height:26px; margin:6px 0; }
      .cell { flex:1; min-width:2px; border-radius:1px; }
      .hollow { height:4px; border:1px solid rgba(255,255,255,.22); background:transparent; align-self:flex-end; }
      .meta { color:#8E949F; font-size:10px; margin-top:6px; word-break:break-all; }
      .warn { margin-top:8px; padding:7px 8px; border-radius:9px; font-size:11px;
              background:rgba(239,68,68,.14); border:1px solid rgba(239,68,68,.4); color:#FDA4AF; }
      .note { margin-top:8px; padding:7px 8px; border-radius:9px; font-size:11px;
              background:rgba(245,158,11,.12); border:1px solid rgba(245,158,11,.35); color:#FCD9A6; }
      button { all:unset; cursor:pointer; color:#8E949F; font-size:14px; line-height:1; padding:0 2px; }
      button:hover { color:#E7E9EE; }
    </style>
    <div class="card">
      <div class="row" style="justify-content:space-between">
        <div class="row"><span class="dot rec"></span><span class="title">VerifAI monitoring</span></div>
        <button id="stop" title="Stop monitoring">×</button>
      </div>
      <div class="score" id="score">—</div>
      <div class="band" id="band">Listening…</div>
      <div class="bar"><div class="fill" id="fill" style="width:0%"></div></div>
      <div class="timeline" id="timeline"></div>
      <div class="meta" id="meta"></div>
      <div id="extra"></div>
    </div>
  `;

  root.getElementById('stop')?.addEventListener('click', () => {
    chrome.runtime.sendMessage({ type: 'guard:stop' }).catch(() => undefined);
  });

  document.documentElement.appendChild(host);
  return root;
}

export function renderGuard(status: GuardStatus): void {
  if (!status.active) {
    host?.remove();
    host = null;
    root = null;
    return;
  }

  const shadow = ensureOverlay();
  const color = BAND_COLORS[status.band] ?? BAND_COLORS.idle;

  const score = shadow.getElementById('score')!;
  score.textContent = status.trust === null ? '—' : `${status.trust}%`;
  score.style.color = color;

  const band = shadow.getElementById('band')!;
  band.textContent = BAND_LABELS[status.band] ?? status.band;
  band.style.color = color;

  const fill = shadow.getElementById('fill') as HTMLElement;
  fill.style.width = `${status.trust ?? 0}%`;
  fill.style.background = color;

  const timeline = shadow.getElementById('timeline')!;
  timeline.innerHTML = '';
  for (const w of status.windows.slice(-40)) {
    const cell = document.createElement('div');
    if (!w.speech || w.audio === null) {
      // Hollow, not green: "we did not listen" is not "we heard nothing wrong".
      cell.className = 'cell hollow';
      cell.title = w.note ? `${w.t.toFixed(0)}s — ${w.note}` : `${w.t.toFixed(0)}s — not scored`;
    } else {
      cell.className = 'cell';
      cell.style.height = `${Math.max(10, w.audio)}%`;
      cell.style.background = w.audio > 70 ? '#EF4444' : w.audio > 30 ? '#F59E0B' : '#10B981';
      cell.title = `${w.t.toFixed(0)}s — ${w.audio}% synthetic`;
    }
    timeline.appendChild(cell);
  }

  const meta = shadow.getElementById('meta')!;
  meta.textContent = [
    status.siteLabel,
    `${status.scored} scored`,
    status.skipped ? `${status.skipped} silent` : '',
    // Where the voice score came from. Worth a permanent line rather than a tooltip: it is
    // the difference between audio staying on this machine and audio being uploaded, and the
    // user should never have to guess which one is happening.
    status.audio?.state === 'verified' ? 'voice: on-device (parity verified)' : 'voice: our backend',
    status.modelSource ? `model: ${status.modelSource}` : '',
  ].filter(Boolean).join(' · ');

  const extra = shadow.getElementById('extra')!;
  extra.innerHTML = '';
  if (status.note) {
    const note = document.createElement('div');
    note.className = 'note';
    note.textContent = status.note;
    extra.appendChild(note);
  }
  if (status.consecutiveHigh >= 3) {
    const warn = document.createElement('div');
    warn.className = 'warn';
    warn.textContent =
      `Synthetic voice signal in ${status.consecutiveHigh} consecutive windows. Verify who you are ` +
      `speaking to another way. This is not proof, and the call is never ended for you.`;
    extra.appendChild(warn);
  }
}

export function removeGuard(): void {
  host?.remove();
  host = null;
  root = null;
}
