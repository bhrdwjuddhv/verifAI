/**
 * Options. Nothing here is cosmetic: the server address decides where files go, and the
 * build panel answers "which model said this" — the question the rest of the project answers
 * everywhere else.
 */

import '../ui/ui.css';
import '../ui/page.css';

import type { UiRequest, UiResponse } from '../shared/protocol';
import {
  CONSENT_VERSION,
  getSettings,
  normalizeOrigin,
  originPattern,
  setSettings,
  type ScanMode,
} from '../shared/settings';
import { fmtBytes } from '../shared/format';
import { AUTO_SITES } from '../shared/sites';
import { clear, h } from '../ui/dom';

declare const __VERSION__: string;
declare const __TARGET__: string;

const modeSelect = document.getElementById('mode') as HTMLSelectElement;
const modeNote = document.getElementById('mode-note')!;
const serverInput = document.getElementById('server') as HTMLInputElement;
const serverStatus = document.getElementById('server-status')!;
const consentState = document.getElementById('consent-state')!;

async function ask(request: UiRequest): Promise<UiResponse> {
  return (await chrome.runtime.sendMessage(request)) as UiResponse;
}

async function boot(): Promise<void> {
  const settings = await getSettings();
  modeSelect.value = settings.mode;
  serverInput.value = settings.serverUrl;
  paintModeNote(settings.mode);

  consentState.textContent =
    settings.consentVersion >= CONSENT_VERSION
      ? 'You have accepted that deep scan uploads files to the server above.'
      : 'Not accepted yet — deep scan will ask before it uploads anything.';

  autoScanBox.checked = settings.autoScan;
  paintAutoNote(settings.mode, settings.autoScan);
  await paintSites();

  await paintPermission(settings.serverUrl);
  await paintCache();
  await paintBuild();
}

const autoScanBox = document.getElementById('auto-scan') as HTMLInputElement;
const autoNote = document.getElementById('auto-note')!;

autoScanBox.addEventListener('change', async () => {
  await setSettings({ autoScan: autoScanBox.checked });
  const settings = await getSettings();
  paintAutoNote(settings.mode, settings.autoScan);
  await paintSites();
});

function paintAutoNote(mode: ScanMode, enabled: boolean): void {
  if (!enabled) {
    autoNote.textContent = 'Off. Nothing is watched, and no content script runs on any site.';
    autoNote.className = 'muted';
    return;
  }
  if (mode !== 'device') {
    autoNote.textContent =
      'Enabled, but inert: the current mode uploads, and auto-scan never does. Switch to on-device above to actually use it.';
    autoNote.className = 'pill-bad';
    return;
  }
  autoNote.textContent = 'Running on the sites allowed below. Everything is scored locally.';
  autoNote.className = 'pill-ok';
}

/**
 * One row per site, each its own grant.
 *
 * Access is requested here rather than declared in the manifest: an extension that asks for
 * youtube.com and instagram.com at install time is asking for a slow store review and a
 * suspicious user, and neither is needed until auto-scan is actually switched on.
 */
async function paintSites(): Promise<void> {
  const target = document.getElementById('sites')!;
  clear(target);

  for (const site of AUTO_SITES) {
    const granted = await chrome.permissions.contains({ origins: site.origins });
    const row = h(
      'div',
      { class: 'site-row' },
      h(
        'div',
        { class: 'stack' },
        h('span', { text: site.label }),
        h('span', { class: 'muted', text: site.description })
      ),
      h('button', {
        class: granted ? '' : 'primary',
        text: granted ? 'Remove access' : 'Allow',
        onclick: async () => {
          if (granted) await chrome.permissions.remove({ origins: site.origins });
          else await chrome.permissions.request({ origins: site.origins });
          await paintSites();
        },
      })
    );
    target.append(row);
  }
}

function paintModeNote(mode: ScanMode): void {
  modeNote.textContent =
    mode === 'device'
      ? 'Images are decoded and scored here, on this machine. Nothing is uploaded and no consent is needed. Whichever detectors are bundled vote; any that are missing abstain and say so, rather than being replaced by a guess.'
      : 'Files are uploaded to the server below, scored there, and the verdict comes back with the per-detector breakdown.';
}

modeSelect.addEventListener('change', async () => {
  const mode = modeSelect.value as ScanMode;
  await setSettings({ mode });
  paintModeNote(mode);
});

document.getElementById('save')!.addEventListener('click', async () => {
  const serverUrl = normalizeOrigin(serverInput.value);
  const origin = originPattern(serverUrl);
  if (!origin) {
    serverStatus.textContent = 'That is not a valid URL.';
    return;
  }
  await setSettings({ serverUrl });
  serverInput.value = serverUrl;
  // Asking here keeps the grant on this click; the service worker cannot request it later.
  if (!(await chrome.permissions.contains({ origins: [origin] }))) {
    await chrome.permissions.request({ origins: [origin] });
  }
  await paintPermission(serverUrl);
});

/**
 * Liveness probe with no file attached. `/api/scan` answers 400 "No file provided." when it
 * is alive, which proves the route and the middleware are both up without uploading anything
 * — at the cost of one request against the 20/min budget.
 */
document.getElementById('test')!.addEventListener('click', async () => {
  const serverUrl = normalizeOrigin(serverInput.value);
  const origin = originPattern(serverUrl);
  if (!origin) {
    serverStatus.textContent = 'That is not a valid URL.';
    return;
  }
  if (!(await chrome.permissions.request({ origins: [origin] }))) {
    serverStatus.textContent = 'Access to that origin was declined, so it cannot be tested.';
    return;
  }

  serverStatus.textContent = 'Testing…';
  try {
    const res = await fetch(`${serverUrl}/api/scan`, {
      method: 'POST',
      body: new FormData(),
      headers: { 'X-VerifAI-Client': `extension/${__VERSION__}` },
      signal: AbortSignal.timeout(15_000),
    });
    const body = (await res.json().catch(() => ({}))) as { error?: string };
    serverStatus.textContent =
      res.status === 400
        ? `Reachable — the scan route answered "${body.error ?? '400'}".`
        : `Answered ${res.status}${body.error ? ` — ${body.error}` : ''}.`;
    serverStatus.className = res.status === 400 ? 'pill-ok' : 'muted';
  } catch {
    serverStatus.textContent = `No answer from ${serverUrl}. Is the app running?`;
    serverStatus.className = 'pill-bad';
  }
});

async function paintPermission(serverUrl: string): Promise<void> {
  const origin = originPattern(serverUrl);
  if (!origin) return;
  const granted = await chrome.permissions.contains({ origins: [origin] });
  serverStatus.textContent = granted
    ? 'Access granted for this origin.'
    : 'No access yet — the first scan will ask.';
  serverStatus.className = granted ? 'pill-ok' : 'muted';
}

async function paintCache(): Promise<void> {
  const stats = await ask({ type: 'ui:cache-stats' });
  const target = document.getElementById('cache-stats')!;
  clear(target);
  if (stats.type !== 'cache-stats') return;
  target.append(
    h('dt', { text: 'Verdicts' }),
    h('dd', { class: 'mono', text: String(stats.entries) }),
    h('dt', { text: 'Heatmaps' }),
    h('dd', { class: 'mono', text: String(stats.heatmaps) }),
    h('dt', { text: 'Approx. size' }),
    h('dd', { class: 'mono', text: fmtBytes(stats.approxBytes) })
  );
}

document.getElementById('clear-cache')!.addEventListener('click', async () => {
  await ask({ type: 'ui:clear-cache' });
  await paintCache();
});

document.getElementById('review-consent')!.addEventListener('click', () => {
  void chrome.tabs.create({ url: chrome.runtime.getURL('onboarding.html') });
});

document.getElementById('check-drift')!.addEventListener('click', async () => {
  const target = document.getElementById('drift-info')!;
  clear(target);
  target.append(h('dt', { text: 'Status' }), h('dd', { text: 'checking…' }));

  const drift = (await ask({ type: 'ui:drift' } as UiRequest)) as unknown as {
    differences?: string[];
    checkedAt?: number | null;
    serverUnavailable?: boolean;
  };

  clear(target);
  if (drift?.serverUnavailable) {
    target.append(
      h('dt', { text: 'Status' }),
      h('dd', { class: 'muted', text: 'the server answered, but its model service is down — nothing to compare' })
    );
    return;
  }
  if (!drift?.checkedAt) {
    target.append(
      h('dt', { text: 'Status' }),
      h('dd', { class: 'muted', text: 'could not reach the server, or access to it has not been granted' })
    );
    return;
  }
  if (!drift.differences?.length) {
    target.append(
      h('dt', { text: 'Status' }),
      h('dd', { class: 'pill-ok', text: 'settings match the server' }),
      h('dt', { text: 'Checked' }),
      h('dd', { class: 'mono', text: new Date(drift.checkedAt).toLocaleString() })
    );
    return;
  }
  target.append(h('dt', { text: 'Drifted' }), h('dd', { class: 'pill-bad', text: `${drift.differences.length} setting(s)` }));
  for (const difference of drift.differences) {
    target.append(h('dt', { text: '' }), h('dd', { class: 'pill-bad', text: difference }));
  }
});

document.getElementById('probe')!.addEventListener('click', async () => {
  const target = document.getElementById('engine-info')!;
  clear(target);
  target.append(h('dt', { text: 'Status' }), h('dd', { text: 'probing…' }));

  const probe = (await ask({ type: 'ui:probe' } as UiRequest)) as unknown as
    | {
        webgpu: boolean;
        sharedArrayBuffer: boolean;
        crossOriginIsolated: boolean;
        backend: string;
        hardwareConcurrency: number;
        wasm: boolean;
        wasmReason?: string;
      }
    | { error: string };

  clear(target);
  if ('error' in probe) {
    target.append(h('dt', { text: 'Status' }), h('dd', { class: 'pill-bad', text: probe.error }));
    return;
  }
  target.append(
    h('dt', { text: 'WebAssembly' }),
    h('dd', {
      class: probe.wasm ? 'pill-ok' : 'pill-bad',
      text: probe.wasm ? 'allowed — detectors can load' : (probe.wasmReason ?? 'blocked'),
    }),
    h('dt', { text: 'Backend' }),
    h('dd', { class: `mono ${probe.webgpu ? 'pill-ok' : ''}`, text: probe.backend }),
    h('dt', { text: 'WebGPU' }),
    h('dd', { text: probe.webgpu ? 'available — inference runs on the GPU' : 'unavailable — CPU only' }),
    h('dt', { text: 'WASM threads' }),
    h('dd', {
      text: probe.sharedArrayBuffer
        ? `available (${probe.hardwareConcurrency} cores)`
        : 'unavailable — extension pages get no cross-origin isolation, so ORT runs single-threaded',
    })
  );
});

interface ModelsManifest {
  builtAt: string;
  fusionWeights: Record<string, number>;
  thresholds: { fakeAbove: number; realBelow: number };
  models: { file: string; bytes: number; sha256: string }[];
  missing: string[];
}

/**
 * What actually shipped in this build, read back from the manifest the build wrote.
 *
 * Describing the bundled models from memory is how a panel ends up claiming a detector that
 * is not there — so this reads hashes off disk instead, and names what is missing.
 */
async function paintBuild(): Promise<void> {
  const target = document.getElementById('build-info')!;
  clear(target);
  target.append(
    h('dt', { text: 'Extension' }),
    h('dd', { class: 'mono', text: `${__VERSION__} (${__TARGET__})` }),
    h('dt', { text: 'Deep scan detectors' }),
    h('dd', { text: 'whatever the configured server reports, per scan' })
  );

  let manifest: ModelsManifest | null = null;
  try {
    const res = await fetch(chrome.runtime.getURL('models/models.json'));
    if (res.ok) manifest = (await res.json()) as ModelsManifest;
  } catch {
    manifest = null;
  }
  if (!manifest) {
    target.append(h('dt', { text: 'On-device models' }), h('dd', { text: 'no manifest in this build' }));
    return;
  }

  for (const model of manifest.models) {
    target.append(
      h('dt', { text: model.file }),
      h('dd', {
        class: 'mono',
        text: `${(model.bytes / 1048576).toFixed(1)}MB · ${model.sha256.slice(0, 12)}`,
        title: `sha256 ${model.sha256}`,
      })
    );
  }
  if (manifest.missing.length) {
    target.append(
      h('dt', { text: 'Missing' }),
      h('dd', { class: 'pill-bad', text: `${manifest.missing.join(', ')} — those detectors abstain` })
    );
  }

  const weights = Object.entries(manifest.fusionWeights)
    .map(([key, value]) => `${key} ${value}`)
    .join(', ');
  target.append(
    h('dt', { text: 'Fusion' }),
    h('dd', { class: 'mono', text: weights }),
    h('dt', { text: 'Thresholds' }),
    h('dd', {
      class: 'mono',
      text: `fake > ${manifest.thresholds.fakeAbove} · real < ${manifest.thresholds.realBelow}`,
    })
  );
}

void boot();
