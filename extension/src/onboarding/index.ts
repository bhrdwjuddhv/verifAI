/**
 * First run.
 *
 * The screen exists for one decision — does the file leave this machine — so that decision is
 * made here rather than buried in options afterwards. Two consequences follow:
 *
 *   - Host permission is requested only when it is actually needed. Asking someone who chose
 *     on-device to grant access to a server they will never contact is how a privacy-first
 *     option ends up feeling worse than the alternative.
 *   - The page has a real finished state. A greyed-out button and a line of small text reads
 *     as "stuck", not "done".
 */

import '../ui/ui.css';
import '../ui/page.css';

import {
  CONSENT_VERSION,
  DEFAULT_SERVER_URL,
  getSettings,
  normalizeOrigin,
  originPattern,
  setSettings,
  type ScanMode,
} from '../shared/settings';

const setup = document.getElementById('setup')!;
const done = document.getElementById('done')!;
const serverStep = document.getElementById('server-step')!;
const serverInput = document.getElementById('server') as HTMLInputElement;
const acceptButton = document.getElementById('accept') as HTMLButtonElement;
const status = document.getElementById('status')!;
const deviceDetail = document.getElementById('device-detail')!;

const modeInputs = [...document.querySelectorAll<HTMLInputElement>('input[name="mode"]')];
const chosenMode = (): ScanMode =>
  (modeInputs.find((input) => input.checked)?.value as ScanMode) ?? 'device';

function syncMode(): void {
  const mode = chosenMode();
  serverStep.hidden = mode !== 'server';
  acceptButton.textContent =
    mode === 'server' ? 'Accept and allow that server' : 'Accept and start scanning';
  for (const input of modeInputs) {
    input.closest('.choice')?.classList.toggle('selected', input.checked);
  }
}

modeInputs.forEach((input) => input.addEventListener('change', syncMode));

/**
 * Says what this build can actually do locally, read from the manifest the build wrote.
 *
 * Describing the bundled detectors from memory is how the copy ends up claiming a model that
 * is not in the package — which is exactly what the previous version of this page did.
 */
async function describeDevice(): Promise<void> {
  try {
    const res = await fetch(chrome.runtime.getURL('models/models.json'));
    if (!res.ok) throw new Error('no manifest');
    const manifest = (await res.json()) as { models: { file: string }[]; missing: string[] };

    const has = (name: string) => manifest.models.some((m) => m.file.startsWith(name));
    const parts: string[] = [];
    if (has('detector')) parts.push('a face-manipulation classifier');
    if (has('yunet')) parts.push('a face detector');
    if (has('npr')) parts.push('the whole-image AI-generation detector');

    const missing = manifest.missing.length
      ? ` The whole-image detector is not in this build, so it abstains and every verdict says so — a fully generated image with no face may not be caught locally.`
      : '';

    deviceDetail.textContent = parts.length
      ? `Runs ${parts.join(' and ')} here, on your machine. The image is never uploaded and no permission is needed.${missing}`
      : 'No detectors are bundled in this build, so on-device scans will report that rather than guess.';
  } catch {
    deviceDetail.textContent =
      'Runs the bundled detectors here, on your machine. The image is never uploaded and no permission is needed.';
  }
}

async function boot(): Promise<void> {
  const settings = await getSettings();
  serverInput.value = settings.serverUrl || DEFAULT_SERVER_URL;
  const preselect = modeInputs.find((input) => input.value === settings.mode);
  if (preselect) preselect.checked = true;
  syncMode();
  await describeDevice();

  if (settings.consentVersion >= CONSENT_VERSION) {
    status.textContent = 'Already set up — changing anything here will just save over it.';
  }
}

acceptButton.addEventListener('click', async () => {
  const mode = chosenMode();
  acceptButton.disabled = true;
  status.textContent = '';

  if (mode === 'server') {
    const serverUrl = normalizeOrigin(serverInput.value);
    const origin = originPattern(serverUrl);
    if (!origin) {
      status.textContent = 'That is not a valid URL.';
      acceptButton.disabled = false;
      return;
    }

    // Must happen on this click: chrome.permissions.request needs a live user gesture.
    const granted = await chrome.permissions.request({ origins: [origin] });
    if (!granted) {
      status.textContent =
        'Without access to that server, deep scan cannot upload anything — nothing was saved. Choose on-device instead, or accept again.';
      acceptButton.disabled = false;
      return;
    }
    await setSettings({ serverUrl, mode, consentVersion: CONSENT_VERSION });
    finish(`Deep scan, sending to ${new URL(serverUrl).host}.`);
    return;
  }

  await setSettings({ mode, consentVersion: CONSENT_VERSION });
  finish('On-device scanning. Nothing you verify will leave this machine.');
});

function finish(summary: string): void {
  document.getElementById('done-summary')!.textContent = summary;
  setup.hidden = true;
  done.hidden = false;
  window.scrollTo({ top: 0 });
}

document.getElementById('open-options')!.addEventListener('click', () => {
  void chrome.tabs.create({ url: chrome.runtime.getURL('options.html') });
});

document.getElementById('close-tab')!.addEventListener('click', () => {
  window.close();
});

void boot();
