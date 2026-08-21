/**
 * User settings, in chrome.storage.sync so they follow a signed-in profile.
 *
 * The cache and scan history live in storage.local instead — they are large, machine-local,
 * and there is no reason to push image hashes to another device.
 */

/**
 * The deployed VerifAI instance. Editable in options, so a self-hoster can point this at
 * their own deployment or at `http://localhost:3000` for development.
 */
export const DEFAULT_SERVER_URL = 'https://verif-ai-blue.vercel.app';

/** Bump when the consent copy changes materially; users are re-asked. */
export const CONSENT_VERSION = 1;

export type ScanMode = 'server' | 'device' | 'ask';

export interface Settings {
  /** Origin of the VerifAI web app. `/api/scan` is appended to it. */
  serverUrl: string;
  /**
   * 'server' uploads the file for a deep scan; 'device' runs the bundled models locally
   * (Phase 2); 'ask' prompts per scan. Default is 'server' because it is the only mode with
   * a trained model behind it today.
   */
  mode: ScanMode;
  /** CONSENT_VERSION the user accepted, or 0. Deep scan is inert below the current version. */
  consentVersion: number;
  /** Phase 4. Off by design, and on-device only when it lands. */
  autoScan: boolean;
}

export const DEFAULTS: Settings = {
  serverUrl: DEFAULT_SERVER_URL,
  mode: 'server',
  consentVersion: 0,
  autoScan: false,
};

const AREA: chrome.storage.StorageArea = chrome.storage.sync ?? chrome.storage.local;

export async function getSettings(): Promise<Settings> {
  const stored = (await AREA.get(DEFAULTS)) as Settings;
  return { ...DEFAULTS, ...stored, serverUrl: normalizeOrigin(stored.serverUrl) };
}

export async function setSettings(patch: Partial<Settings>): Promise<void> {
  if (patch.serverUrl !== undefined) patch.serverUrl = normalizeOrigin(patch.serverUrl);
  await AREA.set(patch);
}

export function hasConsented(settings: Settings): boolean {
  return settings.consentVersion >= CONSENT_VERSION;
}

/** Trailing slashes turn `${base}/api/scan` into a 404 that looks like a server outage. */
export function normalizeOrigin(url: string | undefined): string {
  const raw = (url ?? DEFAULT_SERVER_URL).trim();
  if (!raw) return DEFAULT_SERVER_URL;
  return raw.replace(/\/+$/, '');
}

/** The match pattern we must hold to POST to the configured server. */
export function originPattern(url: string): string | null {
  try {
    return `${new URL(url).origin}/*`;
  } catch {
    return null;
  }
}
