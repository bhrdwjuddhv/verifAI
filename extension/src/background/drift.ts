/**
 * Detects when the extension's local settings have drifted from the server's.
 *
 * On-device mode scores with a bundled copy of the thresholds and fusion weights. Retune
 * FUSION_WEIGHTS on the model service and every installed extension keeps blending by the old
 * numbers — confidently, and wrongly, with nothing on screen to suggest it. This is the check
 * that stops that being silent.
 *
 * It never blocks a scan. It attaches a note, in the same voice the service uses for its own
 * caveats, because a stale weight makes a verdict *questionable*, not absent.
 */

import { getSettings, originPattern } from '../shared/settings';

const STORAGE_KEY = 'server:manifest';

/** The route is cheap and changes rarely; daily is plenty and costs one request. */
export const REFRESH_ALARM = 'verifai-manifest';
export const REFRESH_MINUTES = 24 * 60;

export interface ServerManifest {
  apiVersion?: number;
  minExtensionVersion?: string;
  modelService?: 'ok' | 'unavailable';
  thresholds?: { fakeAbove: number; realBelow: number };
  fusionWeights?: Record<string, number>;
  saliencyGrid?: number;
  detectors?: Record<string, unknown>;
}

interface Stored {
  fetchedAt: number;
  serverUrl: string;
  manifest: ServerManifest;
}

interface BundledManifest {
  fusionWeights: Record<string, number>;
  thresholds: { fakeAbove: number; realBelow: number };
  saliencyGrid: number;
}

let bundled: BundledManifest | null = null;

async function readBundled(): Promise<BundledManifest | null> {
  if (bundled) return bundled;
  try {
    const res = await fetch(chrome.runtime.getURL('models/models.json'));
    if (!res.ok) return null;
    bundled = (await res.json()) as BundledManifest;
    return bundled;
  } catch {
    return null;
  }
}

/**
 * Fetches the server's manifest, if we are allowed to.
 *
 * Silent on failure by design: this runs on a timer, and an unreachable server is not
 * something to interrupt anyone about. The staleness shows up in the options page instead.
 */
export async function refreshManifest(): Promise<ServerManifest | null> {
  const settings = await getSettings();
  const origin = originPattern(settings.serverUrl);
  if (!origin) return null;
  if (!(await chrome.permissions.contains({ origins: [origin] }))) return null;

  try {
    const res = await fetch(`${settings.serverUrl}/api/extension/manifest`, {
      headers: { 'X-VerifAI-Client': 'extension/manifest' },
      signal: AbortSignal.timeout(15_000),
    });
    if (!res.ok) return null;

    const manifest = (await res.json()) as ServerManifest;
    const stored: Stored = { fetchedAt: Date.now(), serverUrl: settings.serverUrl, manifest };
    await chrome.storage.local.set({ [STORAGE_KEY]: stored });
    return manifest;
  } catch {
    return null;
  }
}

export async function storedManifest(): Promise<Stored | null> {
  const got = await chrome.storage.local.get(STORAGE_KEY);
  return (got[STORAGE_KEY] as Stored | undefined) ?? null;
}

export interface Drift {
  /** Human-readable differences, ready to display. Empty means agreement. */
  differences: string[];
  checkedAt: number | null;
  /** True when the server said its own model service was down — nothing to compare against. */
  serverUnavailable: boolean;
}

export async function checkDrift(): Promise<Drift> {
  const stored = await storedManifest();
  const local = await readBundled();
  if (!stored || !local) return { differences: [], checkedAt: stored?.fetchedAt ?? null, serverUnavailable: false };

  const { manifest } = stored;
  if (manifest.modelService === 'unavailable') {
    return { differences: [], checkedAt: stored.fetchedAt, serverUnavailable: true };
  }

  const differences: string[] = [];

  if (manifest.thresholds) {
    if (manifest.thresholds.fakeAbove !== local.thresholds.fakeAbove) {
      differences.push(
        `verdict threshold: server calls it fake above ${manifest.thresholds.fakeAbove}, this build above ${local.thresholds.fakeAbove}`
      );
    }
    if (manifest.thresholds.realBelow !== local.thresholds.realBelow) {
      differences.push(
        `verdict threshold: server calls it real below ${manifest.thresholds.realBelow}, this build below ${local.thresholds.realBelow}`
      );
    }
  }

  if (manifest.fusionWeights) {
    // Only weights the server actually reports are compared; a detector it does not run is a
    // separate matter, and the parity harness is where that belongs.
    for (const [key, serverWeight] of Object.entries(manifest.fusionWeights)) {
      const localWeight = local.fusionWeights[key];
      if (localWeight !== undefined && localWeight !== serverWeight) {
        differences.push(`fusion weight ${key}: server ${serverWeight}, this build ${localWeight}`);
      }
    }
  }

  if (manifest.saliencyGrid !== undefined && manifest.saliencyGrid !== local.saliencyGrid) {
    differences.push(
      `saliency grid: server ${manifest.saliencyGrid}x${manifest.saliencyGrid}, this build ${local.saliencyGrid}x${local.saliencyGrid}`
    );
  }

  return { differences, checkedAt: stored.fetchedAt, serverUnavailable: false };
}

/** The note attached to an on-device verdict when its settings no longer match the server's. */
export function driftNote(drift: Drift): string | null {
  if (!drift.differences.length) return null;
  return `this build's settings no longer match the server's (${drift.differences.join('; ')}), so an on-device score may not match a deep scan of the same image`;
}
