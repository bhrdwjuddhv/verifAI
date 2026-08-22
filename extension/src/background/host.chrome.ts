/**
 * Owns the offscreen document's lifecycle.
 *
 * Only one offscreen document may exist per extension, and `hasDocument()` followed by
 * `createDocument()` is a race the moment two scans start together — both see "no document"
 * and the second create throws. Everything funnels through one promise so that cannot happen,
 * and the "already exists" error is still caught, because the worker can be evicted between
 * the check and the call.
 */

import type { ScanResult } from '../shared/scan-types';

const PATH = 'offscreen.html';

let opening: Promise<void> | null = null;

async function open(): Promise<void> {
  if (await chrome.offscreen.hasDocument()) return;
  try {
    await chrome.offscreen.createDocument({
      url: PATH,
      reasons: [chrome.offscreen.Reason.WORKERS],
      justification: 'Runs the bundled ONNX detectors locally so images are never uploaded.',
    });
  } catch (err) {
    if (!/single offscreen document/i.test(String((err as Error)?.message))) throw err;
  }
}

export function ensureOffscreen(): Promise<void> {
  opening ??= open().finally(() => {
    opening = null;
  });
  return opening;
}

export interface DeviceScan {
  result: ScanResult;
  ms: number;
}

export async function scoreOnDevice(payload: {
  b64: string;
  filename: string;
  mime: string;
  /** Adds the occlusion heatmap: 26 forward passes, so never on by default. */
  explain?: boolean;
}): Promise<DeviceScan> {
  await ensureOffscreen();
  const response = (await chrome.runtime.sendMessage({ type: 'offscreen:score', ...payload })) as
    | { ok: true; result: ScanResult; ms: number }
    | { ok: false; error: string }
    | undefined;

  if (!response) throw new Error('The offscreen worker did not answer.');
  if (!response.ok) throw new Error(response.error);
  return { result: response.result, ms: response.ms };
}

export interface Capabilities {
  crossOriginIsolated: boolean;
  sharedArrayBuffer: boolean;
  webgpu: boolean;
  backend: string;
  hardwareConcurrency: number;
  wasm: boolean;
  wasmReason?: string;
}

/** Spikes S1 and S2 from PLAN.md, surfaced in the options page instead of a console log. */
export async function probeCapabilities(): Promise<Capabilities | { error: string }> {
  await ensureOffscreen();
  return (await chrome.runtime.sendMessage({ type: 'offscreen:probe' })) as Capabilities;
}

export interface AudioSelftestResult {
  status: 'pass' | 'fail' | 'unavailable';
  observed?: number;
  expected?: number;
  tol?: number;
  delta?: number;
  ep?: string;
  reason: string | null;
}

/**
 * Run the v2 audio parity check in the offscreen document, where the graphs live.
 *
 * The result is cached there for the life of the document, so calling this repeatedly costs
 * nothing after the first run — and Live Guard reads the same cached answer when it starts.
 */
export async function audioSelftest(): Promise<AudioSelftestResult> {
  await ensureOffscreen();
  return (await chrome.runtime.sendMessage({ type: 'offscreen:audio-selftest' })) as AudioSelftestResult;
}
