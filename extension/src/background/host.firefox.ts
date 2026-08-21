/**
 * The Firefox half of the inference host.
 *
 * Firefox MV3 has no `chrome.offscreen` — but it does not need one. Its background is an
 * event *page*, not a service worker, so it already has a DOM, `OffscreenCanvas`,
 * `createImageBitmap` and (where enabled) `navigator.gpu`. The scorer runs right here.
 *
 * The interface is identical to host.chrome.ts on purpose: scans.ts imports `#host` and never
 * learns which browser it is on. `scripts/build.mjs` picks the file.
 */

import type { ScanResult } from '../shared/scan-types';
import { score, type ScoreRequest } from '../offscreen/score';
import { backend } from '../offscreen/session';
import { wasmAllowed } from '../shared/wasm';

/** No document to create. Kept so callers do not need a per-target branch. */
export async function ensureOffscreen(): Promise<void> {}

export interface DeviceScan {
  result: ScanResult;
  ms: number;
}

export async function scoreOnDevice(payload: ScoreRequest): Promise<DeviceScan> {
  const started = performance.now();
  const result = await score(payload);
  return { result, ms: Math.round(performance.now() - started) };
}

export interface Capabilities {
  crossOriginIsolated: boolean;
  sharedArrayBuffer: boolean;
  webgpu: boolean;
  backend: string;
  hardwareConcurrency: number;
  /** False when the manifest's CSP forbids WebAssembly — nothing can run until it is fixed. */
  wasm: boolean;
  wasmReason?: string;
}

export async function probeCapabilities(): Promise<Capabilities | { error: string }> {
  let webgpu = false;
  try {
    webgpu = Boolean(await navigator.gpu?.requestAdapter());
  } catch {
    webgpu = false;
  }
  const wasm = wasmAllowed();
  return {
    crossOriginIsolated: self.crossOriginIsolated === true,
    sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    webgpu,
    backend: await backend(),
    hardwareConcurrency: navigator.hardwareConcurrency || 0,
    wasm: wasm.ok,
    wasmReason: wasm.reason,
  };
}
