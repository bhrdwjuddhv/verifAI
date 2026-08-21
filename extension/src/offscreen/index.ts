/**
 * The offscreen document: the only context that runs inference.
 *
 * It exists because a service worker has neither `navigator.gpu` nor a lifetime longer than
 * ~30s idle, and a content script must never block a page's main thread. Messages arrive
 * from the worker; nothing here touches the network.
 */

import { score, type ScoreRequest } from './score';
import { backend } from './session';
import { wasmAllowed } from '../shared/wasm';

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

/** Spikes S1 and S2 from PLAN.md, answerable the moment this document exists. */
export async function probeCapabilities(): Promise<Capabilities> {
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

type Request =
  | { type: 'offscreen:probe' }
  | ({ type: 'offscreen:score' } & ScoreRequest);

function handle(message: Request, _sender: unknown, sendResponse: (value: unknown) => void): boolean {
  if (message?.type === 'offscreen:probe') {
    probeCapabilities().then(sendResponse, (err) => sendResponse({ error: String(err?.message ?? err) }));
    return true;
  }

  if (message?.type === 'offscreen:score') {
    const started = performance.now();
    score(message).then(
      (result) => sendResponse({ ok: true, result, ms: Math.round(performance.now() - started) }),
      // A failure here is reported, never smoothed over: an unscored image must not come
      // back looking like a verdict.
      (err) => sendResponse({ ok: false, error: String(err?.message ?? err) })
    );
    return true;
  }

  return false;
}

chrome.runtime.onMessage.addListener(handle);

// scripts/preview.mjs drives this document in an ordinary tab to exercise ONNX Runtime, the
// wasm load and the real image decode without installing the extension.
(self as unknown as { __verifaiHandler?: typeof handle }).__verifaiHandler = handle;
