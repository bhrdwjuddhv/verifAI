/**
 * The offscreen document: the only context that runs inference.
 *
 * It exists because a service worker has neither `navigator.gpu` nor a lifetime longer than
 * ~30s idle, and a content script must never block a page's main thread. Messages arrive
 * from the worker; nothing here touches the network.
 */

import { score, type ScoreRequest } from './score';
import { startCapture, stopCapture } from './live';
import { backend } from './session';
import { wasmAllowed } from '../shared/wasm';
import { MODEL_SOURCE as AUDIO_MODEL_SOURCE, scoreSamples, selftest } from './audio';
import { speechMetrics } from '../shared/vad';

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


/**
 * Live Guard capture lives here because only a document has getUserMedia and AudioContext.
 *
 * Two possible routes for a completed window, decided by the worker when it starts the
 * session and never mid-call:
 *
 *   on-device — the samples are scored here by the v2 chain and only a probability crosses
 *               back. The call audio never leaves this document, let alone the machine.
 *   backend   — the WAV goes to the worker base64-encoded, because chrome messaging
 *               JSON-serialises and an ArrayBuffer would silently arrive as `{}`.
 */
let liveOnDevice = false;

chrome.runtime.onMessage.addListener((message: any, _sender, sendResponse) => {
  if (message?.type === 'offscreen:audio-selftest') {
    selftest().then(sendResponse, (err) =>
      sendResponse({ status: 'unavailable', reason: String(err?.message ?? err) })
    );
    return true;
  }

  // The worker can revoke on-device routing mid-call — if the chain throws after the
  // self-test passed, the rest of the session must reach the backend, not stop being scored.
  if (message?.type === 'offscreen:live-route') {
    liveOnDevice = message.onDevice === true;
    sendResponse({ ok: true });
    return false;
  }

  if (message?.type === 'offscreen:live-start') {
    liveOnDevice = message.onDevice === true;
    startCapture(message.streamId, message.windowSeconds, {
      onWindow: (wav, samples, sampleRate) => {
        if (liveOnDevice) {
          void scoreWindowOnDevice(samples, sampleRate);
          return;
        }
        let binary = '';
        const bytes = new Uint8Array(wav);
        for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
        chrome.runtime
          .sendMessage({ type: 'offscreen:live-window', wavBase64: btoa(binary) })
          .catch(() => undefined);
      },
      onError: (reason) => {
        chrome.runtime.sendMessage({ type: 'offscreen:live-error', reason }).catch(() => undefined);
      },
    })
      .then(() => sendResponse({ ok: true }))
      .catch((err: Error) => sendResponse({ ok: false, reason: err.message }));
    return true;
  }
  if (message?.type === 'offscreen:live-stop') {
    void stopCapture();
    sendResponse({ ok: true });
    return false;
  }
  return undefined;
});

/**
 * Score one window locally and send the worker the same shape the backend endpoint returns,
 * so the risk engine above does not need to know which path produced it.
 *
 * The VAD runs first, exactly as the service does it — a call is mostly silence, and a guard
 * that scores silence cries wolf. A window the chain cannot score is reported unscored; there
 * is no branch here that produces a number without the model.
 */
async function scoreWindowOnDevice(samples: Float32Array, sampleRate: number): Promise<void> {
  const vad = speechMetrics(samples);
  const base = {
    type: 'offscreen:live-score',
    speechDetected: vad.isSpeech,
    windowSeconds: Math.round((samples.length / sampleRate) * 100) / 100,
    vad: { rmsDbfs: vad.rmsDbfs, spectralFlatness: vad.spectralFlatness, reason: vad.reason },
    modelSource: AUDIO_MODEL_SOURCE,
  };

  if (!vad.isSpeech) {
    chrome.runtime
      .sendMessage({ ...base, fakeProbability: null, notes: [`not scored: ${vad.reason}`] })
      .catch(() => undefined);
    return;
  }

  let probability: number | null = null;
  let failure: string | null = null;
  try {
    probability = await scoreSamples(samples);
  } catch (err) {
    failure = String((err as Error)?.message ?? err);
  }

  if (probability === null) {
    // The gate closed, or the chain threw. Say so and score nothing — the worker decides
    // whether to fall back, and a missing number must never read as a low one.
    chrome.runtime
      .sendMessage({
        ...base,
        fakeProbability: null,
        onDeviceFailed: true,
        notes: [failure ?? 'the on-device chain declined this window'],
      })
      .catch(() => undefined);
    return;
  }

  chrome.runtime
    .sendMessage({
      ...base,
      fakeProbability: Math.round(100 * probability),
      notes: [
        'on-device v2 chain (preproc.onnx -> CNN), parity self-test passed',
        'uncalibrated — treat as a ranking, not a probability',
      ],
    })
    .catch(() => undefined);
}
