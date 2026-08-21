/**
 * ONNX Runtime setup and session cache.
 *
 * Lives in the offscreen document because that is the only extension context that has both
 * `navigator.gpu` and a lifetime longer than the service worker's ~30s idle eviction. The
 * worker orchestrates; nothing here ever runs there.
 *
 * Every path is bundled: `wasmPaths` points at files copied into the package at build time,
 * because MV3 refuses to execute remote code and a CDN fallback would fail silently at the
 * worst moment.
 */

// Types come from the package root; the build script aliases this specifier to the
// non-bundled WebGPU build, because the default entry carries an `eval` that MV3's
// `script-src 'self'` forbids. See scripts/build.mjs.
import * as ort from 'onnxruntime-web';

/** models/face/detector.json, shipped verbatim next to the weights. */
export interface FaceMeta {
  labels: string[];
  imgSize: number;
  mean: number[];
  std: number[];
  expectsFace: boolean;
  temperature: number;
  source: string;
  calibrated: boolean;
  quantized: boolean;
}

export interface NprMeta {
  imgSize: number;
  mean: number[];
  std: number[];
  source: string;
  quantized: boolean;
}

export type Backend = 'webgpu' | 'wasm';

let configured = false;
let backendPromise: Promise<Backend> | null = null;

async function configure(): Promise<Backend> {
  if (!configured) {
    ort.env.wasm.wasmPaths = chrome.runtime.getURL('ort/');
    // No cross-origin isolation in an extension page means no SharedArrayBuffer, and ORT
    // falls back to one thread. Spike S1 in PLAN.md exists to confirm or refute that; until
    // it does, ask for threads only when the platform actually granted isolation.
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    ort.env.logLevel = 'error';
    configured = true;
  }

  let adapter: unknown = null;
  try {
    adapter = await navigator.gpu?.requestAdapter();
  } catch {
    adapter = null;
  }
  return adapter ? 'webgpu' : 'wasm';
}

export function backend(): Promise<Backend> {
  backendPromise ??= configure();
  return backendPromise;
}

async function create(file: string): Promise<ort.InferenceSession> {
  const ep = await backend();
  return ort.InferenceSession.create(chrome.runtime.getURL(file), {
    executionProviders: ep === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'],
    graphOptimizationLevel: 'all',
  });
}

async function readJson<T>(file: string): Promise<T> {
  const res = await fetch(chrome.runtime.getURL(file));
  if (!res.ok) throw new Error(`${file}: ${res.status}`);
  return (await res.json()) as T;
}

export interface Loaded<M> {
  session: ort.InferenceSession;
  meta: M;
}

/**
 * Each loader resolves to null when its model is not in the build, and records why.
 *
 * Null is a real answer here: the service behaves the same way when a detector is missing —
 * it abstains and says so in `notes` rather than substituting a number.
 */
const failures: Record<string, string> = {};

export function loadFailure(key: string): string | undefined {
  return failures[key];
}

let facePromise: Promise<Loaded<FaceMeta> | null> | null = null;
let nprPromise: Promise<Loaded<NprMeta> | null> | null = null;
let yunetPromise: Promise<ort.InferenceSession | null> | null = null;

export function faceModel(): Promise<Loaded<FaceMeta> | null> {
  facePromise ??= (async () => {
    try {
      const meta = await readJson<FaceMeta>('models/detector.json');
      const session = await create('models/detector.onnx');
      return { session, meta };
    } catch (err) {
      failures.face = describe(err);
      return null;
    }
  })();
  return facePromise;
}

export function nprModel(): Promise<Loaded<NprMeta> | null> {
  nprPromise ??= (async () => {
    try {
      const meta = await readJson<NprMeta>('models/npr.json');
      const session = await create('models/npr.onnx');
      return { session, meta };
    } catch (err) {
      failures.npr = describe(err);
      return null;
    }
  })();
  return nprPromise;
}

export function yunetSession(): Promise<ort.InferenceSession | null> {
  yunetPromise ??= (async () => {
    try {
      return await create('models/yunet.onnx');
    } catch (err) {
      failures.yunet = describe(err);
      return null;
    }
  })();
  return yunetPromise;
}

function describe(err: unknown): string {
  const message = err instanceof Error ? err.message : String(err);
  return /404|not found|failed to fetch/i.test(message) ? 'not bundled in this build' : message;
}

/** NCHW float32 tensor from a flat array. */
export function tensor(data: Float32Array, dims: number[]): ort.Tensor {
  return new ort.Tensor('float32', data, dims);
}

export function outputData(value: ort.Tensor): Float32Array {
  return value.data as Float32Array;
}
