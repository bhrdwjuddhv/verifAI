/**
 * On-device voice-clone detection: the v2 chain, and the gate that decides whether to use it.
 *
 *     raw samples -> preproc.onnx -> audio_detector.onnx -> P(fake)
 *
 * The reason this exists at all is that librosa's mel-spectrogram cannot be reproduced in a
 * browser with any confidence, so the previous on-device audio plan was unsafe by
 * construction — a subtly wrong spectrogram yields confident nonsense with nothing in the
 * logs. v2 exports the preprocessing itself to ONNX, so the browser runs the same graph the
 * server does. `selftest()` is what proves that claim on this machine, at runtime, instead of
 * assuming it holds.
 *
 * Nothing here ever invents a number. If the chain will not load, or the self-test misses
 * `expected_prob` by more than `tol`, this reports why and the caller keeps using the backend.
 */

import * as ort from 'onnxruntime-web';
import { backend } from './session';
import { reconstructSelftestAudio, type AudioSelftestMeta } from '../shared/audio-selftest';

const DIR = 'models/audio/';

export interface AudioSelftest {
  status: 'pass' | 'fail' | 'unavailable';
  observed?: number;
  expected?: number;
  tol?: number;
  delta?: number;
  /** Which execution provider actually produced `observed`. */
  ep?: string;
  reason: string | null;
}

interface Chain {
  preproc: ort.InferenceSession;
  cnn: ort.InferenceSession;
  meta: AudioSelftestMeta;
  ep: string;
}

/**
 * Both graphs carry their weights in a `.onnx.data` sidecar, referenced from inside the graph
 * by bare filename — so the names must stay exactly as exported, and each has to be handed to
 * ORT explicitly because there is no directory to discover it in.
 */
async function createSession(file: string, ep: string): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create(chrome.runtime.getURL(DIR + file), {
    executionProviders: ep === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'],
    graphOptimizationLevel: 'all',
    externalData: [{ path: `${file}.data`, data: chrome.runtime.getURL(`${DIR}${file}.data`) }],
  });
}

async function loadChain(ep: string): Promise<Chain> {
  const res = await fetch(chrome.runtime.getURL(DIR + 'audio_selftest.json'));
  if (!res.ok) throw new Error(`audio_selftest.json: ${res.status}`);
  const meta = (await res.json()) as AudioSelftestMeta;
  const [preproc, cnn] = await Promise.all([
    createSession('preproc.onnx', ep),
    createSession('audio_detector.onnx', ep),
  ]);
  return { preproc, cnn, meta, ep };
}

function describe(err: unknown): string {
  const message = err instanceof Error ? err.message : String(err);
  return /404|not found|failed to fetch/i.test(message)
    ? 'the v2 audio chain is not bundled in this build'
    : message;
}

/** raw samples -> preproc -> cnn -> P(fake). Index 0 = Fake, per the v2 detector.json labels. */
async function probability(chain: Chain, samples: Float32Array): Promise<number> {
  const audio = new ort.Tensor('float32', samples, [1, samples.length]);
  const spec = await chain.preproc.run({ [chain.preproc.inputNames[0]]: audio });
  const image = spec[chain.preproc.outputNames[0]];
  const out = await chain.cnn.run({ [chain.cnn.inputNames[0]]: image });
  const logits = out[chain.cnn.outputNames[0]].data as Float32Array;
  const max = Math.max(logits[0], logits[1]);
  const a = Math.exp(logits[0] - max);
  const b = Math.exp(logits[1] - max);
  return a / (a + b);
}

let chain: Chain | null = null;
let gate: AudioSelftest | null = null;
let running: Promise<AudioSelftest> | null = null;

/**
 * Run the parity self-test and cache the outcome. WebGPU is tried first as the prompt asks,
 * but it is not trusted on faith: GPU float32 reduces in a different order, and the tolerance
 * here is 1e-3. If WebGPU misses it, wasm gets a turn before giving up — a machine with a
 * fussy GPU should fall back to slower-but-verified, not to the network.
 */
export function selftest(): Promise<AudioSelftest> {
  if (gate) return Promise.resolve(gate);
  if (running) return running;
  const attempt = (async (): Promise<AudioSelftest> => {
    const preferred = await backend();
    const providers = preferred === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];
    let last: AudioSelftest = { status: 'unavailable', reason: 'no execution provider tried' };

    for (const ep of providers) {
      let candidate: Chain;
      try {
        candidate = await loadChain(ep);
      } catch (err) {
        last = { status: 'unavailable', ep, reason: describe(err) };
        continue;
      }

      const { samples, reason } = reconstructSelftestAudio(candidate.meta);
      if (!samples) {
        last = { status: 'unavailable', ep, reason };
        break; // a bad fixture is not an execution-provider problem; retrying cannot help
      }

      const expected = Number(candidate.meta.expected_prob ?? 0.5);
      const tol = Number(candidate.meta.tol ?? 1e-3);
      let observed: number;
      try {
        observed = await probability(candidate, samples);
      } catch (err) {
        last = { status: 'fail', ep, reason: `chain raised ${describe(err)}` };
        continue;
      }

      const delta = Math.abs(observed - expected);
      if (delta <= tol) {
        chain = candidate;
        return { status: 'pass' as const, observed, expected, tol, delta, ep, reason: null };
      }
      last = {
        status: 'fail',
        observed,
        expected,
        tol,
        delta,
        ep,
        reason: `${ep}: observed ${observed.toFixed(5)} vs expected ${expected.toFixed(5)} (tol ${tol})`,
      };
    }
    return last;
  })().then((result) => {
    gate = result;
    running = null;
    return result;
  });
  running = attempt;
  return attempt;
}

/**
 * Score one window on-device, or null if the gate is not open.
 *
 * Null means "use the backend", never "assume real". The caller must not treat it as a score.
 */
export async function scoreSamples(samples: Float32Array): Promise<number | null> {
  const result = await selftest();
  if (result.status !== 'pass' || !chain) return null;
  return probability(chain, samples);
}

export const MODEL_SOURCE = 'onnx:audio_v2(preproc.onnx -> audio_detector.onnx)';
