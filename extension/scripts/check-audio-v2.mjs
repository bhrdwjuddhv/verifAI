/**
 * The on-device audio gate, checked end to end.
 *
 *   node --experimental-strip-types scripts/check-audio-v2.mjs
 *
 * Three things can silently break on-device voice, and none of them is visible in the UI:
 *
 *   1. the self-test vector is reconstructed wrongly, so the gate measures the wrong input
 *      and "passes" while proving nothing,
 *   2. the chain is bundled without its .onnx.data sidecars, or with them renamed, so it
 *      never loads on a user's machine although it loads here,
 *   3. the ported VAD drifts from the service's, so the two disagree about which windows
 *      count and it reads as a model disagreement.
 *
 * This runs the real bundled graphs through onnxruntime-node. That is not the browser's wasm
 * build, so it does not replace the runtime self-test — it proves the fixture, the bundle and
 * the maths are right, and leaves the runtime gate to prove this machine is.
 */

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import ort from 'onnxruntime-node';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.join(here, '..');
const dir = path.join(repo, 'dist', 'models', 'audio');

if (!fs.existsSync(dir)) {
  console.error(`  ${dir} not found — run \`npm run build\` first.`);
  process.exit(1);
}

// --- 1. the bundle -------------------------------------------------------------------------
// The sidecars are named from inside the graph, so a rename here breaks loading with a
// message about a missing tensor rather than a missing file.
for (const name of [
  'preproc.onnx',
  'preproc.onnx.data',
  'audio_detector.onnx',
  'audio_detector.onnx.data',
  'audio_selftest.json',
]) {
  assert.ok(fs.existsSync(path.join(dir, name)), `${name} must be bundled`);
}

const meta = JSON.parse(fs.readFileSync(path.join(dir, 'audio_selftest.json'), 'utf8'));
assert.equal(meta.pipeline, 'preproc.onnx -> audio_detector.onnx');
assert.ok(meta.tol > 0 && meta.tol <= 0.01, `tol ${meta.tol} is not a meaningful tolerance`);

// --- 2. the fixture ------------------------------------------------------------------------
const { reconstructSelftestAudio } = await import('../src/shared/audio-selftest.ts');

const { samples, reason } = reconstructSelftestAudio(meta);
assert.ok(samples, `the shipped self-test must reconstruct: ${reason}`);
assert.equal(samples.length, meta.num_samples);

let checksum = 0;
for (const s of samples) checksum += Math.abs(s);
assert.ok(
  Math.abs(checksum - meta.full_audio_checksum) < 1e-2,
  `reconstruction checksum ${checksum} != ${meta.full_audio_checksum} — the TS port has ` +
    'drifted from scripts/common/audio_v2.py'
);

// A truncated vector whose prefix does not match must fail closed, not be "reconstructed".
const wrong = reconstructSelftestAudio({ ...meta, audio: new Array(100).fill(0) });
assert.equal(wrong.samples, null, 'a wrong prefix must not reconstruct');
assert.match(wrong.reason ?? '', /prefix/);

// No vector at all is unavailable, never a silent pass.
assert.equal(reconstructSelftestAudio({ num_samples: 0, audio: [] }).samples, null);

// A complete vector passes straight through.
const whole = reconstructSelftestAudio({
  sample_rate: 16000, duration_sec: 0.01, num_samples: 4, audio: [0.1, 0.2, 0.3, 0.4],
});
assert.equal(whole.reason, null);
assert.equal(whole.samples.length, 4);

// --- 3. the chain --------------------------------------------------------------------------
const preproc = await ort.InferenceSession.create(path.join(dir, 'preproc.onnx'));
const cnn = await ort.InferenceSession.create(path.join(dir, 'audio_detector.onnx'));

const spec = await preproc.run({
  [preproc.inputNames[0]]: new ort.Tensor('float32', samples, [1, samples.length]),
});
const image = spec[preproc.outputNames[0]];
assert.deepEqual([...image.dims], [1, 3, 224, 224], 'preproc must emit the CNN input shape');

const out = await cnn.run({ [cnn.inputNames[0]]: image });
const logits = out[cnn.outputNames[0]].data;
const max = Math.max(logits[0], logits[1]);
const a = Math.exp(logits[0] - max);
const observed = a / (a + Math.exp(logits[1] - max));
const delta = Math.abs(observed - meta.expected_prob);

assert.ok(
  delta <= meta.tol,
  `chain observed ${observed} vs expected ${meta.expected_prob} (delta ${delta} > tol ${meta.tol})`
);

// --- 4. the VAD ----------------------------------------------------------------------------
// Reference values from numpy on the same signals (scripts/inference_server.py speech_metrics).
// MINSTD is used for the noise cases because every product stays under 2^53, so Python and
// JavaScript generate the identical sequence — np.random would not be reproducible here.
const { speechMetrics, spectralFlatness } = await import('../src/shared/vad.ts');

const SR = 16000;
const N = 3 * SR;

function minstd(n, amp) {
  const out = new Float32Array(n);
  let x = 42;
  for (let i = 0; i < n; i++) {
    x = (16807 * x) % 2147483647;
    out[i] = amp * (2 * (x / 2147483647) - 1);
  }
  return out;
}

function tones(freqs, gain) {
  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    let v = 0;
    freqs.forEach((f, k) => { v += Math.sin((2 * Math.PI * f * i) / SR) / (k + 1); });
    out[i] = v * gain;
  }
  return out;
}

const cases = [
  // name,              signal,                    dbfs,    flatness, speech
  ['tone_440', tones([440], 0.5), -9.03, 0.0, true],
  ['voiced', tones([150, 300, 450, 600], 0.2), -15.46, 0.0, true],
  ['minstd_noise', minstd(N, 0.1), -24.79, 0.55679, false],
  ['silence', new Float32Array(N), -240.0, null, false],
  ['quiet_noise', minstd(N, 0.0005), -70.81, null, false],
];

for (const [name, signal, dbfs, flatness, speech] of cases) {
  const r = speechMetrics(signal);
  assert.ok(Math.abs(r.rmsDbfs - dbfs) < 0.05, `${name}: dBFS ${r.rmsDbfs} != ${dbfs}`);
  assert.equal(r.isSpeech, speech, `${name}: isSpeech ${r.isSpeech} != ${speech} (${r.reason})`);
  if (flatness !== null) {
    const observedFlat = spectralFlatness(signal);
    assert.ok(
      Math.abs(observedFlat - flatness) < 1e-4,
      `${name}: flatness ${observedFlat} != numpy's ${flatness} on the same prefix`
    );
  }
  if (!speech) assert.ok(r.reason, `${name}: an unscored window must say why`);
}

// Too short to transform is unscored, not scored as quiet.
assert.equal(speechMetrics(new Float32Array(64)).reason, 'window too short');

console.log(
  `audio v2 selfcheck passed — chain delta ${delta.toExponential(1)} (tol ${meta.tol}), ` +
    `fixture checksum ok, VAD matches numpy on ${cases.length} signals`
);
