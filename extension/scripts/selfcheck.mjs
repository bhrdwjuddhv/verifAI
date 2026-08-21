/**
 * Exercises the on-device scoring stack under Node, against the real ONNX files.
 *
 * The counterpart to `python scripts/inference_server.py --selfcheck`: it does not prove the
 * models are accurate, it proves the maths around them is the maths the service uses. Every
 * assertion here corresponds to a bug that produces a plausible number rather than an error —
 * an unnormalised resample, an anchor grid indexed the wrong way, a missing detector dragging
 * the fused score toward 50.
 *
 * Runs the extension's actual source through Node's type stripping, so there is no second
 * implementation to keep in sync.
 *
 *   npm run selfcheck            (needs Node >= 22.18 for .ts imports)
 */

import ort from 'onnxruntime-node';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  fakeIndex,
  fakeProbability,
  frequencyScore,
  fuse,
  labelIsFake,
  softmax,
  verdictFor,
  FUSION_DEFAULTS,
} from '../src/shared/detect.ts';
import { resize, toGray } from '../src/shared/resample.ts';
import { cropFace, flipWidth, toArray } from '../src/shared/preprocess.ts';
import * as yunet from '../src/shared/yunet.ts';
import {
  SALIENCY_GRID,
  encodeOverlay,
  occlusionBatch,
  saliencyFromDrops,
} from '../src/shared/saliency.ts';

const root = fileURLToPath(new URL('..', import.meta.url));
const models = path.join(root, '..', 'models');

let checks = 0;
let failures = 0;

function assert(condition, message) {
  checks++;
  if (condition) return;
  failures++;
  console.error(`  FAIL  ${message}`);
}

function near(actual, expected, tolerance, message) {
  assert(Math.abs(actual - expected) <= tolerance, `${message} (got ${actual}, want ${expected}±${tolerance})`);
}

function section(name) {
  console.log(`\n${name}`);
}

// ---------------------------------------------------------------------------------------

function solid(width, height, [r, g, b]) {
  const data = new Uint8Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    data[i * 3] = r;
    data[i * 3 + 1] = g;
    data[i * 3 + 2] = b;
  }
  return { data, width, height };
}

function checkerboard(size, cell = 1) {
  const data = new Uint8Array(size * size * 3);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const on = (Math.floor(x / cell) + Math.floor(y / cell)) % 2 === 0 ? 255 : 0;
      const i = (y * size + x) * 3;
      data[i] = data[i + 1] = data[i + 2] = on;
    }
  }
  return { data, width: size, height: size };
}

// ---------------------------------------------------------------------------------------

section('resampling (Pillow parity)');
{
  const img = checkerboard(64, 4);
  const same = resize(img.data, 64, 64, 64, 64, 3, 'bilinear');
  assert(
    same.every((v, i) => v === img.data[i]),
    'resize to the same size must be the identity — a shifted sampling grid shows up here first'
  );

  const flat = solid(97, 33, [17, 200, 90]);
  const shrunk = resize(flat.data, 97, 33, 24, 24, 3, 'bilinear');
  assert(
    [...shrunk].every((v, i) => v === [17, 200, 90][i % 3]),
    'a uniform image must resample to the same uniform colour (catches unnormalised coefficients)'
  );

  // The antialiasing test. A naive bilinear samples one texel and returns 0 or 255; Pillow
  // scales the filter support by the reduction factor and returns the mean.
  const fine = checkerboard(512, 1);
  const one = resize(fine.data, 512, 512, 1, 1, 3, 'bilinear');
  near(one[0], 127, 2, 'a 1px checkerboard downscaled to 1x1 must average to mid-grey');

  const gray = toGray(solid(4, 4, [255, 255, 255]).data, 4, 4);
  assert(gray[0] === 255, 'white must convert to L=255');
  const midGray = toGray(solid(1, 1, [10, 200, 30]).data, 1, 1);
  near(midGray[0], (10 * 19595 + 200 * 38470 + 30 * 7471 + 0x8000) >> 16, 0, 'convert("L") uses Pillow fixed-point luma');
}

section('crop geometry (preprocess_faces.crop_face)');
{
  const img = solid(400, 300, [0, 0, 0]);

  const centred = cropFace(img, { x1: 180, y1: 130, x2: 220, y2: 170 }, 0.35);
  assert(centred !== null, 'a centred 40px face must produce a crop');
  assert(centred.width === centred.height, `crop must be square, got ${centred?.width}x${centred?.height}`);
  assert(
    centred.width >= 52 && centred.width <= 56,
    `40px face +35% margin should be ~54px, got ${centred.width}`
  );

  const tiny = cropFace(img, { x1: 195, y1: 145, x2: 210, y2: 160 }, 0.35);
  assert(tiny === null, 'a face under the 24px half-side floor must be rejected, not scaled up');

  const edge = cropFace(img, { x1: 0, y1: 0, x2: 120, y2: 120 }, 0.35);
  assert(edge !== null && edge.width === edge.height, 'a face on the edge must clamp and stay square');
}

section('fusion and verdict bands');
{
  const both = fuse({ face: 90, npr: 70 }, FUSION_DEFAULTS);
  near(both.fused, 80, 0, 'two detectors at 0.5/0.5 average');

  const faceOnly = fuse({ face: 90, npr: null }, FUSION_DEFAULTS);
  near(faceOnly.fused, 90, 0, 'a missing detector must renormalise away, not vote 50');
  assert(!('npr' in faceOnly.used), 'a missing detector must not appear as a voter');

  const zeroWeight = fuse({ frequency: 99 }, FUSION_DEFAULTS);
  assert(zeroWeight.fused === null, 'a weight-0 signal alone must yield no verdict');

  assert(fuse({}, FUSION_DEFAULTS).fused === null, 'no signals means no fused score');

  assert(verdictFor(71).verdict === 'fake', '>70 is fake');
  assert(verdictFor(70).verdict === 'uncertain', 'exactly 70 stays uncertain');
  assert(verdictFor(29).verdict === 'real', '<30 is real');
  assert(verdictFor(30).verdict === 'uncertain', 'exactly 30 stays uncertain');
  near(verdictFor(10).confidence, 90, 0, 'confidence is in the reported verdict, not in "fake"');
}

section('label mapping');
{
  assert(labelIsFake('Fake') === true, '"Fake" is fake');
  assert(labelIsFake('Realism') === false, '"Realism" is real, and must not match "fake"');
  assert(labelIsFake('real_vs_fake') === null, 'a label naming both sides must commit to neither');
  assert(fakeIndex(['Fake', 'Real']) === 0, 'fake index comes from the name, not the position');
  assert(fakeIndex(['Real', 'Deepfake']) === 1, 'and it follows the name when the order flips');
  near(fakeProbability([0.25, 0.75], ['Fake', 'Real']), 0.25, 1e-9, 'P(fake) renormalises over named classes');
  near(softmax([2, 2], 1).reduce((a, b) => a + b, 0), 1, 1e-9, 'softmax sums to one');
  assert(softmax([1, 0], 0.5)[0] > softmax([1, 0], 1)[0], 'a temperature below 1 sharpens the distribution');
}

section('frequency score');
{
  const flat = new Float32Array(256 * 256).fill(0.5);
  near(frequencyScore(flat), 0, 0, 'a flat image has no spectral energy at all');

  const fine = new Float32Array(256 * 256);
  for (let y = 0; y < 256; y++) for (let x = 0; x < 256; x++) fine[y * 256 + x] = (x + y) % 2 === 0 ? 1 : 0;
  const smooth = new Float32Array(256 * 256);
  for (let y = 0; y < 256; y++) for (let x = 0; x < 256; x++) smooth[y * 256 + x] = x / 255;

  assert(
    frequencyScore(fine) > frequencyScore(smooth),
    `a 1px checkerboard must score above a gradient (${frequencyScore(fine)} vs ${frequencyScore(smooth)})`
  );
  near(frequencyScore(fine), 100, 2, 'a Nyquist checkerboard puts nearly all energy above half-Nyquist');
}

section('occlusion saliency (common/xai.py parity)');
{
  const size = 8;
  const grid = 4;
  const arr = new Float32Array(3 * size * size).fill(1);
  const cells = occlusionBatch(arr, size, grid);

  assert(cells.length === grid * grid * 3 * size * size, 'one masked copy per grid cell');

  const perImage = 3 * size * size;
  const step = size / grid;
  let allOk = true;
  for (let k = 0; k < grid * grid; k++) {
    const copy = cells.subarray(k * perImage, (k + 1) * perImage);
    const zeros = [...copy].filter((v) => v === 0).length;
    // One patch, all three channels. Anything else means the mask is the wrong shape.
    if (zeros !== 3 * step * step) allOk = false;
  }
  assert(allOk, `each copy must zero exactly one ${step}x${step} patch across 3 channels`);

  // The masked region must be where the index says it is, not merely the right size.
  const third = cells.subarray(2 * perImage, 3 * perImage);
  const r = Math.floor(2 / grid);
  const c = 2 % grid;
  assert(third[r * step * size + c * step] === 0, 'patch k must sit at (row, col) = divmod(k, grid)');
  assert(third[0] === 1 || (r === 0 && c === 0), 'pixels outside the patch must be untouched');

  assert(saliencyFromDrops(0.5, new Array(grid * grid).fill(0.5), grid) === null,
    'a score that never moves must yield no heatmap, not a flat one');
  assert(saliencyFromDrops(0.5, new Array(grid * grid).fill(0.9), grid) === null,
    'negative drops clip to zero, so a score that only rises yields nothing');

  const drops = new Array(grid * grid).fill(0.5);
  drops[5] = 0.1; // hiding patch 5 cost 0.4
  drops[6] = 0.3; // hiding patch 6 cost 0.2
  const cam = saliencyFromDrops(0.5, drops, grid);
  assert(cam !== null, 'a real drop must produce a map');
  near(cam[5], 1, 1e-6, 'the largest drop normalises to the peak');
  near(cam[6], 0.5, 1e-6, 'a half-sized drop normalises to half');
  near(cam[0], 0, 1e-6, 'patches that changed nothing stay cold');
}

section('overlay rendering (encode_overlay parity)');
{
  const base = { data: new Uint8Array(32 * 24 * 3).fill(20), width: 32, height: 24 };
  const cold = encodeOverlay(new Float32Array(16), 4, base);
  const hot = encodeOverlay(new Float32Array(16).fill(1), 4, base);

  assert(cold.width === base.width && cold.height === base.height,
    `overlay must keep the image size, got ${cold.width}x${cold.height}`);
  assert(!cold.data.every((v, i) => v === hot.data[i]),
    'an all-cold and an all-hot map must not render identically');

  const mean = (img, channel) => {
    let sum = 0;
    for (let i = channel; i < img.data.length; i += 3) sum += img.data[i];
    return sum / (img.data.length / 3);
  };
  assert(mean(hot, 0) > mean(hot, 2), 'a hot map must push the overlay red, not blue');
  // weight = alpha * cam, so a cold map blends nothing: the base image must survive intact.
  assert(cold.data.every((v, i) => v === base.data[i]), 'a cold map must leave the image untouched');

  // Half-hot: the hot half must be redder than the cold half, or the map is not oriented.
  const half = new Float32Array(16);
  for (let row = 0; row < 4; row++) for (let col = 2; col < 4; col++) half[row * 4 + col] = 1;
  const split = encodeOverlay(half, 4, base);
  let left = 0;
  let right = 0;
  for (let y = 0; y < split.height; y++) {
    for (let x = 0; x < split.width; x++) {
      const red = split.data[(y * split.width + x) * 3];
      if (x < split.width / 2) left += red;
      else right += red;
    }
  }
  assert(right > left, 'the overlay must be spatially oriented, not merely coloured');
}

// ---------------------------------------------------------------------------------------

section('face classifier (real ONNX)');
// Newest first, matching scripts/build.mjs and the service's config.first_existing(). A
// selfcheck that tests a different file from the one shipped is worse than no selfcheck.
const faceOnnx = [
  path.join(models, 'face', 'detector_v2.onnx'),
  path.join(models, 'face', 'detector.onnx'),
].find((candidate) => fs.existsSync(candidate)) ?? path.join(models, 'face', 'detector.onnx');
if (!fs.existsSync(faceOnnx)) {
  console.log('  skipped — models/face/detector.onnx is absent');
} else {
  // The meta always sits beside its ONNX, so the pair cannot drift apart.
  const meta = JSON.parse(fs.readFileSync(faceOnnx.replace(/\.onnx$/, '.json'), 'utf8'));
  const session = await ort.InferenceSession.create(faceOnnx);

  const image = checkerboard(320, 8);
  const straight = toArray(image, meta.imgSize, meta.mean, meta.std, false);
  const flipped = flipWidth(straight, meta.imgSize);

  assert(straight.length === 3 * meta.imgSize * meta.imgSize, 'preprocessed tensor is 3xSxS');
  assert(
    flipWidth(flipped, meta.imgSize).every((v, i) => v === straight[i]),
    'flipping twice must return the original — catches a transposed width axis'
  );

  const batch = new Float32Array(straight.length * 2);
  batch.set(straight, 0);
  batch.set(flipped, straight.length);

  const output = await session.run({
    [session.inputNames[0]]: new ort.Tensor('float32', batch, [2, 3, meta.imgSize, meta.imgSize]),
  });
  const logits = output[session.outputNames[0]].data;
  const classes = meta.labels.length;
  assert(logits.length === 2 * classes, `flip-TTA batch must return ${2 * classes} logits, got ${logits.length}`);

  const averaged = new Array(classes).fill(0);
  for (let row = 0; row < 2; row++) {
    const probs = softmax(logits.subarray(row * classes, (row + 1) * classes), meta.temperature);
    near(probs.reduce((a, b) => a + b, 0), 1, 1e-6, 'each TTA row is a distribution');
    for (let c = 0; c < classes; c++) averaged[c] += probs[c] / 2;
  }

  const p = fakeProbability(averaged, meta.labels);
  assert(p >= 0 && p <= 1, `P(fake) must be a probability, got ${p}`);
  console.log(`  ok    ran ${meta.labels.join('/')} at T=${meta.temperature}, P(fake)=${(100 * p).toFixed(1)}%`);

  // The occlusion batch against the real graph. The pure tests cannot show that the exported
  // model accepts a batch of 25 — and if it does not, the explanation fails at the worst time.
  const fakeIdx = fakeIndex(meta.labels);
  const cells = occlusionBatch(straight, meta.imgSize, SALIENCY_GRID);
  const count = SALIENCY_GRID * SALIENCY_GRID;
  const occludedProbs = [];
  const occlusionStart = performance.now();
  for (let start = 0; start < count; start += 8) {
    const n = Math.min(8, count - start);
    const slice = cells.subarray(start * straight.length, (start + n) * straight.length);
    const out = await session.run({
      [session.inputNames[0]]: new ort.Tensor('float32', new Float32Array(slice), [n, 3, meta.imgSize, meta.imgSize]),
    });
    const raw = out[session.outputNames[0]].data;
    for (let row = 0; row < n; row++) {
      occludedProbs.push(softmax(raw.subarray(row * classes, (row + 1) * classes), meta.temperature)[fakeIdx]);
    }
  }
  assert(occludedProbs.length === count, `${count} occlusion cells must yield ${count} scores`);
  assert(occludedProbs.every((v) => v >= 0 && v <= 1), 'every occluded score must be a probability');
  const occlusionMs = Math.round(performance.now() - occlusionStart);
  const spread = Math.max(...occludedProbs) - Math.min(...occludedProbs);
  console.log(`  ok    ${count} occlusion cells through the real graph in ${occlusionMs}ms on CPU`);
  console.log(`        P(fake) spread across cells: ${(100 * spread).toFixed(2)}pp`);
}

section('YuNet face detector (real ONNX)');
const yunetOnnx = path.join(models, 'face_detection_yunet.onnx');
if (!fs.existsSync(yunetOnnx)) {
  console.log('  skipped — models/face_detection_yunet.onnx is absent');
} else {
  const session = await ort.InferenceSession.create(yunetOnnx);
  const image = solid(200, 150, [120, 130, 140]);
  const input = yunet.buildInput(image);

  assert(
    input.width === yunet.INPUT_SIZE && input.height === yunet.INPUT_SIZE,
    'the ONNX declares a fixed 640x640 input, so every image must letterbox into it'
  );
  near(input.inverseScale, 200 / yunet.INPUT_SIZE, 1e-6, 'inverse scale must map detections back to image space');

  const outputs = await session.run({
    input: new ort.Tensor('float32', input.tensor, [1, 3, input.height, input.width]),
  });

  // The assertion that matters: the anchor count per stride has to be exactly the feature
  // grid this decoder walks. If it is not, every decoded box lands somewhere plausible and
  // wrong, and nothing throws.
  const raw = {};
  for (const stride of [8, 16, 32]) {
    const cols = Math.floor(input.width / stride);
    const rows = Math.floor(input.height / stride);
    const cls = outputs[`cls_${stride}`];
    const bbox = outputs[`bbox_${stride}`];
    assert(
      cls.data.length === rows * cols,
      `cls_${stride} should hold ${rows * cols} anchors (${rows}x${cols}), got ${cls.data.length}`
    );
    assert(
      bbox.data.length === rows * cols * 4,
      `bbox_${stride} should hold ${rows * cols * 4} values, got ${bbox.data.length}`
    );
    raw[`cls_${stride}`] = cls.data;
    raw[`obj_${stride}`] = outputs[`obj_${stride}`].data;
    raw[`bbox_${stride}`] = bbox.data;
  }

  const found = yunet.decode(raw, input.width, input.height);
  assert(Array.isArray(found), 'decode must return a list');
  assert(
    found.every((d) => d.score >= yunet.SCORE_THRESHOLD && d.score <= 1),
    'every surviving detection must be inside the score threshold and 1'
  );
  console.log(`  ok    decoded a flat ${image.width}x${image.height} image into ${found.length} detection(s)`);
}

section('NPR whole-image detector (real ONNX)');
const nprOnnx = path.join(models, 'npr_detector.onnx');
if (!fs.existsSync(nprOnnx)) {
  console.log('  skipped — models/npr_detector.onnx is absent (run scripts/export_npr_no_torch.py)');
} else {
  const meta = JSON.parse(fs.readFileSync(path.join(models, 'npr_detector.json'), 'utf8'));
  const session = await ort.InferenceSession.create(nprOnnx);
  const size = meta.imgSize;

  // Loading at all is the graph validation the Python-side onnx.checker could not do here.
  assert(session.inputNames.length === 1, 'one input');
  assert(meta.sigmoid === true, 'NPR emits a single logit, read through a sigmoid');

  const run = async (data, n) => {
    const out = await session.run({
      [session.inputNames[0]]: new ort.Tensor('float32', data, [n, 3, size, size]),
    });
    return [...out[session.outputNames[0]].data];
  };

  const rng = (seed) => {
    // Deterministic LCG: the assertions below compare runs, so the inputs must be repeatable.
    let s = seed >>> 0;
    return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296) * 2 - 1;
  };

  const noise = (seed, n = 1) => {
    const next = rng(seed);
    const data = new Float32Array(n * 3 * size * size);
    for (let i = 0; i < data.length; i++) data[i] = next();
    return data;
  };

  const batched = await run(noise(1, 2), 2);
  assert(batched.length === 2, `dynamic batch must work, got ${batched.length} logits`);

  /**
   * NPR's defining property, and the only structural check that can catch a wrong residual.
   * An image that already IS a 2x nearest up-sample has no neighbouring-pixel residual, so
   * the trunk sees zeros and the logit is the same whatever the picture was.
   */
  const upsampled = (seed) => {
    const next = rng(seed);
    const half = size / 2;
    const small = new Float32Array(3 * half * half);
    for (let i = 0; i < small.length; i++) small[i] = next();
    const big = new Float32Array(3 * size * size);
    for (let c = 0; c < 3; c++) {
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          big[c * size * size + y * size + x] = small[c * half * half + (y >> 1) * half + (x >> 1)];
        }
      }
    }
    return big;
  };

  const flatA = (await run(upsampled(7), 1))[0];
  const flatB = (await run(upsampled(99), 1))[0];
  near(flatA, flatB, 1e-4,
    'two different 2x-upsampled images must give the same logit — the NPR residual zeroes both');

  const natural = (await run(noise(3), 1))[0];
  assert(Math.abs(natural - flatA) > 1e-5,
    `a natural image must not collapse to the degenerate logit (${natural} vs ${flatA})`);

  console.log(`  ok    ${meta.source}${meta.quantized ? ' (int8)' : ' (fp32)'}`);
  console.log(`        upsampled inputs collapse to ${flatA.toFixed(6)}, natural gives ${natural.toFixed(6)}`);
}

// ---------------------------------------------------------------------------------------

console.log(`\n${failures ? 'FAILED' : 'passed'} — ${checks - failures}/${checks} assertions\n`);
process.exit(failures ? 1 : 0);
