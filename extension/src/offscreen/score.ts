/**
 * On-device scoring: the `analyze()` of inference_server.py, plus the response assembly of
 * app/api/scan/route.ts, so the popup renders an on-device scan and a deep scan identically.
 *
 * The two sources it mirrors are named at each step. Where a detector is missing it abstains
 * and says so in `notes` — the same wording the service uses — rather than contributing a
 * number nothing measured.
 */

import { VERDICT_CONFIG, type VerdictCategory } from '@app/lib/verdict';
import type { MetadataSignal, ScanResult } from '../shared/scan-types';
import {
  FUSION_DEFAULTS,
  fakeIndex,
  fakeProbability,
  frequencyScore,
  fuse,
  sigmoid,
  softmax,
  verdictFor,
} from '../shared/detect.ts';
import { cropFace, flipWidth, resample, toArray, type RgbImage } from '../shared/preprocess.ts';
import {
  SALIENCY_GRID,
  encodeOverlay,
  occlusionBatch,
  saliencyFromDrops,
} from '../shared/saliency.ts';
import { resize, rgbaToRgb, toGray } from '../shared/resample.ts';
import * as yunet from '../shared/yunet.ts';
import { backend, faceModel, loadFailure, nprModel, outputData, tensor, yunetSession } from './session';

/** inference_server.py FACE_MARGIN. */
const FACE_MARGIN = 0.35;

const CATEGORY: Record<string, VerdictCategory> = {
  real: 'genuine',
  fake: 'manipulated',
  uncertain: 'uncertain',
};

export interface ScoreRequest {
  /** Base64 file bytes — chrome messaging is JSON, so binary cannot cross directly. */
  b64: string;
  filename: string;
  mime: string;
  /**
   * Compute the occlusion heatmap. Off by default: it costs 26 forward passes, so it is an
   * explicit action rather than something every scan pays for.
   */
  explain?: boolean;
}

export async function score(request: ScoreRequest): Promise<ScanResult> {
  const bytes = base64ToBytes(request.b64);
  const image = await decode(bytes, request.mime);
  const notes: string[] = [];

  const face = await faceModel();
  const npr = await nprModel();
  if (!face && !npr) {
    throw new Error(
      `No on-device model is bundled (face: ${loadFailure('face') ?? 'missing'}, npr: ${loadFailure('npr') ?? 'missing'}).`
    );
  }

  // --- face detection ------------------------------------------------------------------
  let crop: RgbImage | null = null;
  let faceDetected: boolean | null = null;
  const detector = await yunetSession();

  if (detector) {
    const input = yunet.buildInput(image);
    const outputs = await detector.run({
      input: tensor(input.tensor, [1, 3, input.height, input.width]),
    });
    const raw: Record<string, Float32Array> = {};
    for (const name of detector.outputNames) raw[name] = outputData(outputs[name]);

    const best = yunet.largest(
      yunet.toImageSpace(yunet.decode(raw, input.width, input.height), input, image)
    );
    // crop_face rejects anything under 0.90 even though the detector itself accepts 0.70.
    if (best && best.score >= yunet.ACCEPT_THRESHOLD) {
      crop = cropFace(image, best.box, FACE_MARGIN);
      faceDetected = crop !== null;
    } else {
      faceDetected = false;
    }
  }

  // --- NPR: whole frame, no face needed ------------------------------------------------
  let nprPct: number | null = null;
  if (npr) {
    const input = toArray(image, npr.meta.imgSize, npr.meta.mean, npr.meta.std, true);
    const output = await npr.session.run({
      [npr.session.inputNames[0]]: tensor(input, [1, 3, npr.meta.imgSize, npr.meta.imgSize]),
    });
    const logit = outputData(output[npr.session.outputNames[0]])[0];
    nprPct = Math.round(100 * sigmoid(logit)); // official convention: 1 = fake
  }

  // --- face classifier: only where it applies ------------------------------------------
  let facePct: number | null = null;
  let heatmap: string | null = null;
  if (face && !(face.meta.expectsFace && faceDetected === false)) {
    const target = face.meta.expectsFace && crop ? crop : image;
    if (face.meta.expectsFace && !detector) {
      notes.push('face detector unavailable; ran the full frame through a face model');
    }
    facePct = Math.round(100 * (await runFace(face, target)));
    if (request.explain) {
      const overlay = await explainFace(face, target);
      heatmap = overlay ? await toPngDataUrl(overlay) : null;
    }
  } else if (!face) {
    notes.push(`face classifier unavailable (${loadFailure('face') ?? 'not bundled'})`);
  } else {
    notes.push('no face detected — the face classifier does not apply, so it did not vote');
  }

  // --- descriptive statistic, weight 0 --------------------------------------------------
  const gray = resize(toGray(image.data, image.width, image.height), image.width, image.height, 256, 256, 1, 'bilinear');
  const grayFloat = new Float32Array(gray.length);
  for (let i = 0; i < gray.length; i++) grayFloat[i] = gray[i] / 255;
  const frequency = frequencyScore(grayFloat);

  const { fused, used } = fuse({ face: facePct, npr: nprPct, frequency }, FUSION_DEFAULTS);

  const modelSource =
    [facePct !== null && face ? `onnx:${face.meta.source}` : null, nprPct !== null && npr ? `onnx:${npr.meta.source}` : null]
      .filter(Boolean)
      .join(' + ') || 'none';

  const metadata = readMetadataSignal(bytes);
  const signals = {
    modelScore: facePct,
    nprScore: nprPct,
    audioScore: null,
    frequencyScore: frequency,
    faceDetected,
  };

  if (face && !face.meta.calibrated) {
    notes.push('confidence is uncalibrated — treat it as a ranking, not a probability');
  }
  if (request.explain && facePct !== null) {
    notes.push(
      heatmap
        ? 'explanation is occlusion saliency, not Grad-CAM: each patch was hidden in turn and the drop in P(fake) measured. It shows what the model depended on, not what it activated on.'
        : 'no explanation heatmap available: hiding any one region left the score unchanged, so there is nothing honest to draw'
    );
  }
  if (face?.meta.quantized || npr?.meta.quantized) {
    notes.push('int8-quantized build; scores can differ from the full-precision model by a point or two');
  }
  if (nprPct === null) {
    notes.push(
      `NPR whole-image detector unavailable (${loadFailure('npr') ?? 'not bundled'}) — a fully generated image may not be caught by the face model alone`
    );
  }
  if (!detector) {
    notes.push(`face detector unavailable (${loadFailure('yunet') ?? 'not bundled'})`);
  }
  if (Object.keys(used).length > 1 && facePct !== null && nprPct !== null && Math.abs(facePct - nprPct) > 50) {
    notes.push(
      `the detectors disagree sharply (face ${facePct}% vs NPR ${nprPct}%); the fused score sits between them, which is why this may read as uncertain`
    );
  }
  notes.push(`scored on this device (${await backend()}); nothing was uploaded`);

  const category: VerdictCategory = fused === null ? 'uncertain' : CATEGORY[verdictFor(fused).verdict];
  const base = VERDICT_CONFIG[category];
  const detectorCount = Object.keys(used).length;
  const across = detectorCount > 1 ? ` across ${detectorCount} detectors` : '';

  const laymanSummary =
    fused === null
      ? `No verdict: ${notes[0] ?? 'no detector applied to this file'}.`
      : category === 'manipulated'
        ? `The detectors scored this ${fused}% likely AI-generated or manipulated${across}.`
        : category === 'genuine'
          ? `The detectors scored this ${100 - fused}% likely real${across}.`
          : `The combined score is ${fused}% likely AI-generated — inside the uncertain band (30–70%), so it is not calling it either way.`;

  return {
    id: `VRF-${Math.floor(100000 + Math.random() * 900000)}`,
    filename: request.filename,
    fileType: 'image',
    fileSize: `${(bytes.byteLength / (1024 * 1024)).toFixed(1)} MB`,
    score: fused === null ? null : 100 - fused,
    confidence: fused === null ? 0 : Math.round(verdictFor(fused).confidence),
    verdict: { ...base, laymanSummary },
    reasons: buildReasons({ signals, fused, used, modelSource, metadata, notes }),
    signals,
    metadata,
    modelSource,
    fusion: { weights: FUSION_DEFAULTS, used },
    heatmap,
    notes,
    timestamp: new Date().toISOString(),
  };
}

type Face = NonNullable<Awaited<ReturnType<typeof faceModel>>>;

/** The service's VERIFAI_BATCH: 25 occlusion cells at 224px is ~15MB of float32 in one go. */
const CHUNK = 8;

/**
 * One batch of images -> one row of class probabilities each, temperature-scaled.
 *
 * The direct equivalent of the `run` closure the ONNX engine builds, and the reason the
 * verdict and the explanation below cannot drift apart: they call the same thing.
 */
async function runProbs(face: Face, batch: Float32Array, count: number): Promise<number[][]> {
  const size = face.meta.imgSize;
  const classes = face.meta.labels.length;
  const perImage = 3 * size * size;
  const rows: number[][] = [];

  for (let start = 0; start < count; start += CHUNK) {
    const n = Math.min(CHUNK, count - start);
    const slice = batch.subarray(start * perImage, (start + n) * perImage);
    const output = await face.session.run({
      [face.session.inputNames[0]]: tensor(new Float32Array(slice), [n, 3, size, size]),
    });
    const logits = outputData(output[face.session.outputNames[0]]);
    for (let row = 0; row < n; row++) {
      rows.push(softmax(logits.subarray(row * classes, (row + 1) * classes), face.meta.temperature));
    }
  }
  return rows;
}

/** Flip-TTA and temperature exactly as the ONNX path of inference_server.py runs them. */
async function runFace(face: Face, target: RgbImage): Promise<number> {
  const size = face.meta.imgSize;
  const straight = toArray(target, size, face.meta.mean, face.meta.std, false);
  const flipped = flipWidth(straight, size);

  const batch = new Float32Array(straight.length * 2);
  batch.set(straight, 0);
  batch.set(flipped, straight.length);

  const probs = await runProbs(face, batch, 2);
  const classes = face.meta.labels.length;
  const averaged = new Array(classes).fill(0);
  for (const row of probs) {
    for (let c = 0; c < classes; c++) averaged[c] += row[c] / 2;
  }
  return fakeProbability(averaged, face.meta.labels);
}

/**
 * Occlusion saliency over the same input the classifier scored.
 *
 * The baseline is `run(arr)` on the unflipped image, not the flip-TTA average behind the
 * reported score. That is what the service does, and matching it matters more than making
 * the two numbers agree to the decimal.
 */
async function explainFace(face: Face, target: RgbImage): Promise<RgbImage | null> {
  const size = face.meta.imgSize;
  const fake = fakeIndex(face.meta.labels);
  const arr = toArray(target, size, face.meta.mean, face.meta.std, false);

  const baseline = (await runProbs(face, arr, 1))[0][fake];
  const cells = occlusionBatch(arr, size, SALIENCY_GRID);
  const occluded = (await runProbs(face, cells, SALIENCY_GRID * SALIENCY_GRID)).map((row) => row[fake]);

  const cam = saliencyFromDrops(baseline, occluded, SALIENCY_GRID);
  if (!cam) return null;

  return encodeOverlay(cam, SALIENCY_GRID, resample(target, size, size, 'bilinear'));
}

/** RGB buffer -> a `data:image/png;base64` URL, the shape the popup and the app both expect. */
async function toPngDataUrl(img: RgbImage): Promise<string> {
  const pixels = img.width * img.height;
  const rgba = new Uint8ClampedArray(pixels * 4);
  for (let i = 0, s = 0, d = 0; i < pixels; i++, s += 3, d += 4) {
    rgba[d] = img.data[s];
    rgba[d + 1] = img.data[s + 1];
    rgba[d + 2] = img.data[s + 2];
    rgba[d + 3] = 255;
  }

  const canvas = new OffscreenCanvas(img.width, img.height);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('no 2d context for the overlay');
  ctx.putImageData(new ImageData(rgba, img.width, img.height), 0, 0);

  const blob = await canvas.convertToBlob({ type: 'image/png' });
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

/**
 * The reason list from app/api/scan/route.ts.
 *
 * Duplicated rather than imported: that logic lives inside the route handler, and pulling it
 * out would mean editing a live API file. If the route's wording changes, this needs the same
 * edit — the parity harness compares these strings.
 */
function buildReasons(input: {
  signals: ScanResult['signals'];
  fused: number | null;
  used: Record<string, number>;
  modelSource: string;
  metadata: MetadataSignal;
  notes: string[];
}): string[] {
  const { signals, fused, used, modelSource, metadata, notes } = input;
  const reasons: string[] = [];

  if (signals.faceDetected === true) {
    reasons.push('A face was detected; the model ran on the cropped face region.');
  } else if (signals.faceDetected === false) {
    reasons.push('No face was detected in this image.');
  } else {
    reasons.push('Face detection was unavailable on the model service.');
  }

  if (signals.modelScore !== null) {
    reasons.push(`Face classifier: ${signals.modelScore}% likely swapped or manipulated.`);
  } else {
    reasons.push('Face classifier did not apply to this image, so it did not vote.');
  }

  if (signals.nprScore !== null && signals.nprScore !== undefined) {
    reasons.push(
      `Whole-image AI-generation detector (NPR): ${signals.nprScore}% likely generated. This one catches fully synthetic images, including generated faces.`
    );
  }

  if (fused !== null) {
    const blend = Object.entries(used)
      .filter(([, w]) => w > 0)
      .map(([k, w]) => `${k} ${w}`)
      .join(', ');
    reasons.push(
      `Combined score: ${fused}% likely AI-generated or manipulated${blend ? ` (weights: ${blend})` : ''}. Thresholds: above 70 fake, below 30 real, in between uncertain.`
    );
  }

  if (signals.frequencyScore !== null) {
    reasons.push(
      `High-frequency energy share: ${signals.frequencyScore}%. A descriptive statistic only — sharp real photos score high too, and it does not affect the verdict.`
    );
  }

  reasons.push(describeMetadata(metadata));
  reasons.push(`Model used: ${modelSource}.`);
  reasons.push(...notes);
  return reasons;
}

/** Ported from lib/models/model_service.ts, which cannot be imported without pulling in zustand. */
function readMetadataSignal(bytes: Uint8Array): MetadataSignal {
  const head = bytes.subarray(0, Math.min(bytes.byteLength, 256 * 1024));
  const text = new TextDecoder('latin1').decode(head);
  return {
    c2paManifestPresent: text.includes('c2pa') || text.includes('jumb'),
    exifPresent: text.includes('Exif\u0000\u0000') || text.includes('eXIf'),
  };
}

function describeMetadata(m: MetadataSignal): string {
  const parts = [
    m.c2paManifestPresent ? 'a C2PA manifest is present' : 'no C2PA manifest',
    m.exifPresent ? 'EXIF metadata is present' : 'no EXIF metadata',
  ];
  return `Metadata signal (supporting only, not part of the verdict): ${parts.join(', ')}. Metadata is stripped by most platforms and can be forged, so neither presence nor absence proves anything.`;
}

/**
 * `imageOrientation: 'none'` is not cosmetic: the browser's default is to apply the EXIF
 * rotation, and PIL's `Image.open` does not. Without this, a phone photo is scored rotated
 * on-device and upright on the server, and the two disagree for a reason no one would guess.
 */
async function decode(bytes: Uint8Array, mime: string): Promise<RgbImage> {
  const bitmap = await createImageBitmap(new Blob([bytes as BlobPart], { type: mime }), {
    imageOrientation: 'none',
    premultiplyAlpha: 'none',
    colorSpaceConversion: 'none',
  });
  try {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('no 2d context in the offscreen document');
    ctx.drawImage(bitmap, 0, 0);
    const { data } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    return { data: rgbaToRgb(data, bitmap.width * bitmap.height), width: bitmap.width, height: bitmap.height };
  } finally {
    bitmap.close();
  }
}

function base64ToBytes(b64: string): Uint8Array {
  const binary = atob(b64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}
