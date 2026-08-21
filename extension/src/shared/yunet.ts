/**
 * YuNet post-processing, ported from OpenCV's FaceDetectorYN.
 *
 * ONNX Runtime hands back twelve raw tensors (cls/obj/bbox/kps at strides 8, 16 and 32) and
 * nothing else — the anchor decode, the `sqrt(cls * obj)` score and the NMS all live inside
 * OpenCV's C++, so they have to be rebuilt here. Getting the anchor indexing wrong does not
 * throw; it produces plausible boxes in the wrong places, which is the failure mode worth
 * guarding against.
 *
 * Reference: opencv/modules/objdetect/src/face_detect.cpp (postProcess).
 */

import type { Box, RgbImage } from './preprocess.ts';
import { resize } from './resample.ts';

const STRIDES = [8, 16, 32] as const;

/** OpenCV's defaults, and what inference_server.py constructs the detector with. */
export const SCORE_THRESHOLD = 0.7;
export const NMS_THRESHOLD = 0.3;

/**
 * preprocess_faces.crop_face rejects any detection under 0.90 — so with YuNet the effective
 * gate is 0.90, and the 0.7 above only decides which boxes reach NMS.
 */
export const ACCEPT_THRESHOLD = 0.9;

/**
 * The 2023mar YuNet ONNX declares a fixed 1x3x640x640 input.
 *
 * OpenCV gets away with `setInputSize` because its DNN module reshapes the graph; ONNX
 * Runtime will not, so every image is letterboxed into 640x640 instead. That is a known and
 * deliberate divergence from the service, which detects at native resolution: a face far
 * below 1/640th of the frame can be found there and missed here. It changes which face is
 * cropped, never how the crop is scored.
 */
export const INPUT_SIZE = 640;

export interface Detection {
  box: Box;
  score: number;
}

export interface YuNetInput {
  /** NCHW float32, BGR, raw 0-255 — blobFromImage with no scaling, which is what YuNet wants. */
  tensor: Float32Array;
  width: number;
  height: number;
  /** Multiply detected coordinates by this to get back to the original image. */
  inverseScale: number;
}

/**
 * Letterboxes into the model's fixed 640x640 input.
 *
 * Uniform scale with zero padding on the right and bottom only: aspect ratio is preserved,
 * so a face is never stretched, and because the origin does not move, mapping a detection
 * back to image coordinates is a single multiply.
 */
export function buildInput(img: RgbImage, side = INPUT_SIZE): YuNetInput {
  const scale = side / Math.max(img.width, img.height);

  const scaledWidth = Math.max(1, Math.round(img.width * scale));
  const scaledHeight = Math.max(1, Math.round(img.height * scale));
  const source =
    scaledWidth === img.width && scaledHeight === img.height
      ? img.data
      : resize(img.data, img.width, img.height, scaledWidth, scaledHeight, 3, 'bilinear');

  const width = side;
  const height = side;

  const plane = width * height;
  const tensor = new Float32Array(3 * plane);
  for (let y = 0; y < scaledHeight; y++) {
    for (let x = 0; x < scaledWidth; x++) {
      const src = (y * scaledWidth + x) * 3;
      const dst = y * width + x;
      // blobFromImage does not swap channels, and OpenCV images are BGR.
      tensor[dst] = source[src + 2];
      tensor[plane + dst] = source[src + 1];
      tensor[2 * plane + dst] = source[src];
    }
  }

  return { tensor, width, height, inverseScale: 1 / scale };
}

export function decode(
  outputs: Record<string, Float32Array>,
  width: number,
  height: number,
  scoreThreshold = SCORE_THRESHOLD
): Detection[] {
  const found: Detection[] = [];

  for (const stride of STRIDES) {
    const cls = outputs[`cls_${stride}`];
    const obj = outputs[`obj_${stride}`];
    const bbox = outputs[`bbox_${stride}`];
    if (!cls || !obj || !bbox) continue;

    const cols = Math.floor(width / stride);
    const rows = Math.floor(height / stride);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const i = r * cols + c;
        const clsScore = clamp01(cls[i]);
        const objScore = clamp01(obj[i]);
        const score = Math.sqrt(clsScore * objScore);
        if (score < scoreThreshold) continue;

        const cx = (c + bbox[i * 4]) * stride;
        const cy = (r + bbox[i * 4 + 1]) * stride;
        const w = Math.exp(bbox[i * 4 + 2]) * stride;
        const h = Math.exp(bbox[i * 4 + 3]) * stride;

        found.push({
          box: { x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2 },
          score,
        });
      }
    }
  }

  return nms(found, NMS_THRESHOLD);
}

/** Scales detections back onto the original image and clamps them inside it. */
export function toImageSpace(detections: Detection[], input: YuNetInput, img: RgbImage): Detection[] {
  return detections.map((d) => ({
    score: d.score,
    box: {
      x1: Math.max(0, d.box.x1 * input.inverseScale),
      y1: Math.max(0, d.box.y1 * input.inverseScale),
      x2: Math.min(img.width, d.box.x2 * input.inverseScale),
      y2: Math.min(img.height, d.box.y2 * input.inverseScale),
    },
  }));
}

/** The service takes the largest face, not the most confident one. */
export function largest(detections: Detection[]): Detection | null {
  let best: Detection | null = null;
  let bestArea = -1;
  for (const d of detections) {
    const area = (d.box.x2 - d.box.x1) * (d.box.y2 - d.box.y1);
    if (area > bestArea) {
      bestArea = area;
      best = d;
    }
  }
  return best;
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function nms(detections: Detection[], threshold: number): Detection[] {
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const kept: Detection[] = [];

  for (const candidate of sorted) {
    let overlaps = false;
    for (const keeper of kept) {
      if (iou(candidate.box, keeper.box) > threshold) {
        overlaps = true;
        break;
      }
    }
    if (!overlaps) kept.push(candidate);
  }
  return kept;
}

function iou(a: Box, b: Box): number {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const overlap = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (overlap <= 0) return 0;

  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return overlap / (areaA + areaB - overlap);
}
