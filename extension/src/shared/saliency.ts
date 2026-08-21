/**
 * Occlusion saliency — the explanation the torch-free path uses.
 *
 * Grad-CAM needs gradients and ONNX Runtime is forward-only, so the service's ONNX engine
 * answers "where did the model look?" by hiding one patch at a time and measuring how far
 * P(fake) falls. This is a port of `occlusion_overlay` and `encode_overlay` from
 * inference_server.py and common/xai.py — the same feature the server ships, not a lesser one.
 *
 * Two details are load-bearing and easy to lose:
 *   - patches are filled with 0 *after* normalisation, which is the dataset mean — "no
 *     information" rather than "black square", which would itself be a strong feature;
 *   - a peak of zero returns null. No heatmap is an honest answer; a smooth, plausible,
 *     meaningless one is not.
 */

import type { RgbImage } from './preprocess.ts';
import { resize } from './resample.ts';

/** inference_server.py SALIENCY_GRID (VERIFAI_SALIENCY_GRID). 25 patches + 1 baseline. */
export const SALIENCY_GRID = 5;

/** common/xai.py encode_overlay default. */
export const OVERLAY_ALPHA = 0.45;

/**
 * `np.repeat(arr, grid*grid, axis=0)` with patch k zeroed in copy k.
 *
 * Returns one contiguous (grid² , 3, size, size) batch, so the whole thing is a single set of
 * forwards rather than 25 round trips.
 */
export function occlusionBatch(arr: Float32Array, size: number, grid = SALIENCY_GRID): Float32Array {
  const cells = grid * grid;
  const perImage = 3 * size * size;
  const out = new Float32Array(cells * perImage);
  const step = size / grid;

  for (let k = 0; k < cells; k++) {
    const offset = k * perImage;
    out.set(arr, offset);

    const r = Math.floor(k / grid);
    const c = k % grid;
    const y0 = Math.trunc(r * step);
    const y1 = Math.trunc((r + 1) * step);
    const x0 = Math.trunc(c * step);
    const x1 = Math.trunc((c + 1) * step);

    for (let channel = 0; channel < 3; channel++) {
      const plane = offset + channel * size * size;
      for (let y = y0; y < y1; y++) {
        out.fill(0, plane + y * size + x0, plane + y * size + x1);
      }
    }
  }
  return out;
}

/**
 * How far P(fake) fell for each hidden patch, clipped at zero and scaled to its own peak.
 *
 * Null when nothing moved the score: the model did not depend on any one region, and there is
 * no honest picture to draw.
 */
export function saliencyFromDrops(base: number, occluded: ArrayLike<number>, grid = SALIENCY_GRID): Float32Array | null {
  const cam = new Float32Array(grid * grid);
  let peak = 0;
  for (let k = 0; k < cam.length; k++) {
    const drop = base - occluded[k];
    cam[k] = drop > 0 ? drop : 0;
    if (cam[k] > peak) peak = cam[k];
  }
  if (peak <= 0) return null;
  for (let k = 0; k < cam.length; k++) cam[k] /= peak;
  return cam;
}

/**
 * Blends a 0-1 saliency grid over the image as a red-to-blue heat overlay.
 *
 * Red where the model looked, blue where it did not, and the blend is weighted by the map
 * itself so cold regions stay recognisable instead of being washed flat.
 */
export function encodeOverlay(
  cam: Float32Array,
  grid: number,
  baseImage: RgbImage,
  alpha = OVERLAY_ALPHA
): RgbImage {
  const { width, height } = baseImage;

  // Pillow round-trips the map through uint8 before resizing; matching that keeps the overlay
  // pixel-comparable to the server's.
  const small = new Uint8Array(grid * grid);
  for (let i = 0; i < small.length; i++) {
    small[i] = Math.trunc(255 * Math.min(1, Math.max(0, cam[i])));
  }
  const upscaled = resize(small, grid, grid, width, height, 1, 'bilinear');

  const out = new Uint8Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    const heat = upscaled[i] / 255;

    const heatR = heat;
    const heatB = 1 - heat;
    const heatG = Math.min(1, Math.max(0, 1 - Math.abs(2 * heat - 1))) * 0.6;

    const weight = alpha * heat;
    const p = i * 3;
    out[p] = clamp255(255 * ((baseImage.data[p] / 255) * (1 - weight) + heatR * weight));
    out[p + 1] = clamp255(255 * ((baseImage.data[p + 1] / 255) * (1 - weight) + heatG * weight));
    out[p + 2] = clamp255(255 * ((baseImage.data[p + 2] / 255) * (1 - weight) + heatB * weight));
  }

  return { data: out, width, height };
}

function clamp255(v: number): number {
  const n = Math.trunc(v);
  return n < 0 ? 0 : n > 255 ? 255 : n;
}
