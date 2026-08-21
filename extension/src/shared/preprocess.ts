/**
 * `to_array` from inference_server.py, and the crop geometry from preprocess_faces.py.
 *
 * Pure functions over plain RGB buffers so the offscreen document and the parity harness
 * run identical code. Nothing here touches a canvas — decoding is the caller's job.
 */

import { IMAGENET_MEAN, IMAGENET_STD } from './detect.ts';
import { resize } from './resample.ts';

export interface RgbImage {
  /** Interleaved RGB, 3 bytes per pixel. */
  data: Uint8Array;
  width: number;
  height: number;
}

/**
 * PIL image -> (1, 3, size, size) float32, normalised.
 *
 * `crop` centre-crops at native resolution instead of resizing. NPR measures the generator's
 * up-sampling artifact, and resizing would lay our own resampling over exactly the signal
 * being read — so it only scales up when the image is smaller than the crop.
 */
export function toArray(
  img: RgbImage,
  size: number,
  mean: number[] = IMAGENET_MEAN,
  std: number[] = IMAGENET_STD,
  crop = false
): Float32Array {
  let work = img;

  if (crop) {
    if (work.width < size || work.height < size) {
      const s = size / Math.min(work.width, work.height);
      work = resample(
        work,
        Math.max(size, Math.trunc(work.width * s)),
        Math.max(size, Math.trunc(work.height * s)),
        'bicubic'
      );
    }
    const left = Math.floor((work.width - size) / 2);
    const top = Math.floor((work.height - size) / 2);
    work = cropRect(work, left, top, size, size);
  } else {
    work = resample(work, size, size, 'bilinear');
  }

  const out = new Float32Array(3 * size * size);
  const plane = size * size;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const src = (y * size + x) * 3;
      const dst = y * size + x;
      for (let c = 0; c < 3; c++) {
        out[c * plane + dst] = (work.data[src + c] / 255 - mean[c]) / std[c];
      }
    }
  }
  return out;
}

export function resample(img: RgbImage, width: number, height: number, filter: 'bilinear' | 'bicubic'): RgbImage {
  if (width === img.width && height === img.height) return img;
  return { data: resize(img.data, img.width, img.height, width, height, 3, filter), width, height };
}

export function cropRect(img: RgbImage, left: number, top: number, width: number, height: number): RgbImage {
  const data = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    const from = ((top + y) * img.width + left) * 3;
    data.set(img.data.subarray(from, from + width * 3), y * width * 3);
  }
  return { data, width, height };
}

/** `arr[:, :, :, ::-1]` — the horizontal flip half of the service's flip-TTA. */
export function flipWidth(nchw: Float32Array, size: number): Float32Array {
  const out = new Float32Array(nchw.length);
  const plane = size * size;
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < size; y++) {
      const row = c * plane + y * size;
      for (let x = 0; x < size; x++) out[row + x] = nchw[row + size - 1 - x];
    }
  }
  return out;
}

export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/**
 * `crop_face` from preprocess_faces.py: a square crop around the largest face, or null.
 *
 * The margin matters more than it looks — face-swap blending seams sit on the jawline and
 * hairline, so a tight crop cuts away the single most discriminative region. The `half < 24`
 * rejection matters just as much: it is the difference between "no face" and a garbage 12px
 * crop being scored as evidence.
 */
export function cropFace(img: RgbImage, box: Box, margin: number): RgbImage | null {
  const cx = (box.x1 + box.x2) / 2;
  const cy = (box.y1 + box.y2) / 2;
  let half = (Math.max(box.x2 - box.x1, box.y2 - box.y1) * (1 + margin)) / 2;

  // Clamp to the image, keeping the box square so the later resize never stretches.
  half = Math.min(half, cx, cy, img.width - cx, img.height - cy);
  if (half < 24) return null;

  const left = Math.trunc(cx - half);
  const top = Math.trunc(cy - half);
  const right = Math.trunc(cx + half);
  const bottom = Math.trunc(cy + half);
  return cropRect(img, left, top, right - left, bottom - top);
}
