/**
 * Pillow-compatible image resampling.
 *
 * This file exists because of standing rule 2 in PLAN.md: on-device and server must agree.
 * The server preprocesses with `PIL.Image.resize(..., BILINEAR)`, and a canvas `drawImage`
 * downscale is a *different* filter — close enough to look identical and far enough to move
 * a fused score by several points, which is the entire ±2 parity budget.
 *
 * So this is a literal port of Pillow's Resample.c: the same triangle filter with
 * support scaled by the reduction factor, the same 22-bit fixed-point coefficients, the same
 * clip to uint8 between the horizontal and vertical passes. The fixed-point detail is not
 * pedantry — floating-point accumulation drifts from Pillow by a level or two per pixel, and
 * those levels land in the model's input.
 *
 * Reference: Pillow src/libImaging/Resample.c (precompute_coeffs, normalize_coeffs_8bpc,
 * ImagingResampleHorizontal_8bpc).
 */

const PRECISION_BITS = 32 - 8 - 2; // 22, as in Resample.c

export type FilterName = 'bilinear' | 'bicubic';

interface Filter {
  support: number;
  fn: (x: number) => number;
}

const FILTERS: Record<FilterName, Filter> = {
  bilinear: {
    support: 1.0,
    fn: (x) => {
      const a = Math.abs(x);
      return a < 1.0 ? 1.0 - a : 0.0;
    },
  },
  bicubic: {
    support: 2.0,
    // Catmull-Rom with a = -0.5, matching Pillow's bicubic_filter exactly.
    fn: (x) => {
      const a = -0.5;
      const t = Math.abs(x);
      if (t < 1.0) return ((a + 2.0) * t - (a + 3.0)) * t * t + 1;
      if (t < 2.0) return (((t - 5) * t + 8) * t - 4) * a;
      return 0.0;
    },
  },
};

interface Coeffs {
  ksize: number;
  bounds: Int32Array; // [xmin, xmax] pairs
  kk: Int32Array; // fixed-point weights, ksize per output pixel
}

function precomputeCoeffs(inSize: number, outSize: number, filter: Filter): Coeffs {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1.0);
  const support = filter.support * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;

  const bounds = new Int32Array(outSize * 2);
  const kk = new Int32Array(outSize * ksize);
  const weights = new Float64Array(ksize);

  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1.0 / filterscale;

    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;

    let total = 0;
    for (let x = 0; x < xmax; x++) {
      const w = filter.fn((x + xmin - center + 0.5) * ss);
      weights[x] = w;
      total += w;
    }

    for (let x = 0; x < xmax; x++) {
      const w = total !== 0 ? weights[x] / total : 0;
      // normalize_coeffs_8bpc: round away from zero into 22-bit fixed point.
      kk[xx * ksize + x] =
        w < 0 ? Math.trunc(-0.5 + w * (1 << PRECISION_BITS)) : Math.trunc(0.5 + w * (1 << PRECISION_BITS));
    }

    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }

  return { ksize, bounds, kk };
}

const ROUND = 1 << (PRECISION_BITS - 1);

function clip8(value: number): number {
  const v = Math.floor(value / (1 << PRECISION_BITS));
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

/**
 * Resize interleaved 8-bit pixel data. `channels` is 1 (grayscale) or 3 (RGB).
 *
 * Horizontal pass then vertical, with the intermediate clipped back to 8 bits — Pillow does
 * the same, and skipping the intermediate clip is enough to shift the output.
 */
export function resize(
  src: Uint8Array | Uint8ClampedArray,
  srcWidth: number,
  srcHeight: number,
  outWidth: number,
  outHeight: number,
  channels: number,
  filterName: FilterName = 'bilinear'
): Uint8Array {
  const filter = FILTERS[filterName];

  const horizontal = new Uint8Array(outWidth * srcHeight * channels);
  if (outWidth === srcWidth) {
    horizontal.set(src.subarray(0, outWidth * srcHeight * channels));
  } else {
    const { ksize, bounds, kk } = precomputeCoeffs(srcWidth, outWidth, filter);
    for (let y = 0; y < srcHeight; y++) {
      const rowIn = y * srcWidth * channels;
      const rowOut = y * outWidth * channels;
      for (let xx = 0; xx < outWidth; xx++) {
        const xmin = bounds[xx * 2];
        const xmax = bounds[xx * 2 + 1];
        const kOffset = xx * ksize;
        for (let c = 0; c < channels; c++) {
          let sum = ROUND;
          for (let k = 0; k < xmax; k++) {
            sum += src[rowIn + (xmin + k) * channels + c] * kk[kOffset + k];
          }
          horizontal[rowOut + xx * channels + c] = clip8(sum);
        }
      }
    }
  }

  if (outHeight === srcHeight) return horizontal;

  const out = new Uint8Array(outWidth * outHeight * channels);
  const { ksize, bounds, kk } = precomputeCoeffs(srcHeight, outHeight, filter);
  for (let yy = 0; yy < outHeight; yy++) {
    const ymin = bounds[yy * 2];
    const ymax = bounds[yy * 2 + 1];
    const kOffset = yy * ksize;
    for (let x = 0; x < outWidth; x++) {
      for (let c = 0; c < channels; c++) {
        let sum = ROUND;
        for (let k = 0; k < ymax; k++) {
          sum += horizontal[((ymin + k) * outWidth + x) * channels + c] * kk[kOffset + k];
        }
        out[(yy * outWidth + x) * channels + c] = clip8(sum);
      }
    }
  }
  return out;
}

/**
 * RGB -> L, using Pillow's fixed-point ITU-R 601-2 luma.
 *
 * `L = (R*19595 + G*38470 + B*7471 + 0x8000) >> 16` is what convert("L") actually computes;
 * the float form of the same constants rounds differently on roughly one pixel in three.
 */
export function toGray(rgb: Uint8Array | Uint8ClampedArray, width: number, height: number): Uint8Array {
  const out = new Uint8Array(width * height);
  for (let i = 0, p = 0; i < out.length; i++, p += 3) {
    out[i] = (rgb[p] * 19595 + rgb[p + 1] * 38470 + rgb[p + 2] * 7471 + 0x8000) >> 16;
  }
  return out;
}

/** Drops alpha the way `Image.open(...).convert("RGB")` does — no compositing over white. */
export function rgbaToRgb(rgba: Uint8ClampedArray | Uint8Array, pixels: number): Uint8Array {
  const out = new Uint8Array(pixels * 3);
  for (let i = 0, s = 0, d = 0; i < pixels; i++, s += 4, d += 3) {
    out[d] = rgba[s];
    out[d + 1] = rgba[s + 1];
    out[d + 2] = rgba[s + 2];
  }
  return out;
}
