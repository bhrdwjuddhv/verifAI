/**
 * Generates the toolbar icons as PNGs, with no image library.
 *
 * A build that needs Sharp or canvas to produce four tiny squares is a build that breaks on
 * someone else's machine. node:zlib is enough: a PNG is a signature, three chunks, and a
 * CRC — and the mark itself is a lens ring, drawn by supersampled coverage so it stays clean
 * at 16px.
 *
 * Run: npm run icons   (output is committed; this only needs re-running if the mark changes)
 */

import zlib from 'node:zlib';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const outDir = path.join(fileURLToPath(new URL('..', import.meta.url)), 'icons');

const BG = [14, 17, 22]; // ink, matches the app's dark ground
const RING = [168, 216, 240]; // brand.accent
const DOT = [240, 249, 255];

const crcTable = (() => {
  const t = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c;
  }
  return t;
})();

function crc32(buf) {
  let c = -1;
  for (let i = 0; i < buf.length; i++) c = crcTable[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([len, body, crc]);
}

function png(size, rgba) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // truecolour + alpha
  // 10..12 stay 0: deflate, adaptive filtering, no interlace

  const stride = size * 4;
  const raw = Buffer.alloc((stride + 1) * size);
  for (let y = 0; y < size; y++) {
    raw[y * (stride + 1)] = 0; // filter: none
    rgba.copy(raw, y * (stride + 1) + 1, y * stride, (y + 1) * stride);
  }

  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', zlib.deflateSync(raw, { level: 9 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

/** Coverage of one pixel, sampled on a 4x4 grid — cheap anti-aliasing. */
function coverage(x, y, inside) {
  let hits = 0;
  for (let sy = 0; sy < 4; sy++) {
    for (let sx = 0; sx < 4; sx++) {
      if (inside(x + (sx + 0.5) / 4, y + (sy + 0.5) / 4)) hits++;
    }
  }
  return hits / 16;
}

function over(dst, i, rgb, alpha) {
  if (alpha <= 0) return;
  const a = dst[i + 3] / 255;
  const outA = alpha + a * (1 - alpha);
  for (let c = 0; c < 3; c++) {
    dst[i + c] = Math.round((rgb[c] * alpha + dst[i + c] * a * (1 - alpha)) / (outA || 1));
  }
  dst[i + 3] = Math.round(outA * 255);
}

function draw(size) {
  const buf = Buffer.alloc(size * size * 4);
  const c = size / 2;
  const radius = size * 0.22; // rounded-square corner
  const ringOuter = size * 0.345;
  const ringInner = size * 0.235;
  const dot = size * 0.105;

  const inSquare = (x, y) => {
    const dx = Math.max(Math.abs(x - c) - (c - radius), 0);
    const dy = Math.max(Math.abs(y - c) - (c - radius), 0);
    return Math.hypot(dx, dy) <= radius && Math.abs(x - c) <= c && Math.abs(y - c) <= c;
  };
  const dist = (x, y) => Math.hypot(x - c, y - c);
  const inRing = (x, y) => dist(x, y) <= ringOuter && dist(x, y) >= ringInner;
  const inDot = (x, y) => dist(x, y) <= dot;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      over(buf, i, BG, coverage(x, y, inSquare));
      over(buf, i, RING, coverage(x, y, inRing));
      over(buf, i, DOT, coverage(x, y, inDot));
    }
  }
  return buf;
}

fs.mkdirSync(outDir, { recursive: true });
for (const size of [16, 32, 48, 128]) {
  const file = path.join(outDir, `icon-${size}.png`);
  fs.writeFileSync(file, png(size, draw(size)));
  console.log(`wrote ${path.basename(file)}`);
}
