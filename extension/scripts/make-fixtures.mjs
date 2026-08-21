/**
 * Generates deterministic test images with deliberately different frequency content.
 *
 * Not a substitute for the real fixture set PLAN.md asks for — these carry no ground truth
 * about whether anything is AI-generated. They exist to compare *implementations*: NPR reads
 * the resampling fingerprint in an image, so a set that ranges from "already a 2x nearest
 * up-sample" (no residual at all) to "pixel-level checkerboard" (maximum residual) exercises
 * the whole range of what it responds to. If the on-device port and the server agree across
 * all of them, they agree about NPR.
 *
 *   node scripts/make-fixtures.mjs
 */

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';

const out = path.join(fileURLToPath(new URL('..', import.meta.url)), 'fixtures');
const SIZE = 256;

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

/** 8-bit RGB PNG. Lossless, so the fixture is exactly the pixels intended. */
function png(size, rgb) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8;
  ihdr[9] = 2; // truecolour, no alpha
  const stride = size * 3;
  const raw = Buffer.alloc((stride + 1) * size);
  for (let y = 0; y < size; y++) {
    raw[y * (stride + 1)] = 0;
    rgb.copy(raw, y * (stride + 1) + 1, y * stride, (y + 1) * stride);
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', zlib.deflateSync(raw, { level: 9 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

/** Deterministic LCG, so two runs produce byte-identical fixtures. */
function rand(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

function make(fn) {
  const buf = Buffer.alloc(SIZE * SIZE * 3);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const [r, g, b] = fn(x, y);
      const i = (y * SIZE + x) * 3;
      buf[i] = r;
      buf[i + 1] = g;
      buf[i + 2] = b;
    }
  }
  return buf;
}

const images = {
  // Smooth: almost no high-frequency detail, so almost no NPR residual either.
  'synth-gradient.png': make((x, y) => [(x * 255) / SIZE, (y * 255) / SIZE, 128]),

  // The degenerate case: already a 2x nearest up-sample, so the residual is exactly zero.
  // Whatever the model says here, both implementations must say the same thing.
  'synth-upsampled.png': (() => {
    const next = rand(7);
    const half = SIZE / 2;
    const small = Array.from({ length: half * half * 3 }, () => Math.floor(next() * 256));
    return make((x, y) => {
      const i = ((y >> 1) * half + (x >> 1)) * 3;
      return [small[i], small[i + 1], small[i + 2]];
    });
  })(),

  // Maximum residual: detail at the Nyquist limit, nothing a resampler could have produced.
  'synth-checker.png': make((x, y) => {
    const on = (x + y) % 2 === 0 ? 235 : 20;
    return [on, on, on];
  }),

  // Photo-ish: broadband detail over structure, the closest of these to a real photograph.
  'synth-noisy-blobs.png': (() => {
    const next = rand(42);
    return make((x, y) => {
      const blob = Math.sin(x / 21) * Math.cos(y / 17) * 60 + 128;
      const grain = (next() - 0.5) * 40;
      const v = Math.max(0, Math.min(255, blob + grain));
      return [v, v * 0.92, v * 0.8];
    });
  })(),
};

fs.mkdirSync(out, { recursive: true });
for (const [name, rgb] of Object.entries(images)) {
  const file = path.join(out, name);
  fs.writeFileSync(file, png(SIZE, rgb));
  console.log(`  ${name.padEnd(24)} ${(fs.statSync(file).size / 1024).toFixed(0)}KB`);
}
console.log(`\n${Object.keys(images).length} fixtures in ${path.relative(process.cwd(), out)}`);
console.log('next: npm run parity:capture && npm run parity\n');
