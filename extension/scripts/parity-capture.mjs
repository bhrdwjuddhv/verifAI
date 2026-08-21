/**
 * Captures the server's verdict for every fixture image, as the reference to compare against.
 *
 * Standing rule 2 in PLAN.md is that on-device and server must agree; this is the half that
 * asks the server. It POSTs each fixture to a real `/api/scan` and records the full response,
 * then writes everything — images included, base64 — into dist/parity-fixtures.json so the
 * browser half needs one fetch and no file access.
 *
 *   npm run parity:capture                      # against the configured server
 *   npm run parity:capture -- --server http://localhost:3000 --dir fixtures
 *
 * Pacing is deliberate: middleware.ts allows 20 requests/min/IP across all of /api/*, shared
 * with anyone browsing the site from the same address. 5s between requests keeps this at 12/min,
 * the same budget the extension's own queue uses.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const hit = args.find((a) => a.startsWith(`--${name}=`));
  if (hit) return hit.split('=').slice(1).join('=');
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};

const server = flag('server', 'https://verif-ai-blue.vercel.app').replace(/\/+$/, '');
const dir = path.resolve(root, flag('dir', 'fixtures'));
const SPACING_MS = 5000;

const MIME = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.bmp': 'image/bmp',
  '.gif': 'image/gif',
};

if (!fs.existsSync(dir)) {
  console.error(`No fixture directory at ${dir}.

Put images in it — PLAN.md P4 asks for 30: 10 real, 10 face-swap, 10 fully generated, at
least 5 with no face. The mix matters more than the count: a set with no faces never
exercises the crop path, and a set of only faces never exercises NPR alone.`);
  process.exit(1);
}

const files = fs
  .readdirSync(dir)
  .filter((f) => MIME[path.extname(f).toLowerCase()])
  .sort();

if (!files.length) {
  console.error(`No images in ${dir}.`);
  process.exit(1);
}

console.log(`\ncapturing ${files.length} fixture(s) from ${server}\n`);

const captured = [];
let failures = 0;

for (const [index, file] of files.entries()) {
  const bytes = fs.readFileSync(path.join(dir, file));
  const mime = MIME[path.extname(file).toLowerCase()];

  const form = new FormData();
  form.append('file', new Blob([bytes], { type: mime }), file);

  process.stdout.write(`  ${String(index + 1).padStart(2)}/${files.length} ${file.padEnd(34)} `);

  let entry;
  try {
    const res = await fetch(`${server}/api/scan`, {
      method: 'POST',
      body: form,
      headers: { 'X-VerifAI-Client': 'parity-capture/1' },
      signal: AbortSignal.timeout(90_000),
    });
    const body = await res.json().catch(() => ({}));

    if (!res.ok) {
      failures++;
      // Recorded rather than dropped: "the service refused this file" is itself a reference
      // the device side has to reproduce.
      entry = { file, mime, ok: false, status: res.status, error: body.error ?? `HTTP ${res.status}` };
      console.log(`${res.status} ${body.error ?? ''}`);
    } else {
      entry = { file, mime, ok: true, reference: body };
      const fake = body.score === null ? '—' : `${100 - body.score}%`;
      console.log(`${String(fake).padStart(4)} fake · ${body.verdict?.shortLabel ?? '?'}`);
    }
  } catch (err) {
    failures++;
    entry = { file, mime, ok: false, status: 0, error: String(err?.message ?? err) };
    console.log(`unreachable — ${entry.error}`);
  }

  entry.b64 = bytes.toString('base64');
  captured.push(entry);

  if (index < files.length - 1) await new Promise((r) => setTimeout(r, SPACING_MS));
}

const out = path.join(root, 'dist');
fs.mkdirSync(out, { recursive: true });
fs.writeFileSync(
  path.join(out, 'parity-fixtures.json'),
  JSON.stringify({ server, capturedAt: new Date().toISOString(), fixtures: captured })
);

console.log(`\nwrote dist/parity-fixtures.json (${captured.length} fixtures, ${failures} server failure(s))`);
console.log('now run `npm run parity` and open the page it prints\n');
