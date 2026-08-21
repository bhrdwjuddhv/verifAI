/**
 * Builds and zips a store-ready package.
 *
 * The ZIP writer is here rather than a dependency for the same reason gen-icons.mjs draws its
 * own PNGs: a release step that needs `archiver` to be installed correctly is a release step
 * that breaks on someone else's machine, and the format is a header, deflate, and a central
 * directory.
 *
 *   npm run pack                    # chrome (also the Edge package)
 *   npm run pack -- --target=firefox
 *
 * Dev scaffolding never ships: preview pages and captured fixtures are excluded by name, and
 * the script fails loudly if one slips through rather than quietly uploading it to a store.
 */

import { execFileSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const dist = path.join(root, 'dist');

const args = process.argv.slice(2);
const target = (args.find((a) => a.startsWith('--target=')) || '--target=chrome').split('=')[1];
if (!['chrome', 'firefox'].includes(target)) {
  console.error(`unknown --target=${target}`);
  process.exit(1);
}

const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));

/** Anything a developer generated locally. None of it belongs in a store upload. */
const EXCLUDE = [/^preview-/, /^parity-fixtures\.json$/, /\.map$/, /^\.DS_Store$/];

console.log(`\nbuilding ${target}…\n`);
execFileSync(process.execPath, [path.join(root, 'scripts/build.mjs'), `--target=${target}`], {
  stdio: 'inherit',
  cwd: root,
});

function walk(dir, prefix = '') {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
    const name = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (EXCLUDE.some((rule) => rule.test(entry.name))) continue;
    if (entry.isDirectory()) out.push(...walk(path.join(dir, entry.name), name));
    else out.push({ name, absolute: path.join(dir, entry.name) });
  }
  return out;
}

const files = walk(dist);
if (!files.length) {
  console.error('dist is empty');
  process.exit(1);
}

// ---------------------------------------------------------------------------------------
// Minimal ZIP writer (deflate, no zip64 — the package is tens of megabytes, not gigabytes)

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

/** MS-DOS date/time. Fixed, so two builds of the same input produce the same archive. */
const DOS_TIME = 0;
const DOS_DATE = ((2026 - 1980) << 9) | (1 << 5) | 1;

function zip(entries) {
  const locals = [];
  const central = [];
  let offset = 0;

  for (const entry of entries) {
    const raw = fs.readFileSync(entry.absolute);
    const deflated = zlib.deflateRawSync(raw, { level: 9 });
    // A store or two rejects entries that grew; fall back to stored when deflate loses.
    const useDeflate = deflated.length < raw.length;
    const body = useDeflate ? deflated : raw;
    const method = useDeflate ? 8 : 0;
    const name = Buffer.from(entry.name, 'utf8');
    const sum = crc32(raw);

    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4); // version needed
    local.writeUInt16LE(0, 6); // flags
    local.writeUInt16LE(method, 8);
    local.writeUInt16LE(DOS_TIME, 10);
    local.writeUInt16LE(DOS_DATE, 12);
    local.writeUInt32LE(sum, 14);
    local.writeUInt32LE(body.length, 18);
    local.writeUInt32LE(raw.length, 22);
    local.writeUInt16LE(name.length, 26);
    local.writeUInt16LE(0, 28);

    locals.push(local, name, body);

    const header = Buffer.alloc(46);
    header.writeUInt32LE(0x02014b50, 0);
    header.writeUInt16LE(20, 4); // version made by
    header.writeUInt16LE(20, 6); // version needed
    header.writeUInt16LE(0, 8);
    header.writeUInt16LE(method, 10);
    header.writeUInt16LE(DOS_TIME, 12);
    header.writeUInt16LE(DOS_DATE, 14);
    header.writeUInt32LE(sum, 16);
    header.writeUInt32LE(body.length, 20);
    header.writeUInt32LE(raw.length, 24);
    header.writeUInt16LE(name.length, 28);
    header.writeUInt16LE(0, 30); // extra
    header.writeUInt16LE(0, 32); // comment
    header.writeUInt16LE(0, 34); // disk
    header.writeUInt16LE(0, 36); // internal attrs
    header.writeUInt32LE(0, 38); // external attrs
    header.writeUInt32LE(offset, 42);

    central.push(header, name);
    offset += local.length + name.length + body.length;
  }

  const directory = Buffer.concat(central);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(0, 4);
  end.writeUInt16LE(0, 6);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(directory.length, 12);
  end.writeUInt32LE(offset, 16);
  end.writeUInt16LE(0, 20);

  return Buffer.concat([...locals, directory, end]);
}

// ---------------------------------------------------------------------------------------

const archive = zip(files);
const out = path.join(root, `verifai-${pkg.version}-${target}.zip`);
fs.writeFileSync(out, archive);

const manifest = JSON.parse(fs.readFileSync(path.join(dist, 'manifest.json'), 'utf8'));
const models = files.filter((f) => f.name.startsWith('models/')).length;

console.log(`\npackaged ${path.basename(out)}`);
console.log(`  ${files.length} files · ${(archive.length / 1048576).toFixed(1)}MB compressed`);
console.log(`  sha256 ${crypto.createHash('sha256').update(archive).digest('hex')}`);
console.log(`  manifest v${manifest.version} · ${manifest.permissions.join(', ')}`);
console.log(`  optional hosts: ${(manifest.optional_host_permissions || []).join(', ') || 'none'}`);
console.log(`  bundled model files: ${models}`);

// A last look for anything that should never have been in a store upload.
const suspicious = files.filter((f) => /preview|fixture|\.map$/.test(f.name));
if (suspicious.length) {
  console.error(`\nrefusing to ship dev files: ${suspicious.map((f) => f.name).join(', ')}`);
  process.exit(1);
}
console.log('');
