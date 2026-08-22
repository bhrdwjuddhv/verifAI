/**
 * Builds the extension into dist/.
 *
 * Two Vite passes, deliberately, because MV3 wants two different module formats:
 *
 *   1. content.js  — IIFE, everything inlined. Scripts injected with chrome.scripting are
 *                    classic scripts; an ESM bundle with a relative import would throw on
 *                    the first line and the failure surfaces as "nothing happened".
 *   2. everything  — ESM. The service worker declares "type": "module", and the HTML pages
 *      else          are ordinary module documents, so shared chunks are fine here.
 *
 * Pass 1 owns the clean, pass 2 must not wipe it (emptyOutDir: false).
 *
 * Usage: node scripts/build.mjs [--target=chrome|firefox] [--watch]
 */

import { build } from 'vite';
import { fileURLToPath } from 'node:url';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const root = fileURLToPath(new URL('..', import.meta.url));
const dist = path.join(root, 'dist');
const repo = path.join(root, '..');

const args = process.argv.slice(2);
const target = (args.find((a) => a.startsWith('--target=')) || '--target=chrome').split('=')[1];
const watch = args.includes('--watch');

if (!['chrome', 'firefox'].includes(target)) {
  console.error(`unknown --target=${target}; expected chrome or firefox`);
  process.exit(1);
}

const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));

/** Shared build options. `configFile: false` because every config lives in this file. */
const common = {
  root,
  configFile: false,
  publicDir: false,
  // The repo root has a postcss.config.js that loads the app's Tailwind. Vite walks upward
  // looking for one, finds it, and would process the extension's plain CSS through the
  // website's build. An inline (empty) config stops the search.
  css: { postcss: { plugins: [] } },
  // Lets the extension import the app's own verdict vocabulary (lib/verdict.ts has no
  // dependencies) instead of keeping a second copy of the labels and colours.
  resolve: {
    alias: {
      // Lets the extension import the app's own verdict vocabulary (lib/verdict.ts has no
      // dependencies) instead of keeping a second copy of the labels and colours.
      '@app': repo,
      // onnxruntime-web's default entry is ort.bundle.min.mjs, which inlines the emscripten
      // glue and contains an `eval` — forbidden by MV3's `script-src 'self'`, and it fails at
      // runtime, not at build time. The non-bundled WebGPU build has no eval, loads its glue
      // and wasm from `ort.env.wasm.wasmPaths` (i.e. from inside the package), and carries the
      // CPU kernels alongside the GPU ones so one artifact serves both execution providers.
      // The package's `exports` map does not expose dist/, hence an alias rather than a deep
      // import — and the version is pinned exactly so this path cannot move underneath us.
      'onnxruntime-web': path.join(root, 'node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs'),
      // Chrome runs inference in an offscreen document; Firefox's background is already a
      // page and runs it directly. One alias, so nothing above the host has to know which.
      '#host': path.join(root, `src/background/host.${target === 'firefox' ? 'firefox' : 'chrome'}.ts`),
    },
  },
  define: {
    __VERSION__: JSON.stringify(pkg.version),
    __TARGET__: JSON.stringify(target),
    // Every bundle in one build carries the same stamp. The popup compares its own against
    // the service worker's, which is the only reliable way to notice that Chrome is still
    // running a worker from an earlier build — the failure that otherwise surfaces as an
    // inexplicable "unknown request".
    __BUILD__: JSON.stringify(new Date().toISOString()),
  },
};

async function buildContent() {
  await build({
    ...common,
    build: {
      outDir: 'dist',
      emptyOutDir: true,
      target: 'chrome116',
      minify: false,
      sourcemap: false,
      lib: {
        entry: path.join(root, 'src/content/index.ts'),
        formats: ['iife'],
        name: 'VerifAIContent',
        fileName: () => 'content.js',
      },
      watch: watch ? {} : null,
    },
  });
}

async function buildMain() {
  await build({
    ...common,
    build: {
      outDir: 'dist',
      emptyOutDir: false,
      target: 'chrome116',
      minify: false,
      sourcemap: false,
      modulePreload: false,
      rollupOptions: {
        input: {
          background: path.join(root, 'src/background/index.ts'),
          popup: path.join(root, 'popup.html'),
          options: path.join(root, 'options.html'),
          onboarding: path.join(root, 'onboarding.html'),
          // Chrome only: Firefox runs inference in its background page, so shipping an
          // offscreen document there would be a page nothing can ever open.
          ...(target === 'chrome' ? { offscreen: path.join(root, 'offscreen.html') } : {}),
        },
        output: {
          // The manifest names background.js literally, so it cannot carry a hash.
          entryFileNames: (chunk) => (chunk.name === 'background' ? 'background.js' : 'assets/[name].js'),
          chunkFileNames: 'assets/[name].js',
          assetFileNames: 'assets/[name][extname]',
        },
      },
      watch: watch ? {} : null,
    },
  });
}

/** Chrome's manifest is the source of truth; Firefox's is derived so the two cannot drift. */
function writeManifest() {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'manifest.chrome.json'), 'utf8'));
  manifest.version = pkg.version;

  if (target === 'firefox') {
    // Firefox has no offscreen API — it keeps a background page with DOM access instead,
    // which is where onnxruntime-web runs there (see PLAN.md, Phase 5).
    manifest.permissions = manifest.permissions.filter((p) => p !== 'offscreen');
    manifest.background = { scripts: ['background.js'], type: 'module' };
    manifest.browser_specific_settings = {
      gecko: { id: 'verifai@verifai.app', strict_min_version: '121.0' },
    };
    delete manifest.minimum_chrome_version;
  }

  fs.writeFileSync(path.join(dist, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
}

function copyDir(from, to) {
  if (!fs.existsSync(from)) return false;
  fs.mkdirSync(to, { recursive: true });
  for (const entry of fs.readdirSync(from, { withFileTypes: true })) {
    const src = path.join(from, entry.name);
    const dst = path.join(to, entry.name);
    if (entry.isDirectory()) copyDir(src, dst);
    else fs.copyFileSync(src, dst);
  }
  return true;
}

/**
 * Model weights are copied from the repo at build time rather than committed under
 * extension/ — one copy of a 16MB ONNX in git is enough, and the Vercel deploy should not
 * carry a second. Missing models are a warning, not an error: Phase 1 (server mode) does
 * not need them, and on-device mode reports "unavailable" rather than guessing.
 */
function copyModels() {
  const out = path.join(dist, 'models');
  fs.mkdirSync(out, { recursive: true });

  // The v2 audio chain. Its filenames are NOT renamed on the way in, unlike everything else
  // here: each .onnx names its weight sidecar from inside the graph as a bare filename, so
  // `audio.onnx` looking for `audio_detector.onnx.data` would fail to load with a message
  // about a missing tensor rather than a missing file. Subdirectory instead of a prefix,
  // since detector.json is already taken by the face model.
  const V2 = 'models/audio/v2/models/audio/';
  const audioChain = [
    'preproc.onnx',
    'preproc.onnx.data',
    'audio_detector.onnx',
    'audio_detector.onnx.data',
    'audio_selftest.json',
  ].map((name) => [V2 + name, 'audio/' + name]);

  const wanted = [
    // The trained face model, newest first. The build ships whichever exists; on-device mode
    // reports the filename it loaded, so a stale bundle is visible rather than silent.
    ['models/face/detector_v2.onnx', 'detector.onnx'],
    ['models/face/detector_v2.json', 'detector.json'],
    ['models/npr_detector.onnx', 'npr.onnx'],
    ['models/npr_detector.json', 'npr.json'],
    ['models/face_detection_yunet.onnx', 'yunet.onnx'],
    ...audioChain,
  ];

  const missing = [];
  const shipped = [];
  for (const [from, name] of wanted) {
    const src = path.join(repo, from);
    if (fs.existsSync(src)) {
      const dst = path.join(out, name);
      fs.mkdirSync(path.dirname(dst), { recursive: true });
      fs.copyFileSync(src, dst);
      const bytes = fs.readFileSync(src);
      shipped.push({
        file: name,
        from,
        bytes: bytes.length,
        sha256: crypto.createHash('sha256').update(bytes).digest('hex'),
      });
    } else {
      missing.push(from);
    }
  }
  if (missing.length) {
    console.warn(`  ! on-device models missing (server mode still works): ${missing.join(', ')}`);
  }

  // "Which model said this" is a question the rest of the project answers everywhere, so the
  // build records exactly what it shipped — hashes included — and the options page reads it
  // back rather than describing the models from memory.
  fs.writeFileSync(
    path.join(out, 'models.json'),
    JSON.stringify(
      {
        builtAt: new Date().toISOString(),
        extensionVersion: pkg.version,
        target,
        // Mirrors common/config.py. If the server retunes these, the two disagree until this
        // is updated — which is what app/api/extension/manifest is meant to prevent.
        fusionWeights: { face: 0.5, npr: 0.5, frequency: 0 },
        thresholds: { fakeAbove: 70, realBelow: 30 },
        saliencyGrid: 5,
        models: shipped,
        missing,
      },
      null,
      2
    ) + '\n'
  );
}

/**
 * onnxruntime-web loads its glue and wasm at runtime from `ort.env.wasm.wasmPaths`; MV3
 * forbids fetching either from a CDN, so both are packaged.
 *
 * Only the jsep pair is copied. That is the binary behind ort.webgpu.min.mjs, and it carries
 * the CPU kernels too — so one 21MB artifact covers the WebGPU path and the wasm fallback,
 * rather than shipping two.
 */
function copyOrt() {
  const from = path.join(root, 'node_modules/onnxruntime-web/dist');
  if (!fs.existsSync(from)) {
    console.warn('  ! onnxruntime-web not installed — on-device mode is not built in');
    return;
  }
  const out = path.join(dist, 'ort');
  fs.mkdirSync(out, { recursive: true });
  for (const file of ['ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm']) {
    fs.copyFileSync(path.join(from, file), path.join(out, file));
  }
}

async function run() {
  console.log(`building verifai-extension v${pkg.version} for ${target}`);
  await buildContent();
  await buildMain();
  writeManifest();
  copyDir(path.join(root, 'icons'), path.join(dist, 'icons'));
  copyModels();
  copyOrt();
  console.log(`\ndone -> ${path.relative(process.cwd(), dist)}`);
  if (watch) console.log('watching for changes…');
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
