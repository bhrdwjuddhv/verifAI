/**
 * Renders the built popup and options pages in an ordinary browser, with a stubbed chrome.*
 * and a realistic scan result.
 *
 * A dev-only harness: the popup is 380px of layout that otherwise can only be looked at by
 * loading the extension, right-clicking something, and hoping. This makes the UI reviewable
 * in one command, and it never ships — it writes into dist/, which is gitignored.
 *
 *   node scripts/preview.mjs        # writes dist/preview-popup.html, serves dist on :8787
 */

import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const dist = path.join(root, 'dist');
const PORT = Number(process.env.PORT || 8787);

if (!fs.existsSync(path.join(dist, 'popup.html'))) {
  console.error('run `npm run build` first');
  process.exit(1);
}

const RESULT = {
  id: 'VRF-481920',
  filename: 'portrait_04.jpg',
  fileType: 'image',
  fileSize: '0.4 MB',
  // The API sends "how real"; 12 means a fused P(fake) of 88.
  score: 12,
  confidence: 88,
  verdict: {
    category: 'manipulated',
    label: 'AI-Generated or Manipulated',
    shortLabel: 'AI / Manipulated',
    badgeBg: '', badgeText: '', badgeBorder: '',
    ringColor: '#EF4444',
    glowColor: 'rgba(239, 68, 68, 0.25)',
    description: 'The detector scored this above the AI-generated threshold.',
    recommendation: 'Do not treat this file as authentic without a second, independent check.',
    laymanSummary: 'The detectors scored this 88% likely AI-generated or manipulated across 2 detectors.',
  },
  reasons: [
    'A face was detected; the model ran on the cropped face region.',
    'Face classifier: 91% likely swapped or manipulated.',
    'Whole-image AI-generation detector (NPR): 85% likely generated. This one catches fully synthetic images, including generated faces.',
    'Combined score: 88% likely AI-generated or manipulated (weights: face 0.5, npr 0.5). Thresholds: above 70 fake, below 30 real, in between uncertain.',
    'High-frequency energy share: 43%. A descriptive statistic only.',
    'No C2PA manifest and no EXIF. Every social network strips both from real photos too.',
    'Model used: onnx:trained_checkpoint + onnx:npr.',
  ],
  signals: { modelScore: 91, nprScore: 85, audioScore: null, frequencyScore: 43, faceDetected: true },
  metadata: { c2paManifestPresent: false, exifPresent: false },
  modelSource: 'onnx:trained_checkpoint + onnx:npr',
  fusion: { weights: { face: 0.5, npr: 0.5, frequency: 0 }, used: { face: 0.5, npr: 0.5 } },
  heatmap: null,
  notes: [
    'confidence is uncalibrated — treat it as a ranking, not a probability',
    'the detectors agree closely here; that is not independent confirmation, both saw the same pixels',
  ],
  timestamp: new Date().toISOString(),
};

const SCANS = [
  {
    id: 'a', createdAt: Date.now(), mediaUrl: 'https://cdn.example.com/media/portrait_04.jpg',
    kind: 'image', source: 'device', phase: 'done', cached: false, bytes: 412_233, ms: 812, result: RESULT,
  },
  {
    id: 'c', createdAt: Date.now() - 60_000, mediaUrl: 'https://cdn.example.com/clip.mp4',
    kind: 'video', source: 'device', phase: 'error', errorKind: 'not-implemented', offerServer: true,
    error: 'Scoring a whole video means sampling frames across it, which only the server does.',
  },
  {
    id: 'b', createdAt: Date.now() - 120_000, mediaUrl: 'blob:https://web.whatsapp.com/9f2c',
    kind: 'image', source: 'server', phase: 'needs-consent',
  },
];

const STUB = `
<script>
  const scans = ${JSON.stringify(SCANS)};
  const settings = { serverUrl: 'https://verif-ai-blue.vercel.app', consentVersion: 0, autoScan: false };
  const noop = () => {};
  window.chrome = {
    runtime: {
      sendMessage: async (m) => {
        if (m.type === 'ui:get-state') return { type: 'state', scans };
        if (m.type === 'ui:cache-stats') return { type: 'cache-stats', entries: 37, heatmaps: 9, approxBytes: 612_000 };
        if (m.type === 'ui:auto-status') return { type: 'ok', ok: false, reason: 'auto-scan is off', url: 'https://www.youtube.com/shorts/abc' };
        return { type: 'ok' };
      },
      onMessage: { addListener: noop },
      getURL: (p) => p,
    },
    storage: { sync: { get: async (d) => ({ ...d, ...settings }), set: async () => {} } },
    tabs: { create: noop },
    permissions: { contains: async () => true, request: async () => true },
  };
</script>`;

/**
 * A harness for the offscreen document, which has no UI of its own.
 *
 * It stubs just enough of chrome.* to let the real built bundle run in an ordinary tab,
 * draws a test image on a canvas so no fixture file is needed, and pushes it through the
 * same message the service worker sends. This is the only way to exercise ONNX Runtime,
 * the wasm load and the real image decode without installing the extension.
 */
const OFFSCREEN_HARNESS = `<!doctype html>
<html><head><meta charset="utf-8"><title>offscreen harness</title>
<style>body{font:13px ui-monospace,monospace;background:#0A0C12;color:#E6E8EF;padding:20px;white-space:pre-wrap}</style>
</head><body>
<div id="log">booting…</div>
<script>
  const log = (m) => { document.getElementById('log').textContent += String.fromCharCode(10) + m; };
  let handler = null;
  window.chrome = {
    runtime: {
      getURL: (p) => '/' + p,
      onMessage: { addListener: (fn) => { handler = fn; } },
    },
  };
  window.addEventListener('error', (e) => log('ERROR ' + e.message));
  window.addEventListener('unhandledrejection', (e) => log('REJECTED ' + e.reason));
<\/script>
<script type="module" src="/assets/offscreen.js"><\/script>
<script type="module">
  const log = (m) => { document.getElementById('log').textContent += String.fromCharCode(10) + m; };
  const send = (message) => new Promise((resolve) => {
    const fn = window.__verifaiHandler;
    if (!fn) { resolve({ error: 'the offscreen bundle registered no listener' }); return; }
    fn(message, null, resolve);
  });

  // Wait a tick for the module's top-level listener registration.
  await new Promise((r) => setTimeout(r, 50));
  document.getElementById('log').textContent = 'probing…';

  const caps = await send({ type: 'offscreen:probe' });
  log('capabilities ' + JSON.stringify(caps));

  // A synthetic photo-ish image: no fixture file, and deterministic enough to compare runs.
  const canvas = document.createElement('canvas');
  canvas.width = 512; canvas.height = 384;
  const ctx = canvas.getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, 512, 384);
  grad.addColorStop(0, '#8899aa'); grad.addColorStop(1, '#332211');
  ctx.fillStyle = grad; ctx.fillRect(0, 0, 512, 384);
  ctx.fillStyle = '#d8b898'; ctx.beginPath(); ctx.ellipse(256, 180, 70, 95, 0, 0, 7); ctx.fill();
  ctx.fillStyle = '#221a14';
  ctx.beginPath(); ctx.ellipse(232, 160, 9, 6, 0, 0, 7); ctx.fill();
  ctx.beginPath(); ctx.ellipse(280, 160, 9, 6, 0, 0, 7); ctx.fill();
  ctx.beginPath(); ctx.ellipse(256, 215, 22, 9, 0, 0, 7); ctx.fill();

  const blob = await new Promise((r) => canvas.toBlob(r, 'image/png'));
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = ''; for (const b of bytes) binary += String.fromCharCode(b);

  log('scoring a ' + bytes.length + ' byte PNG, three passes…');
  const b64 = btoa(binary);
  let out = null;
  for (let pass = 1; pass <= 3; pass++) {
    const started = performance.now();
    out = await send({ type: 'offscreen:score', b64, filename: 'harness.png', mime: 'image/png' });
    // Pass 1 pays for the 21MB wasm and both session creations; passes 2-3 are the number
    // that matters for a user scanning a second image.
    log('pass ' + pass + ': ' + Math.round(performance.now() - started) + 'ms' + (pass === 1 ? ' (cold: wasm + session load)' : ' (warm)'));
  }
  log(JSON.stringify(out, null, 2));
  document.title = out && out.ok ? 'HARNESS OK' : 'HARNESS FAILED';
<\/script>
</body></html>`;

fs.writeFileSync(path.join(dist, 'preview-offscreen.html'), OFFSCREEN_HARNESS);

// The parity page is a real file rather than a template string: it is the one dev page with
// enough logic to be worth editing directly.
for (const page of ['parity', 'feed']) {
  fs.copyFileSync(
    path.join(root, `scripts/templates/${page}.html`),
    path.join(dist, `preview-${page}.html`)
  );
}

for (const page of ['popup', 'options', 'onboarding']) {
  const html = fs.readFileSync(path.join(dist, `${page}.html`), 'utf8');
  fs.writeFileSync(
    path.join(dist, `preview-${page}.html`),
    html.replace('</head>', `${STUB}\n  </head>`)
  );
}

// ORT loads its glue as a module and its binary as wasm; both are refused outright without
// the right Content-Type, which looks exactly like "the model is not bundled".
/**
 * What Chrome enforces on extension pages, taken from the manifest we ship, plus
 * 'unsafe-inline' — the harness pages stub chrome.* with inline scripts, which real extension
 * pages never do.
 *
 * That addition is deliberately narrow: 'unsafe-inline' does not permit WebAssembly
 * compilation, so the part of the policy that actually matters here — whether
 * 'wasm-unsafe-eval' is present — is reproduced exactly. Serve this and the harness fails
 * the same way the extension does when the manifest is wrong.
 */
const EXTENSION_CSP = (() => {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'manifest.chrome.json'), 'utf8'));
  const policy = manifest.content_security_policy?.extension_pages ?? "script-src 'self'; object-src 'self'";
  return policy.replace("script-src 'self'", "script-src 'self' 'unsafe-inline'");
})();

const TYPES = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.css': 'text/css',
  '.png': 'image/png',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
};

http
  .createServer((req, res) => {
    const file = path.join(dist, decodeURIComponent((req.url || '/').split('?')[0]));
    if (!file.startsWith(dist) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404).end('not found');
      return;
    }
    // The preview pages are served under the *extension's* CSP, not a permissive localhost
    // one. Without this the harness happily runs WebAssembly that the real extension is
    // forbidden to compile — which is exactly how a CSP failure reached a user instead of
    // this script. Read from the manifest so the two can never diverge.
    const headers = { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream' };
    if (path.extname(file) === '.html') headers['Content-Security-Policy'] = EXTENSION_CSP;

    res.writeHead(200, headers);
    fs.createReadStream(file).pipe(res);
  })
  .listen(PORT, () => {
    console.log(`preview on http://localhost:${PORT}/preview-popup.html`);
    console.log(`            http://localhost:${PORT}/preview-offscreen.html`);
    console.log(`            http://localhost:${PORT}/preview-parity.html`);
    console.log(`            http://localhost:${PORT}/preview-feed.html`);
    console.log(`            http://localhost:${PORT}/preview-options.html`);
    console.log(`            http://localhost:${PORT}/preview-onboarding.html`);
  });
