/**
 * Live Guard checks: model provenance, call-site detection, and that the shipped bundle
 * actually contains our trained model.
 *
 *   node scripts/check-liveguard.mjs
 *
 * The provenance rule is the one worth a test. Everything else in Live Guard degrades
 * visibly when it breaks; a verdict quietly produced by the wrong model looks exactly like a
 * verdict produced by the right one.
 */

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));   // extension/scripts
const repo = path.join(here, '..');                          // extension/

// --- provenance ---------------------------------------------------------------------------
const { isOurModel, rejectForeignModel } = await import('../src/shared/liveguard.ts');

assert.equal(isOurModel('onnx:trained_checkpoint(detector_v2.onnx)'), true, 'the face model is ours');
assert.equal(isOurModel('onnx:trained_audio_checkpoint(audio_detector.onnx)'), true, 'the voice model is ours');
assert.equal(isOurModel('onnx:npr:npr_detector.pth+int8'), true, 'NPR is ours');
assert.equal(
  isOurModel('onnx:trained_checkpoint(detector_v2.onnx) + onnx:npr:npr_detector.pth+int8'),
  true,
  'a fused source naming both is ours'
);

assert.equal(isOurModel('hf_fallback:prithivMLmods/Deep-Fake-Detector-v2-Model'), false);
assert.equal(isOurModel('onnx:hf_fallback:prithivMLmods/Deep-Fake-Detector-v2-Model+int8'), false);
assert.equal(isOurModel('some-vendor-api/v1'), false);
assert.equal(isOurModel(null), false);
assert.equal(isOurModel(''), false);
assert.equal(isOurModel(undefined), false);

assert.equal(rejectForeignModel('onnx:trained_checkpoint(detector_v2.onnx)'), null, 'ours passes');
assert.match(
  rejectForeignModel('onnx:hf_fallback:prithivMLmods/Deep-Fake-Detector-v2-Model+int8') ?? '',
  /third-party fallback/,
  'the HF fallback must be named and refused, not silently shown'
);
assert.match(rejectForeignModel(null) ?? '', /did not say which model/);
assert.match(rejectForeignModel('mystery-model') ?? '', /unrecognized model/);

// --- call sites ---------------------------------------------------------------------------
const { siteForUrl, isInCall, CALL_ORIGINS } = await import('../src/shared/callsites.ts');

assert.equal(siteForUrl('https://meet.google.com/abc-defg-hij')?.id, 'meet');
assert.equal(siteForUrl('https://discord.com/channels/123/456')?.id, 'discord');
assert.equal(siteForUrl('https://teams.microsoft.com/_#/call/123')?.id, 'teams');
assert.equal(siteForUrl('https://us02web.zoom.us/wc/123/join')?.id, 'zoom', 'zoom subdomains match');
assert.equal(siteForUrl('https://example.com/'), null);
assert.equal(siteForUrl(undefined), null);
assert.equal(siteForUrl('not a url'), null);

// In a call vs merely on the platform — starting a guard on a home page would monitor nothing.
assert.equal(isInCall('https://meet.google.com/abc-defg-hij'), true);
assert.equal(isInCall('https://meet.google.com/'), false, 'the Meet landing page is not a call');
assert.equal(isInCall('https://discord.com/channels/123/456'), true);
assert.equal(isInCall('https://discord.com/app'), false);
assert.equal(isInCall('https://example.com/'), false);

// A near-miss hostname must not match: evil-meet.google.com.attacker.test is not Meet.
assert.equal(siteForUrl('https://meet.google.com.attacker.test/abc-defg-hij'), null);

// --- the shipped bundle -------------------------------------------------------------------
const dist = path.join(repo, 'dist');
if (fs.existsSync(dist)) {
  const manifest = JSON.parse(fs.readFileSync(path.join(dist, 'manifest.json'), 'utf8'));
  assert.ok(manifest.permissions.includes('tabCapture'), 'tabCapture must be in the manifest');
  for (const origin of CALL_ORIGINS) {
    assert.ok(
      manifest.optional_host_permissions.includes(origin),
      `${origin} must be an optional host permission, not a granted one`
    );
  }

  const metaPath = path.join(dist, 'models', 'detector.json');
  assert.ok(fs.existsSync(metaPath), 'the face model must be bundled');
  const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
  assert.ok(
    isOurModel(`onnx:${meta.source}`),
    `the bundled face model must be ours, got source="${meta.source}"`
  );
  assert.equal(meta.source, 'trained_checkpoint', 'must be the trained model, not the HF fallback');
  assert.ok(meta.calibrated === true, 'the v2 export carries its fitted temperature');

  const nprMeta = JSON.parse(fs.readFileSync(path.join(dist, 'models', 'npr.json'), 'utf8'));
  assert.ok(isOurModel(`onnx:${nprMeta.source}`), `NPR bundle must be ours, got "${nprMeta.source}"`);
  console.log(`  bundle: face source="${meta.source}" calibrated=${meta.calibrated}, npr="${nprMeta.source}"`);
} else {
  // Not a pass. The bundle assertions are the point of this file.
  console.error(`  dist/ not found at ${dist} — run \`npm run build\` first.`);
  process.exit(1);
}

console.log('live guard selfcheck passed');
