/**
 * Puts the on-device weights where the build expects them.
 *
 * Sources are the ones scripts/Dockerfile already uses to provision the model service, so a
 * machine running this ends up with the same detectors the deployed server has — which is the
 * only way an on-device verdict and a deep-scan verdict can agree.
 *
 *   npm run models
 *
 * The face classifier is committed to git and needs nothing. YuNet is a plain download. NPR
 * publishes only a PyTorch checkpoint, so it has to be converted — but that no longer means
 * installing torch: scripts/export_npr_no_torch.py builds the ONNX graph from the checkpoint
 * with numpy and onnx alone.
 */

import { execFileSync, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const repo = path.join(root, '..');
const models = path.join(repo, 'models');

const YUNET_URL =
  'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx';
const NPR_URL =
  'https://github.com/chuangchuangtan/NPR-DeepfakeDetection/raw/main/model_epoch_last_3090.pth';

async function download(url, target) {
  process.stdout.write(`  fetching ${path.basename(target)} … `);
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  const bytes = new Uint8Array(await res.arrayBuffer());
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, bytes);
  console.log(`${(bytes.length / 1024).toFixed(0)}KB`);
}

function report(label, file, note) {
  const exists = fs.existsSync(file);
  const size = exists ? ` (${(fs.statSync(file).size / 1048576).toFixed(1)}MB)` : '';
  console.log(`  ${exists ? '[x]' : '[ ]'} ${label}${size}${exists ? '' : ` — ${note}`}`);
  return exists;
}

/** The first interpreter that can import what the exporter needs. */
function findPython() {
  const candidates = [
    ['python', []],
    ['python3', []],
    ['py', ['-3']],
  ];
  for (const [command, prefix] of candidates) {
    const probe = spawnSync(command, [...prefix, '-c', 'import numpy, onnx'], { encoding: 'utf8' });
    if (probe.status === 0) return { command, prefix };
  }
  return null;
}

console.log('\nOn-device detectors\n');

report(
  'face classifier   models/face/detector.onnx',
  path.join(models, 'face', 'detector.onnx'),
  'committed to git; re-clone or check out the branch again'
);

// --- YuNet: a plain ONNX download ---------------------------------------------------------
const yunet = path.join(models, 'face_detection_yunet.onnx');
if (!fs.existsSync(yunet)) await download(YUNET_URL, yunet);
report('face detector     models/face_detection_yunet.onnx', yunet, 'download failed');

// --- NPR: download the checkpoint, then convert it -----------------------------------------
const npr = path.join(models, 'npr_detector.onnx');
if (!fs.existsSync(npr)) {
  const checkpoint = path.join(models, 'npr_detector.pth');
  if (!fs.existsSync(checkpoint)) await download(NPR_URL, checkpoint);

  const python = findPython();
  if (!python) {
    console.log(`
  ! NPR ships as a PyTorch checkpoint and has to be converted once. The converter needs
    Python with numpy and onnx — no torch:

      pip install numpy onnx
      npm run models        (run this again afterwards)
`);
  } else {
    process.stdout.write('  converting npr_detector.pth -> ONNX … ');
    try {
      execFileSync(
        python.command,
        [...python.prefix, path.join('scripts', 'export_npr_no_torch.py')],
        { cwd: repo, stdio: 'pipe' }
      );
      console.log('done');
    } catch (err) {
      console.log('failed');
      console.log(`    ${String(err.stderr ?? err.message).trim().split('\n').slice(-3).join('\n    ')}`);
    }
  }
}
report('NPR whole-image   models/npr_detector.onnx', npr, 'conversion did not run — see above');

console.log('\nRun `npm run build` to copy whatever is present into dist/models/.\n');
