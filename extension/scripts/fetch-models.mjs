/**
 * Puts the on-device weights where the build expects them.
 *
 * Sources are the ones scripts/Dockerfile already uses to provision the model service, so a
 * machine running this ends up with the same detectors the deployed server has — which is
 * the only way an on-device verdict and a deep-scan verdict can agree.
 *
 *   npm run models
 *
 * The face classifier is committed to git and needs nothing. YuNet is a plain ONNX download.
 * NPR is the awkward one: only a PyTorch checkpoint is published, so producing the ONNX needs
 * torch — which is why this script tells you how rather than pretending it can.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const models = path.join(fileURLToPath(new URL('..', import.meta.url)), '..', 'models');

const YUNET_URL =
  'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx';

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

console.log('\nOn-device detectors\n');

report('face classifier   models/face/detector.onnx', path.join(models, 'face', 'detector.onnx'), 'committed to git; run `git lfs pull` or re-clone');

const yunet = path.join(models, 'face_detection_yunet.onnx');
if (!fs.existsSync(yunet)) {
  await download(YUNET_URL, yunet);
}
report('face detector     models/face_detection_yunet.onnx', yunet, 'download failed');

const npr = path.join(models, 'npr_detector.onnx');
if (!report('NPR whole-image   models/npr_detector.onnx', npr, 'needs an export, see below')) {
  console.log(`
  NPR ships only as a PyTorch checkpoint, and exporting it needs torch — which has no
  wheels for Python 3.14. Either route produces models/npr_detector.onnx:

    Docker (what the deployed service uses):
      docker build -f scripts/Dockerfile --target builder -t verifai-models scripts/
      docker create --name m verifai-models && docker cp m:/build/models/npr_detector.onnx models/

    Python 3.11 or 3.12:
      pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
      pip install onnx onnxruntime onnxscript
      curl -L -o models/npr_detector.pth \\
        https://github.com/chuangchuangtan/NPR-DeepfakeDetection/raw/main/model_epoch_last_3090.pth
      python scripts/export_onnx.py --npr

  Do NOT pass --no-quantize: the deployed service reports "onnx:npr:npr_detector.pth+int8",
  so the int8 export (the exporter's default) is the one that matches it.

  Until then on-device mode runs face-only and says so in every verdict's notes — while the
  deployed server fuses face and NPR at 0.5/0.5, so the two will not agree.`);
}

console.log('\nRun `npm run build` to copy whatever is present into dist/models/.\n');
