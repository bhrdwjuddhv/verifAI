/**
 * First run. No decisions.
 *
 * This page used to make people choose a scan mode before they had scanned anything — a trade
 * between privacy and capability that nobody can weigh cold. On-device is now the default and
 * the only thing that happens by itself, so the page explains rather than asks. The single
 * question worth asking — "may this file be uploaded?" — is asked at the moment it applies,
 * in the popup, naming the destination.
 */

import '../ui/ui.css';
import '../ui/page.css';

import { clear, h } from '../ui/dom';

/**
 * Says what this build can actually do, read from the manifest the build wrote.
 *
 * Describing the bundled detectors from memory is how the copy ends up claiming a model that
 * is not in the package — which is exactly what an earlier version of this page did.
 */
async function paintDetectors(): Promise<void> {
  const target = document.getElementById('detectors')!;
  clear(target);

  const names: Record<string, string> = {
    'detector.onnx': 'Face manipulation classifier',
    'yunet.onnx': 'Face detector',
    'npr.onnx': 'Whole-image AI-generation detector',
  };

  try {
    const res = await fetch(chrome.runtime.getURL('models/models.json'));
    if (!res.ok) throw new Error('no manifest');
    const manifest = (await res.json()) as { models: { file: string }[]; missing: string[] };

    for (const model of manifest.models) {
      const label = names[model.file];
      if (label) target.append(h('dt', { text: label }), h('dd', { class: 'pill-ok', text: 'bundled' }));
    }
    for (const absent of manifest.missing) {
      target.append(
        h('dt', { text: absent.split('/').pop() ?? absent }),
        h('dd', { class: 'muted', text: 'not in this build — it abstains, and every verdict says so' })
      );
    }
  } catch {
    target.append(h('dt', { text: 'Detectors' }), h('dd', { class: 'muted', text: 'see options' }));
  }
}

document.getElementById('done')!.addEventListener('click', () => {
  window.close();
});

document.getElementById('open-options')!.addEventListener('click', () => {
  void chrome.tabs.create({ url: chrome.runtime.getURL('options.html') });
});

void paintDetectors();
