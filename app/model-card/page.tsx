import React from 'react';
import Link from 'next/link';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Model Card — VerifAI',
  description:
    'What model VerifAI runs, what it was trained on, what its accuracy is, and what it cannot do.',
};

/**
 * Every number on this page is either measured or marked TBD. If you are tempted to fill a
 * TBD in from a model card, a paper or a dataset README: don't. It only goes here after it
 * has been measured on this pipeline.
 */
const Row: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 py-3 border-b border-ink-800/80">
    <dt className="text-xs uppercase tracking-wider text-ink-400 font-semibold">{label}</dt>
    <dd className="sm:col-span-2 text-sm text-ink-200 font-normal leading-relaxed">{children}</dd>
  </div>
);

const TBD: React.FC<{ how: string }> = ({ how }) => (
  <span>
    <span className="font-mono text-amber-300">TBD — pending evaluation</span>
    <span className="text-ink-400"> ({how})</span>
  </span>
);

export default function ModelCardPage() {
  return (
    <main className="min-h-screen bg-ink-950 text-white py-16">
      <div className="max-w-3xl mx-auto px-4 sm:px-6">
        <Link href="/" className="text-xs text-brand-blue-300 hover:underline">
          ← Back to VerifAI
        </Link>

        <h1 className="text-3xl sm:text-4xl font-bold mt-4">Model Card</h1>
        <p className="text-sm text-ink-300 mt-3 leading-relaxed font-normal">
          VerifAI runs one image classifier over the largest detected face. This page says what that
          model is, what it was trained on, and what is not yet known about it. Where a number has
          not been measured on this pipeline, it says so instead of guessing.
        </p>

        <section className="mt-10">
          <h2 className="text-lg font-semibold mb-2">Active model</h2>
          <dl>
            <Row label="Default">
              <span className="font-mono text-brand-blue-300">
                hf_fallback:prithivMLmods/Deep-Fake-Detector-v2-Model
              </span>
              <br />
              A ViT-base face deepfake classifier from Hugging Face (224px, labels{' '}
              <span className="font-mono">Realism</span> / <span className="font-mono">Deepfake</span>
              ). Used when no locally trained checkpoint is present.
            </Row>
            <Row label="Trained option">
              <span className="font-mono text-brand-blue-300">trained_checkpoint</span> — EfficientNet
              (B0 or B4) fine-tuned by <span className="font-mono">scripts/train_deepfake_detector.py</span>,
              with temperature scaling fitted on a held-out split.
            </Row>
            <Row label="Which one ran">
              Every scan result reports its own <span className="font-mono">modelSource</span>. The
              report panel shows it under “Model used”.
            </Row>
            <Row label="Input">
              One image. The largest detected face is cropped with a 35% margin (MTCNN) and resized.
              If no face is found, the verdict is <em>Uncertain</em> — the model is not applied.
            </Row>
          </dl>
        </section>

        <section className="mt-10">
          <h2 className="text-lg font-semibold mb-2">Training and evaluation data</h2>
          <dl>
            <Row label="Fallback model">
              Trained by its author on their own deepfake dataset. We did not train it and have not
              reproduced its reported numbers, so none are repeated here.
            </Row>
            <Row label="Our checkpoint">
              Face-cropped Kaggle deepfake sets (manjilkarki/deepfake-and-real-images and others
              listed in <span className="font-mono">scripts/datasets.yaml</span>), merged by{' '}
              <span className="font-mono">scripts/preprocess_faces.py</span>. Split is group-aware:
              near-duplicate images are hashed and kept on one side, so validation is not scored on
              re-compressions of the training images.
            </Row>
            <Row label="Evaluation set">
              A held-out dataset from a different source, passed with{' '}
              <span className="font-mono">--eval-dir</span>. Not yet run.
            </Row>
          </dl>
        </section>

        <section className="mt-10">
          <h2 className="text-lg font-semibold mb-2">Measured performance</h2>
          <dl>
            <Row label="Cross-dataset balanced accuracy">
              <TBD how="python scripts/train_deepfake_detector.py --eval-only --eval-dir <unseen dataset>" />
            </Row>
            <Row label="Cross-dataset AUC">
              <TBD how="same command; reported next to the in-distribution number" />
            </Row>
            <Row label="In-distribution accuracy">
              <TBD how="printed by the trainer and stored in the checkpoint as val_metrics" />
            </Row>
            <Row label="Calibration">
              The trainer fits a temperature on the holdout, and the service reports{' '}
              <span className="font-mono">calibrated: true</span> only when one was applied. The
              Hugging Face fallback is <strong>uncalibrated</strong> — its confidence is a ranking,
              not a probability.
            </Row>
          </dl>
          <p className="text-xs text-ink-400 mt-4 leading-relaxed font-normal">
            In-distribution accuracy on a Kaggle split mostly measures how well a model memorized one
            generator’s fingerprint. The cross-dataset number is the one that predicts field
            behaviour, and it is normally much lower. That is why both rows exist, and why neither is
            filled in yet.
          </p>
        </section>

        <section className="mt-10">
          <h2 className="text-lg font-semibold mb-2">Limits</h2>
          <ul className="space-y-2 text-sm text-ink-300 font-normal leading-relaxed list-disc pl-5">
            <li>Faces only. No face, no verdict.</li>
            <li>Images only. Video, audio and lip-sync detection are not implemented.</li>
            <li>
              Generators move faster than detectors. A model trained on today’s face swaps degrades on
              tomorrow’s, and the drop does not announce itself.
            </li>
            <li>
              Compression, resizing, screenshots and social-network re-encoding all erode the signal.
            </li>
            <li>
              The frequency statistic shown next to the verdict is descriptive, not diagnostic. Sharp
              real photos score high on it too, which is exactly why it does not move the verdict.
            </li>
            <li>
              A “Real” verdict means the detector found nothing, not that the file is authentic.
            </li>
          </ul>
        </section>

        <section className="mt-10 mb-10">
          <h2 className="text-lg font-semibold mb-2">Explainability</h2>
          <p className="text-sm text-ink-300 font-normal leading-relaxed">
            Each verdict ships a Grad-CAM overlay showing which regions moved the model’s score. It
            reveals where the model looked — it is not evidence that those regions were edited, and a
            confident-looking heatmap over a wrong verdict is still a wrong verdict.
          </p>
        </section>
      </div>
    </main>
  );
}
