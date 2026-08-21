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
          VerifAI runs two image detectors that answer different questions, and combines them into
          one score. This page says what each model is, what it was trained on, and what is not yet
          known about it. Where a number has not been measured on this pipeline, it says so instead
          of guessing.
        </p>

        <section className="mt-10">
          <h2 className="text-lg font-semibold mb-2">Detectors</h2>
          <dl>
            <Row label="1. Face classifier">
              Answers “was this face swapped or manipulated”. Runs on the largest detected face,
              cropped with a 35% margin. Abstains — casts no vote — when no face is found.
              <br />
              Currently <span className="font-mono text-brand-blue-300">onnx:trained_checkpoint</span>:
              EfficientNet-B0 fine-tuned on Kaggle deepfake data, classes{' '}
              <span className="font-mono">Fake</span>/<span className="font-mono">Real</span>, with a
              temperature of 0.72 fitted on its own held-out split — so its confidence is calibrated,
              unlike the third-party fallback it replaced.
            </Row>
            <Row label="2. NPR (whole image)">
              <span className="font-mono text-brand-blue-300">NPR</span>, Tan et al., CVPR 2024,
              ported from the official repo. Answers “did a generator’s decoder make these pixels”
              by reading the up-sampling artifact, so it catches fully synthetic images the face
              classifier is blind to — including generated faces. Needs no face. Weights are the
              authors’ ProGAN-trained checkpoint unless you train your own.
            </Row>
            <Row label="3. Voice model">
              Mel-spectrogram → classifier, for audio uploads. <strong>Not trained yet</strong>: the
              route returns 501 with the reason until <span className="font-mono">models/audio_detector.onnx</span>{' '}
              exists. See <span className="font-mono">youhavetodo.md</span>.
            </Row>
            <Row label="How they combine">
              A weighted mean over whichever detectors actually ran (
              <span className="font-mono">FUSION_WEIGHTS</span>, default face 0.5 / NPR 0.5), then
              the verdict bands. Every per-detector score stays visible in the report, so a
              disagreement is legible rather than averaged away.
            </Row>
            <Row label="Supporting signals">
              High-frequency FFT energy, video frame-to-frame variance, and C2PA/EXIF presence are
              measured and displayed but carry <strong>no weight</strong> in the verdict — none of
              them has an established direction on this data.
            </Row>
          </dl>
        </section>

        <section className="mt-10">
          <h2 className="text-lg font-semibold mb-2">Model files</h2>
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
            <Row label="Lean deployment">
              <span className="font-mono text-brand-blue-300">onnx:…+int8</span> — the same weights
              exported to ONNX and int8-quantized so the service fits a 512MB instance. Scores shift
              by a point or two versus full precision; the response says so in its notes.
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
              One image, one short video clip, or one voice clip. Video is sampled (default 16
              frames, evenly spaced), each frame scored by the image detectors, and the results
              averaged. If no face is found, the face classifier abstains and NPR decides.
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
            <Row label="Face classifier, in-distribution">
              <span className="font-mono text-emerald-300">balanced accuracy 0.717 · AUC 0.779</span>
              <span className="text-ink-400">
                {' '}— measured on the held-out split of its own training data (39,898 validation
                images; recall 0.729 fake / 0.706 real), reported by the trainer and stored in the
                checkpoint as <span className="font-mono">val_metrics</span>.
              </span>
            </Row>
            <Row label="Face classifier, cross-dataset">
              <TBD how="python scripts/train_deepfake_detector.py --eval-only --eval-dir <unseen dataset>" />
            </Row>
            <Row label="NPR, cross-generator">
              <TBD how="python scripts/train_npr.py --eval-only --eval-dir <unseen generators>" />
            </Row>
            <Row label="Fused, on a labelled set">
              <TBD how="python scripts/tune_fusion.py --data-dir <labelled set> — also picks the weights" />
            </Row>
            <Row label="Voice model, clean audio">
              <span className="font-mono text-amber-300">reported 1.000 — not trustworthy</span>
              <span className="text-ink-400">
                {' '}The trained checkpoint records perfect validation accuracy on 14,721 samples,
                with <span className="font-mono">split: &quot;stratified&quot;</span> — i.e. not
                source-held-out. A perfect score on ASVspoof + FoR is evidence that the same
                corpora sat on both sides of the split, not that the detector is perfect. Every
                voice verdict carries this caveat in its notes.
              </span>
            </Row>
            <Row label="Voice model, call conditions">
              <TBD how="python scripts/eval_call_audio.py --data-dir <held-out corpus> — G.711, narrowband, packet loss, noise" />
              <span className="text-ink-400">
                {' '}This is the number that describes Live Guard. A model measured on studio WAVs
                has not been measured for a phone call: G.711 companding, the 3.4kHz telephone
                band and dropped packets all remove the high-frequency detail these detectors
                lean on.
              </span>
            </Row>
            <Row label="Cross-dataset AUC">
              <TBD how="same command; reported next to the in-distribution number" />
            </Row>

            <Row label="Calibration">
              The active face model carries a fitted temperature (0.72) and the service reports{' '}
              <span className="font-mono">calibrated: true</span>. NPR and the Hugging Face fallback
              are <strong>uncalibrated</strong> — their confidence is a ranking, not a probability.
            </Row>
          </dl>
          <p className="text-xs text-ink-400 mt-4 leading-relaxed font-normal">
            0.717 balanced accuracy is a real measurement, and it is also a modest one — roughly
            seven correct calls in ten, on data drawn from the same source it trained on. Accuracy on
            a <em>different</em> generator is normally lower still, and that row is empty because
            nobody has run it. Treat the in-distribution number as an upper bound, not a promise.
          </p>
        </section>

        <section className="mt-10">
          <h2 className="text-lg font-semibold mb-2">Limits</h2>
          <ul className="space-y-2 text-sm text-ink-300 font-normal leading-relaxed list-disc pl-5">
            <li>
              Images and short video. Voice has a route but no trained model; lip-sync detection is
              not implemented at all.
            </li>
            <li>
              Video re-encoding weakens NPR badly — measured here: an AI image scoring 100 as a
              still scored 0 once re-encoded into an mp4. On video the face classifier does most of
              the work.
            </li>
            <li>
              NPR is trained on generators up to a point in time. A brand-new generator with a
              different up-sampling scheme may not leave the artifact it looks for.
            </li>
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
            Each verdict ships a heatmap showing which regions moved the model’s score: Grad-CAM in
            the full build, occlusion saliency (hide a patch, measure the drop) in the lean ONNX
            build. Either way it reveals where the model looked — it is not evidence that those
            regions were edited, and a confident-looking heatmap over a wrong verdict is still a
            wrong verdict.
          </p>
        </section>
      </div>
    </main>
  );
}
