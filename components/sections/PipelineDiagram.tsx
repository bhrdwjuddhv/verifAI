'use client';

import React from 'react';
import { SectionHeading } from '@/components/ui/SectionHeading';
import { DiagramCanvas } from '@/components/diagrams/DiagramCanvas';
import { DiagramNode } from '@/components/diagrams/DiagramNode';
import { NodeTooltipData } from '@/lib/store';
import { SCORING_LEGEND } from '@/lib/verdict';
import { Eye, Binary, Mic, FileSearch, ShieldCheck } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';

const PIPELINE_NODES: { data: NodeTooltipData; icon: React.ReactNode }[] = [
  {
    data: {
      id: 'pipe-face',
      title: 'XceptionNet Face CNN',
      layer: 'Spatial Model',
      tech: 'XceptionNet / EfficientNet-B4',
      description: 'Detects unnatural facial blending boundaries, warped eye glints, and temporal jitter across video frames.',
      metrics: 'FF++ benchmark: not yet evaluated',
    },
    icon: <Eye className="w-4 h-4" />,
  },
  {
    data: {
      id: 'pipe-gan',
      title: 'GAN Frequency Fingerprint',
      layer: 'Spectral Forensics',
      tech: '2D FFT / High-Pass Residual Filter',
      description: 'Examines high-frequency 2D Fourier spectrum for periodic checkerboard grid artifacts left by upsampling layers.',
      metrics: 'StyleGAN / Latent Diffusion',
    },
    icon: <Binary className="w-4 h-4" />,
  },
  {
    data: {
      id: 'pipe-audio',
      title: 'RawNet2 Audio Spoofing',
      layer: 'Acoustic Model',
      tech: 'RawNet2 / Wav2Vec2',
      description: 'Analyzes raw audio waveforms for synthetic pitch harmonics, neural vocoder artifacts, and missing breath pauses.',
      metrics: 'ASVspoof 2021: not yet evaluated',
    },
    icon: <Mic className="w-4 h-4" />,
  },
  {
    data: {
      id: 'pipe-metadata',
      title: 'C2PA & EXIF Forensics',
      layer: 'Provenance Layer',
      tech: 'C2PA Rust SDK, ExifTool',
      description: 'Verifies cryptographic W3C/C2PA digital asset manifests, hardware camera EXIF headers, and SynthID watermarks.',
      metrics: 'Cryptographic CA Root',
    },
    icon: <FileSearch className="w-4 h-4" />,
  },
  {
    data: {
      id: 'pipe-explain',
      title: 'Grad-CAM & SHAP Heatmaps',
      layer: 'Explainability Engine',
      tech: 'Grad-CAM++, Kernel SHAP',
      description: 'Generates pixel-level visual highlight maps showing exact image regions that triggered synthetic classification.',
      metrics: 'Visual Interpretability',
    },
    icon: <ShieldCheck className="w-4 h-4" />,
  },
];

export const PipelineDiagram: React.FC = () => {
  return (
    <section id="pipeline" className="py-20 bg-ink-950 text-white border-y border-ink-800/60">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="Forensic Intelligence"
          title="Multi-Modal Detection & Fusion Pipeline"
          subtitle="Combining spatial deep learning with spectral frequency analysis, acoustic voice verification, and C2PA provenance."
          dark
        />

        <DiagramCanvas
          title="Detection Ensemble Pipeline Flow"
          subtitle="Explore the forensic neural networks and signal processors analyzing content in parallel."
          dark
        >
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 pt-2">
            {PIPELINE_NODES.map((node) => (
              <DiagramNode key={node.data.id} data={node.data} icon={node.icon} x={0} y={0} dark />
            ))}
          </div>
        </DiagramCanvas>

        {/* Verdict Legend */}
        <div className="mt-10 p-5 rounded-2xl glass-card border border-ink-800 flex flex-col md:flex-row items-center justify-between gap-4">
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-ink-300">
              Verdict Threshold Legend
            </h4>
            <p className="text-xs text-ink-400 mt-0.5 font-normal">
              Score scale evaluated across all forensic modules and APIs.
            </p>
          </div>

          <div className="flex items-center gap-2.5 flex-wrap">
            {SCORING_LEGEND.map((legend) => (
              <div key={legend.range} className="flex items-center gap-2">
                <Badge variant={legend.category} size="md">
                  {legend.range}: {legend.label}
                </Badge>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};
