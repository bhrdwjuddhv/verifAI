'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { SectionHeading } from '@/components/ui/SectionHeading';
import { Badge } from '@/components/ui/Badge';
import { CheckCircle, Clock, Sparkles, Database } from 'lucide-react';

export interface PhaseItem {
  phase: string;
  title: string;
  status: 'Completed' | 'In Progress' | 'Upcoming';
  items: string[];
}

const PHASES: PhaseItem[] = [
  {
    phase: 'Phase 1',
    title: 'Foundation & Core Engine',
    status: 'Completed',
    items: [
      'Next.js 14 App Router & Tailwind design token system',
      'FastAPI REST gateway & Celery task worker architecture',
      'OpenCV keyframe extraction & FFmpeg audio demuxing',
    ],
  },
  {
    phase: 'Phase 2',
    title: 'Model Integration',
    status: 'Completed',
    items: [
      'XceptionNet & EfficientNet-B4 spatial deepfake CNNs',
      'RawNet2 & Wav2Vec2 acoustic voice spoofing detection',
      'Error Level Analysis (ELA) & 2D FFT spectral residual check',
    ],
  },
  {
    phase: 'Phase 3',
    title: 'Forensics & Dashboard UI',
    status: 'Completed',
    items: [
      'Grad-CAM explainability heatmap generator',
      'Interactive Recharts sub-module breakdown panel',
      'C2PA provenance manifest cryptographic validator',
    ],
  },
  {
    phase: 'Phase 4',
    title: 'Browser & Mobile Extensions',
    status: 'In Progress',
    items: [
      'Chrome Extension Manifest V3 right-click context menu',
      'React Native Expo mobile app with native share-sheet hook',
      'Offline hash lookup cache for instant repeat scans',
    ],
  },
  {
    phase: 'Phase 5',
    title: 'Deploy & Open Source Community',
    status: 'Upcoming',
    items: [
      'Zero-cost cloud GPU hosting on HuggingFace Spaces & Render',
      'Open-source public GitHub release & API documentation',
      'Newsroom batch verification API portal',
    ],
  },
];

export const Roadmap: React.FC = () => {
  return (
    <section id="roadmap" className="py-24 bg-ink-50/60 dark:bg-ink-950/60 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="Development Milestones"
          title="VerifAI Strategic Product Roadmap"
          subtitle="Tracking our progress from initial prototype architecture to full open-source ecosystem rollout."
        />

        {/* Timeline Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6 relative">
          {PHASES.map((phase, idx) => (
            <motion.div
              key={phase.phase}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: idx * 0.1 }}
              className="p-6 rounded-2xl bg-white dark:bg-ink-900 border border-ink-200 dark:border-ink-800 shadow-glass flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between gap-2 mb-3">
                  <span className="text-xs font-mono font-bold text-brand-blue-500 uppercase tracking-widest">
                    {phase.phase}
                  </span>
                  <span
                    className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                      phase.status === 'Completed'
                        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                        : phase.status === 'In Progress'
                        ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
                        : 'bg-ink-800 text-ink-400 border border-ink-700'
                    }`}
                  >
                    {phase.status}
                  </span>
                </div>

                <h3 className="text-base font-bold text-ink-950 dark:text-white mb-3">
                  {phase.title}
                </h3>

                <ul className="space-y-2 text-xs text-ink-600 dark:text-ink-300">
                  {phase.items.map((item, i) => (
                    <li key={i} className="flex items-start gap-1.5 leading-relaxed">
                      <span className="w-1 h-1 rounded-full bg-brand-blue-400 mt-1.5 flex-shrink-0" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Dataset Note */}
        <div className="mt-12 p-5 rounded-2xl bg-brand-blue-500/10 border border-brand-blue-400/30 flex items-center gap-3 text-xs text-brand-blue-300">
          <Database className="w-4 h-4 flex-shrink-0 text-brand-blue-400" />
          <span>
            <strong>Benchmark Datasets Used:</strong> Models are trained and benchmarked on FaceForensics++, DeepFake Detection Challenge (DFDC), Celeb-DF, and ASVspoof 2021 evaluations.
          </span>
        </div>
      </div>
    </section>
  );
};
