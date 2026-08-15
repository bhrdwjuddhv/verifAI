'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { SectionHeading } from '@/components/ui/SectionHeading';
import { UploadCloud, Hash, Cpu, Layers, FileCheck, Share2 } from 'lucide-react';

export interface StepItem {
  number: string;
  title: string;
  description: string;
  icon: React.ReactNode;
}

const STEPS: StepItem[] = [
  {
    number: '01',
    title: 'Ingest Media',
    description: 'Upload image, video, audio, or paste a public URL.',
    icon: <UploadCloud className="w-5 h-5 fill-brand-blue-400/20 stroke-[2.2]" />,
  },
  {
    number: '02',
    title: 'Hash Verification',
    description: 'SHA-256 fingerprint check against verified cache.',
    icon: <Hash className="w-5 h-5 stroke-[2.5]" />,
  },
  {
    number: '03',
    title: 'Parallel Forensics',
    description: 'CNN boundary check, GAN frequency residual & audio spoofing.',
    icon: <Cpu className="w-5 h-5 fill-brand-blue-400/20 stroke-[2]" />,
  },
  {
    number: '04',
    title: 'Ensemble Fusion',
    description: 'Multi-modal weighted probability evaluation.',
    icon: <Layers className="w-5 h-5 fill-brand-blue-400/20 stroke-[2]" />,
  },
  {
    number: '05',
    title: 'Explainable Report',
    description: 'Grad-CAM heatmaps & spatial evidence highlights.',
    icon: <FileCheck className="w-5 h-5 fill-brand-blue-400/20 stroke-[2]" />,
  },
  {
    number: '06',
    title: 'Shareable Badge',
    description: 'Generate cryptographic trust badge link.',
    icon: <Share2 className="w-5 h-5 fill-brand-blue-400/20 stroke-[2]" />,
  },
];

export const HowItWorks: React.FC = () => {
  return (
    <section id="how-it-works" className="py-20 relative overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="Verification Workflow"
          title="From Raw Media to Explainable Verdict"
          subtitle="A deterministic multi-stage detection pipeline engineered for speed, mathematical rigor, and transparency."
        />

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 relative">
          {STEPS.map((step, idx) => (
            <motion.div
              key={step.number}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.45, delay: idx * 0.08 }}
              whileHover={{ y: -4 }}
              className="glass-card rounded-2xl p-5 border-gradient shimmer transition-all duration-300 group"
            >
              <div className="flex items-center justify-between mb-4">
                <span className="text-2xl font-semibold font-mono text-brand-blue-400/80 group-hover:text-brand-blue-300 transition-colors">
                  {step.number}
                </span>
                <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-brand-blue-400/30 via-brand-blue-500/15 to-transparent border border-brand-blue-400/40 flex items-center justify-center text-brand-blue-300 shadow-glow-sm group-hover:scale-110 transition-transform">
                  {step.icon}
                </div>
              </div>

              <h3 className="text-base font-semibold text-white mb-1.5">
                {step.title}
              </h3>
              <p className="text-xs text-ink-300 leading-relaxed font-normal">
                {step.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};
