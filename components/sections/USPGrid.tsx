'use client';

import React from 'react';
import { SectionHeading } from '@/components/ui/SectionHeading';
import { AnimatedCard } from '@/components/ui/AnimatedCard';
import { Layers, Eye, ShieldAlert, Code2, Share2, Users } from 'lucide-react';

export interface USPItem {
  icon: React.ReactNode;
  title: string;
  description: string;
}

const USPS: USPItem[] = [
  {
    icon: <Layers className="w-6 h-6" />,
    title: 'Multi-Modal Not Single-Signal',
    description: 'Combines spatial CNNs, spectral frequency analysis, acoustic voice models, and C2PA metadata rather than relying on one fragile test.',
  },
  {
    icon: <Eye className="w-6 h-6" />,
    title: 'Explainable Not a Black Box',
    description: 'Provides Grad-CAM spatial visual heatmaps and itemized forensic reasons so you know exactly *why* content was flagged.',
  },
  {
    icon: <ShieldAlert className="w-6 h-6" />,
    title: 'Verify Before You Share, Not After',
    description: 'Seamless browser extension and mobile share-sheet integrations let you evaluate content in real time before spreading misinfo.',
  },
  {
    icon: <Code2 className="w-6 h-6" />,
    title: 'Fully Free & Open-Source Stack',
    description: 'No proprietary paywalls or subscriptions. Built with open-source PyTorch, OpenCV, Next.js, and FastAPI.',
  },
  {
    icon: <Share2 className="w-6 h-6" />,
    title: 'Cryptographic Shareable Badges',
    description: 'Generate immutable verification links and cryptographic badges that prove authenticity when sharing online.',
  },
  {
    icon: <Users className="w-6 h-6" />,
    title: 'Built for Everyday Users & Newsrooms',
    description: 'Intuitive traffic-light badges for everyday readers alongside batch REST APIs and audit trails for investigative reporters.',
  },
];

export const USPGrid: React.FC = () => {
  return (
    <section id="usp" className="py-24 relative bg-white dark:bg-ink-900 border-y border-ink-200/80 dark:border-ink-800/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="Why VerifAI"
          title="Engineered for Trust, Transparency, and Freedom"
          subtitle="Redefining how digital media authenticity is verified across the global web."
        />

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {USPS.map((usp, idx) => (
            <AnimatedCard key={usp.title} delay={idx * 0.1}>
              <div className="p-3 rounded-2xl bg-brand-blue-500/10 text-brand-blue-500 w-fit mb-4">
                {usp.icon}
              </div>
              <h3 className="text-xl font-bold text-ink-950 dark:text-white mb-2">
                {usp.title}
              </h3>
              <p className="text-sm text-ink-600 dark:text-ink-300 leading-relaxed">
                {usp.description}
              </p>
            </AnimatedCard>
          ))}
        </div>
      </div>
    </section>
  );
};
