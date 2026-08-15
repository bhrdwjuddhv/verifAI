'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { SectionHeading } from '@/components/ui/SectionHeading';
import { AnimatedCard } from '@/components/ui/AnimatedCard';
import { Badge } from '@/components/ui/Badge';
import { useAppStore } from '@/lib/store';
import { Code, Server, Cpu, Search, Database, Terminal, ShieldCheck } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface TechItem {
  id: string;
  name: string;
  category: 'frontend' | 'backend' | 'aiml' | 'forensics' | 'data' | 'devops';
  categoryLabel: string;
  purpose: string;
  icon: React.ReactNode;
}

const TECH_STACK_ITEMS: TechItem[] = [
  {
    id: 'fe-next',
    name: 'Next.js & Tailwind CSS',
    category: 'frontend',
    categoryLabel: 'Frontend',
    purpose: 'Server-side rendering, layout optimization, design-token system.',
    icon: <Code className="w-4 h-4 stroke-[2.2]" />,
  },
  {
    id: 'fe-ext',
    name: 'Chrome Extension MV3',
    category: 'frontend',
    categoryLabel: 'Frontend',
    purpose: 'Right-click context menu media verification in browser.',
    icon: <Code className="w-4 h-4 stroke-[2.2]" />,
  },
  {
    id: 'fe-expo',
    name: 'React Native & Expo',
    category: 'frontend',
    categoryLabel: 'Frontend',
    purpose: 'Cross-platform mobile client with native system share-sheet hook.',
    icon: <Code className="w-4 h-4 stroke-[2.2]" />,
  },
  {
    id: 'be-fastapi',
    name: 'FastAPI (Python)',
    category: 'backend',
    categoryLabel: 'Backend',
    purpose: 'Asynchronous REST API backend with OpenAPI schemas.',
    icon: <Server className="w-4 h-4 stroke-[2.2]" />,
  },
  {
    id: 'be-celery',
    name: 'Celery & Redis Workers',
    category: 'backend',
    categoryLabel: 'Backend',
    purpose: 'Distributed async task queues for video inference.',
    icon: <Server className="w-4 h-4 stroke-[2.2]" />,
  },
  {
    id: 'ai-pytorch',
    name: 'PyTorch & TorchVision',
    category: 'aiml',
    categoryLabel: 'AI / ML',
    purpose: 'Deep learning framework for spatial CNN & acoustic models.',
    icon: <Cpu className="w-4 h-4 fill-brand-blue-400/20 stroke-[2]" />,
  },
  {
    id: 'ai-onnx',
    name: 'ONNX Runtime Engine',
    category: 'aiml',
    categoryLabel: 'AI / ML',
    purpose: 'Accelerated cross-platform neural network inference execution.',
    icon: <Cpu className="w-4 h-4 fill-brand-blue-400/20 stroke-[2]" />,
  },
  {
    id: 'for-exif',
    name: 'ExifTool & ELA Module',
    category: 'forensics',
    categoryLabel: 'Forensics',
    purpose: 'EXIF metadata extraction and Error Level Analysis compression check.',
    icon: <Search className="w-4 h-4 stroke-[2.2]" />,
  },
  {
    id: 'for-c2pa',
    name: 'C2PA Rust Toolkit',
    category: 'forensics',
    categoryLabel: 'Forensics',
    purpose: 'W3C Coalition for Content Provenance and Authenticity validation.',
    icon: <Search className="w-4 h-4 stroke-[2.2]" />,
  },
  {
    id: 'da-pg',
    name: 'PostgreSQL & MinIO',
    category: 'data',
    categoryLabel: 'Data & Storage',
    purpose: 'Relational data persistence and S3-compatible media storage.',
    icon: <Database className="w-4 h-4 stroke-[2.2]" />,
  },
  {
    id: 'do-docker',
    name: 'Docker & HuggingFace',
    category: 'devops',
    categoryLabel: 'DevOps',
    purpose: 'Containerized orchestration and zero-cost cloud GPU hosting.',
    icon: <Terminal className="w-4 h-4 stroke-[2.2]" />,
  },
];

const CATEGORIES = [
  { id: 'all', label: 'All Layers' },
  { id: 'frontend', label: 'Frontend' },
  { id: 'backend', label: 'Backend' },
  { id: 'aiml', label: 'AI / ML' },
  { id: 'forensics', label: 'Forensics' },
  { id: 'data', label: 'Data' },
  { id: 'devops', label: 'DevOps' },
];

export const TechStack: React.FC = () => {
  const { techFilter, setTechFilter } = useAppStore();

  const filteredItems =
    techFilter === 'all'
      ? TECH_STACK_ITEMS
      : TECH_STACK_ITEMS.filter((item) => item.category === techFilter);

  return (
    <section id="tech-stack" className="py-20 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="Open Source Foundation"
          title="Open-Source Technology Stack"
          subtitle="Built entirely on battle-tested open-source libraries and frameworks — zero proprietary software licenses required."
        />

        {/* Tab Category Filter */}
        <div className="flex items-center justify-center gap-2 flex-wrap mb-10">
          {CATEGORIES.map((cat) => {
            const isSelected = techFilter === cat.id;
            return (
              <button
                key={cat.id}
                type="button"
                onClick={() => setTechFilter(cat.id)}
                className={cn(
                  'px-3.5 py-1.5 text-xs font-medium rounded-xl transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-blue-400 select-none',
                  isSelected
                    ? 'bg-brand-blue-400 text-ink-950 font-semibold shadow-sm'
                    : 'glass-card text-ink-300 hover:text-white border-ink-800'
                )}
              >
                {cat.label}
              </button>
            );
          })}
        </div>

        {/* Tech Grid */}
        <motion.div layout className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          <AnimatePresence>
            {filteredItems.map((item) => (
              <AnimatedCard key={item.id} className="flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between gap-2 mb-3">
                    <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-blue-400/30 via-brand-blue-500/15 to-transparent border border-brand-blue-400/40 flex items-center justify-center text-brand-blue-300 shadow-glow-sm">
                      {item.icon}
                    </div>
                    <Badge variant="blue" size="sm" dot={false}>
                      Free / OSS
                    </Badge>
                  </div>

                  <h3 className="text-base font-semibold text-white">
                    {item.name}
                  </h3>
                  <p className="text-xs text-ink-300 mt-1.5 leading-relaxed font-normal">
                    {item.purpose}
                  </p>
                </div>

                <div className="mt-4 pt-3 border-t border-ink-800/60 text-[11px] font-mono text-ink-400">
                  Layer: {item.categoryLabel}
                </div>
              </AnimatedCard>
            ))}
          </AnimatePresence>
        </motion.div>

        {/* Callout Banner */}
        <div className="mt-14 p-6 rounded-2xl glass-card border border-brand-blue-400/30 text-white flex flex-col md:flex-row items-center justify-between gap-5 shadow-glow-sm">
          <div className="flex items-center gap-3.5">
            <div className="w-11 h-11 rounded-2xl bg-gradient-to-br from-brand-blue-400/35 via-brand-blue-500/20 to-transparent border border-brand-blue-400/40 flex items-center justify-center text-brand-blue-300 shadow-glow-sm flex-shrink-0">
              <ShieldCheck className="w-6 h-6 fill-brand-blue-400/20 stroke-[2]" />
            </div>
            <div>
              <h4 className="text-base font-semibold">Zero License Cost Guarantee</h4>
              <p className="text-xs text-ink-300 mt-0.5 max-w-xl font-normal">
                Every tool in this stack is open-source — allowing newsrooms and developers to build and host at zero software licensing cost.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
