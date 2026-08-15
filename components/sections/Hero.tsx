'use client';

import React from 'react';
import dynamic from 'next/dynamic';
import { motion } from 'framer-motion';
import { ArrowRight, Play, Sparkles, ShieldCheck, CheckCircle2, Lock } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { ScanDemo } from '@/components/scan/ScanDemo';

const HeroScene = dynamic(() => import('@/components/three/HeroScene'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full min-h-[400px] bg-gradient-to-tr from-brand-blue-300/20 via-brand-blue-400/10 to-transparent rounded-full blur-3xl" />
  ),
});

export const Hero: React.FC = () => {
  const scrollToScan = () => {
    const demoEl = document.getElementById('live-scan-demo');
    if (demoEl) {
      demoEl.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const scrollToHowItWorks = () => {
    const howEl = document.getElementById('how-it-works');
    if (howEl) {
      howEl.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section id="hero" className="relative pt-32 pb-20 sm:pt-36 sm:pb-28 overflow-hidden">
      {/* 3D WebGL Background Scene */}
      <div className="absolute top-0 right-0 w-full lg:w-1/2 h-[600px] pointer-events-none z-0 opacity-80">
        <HeroScene />
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="text-center max-w-3xl mx-auto space-y-6">
          {/* Badge Tag */}
          <motion.div
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            className="inline-block"
          >
            <Badge variant="blue" size="md" className="gap-2 px-3.5 py-1">
              <Sparkles className="w-3.5 h-3.5 text-brand-blue-400" />
              Open-Source Media Forensics Engine
            </Badge>
          </motion.div>

          {/* Headline - Refined typography weight */}
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
            className="text-4xl sm:text-5xl md:text-6xl font-semibold tracking-tight leading-[1.1] text-white"
          >
            Know what is real <br className="hidden sm:inline" />
            before you <span className="text-gradient-blue">hit share.</span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="text-base sm:text-lg text-ink-300 max-w-2xl mx-auto leading-relaxed font-normal"
          >
            An open-source digital forensics platform that detects manipulated or AI-generated images, video, and voice with transparent, explainable scores.
          </motion.p>

          {/* Action CTAs */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="flex flex-col sm:flex-row items-center justify-center gap-3.5 pt-2"
          >
            <Button
              variant="primary"
              size="lg"
              onClick={scrollToScan}
              rightIcon={<ArrowRight className="w-4 h-4" />}
              className="w-full sm:w-auto"
            >
              Try a Live Scan
            </Button>

            <Button
              variant="secondary"
              size="lg"
              onClick={scrollToHowItWorks}
              leftIcon={<Play className="w-4 h-4 text-brand-blue-400" />}
              className="w-full sm:w-auto"
            >
              See How It Works
            </Button>
          </motion.div>

          {/* Metrics Ribbon */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="pt-4 flex items-center justify-center gap-6 text-xs text-ink-400 flex-wrap"
          >
            <div className="flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              <span>Multi-Modal Forensics</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Lock className="w-3.5 h-3.5 text-brand-blue-400" />
              <span>C2PA Manifest Verified</span>
            </div>
            <div className="flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-amber-400" />
              <span>Zero-Log Privacy</span>
            </div>
          </motion.div>
        </div>

        {/* Live Interactive Forensic Scan Sandbox */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="mt-14 sm:mt-20"
        >
          <ScanDemo />
        </motion.div>
      </div>
    </section>
  );
};
