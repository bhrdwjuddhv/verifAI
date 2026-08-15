'use client';

import React from 'react';
import dynamic from 'next/dynamic';
import { LazyMotion, domAnimation, motion, AnimatePresence, useScroll, useSpring } from 'framer-motion';
import { Nav } from '@/components/sections/Nav';
import { Hero } from '@/components/sections/Hero';
import { HowItWorks } from '@/components/sections/HowItWorks';
import { ArchitectureDiagram } from '@/components/sections/ArchitectureDiagram';
import { PipelineDiagram } from '@/components/sections/PipelineDiagram';
import { TechStack } from '@/components/sections/TechStack';
import { ThreeWaysIn } from '@/components/sections/ThreeWaysIn';
import { FinalCTA } from '@/components/sections/FinalCTA';
import { Footer } from '@/components/sections/Footer';
import { useAppStore } from '@/lib/store';
import { CheckCircle2 } from 'lucide-react';

// Dynamic Full-Page 3D Blockchain Background
const BlockchainBackground = dynamic(() => import('@/components/three/BlockchainBackground'), {
  ssr: false,
});

export default function Home() {
  const { toastMessage } = useAppStore();

  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, {
    stiffness: 120,
    damping: 25,
    restDelta: 0.001,
  });

  return (
    <LazyMotion features={domAnimation}>
      <div className="relative min-h-screen flex flex-col bg-ink-950 text-ink-100 selection:bg-brand-blue-500/30 selection:text-white overflow-x-hidden">
        {/* Minimal Top Scroll Progress Line */}
        <motion.div
          id="scroll-progress"
          style={{ scaleX }}
        />

        {/* Full-Viewport 3D Blockchain Network Background Layer */}
        <BlockchainBackground />

        {/* Subtle Ambient Orbs Overlay */}
        <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden opacity-40">
          <div className="orb orb-1 top-[-15%] left-[-10%]" />
          <div className="orb orb-2 top-[40%] right-[-15%]" />
          <div className="orb orb-3 bottom-[-10%] left-[25%]" />
          <div className="absolute inset-0 dot-grid opacity-15 pointer-events-none" />
        </div>

        {/* Navigation */}
        <Nav />

        {/* Main Content */}
        <main className="relative z-10 flex-grow">
          <Hero />
          <HowItWorks />
          <ArchitectureDiagram />
          <PipelineDiagram />
          <TechStack />
          <ThreeWaysIn />
          <FinalCTA />
        </main>

        {/* Footer */}
        <Footer />

        {/* Global Toast */}
        <AnimatePresence>
          {toastMessage && (
            <motion.div
              initial={{ opacity: 0, y: 40, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 20, scale: 0.95 }}
              transition={{ duration: 0.25, ease: 'easeOut' }}
              className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 px-5 py-3 rounded-xl glass-card border border-brand-blue-400/40 shadow-glow-sm flex items-center gap-2.5"
            >
              <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
              <span className="text-xs font-medium text-white">{toastMessage}</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </LazyMotion>
  );
}
