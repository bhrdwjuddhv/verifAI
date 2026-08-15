'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { SectionHeading } from '@/components/ui/SectionHeading';
import { AlertTriangle, Lock, EyeOff, Radio } from 'lucide-react';
import { AnimatedCard } from '@/components/ui/AnimatedCard';

export const Problem: React.FC = () => {
  return (
    <section id="problem" className="py-24 bg-ink-50/50 dark:bg-ink-950/60 relative overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="The Deepfake Crisis"
          title="Synthetic Media is Eroding Universal Digital Trust"
          subtitle="Generative AI makes hyper-realistic deepfakes trivial to produce at scale — threatening elections, investigative journalism, brand reputation, and everyday social communication."
        />

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          {/* Left Column: Key Problem Vectors */}
          <div className="lg:col-span-6 space-y-6">
            <AnimatedCard delay={0.1}>
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-xl bg-rose-500/10 text-rose-500 flex-shrink-0">
                  <AlertTriangle className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-ink-950 dark:text-white">
                    Zero-Cost Synthetic Fabrication
                  </h3>
                  <p className="text-sm text-ink-600 dark:text-ink-300 mt-1 leading-relaxed">
                    Diffusion models and voice cloners can generate high-fidelity fake evidence in seconds, rendering traditional visual inspection obsolete.
                  </p>
                </div>
              </div>
            </AnimatedCard>

            <AnimatedCard delay={0.2}>
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-xl bg-amber-500/10 text-amber-500 flex-shrink-0">
                  <EyeOff className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-ink-950 dark:text-white">
                    Black-Box Detection Tools
                  </h3>
                  <p className="text-sm text-ink-600 dark:text-ink-300 mt-1 leading-relaxed">
                    Existing commercial detectors output binary percentage scores without explaining *why* content was flagged, leaving newsrooms and users unable to verify findings.
                  </p>
                </div>
              </div>
            </AnimatedCard>

            <AnimatedCard delay={0.3}>
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-xl bg-brand-blue-500/10 text-brand-blue-500 flex-shrink-0">
                  <Lock className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-ink-950 dark:text-white">
                    Proprietary Paywalls & Vendor Lock-In
                  </h3>
                  <p className="text-sm text-ink-600 dark:text-ink-300 mt-1 leading-relaxed">
                    High enterprise licensing fees lock journalists and everyday creators out of verification tools when truth matters most.
                  </p>
                </div>
              </div>
            </AnimatedCard>
          </div>

          {/* Right Column: Abstract Corrupted Signal Glitch Graphic */}
          <div className="lg:col-span-6">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="relative rounded-3xl bg-ink-950 p-8 border border-ink-800 shadow-dark-glass overflow-hidden flex flex-col items-center justify-center min-h-[380px]"
            >
              {/* Glitch Overlay Boxes */}
              <motion.div
                animate={{
                  x: [0, -5, 5, -2, 0],
                  opacity: [0.8, 0.4, 0.9, 0.6, 0.8],
                }}
                transition={{ repeat: Infinity, duration: 2, ease: 'easeInOut' }}
                className="absolute inset-x-8 top-12 h-24 bg-rose-500/10 border-y border-rose-500/30 rounded-xl"
              />

              <div className="relative z-10 text-center space-y-4">
                <div className="inline-flex p-4 rounded-2xl bg-brand-blue-500/10 border border-brand-blue-400/30 text-brand-blue-300">
                  <Radio className="w-10 h-10 animate-pulse" />
                </div>

                <div className="space-y-1">
                  <h4 className="text-xl font-bold font-mono text-white tracking-tight">
                    SIGNAL CORRUPTION DETECTED
                  </h4>
                  <p className="text-xs font-mono text-rose-400">
                    ERR_SYNTHETIC_NOISE_INJECTION // 0x94F2A
                  </p>
                </div>

                {/* Animated Glitch Bars */}
                <div className="w-48 mx-auto space-y-2 pt-2">
                  <motion.div
                    animate={{ width: ['20%', '90%', '40%', '85%'] }}
                    transition={{ repeat: Infinity, duration: 1.5, ease: 'linear' }}
                    className="h-1.5 bg-rose-500 rounded-full"
                  />
                  <motion.div
                    animate={{ width: ['80%', '30%', '95%', '50%'] }}
                    transition={{ repeat: Infinity, duration: 2, ease: 'linear' }}
                    className="h-1.5 bg-brand-blue-400 rounded-full"
                  />
                  <motion.div
                    animate={{ width: ['40%', '75%', '10%', '60%'] }}
                    transition={{ repeat: Infinity, duration: 1.2, ease: 'linear' }}
                    className="h-1.5 bg-amber-400 rounded-full"
                  />
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
};
