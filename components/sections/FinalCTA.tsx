'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/Button';
import { ArrowRight, ShieldCheck } from 'lucide-react';

export const FinalCTA: React.FC = () => {
  const scrollToScan = () => {
    const demoEl = document.getElementById('live-scan-demo');
    if (demoEl) {
      demoEl.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section className="py-20 bg-ink-950 text-white relative overflow-hidden">
      {/* Baby blue ambient glow behind section */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[550px] h-[280px] bg-brand-blue-400/15 rounded-full blur-[120px] pointer-events-none" />

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10 space-y-6">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="w-14 h-14 rounded-2xl bg-gradient-to-br from-brand-blue-400/35 via-brand-blue-500/20 to-transparent border border-brand-blue-400/40 text-brand-blue-300 flex items-center justify-center mx-auto shadow-glow-md"
        >
          <ShieldCheck className="w-7 h-7 fill-brand-blue-400/20 stroke-[2]" />
        </motion.div>

        <motion.h2
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="text-3xl sm:text-4xl md:text-5xl font-semibold tracking-tight"
        >
          Start verifying in <span className="text-gradient-blue">seconds.</span>
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="text-sm sm:text-base text-ink-300 max-w-xl mx-auto leading-relaxed font-normal"
        >
          Protect truth and restore digital media integrity with transparent, explainable open-source forensics.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="pt-3"
        >
          <Button
            variant="primary"
            size="lg"
            onClick={scrollToScan}
            rightIcon={<ArrowRight className="w-4 h-4" />}
          >
            Run a Live Scan Now
          </Button>
        </motion.div>
      </div>
    </section>
  );
};
