'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { SectionHeading } from '@/components/ui/SectionHeading';
import { AnimatedCard } from '@/components/ui/AnimatedCard';
import { Toggle } from '@/components/ui/Toggle';
import { useAppStore } from '@/lib/store';
import { Globe, Chrome, Smartphone, User, Newspaper, CheckCircle2, ShieldCheck, Database, Key } from 'lucide-react';

export const ThreeWaysIn: React.FC = () => {
  const { persona, setPersona } = useAppStore();

  return (
    <section id="three-ways-in" className="py-20 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="Access Vectors"
          title="Three Ways In — Built for Every User Workflow"
          subtitle="Whether you are scrolling social media on mobile, researching stories in a newsroom, or building developer integrations."
        />

        {/* Three Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-14">
          <AnimatedCard delay={0.1}>
            <div className="w-11 h-11 rounded-2xl bg-gradient-to-br from-brand-blue-400/35 via-brand-blue-500/20 to-transparent border border-brand-blue-400/40 flex items-center justify-center text-brand-blue-300 shadow-glow-sm mb-4">
              <Globe className="w-6 h-6 stroke-[2]" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-1.5">Web Application</h3>
            <p className="text-xs text-ink-300 leading-relaxed font-normal">
              Full-featured web portal with drag-and-drop uploads, URL stream analysis, interactive heatmap inspection, and trust badge export.
            </p>
          </AnimatedCard>

          <AnimatedCard delay={0.2}>
            <div className="w-11 h-11 rounded-2xl bg-gradient-to-br from-brand-blue-400/35 via-brand-blue-500/20 to-transparent border border-brand-blue-400/40 flex items-center justify-center text-brand-blue-300 shadow-glow-sm mb-4">
              <Chrome className="w-6 h-6 stroke-[2]" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-1.5">Browser Extension</h3>
            <p className="text-xs text-ink-300 leading-relaxed font-normal">
              Manifest V3 Chrome add-on. Right-click any image or video while browsing X, Reddit, or YouTube to instantly analyze authenticity.
            </p>
          </AnimatedCard>

          <AnimatedCard delay={0.3}>
            <div className="w-11 h-11 rounded-2xl bg-gradient-to-br from-brand-blue-400/35 via-brand-blue-500/20 to-transparent border border-brand-blue-400/40 flex items-center justify-center text-brand-blue-300 shadow-glow-sm mb-4">
              <Smartphone className="w-6 h-6 stroke-[2]" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-1.5">Mobile Share-Sheet</h3>
            <p className="text-xs text-ink-300 leading-relaxed font-normal">
              Native iOS & Android integration. Share any post or image directly to VerifAI from your favorite social apps with one tap.
            </p>
          </AnimatedCard>
        </div>

        {/* Persona Toggle */}
        <div className="glass-card rounded-3xl p-7 border border-ink-800">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 mb-7">
            <div>
              <h3 className="text-lg font-semibold text-white">Tailored Feature Matrix</h3>
              <p className="text-xs text-ink-400 font-normal">Switch persona view to explore features</p>
            </div>

            <Toggle
              options={[
                { id: 'users', label: 'For Everyday Users', icon: <User className="w-4 h-4" /> },
                { id: 'newsrooms', label: 'For Newsrooms & Platforms', icon: <Newspaper className="w-4 h-4" /> },
              ]}
              value={persona}
              onChange={(val) => setPersona(val as 'users' | 'newsrooms')}
              layoutId="persona-toggle-pill"
            />
          </div>

          <AnimatePresence mode="wait">
            {persona === 'users' ? (
              <motion.div
                key="persona-users"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.25 }}
                className="grid grid-cols-1 md:grid-cols-2 gap-5"
              >
                <div className="space-y-3.5">
                  <div className="flex items-start gap-3">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-xs text-white">Plain-Language Verdicts</h4>
                      <p className="text-xs text-ink-300 font-normal mt-0.5">Simple color status (Authentic, Suspicious, Manipulated) with zero jargon.</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-xs text-white">One-Tap Shareable Badges</h4>
                      <p className="text-xs text-ink-300 font-normal mt-0.5">Copy short trust links to attach verification proof when posting online.</p>
                    </div>
                  </div>
                </div>

                <div className="space-y-3.5">
                  <div className="flex items-start gap-3">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-xs text-white">"Why Was This Flagged" Highlights</h4>
                      <p className="text-xs text-ink-300 font-normal mt-0.5">Understand exact reasons (e.g. facial boundaries or missing camera tags) in plain points.</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-xs text-white">Free Forever Access</h4>
                      <p className="text-xs text-ink-300 font-normal mt-0.5">No sign-up required for basic public scans. Zero subscriptions or paywalls.</p>
                    </div>
                  </div>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="persona-newsrooms"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.25 }}
                className="grid grid-cols-1 md:grid-cols-2 gap-5"
              >
                <div className="space-y-3.5">
                  <div className="flex items-start gap-3">
                    <ShieldCheck className="w-4 h-4 text-brand-blue-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-xs text-white">High-Throughput Bulk API</h4>
                      <p className="text-xs text-ink-300 font-normal mt-0.5">Automated REST endpoints for batch scanning breaking news feeds.</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <Database className="w-4 h-4 text-brand-blue-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-xs text-white">Immutable Forensic Audit Trails</h4>
                      <p className="text-xs text-ink-300 font-normal mt-0.5">Cryptographically signed verification logs suitable for editorial legal compliance.</p>
                    </div>
                  </div>
                </div>

                <div className="space-y-3.5">
                  <div className="flex items-start gap-3">
                    <Key className="w-4 h-4 text-brand-blue-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-xs text-white">Customizable Confidence Thresholds</h4>
                      <p className="text-xs text-ink-300 font-normal mt-0.5">Adjust model sensitivity weights to match newsroom risk tolerance.</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <ShieldCheck className="w-4 h-4 text-brand-blue-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-xs text-white">Self-Hosted On-Premise Deployment</h4>
                      <p className="text-xs text-ink-300 font-normal mt-0.5">Deploy dockerized backend microservices within air-gapped secure networks.</p>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </section>
  );
};
