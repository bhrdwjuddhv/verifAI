'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAppStore } from '@/lib/store';
import { UploadZone } from './UploadZone';
import { ReportPanel } from './ReportPanel';
import { Cpu, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface ScanDemoProps {
  className?: string;
}

export const ScanDemo: React.FC<ScanDemoProps> = ({ className }) => {
  const {
    scanStatus,
    setScanStatus,
    progress,
    setProgress,
    scanResult,
    setScanResult,
    scanError,
    setScanError,
    resetScan,
  } = useAppStore();

  const handleStartScan = async (file: File | null, url: string) => {
    try {
      setScanError(null);
      setScanStatus('uploading');
      setProgress(15);

      const formData = new FormData();
      if (file) {
        formData.append('file', file);
      }

      let currentProg = 15;
      const interval = setInterval(() => {
        currentProg += 20;
        if (currentProg >= 85) {
          currentProg = 90;
          clearInterval(interval);
        }
        setProgress(currentProg);
      }, 300);

      setScanStatus('analyzing');

      let response: Response;
      if (file) {
        response = await fetch('/api/scan', {
          method: 'POST',
          body: formData,
        });
      } else {
        response = await fetch('/api/scan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url }),
        });
      }

      clearInterval(interval);
      setProgress(100);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const detail = errorData.detail ? ` ${errorData.detail}` : '';
        throw new Error(`${errorData.error || 'Analysis failed'}.${detail}`);
      }

      const data = await response.json();
      setScanResult(data);
      setScanStatus('complete');
    } catch (err: any) {
      setScanError(err?.message || 'Scan failed. Please try again.');
      setScanStatus('idle');
    }
  };

  return (
    <div
      id="live-scan-demo"
      className={cn(
        'w-full max-w-4xl mx-auto rounded-3xl p-6 sm:p-8 bg-ink-950/95 border border-brand-blue-400/40 shadow-glow-lg text-white relative overflow-hidden backdrop-blur-2xl glass-card border-gradient',
        className
      )}
    >
      {/* Glow highlight blur in backdrop */}
      <div className="absolute -top-24 -right-24 w-72 h-72 bg-brand-blue-400/15 rounded-full blur-3xl pointer-events-none" />

      {/* Header Bar */}
      <div className="flex items-center justify-between border-b border-ink-800 pb-4 mb-6">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-brand-blue-400 animate-pulse" />
          <span className="font-bold text-base sm:text-lg tracking-tight">
            VerifAI Forensic Sandbox
          </span>
        </div>
        <span className="text-xs font-mono text-brand-blue-300 bg-brand-blue-500/10 px-3 py-1 rounded-full border border-brand-blue-500/20">
          Images, short video &amp; voice
        </span>
      </div>

      <AnimatePresence mode="wait">
        {scanStatus === 'idle' && (
          <motion.div
            key="idle"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
          >
            {/* Analysis unavailable: say so. Never show a verdict we did not get. */}
            {scanError && (
              <div className="mb-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs leading-relaxed">
                <span className="font-semibold">Analysis unavailable — no verdict was produced.</span>{' '}
                {scanError}
              </div>
            )}
            <UploadZone onStartScan={handleStartScan} />
          </motion.div>
        )}

        {(scanStatus === 'uploading' || scanStatus === 'analyzing') && (
          <motion.div
            key="analyzing"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="py-12 flex flex-col items-center justify-center text-center space-y-6"
          >
            <div className="relative w-20 h-20 flex items-center justify-center">
              <div className="absolute inset-0 rounded-full border-4 border-ink-800" />
              <div className="absolute inset-0 rounded-full border-4 border-brand-blue-400 border-t-transparent animate-spin" />
              <Cpu className="w-8 h-8 text-brand-blue-300" />
            </div>

            <div className="space-y-2 max-w-md">
              <h4 className="text-xl font-bold text-white">
                {scanStatus === 'uploading' ? 'Uploading...' : 'Running the detector...'}
              </h4>
              <p className="text-xs text-ink-400">
                Face-swap classifier on the cropped face, AI-generation detector on the whole frame; video is sampled frame by frame.
              </p>
            </div>

            {/* Progress Bar */}
            <div className="w-full max-w-md bg-ink-900 h-2 rounded-full overflow-hidden border border-ink-800">
              <motion.div
                className="h-full bg-gradient-to-r from-brand-blue-300 to-brand-blue-500"
                initial={{ width: '0%' }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
          </motion.div>
        )}

        {scanStatus === 'complete' && scanResult && (
          <motion.div
            key="complete"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
          >
            <ReportPanel result={scanResult} onReset={resetScan} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
