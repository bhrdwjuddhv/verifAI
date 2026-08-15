'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';
import { useAppStore } from '@/lib/store';
import { X, Cpu, Layers, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/Button';

export interface DiagramCanvasProps {
  children: React.ReactNode;
  title?: string;
  subtitle?: string;
  dark?: boolean;
  className?: string;
}

export const DiagramCanvas: React.FC<DiagramCanvasProps> = ({
  children,
  title,
  subtitle,
  dark = true,
  className,
}) => {
  const { selectedNode, setSelectedNode } = useAppStore();

  return (
    <div
      onClick={() => setSelectedNode(null)}
      className={cn(
        'relative rounded-3xl p-6 sm:p-8 md:p-10 border overflow-hidden transition-all',
        dark
          ? 'bg-ink-950/90 border-ink-800 text-white shadow-dark-glass'
          : 'bg-ink-50/70 border-ink-200 text-ink-900 shadow-glass',
        className
      )}
    >
      {/* Background SVG Grid pattern */}
      <div className="absolute inset-0 opacity-15 pointer-events-none">
        <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern
              id="grid-pattern"
              width="32"
              height="32"
              patternUnits="userSpaceOnUse"
            >
              <circle cx="2" cy="2" r="1" fill="currentColor" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid-pattern)" />
        </svg>
      </div>

      {(title || subtitle) && (
        <div className="relative z-10 mb-8 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-ink-800/60 pb-6">
          <div>
            {title && (
              <h3 className="text-xl sm:text-2xl font-bold flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-brand-blue-400" />
                {title}
              </h3>
            )}
            {subtitle && (
              <p className="text-sm text-ink-400 mt-1 max-w-xl">{subtitle}</p>
            )}
          </div>
          <div className="text-xs text-brand-blue-300 bg-brand-blue-500/10 border border-brand-blue-500/20 px-3 py-1.5 rounded-full flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5" />
            <span>Interactive Node Graph — Click any node for specs</span>
          </div>
        </div>
      )}

      {/* Main Diagram Area */}
      <div className="relative z-10 min-h-[420px]">{children}</div>

      {/* Interactive Tooltip Card */}
      <AnimatePresence>
        {selectedNode && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            onClick={(e) => e.stopPropagation()}
            className="fixed bottom-6 right-6 z-50 max-w-md w-full p-6 rounded-2xl bg-ink-900/95 text-white border border-brand-blue-400/50 shadow-glow-md backdrop-blur-xl"
          >
            <div className="flex items-start justify-between gap-3 mb-2">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-brand-blue-400" />
                <span className="text-xs font-bold uppercase tracking-wider text-brand-blue-400">
                  {selectedNode.layer}
                </span>
              </div>
              <button
                type="button"
                onClick={() => setSelectedNode(null)}
                className="p-1 rounded-lg text-ink-400 hover:text-white hover:bg-ink-800 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <h4 className="text-lg font-bold text-white">{selectedNode.title}</h4>
            <p className="text-xs font-semibold text-brand-blue-300 mt-0.5">
              Technology: {selectedNode.tech}
            </p>

            <p className="text-sm text-ink-300 mt-3 leading-relaxed">
              {selectedNode.description}
            </p>

            {selectedNode.metrics && (
              <div className="mt-4 pt-3 border-t border-ink-800 flex items-center justify-between text-xs">
                <span className="text-ink-400">Target Metrics:</span>
                <span className="font-mono text-brand-blue-300 font-semibold">
                  {selectedNode.metrics}
                </span>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
