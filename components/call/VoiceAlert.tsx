'use client';

import React from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { AlertTriangle, X } from 'lucide-react';

/**
 * Shown only when synthetic signal persists — never on a single window, and it never acts on
 * the call. The user decides; the tool tells them what it heard.
 */
export const VoiceAlert: React.FC<{
  visible: boolean;
  consecutive: number;
  onDismiss: () => void;
}> = ({ visible, consecutive, onDismiss }) => (
  <AnimatePresence>
    {visible && (
      <motion.div
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -8 }}
        className="rounded-2xl border border-rose-500/40 bg-rose-500/10 p-4 flex items-start gap-3"
      >
        <AlertTriangle className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
        <div className="flex-1 text-sm">
          <p className="font-semibold text-rose-300">
            Synthetic voice signal in {consecutive} consecutive windows
          </p>
          <p className="text-xs text-rose-200/80 mt-1 leading-relaxed font-normal">
            The detector has flagged the incoming audio repeatedly. That is a reason to verify who
            you are speaking to by another channel — a known number, a shared secret, a question an
            impostor could not answer. It is not proof, and detectors do produce false alarms on
            poor connections.
          </p>
        </div>
        <button
          type="button"
          onClick={onDismiss}
          className="text-rose-300/70 hover:text-rose-200 transition-colors"
          aria-label="Dismiss"
        >
          <X className="w-4 h-4" />
        </button>
      </motion.div>
    )}
  </AnimatePresence>
);
