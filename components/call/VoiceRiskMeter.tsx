'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { ShieldCheck, ShieldAlert, HelpCircle, MicOff } from 'lucide-react';

export type RiskBand = 'low' | 'uncertain' | 'suspicious' | 'high' | 'idle';

export const BAND_STYLE: Record<RiskBand, { label: string; color: string; text: string; border: string }> = {
  low: { label: 'No synthetic signal', color: '#10B981', text: 'text-emerald-400', border: 'border-emerald-500/30' },
  uncertain: { label: 'Uncertain', color: '#94A3B8', text: 'text-slate-300', border: 'border-slate-400/30' },
  suspicious: { label: 'Possible synthetic voice', color: '#F59E0B', text: 'text-amber-400', border: 'border-amber-500/30' },
  high: { label: 'Strong synthetic signal', color: '#EF4444', text: 'text-rose-400', border: 'border-rose-500/30' },
  idle: { label: 'Listening…', color: '#5C6370', text: 'text-ink-400', border: 'border-ink-700' },
};

export interface VoiceRiskMeterProps {
  /** Trust score 0-100 (100 = no synthetic signal). Null before the first scored window. */
  trust: number | null;
  band: RiskBand;
  windowsScored: number;
  windowsSkipped: number;
  modelSource: string | null;
  /** A band change in progress: shown so the delay reads as deliberate, not as lag. */
  pending?: { band: RiskBand; count: number; needed: number } | null;
  /** Which signals the risk was fused from. */
  fusedWith?: string;
}

export const VoiceRiskMeter: React.FC<VoiceRiskMeterProps> = ({
  trust,
  band,
  windowsScored,
  windowsSkipped,
  modelSource,
  pending = null,
  fusedWith,
}) => {
  const style = BAND_STYLE[band];
  const Icon = band === 'high' || band === 'suspicious' ? ShieldAlert : band === 'low' ? ShieldCheck : HelpCircle;

  return (
    <div className={`rounded-2xl border ${style.border} bg-ink-900/60 p-5`}>
      <div className="flex items-center gap-2 mb-3">
        <Icon className={`w-4 h-4 ${style.text}`} />
        <span className="text-xs font-semibold uppercase tracking-wider text-ink-300">Live voice check</span>
      </div>

      <div className="flex items-end gap-3">
        <span className="text-4xl font-extrabold font-mono text-white tabular-nums">
          {trust === null ? '—' : trust}
          {trust !== null && <span className="text-lg text-ink-400">%</span>}
        </span>
        <span className={`text-sm font-semibold ${style.text} pb-1.5`}>{style.label}</span>
      </div>

      <div className="mt-3 h-2 w-full rounded-full bg-ink-800 overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: style.color }}
          animate={{ width: `${trust ?? 0}%` }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
        />
      </div>

      {pending && (
        <p className="text-[11px] text-amber-300/90 mt-2 font-normal">
          {BAND_STYLE[pending.band].label} in {pending.count} of {pending.needed} windows needed to
          change the reading — holding until the evidence persists.
        </p>
      )}

      <p className="text-[11px] text-ink-400 mt-3 font-normal leading-relaxed">
        {windowsScored} window{windowsScored === 1 ? '' : 's'} scored{fusedWith ? ` (${fusedWith})` : ''}
        {windowsSkipped > 0 && (
          <span className="inline-flex items-center gap-1">
            {' '}· <MicOff className="w-3 h-3" /> {windowsSkipped} skipped as silence
          </span>
        )}
        {modelSource && <><br />Model: <span className="font-mono">{modelSource}</span></>}
      </p>

      <p className="text-[11px] text-ink-500 mt-2 font-normal leading-relaxed">
        This informs, it does not decide. It never ends your call, and it cannot prove a voice is
        genuine — only that it found no synthetic signal in what it heard.
      </p>
    </div>
  );
};
