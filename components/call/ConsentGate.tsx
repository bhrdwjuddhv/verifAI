'use client';

import React from 'react';
import { ShieldCheck, Upload, MonitorOff } from 'lucide-react';
import { Button } from '@/components/ui/Button';

/**
 * Monitoring is off until this is answered. The choice that matters is where the audio goes,
 * so it is stated plainly rather than buried in a toggle.
 */
export const ConsentGate: React.FC<{
  onAccept: () => void;
  onDecline: () => void;
}> = ({ onAccept, onDecline }) => (
  <div className="rounded-2xl border border-brand-blue-400/30 bg-ink-900/70 p-5 space-y-4">
    <div className="flex items-center gap-2">
      <ShieldCheck className="w-5 h-5 text-brand-blue-400" />
      <h3 className="font-semibold text-white">Before monitoring starts</h3>
    </div>

    <ul className="space-y-2.5 text-xs text-ink-300 font-normal leading-relaxed">
      <li className="flex items-start gap-2">
        <Upload className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
        <span>
          <strong className="text-white">Audio leaves this device.</strong> Three-second windows of
          the <em>incoming</em> audio are sent to your VerifAI model service to be scored. They are
          processed in memory and not stored, but they are transmitted. Your microphone is not
          analysed — only what the other side sends.
        </span>
      </li>
      <li className="flex items-start gap-2">
        <MonitorOff className="w-4 h-4 text-brand-blue-400 flex-shrink-0 mt-0.5" />
        <span>
          <strong className="text-white">It informs, it does not decide.</strong> The call is never
          ended automatically. A high score is a prompt to verify by another channel, not proof.
        </span>
      </li>
      <li className="flex items-start gap-2">
        <ShieldCheck className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
        <span>
          <strong className="text-white">Tell the other person.</strong> Recording or analysing a
          call without consent is illegal in many places. This is your responsibility, not the
          tool's.
        </span>
      </li>
    </ul>

    <div className="flex items-center gap-2 pt-1">
      <Button variant="primary" size="sm" onClick={onAccept}>
        Start monitoring
      </Button>
      <Button variant="ghost" size="sm" onClick={onDecline}>
        Call without monitoring
      </Button>
    </div>
  </div>
);
