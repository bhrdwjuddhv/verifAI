'use client';

import React, { useEffect, useState } from 'react';
import { motion, useMotionValue, animate } from 'framer-motion';
import { getVerdict, VerdictInfo } from '@/lib/verdict';
import { Badge } from '@/components/ui/Badge';

export interface ScoreRingProps {
  /** Null when the model did not apply — the ring shows "—" rather than a made-up number. */
  score: number | null;
  /** The verdict the service actually returned. Falls back to the score bands when absent. */
  verdict?: VerdictInfo;
  size?: number;
  strokeWidth?: number;
  className?: string;
}

export const ScoreRing: React.FC<ScoreRingProps> = ({
  score,
  verdict: given,
  size = 180,
  strokeWidth = 14,
  className,
}) => {
  const [displayScore, setDisplayScore] = useState(0);
  const motionScore = useMotionValue(0);

  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;

  const verdict = given ?? getVerdict(score ?? 50);

  useEffect(() => {
    const controls = animate(motionScore, score ?? 0, {
      duration: 1.6,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: (latest) => setDisplayScore(Math.round(latest)),
    });
    return () => controls.stop();
  }, [score, motionScore]);

  const strokeDashoffset = circumference - (displayScore / 100) * circumference;

  return (
    <div className={`relative flex flex-col items-center justify-center ${className}`}>
      <div className="relative" style={{ width: size, height: size }}>
        <svg className="w-full h-full transform -rotate-90">
          {/* Track Circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="currentColor"
            strokeWidth={strokeWidth}
            fill="transparent"
            className="text-ink-800/80"
          />
          {/* Animated Value Arc */}
          <motion.circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={verdict.ringColor}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            style={{
              filter: `drop-shadow(0px 0px 8px ${verdict.glowColor})`,
            }}
          />
        </svg>

        {/* Center Content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
          <span className="text-4xl font-extrabold tracking-tight text-white font-mono">
            {score === null ? '—' : `${displayScore}%`}
          </span>
          <span className="text-[10px] uppercase font-bold tracking-widest text-ink-400 mt-0.5">
            {score === null ? 'Not measured' : 'Likelihood Real'}
          </span>
        </div>
      </div>

      <div className="mt-3">
        <Badge variant={verdict.category} size="md">
          {verdict.label}
        </Badge>
      </div>
    </div>
  );
};
