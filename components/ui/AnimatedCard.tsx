'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

export interface AnimatedCardProps {
  children: React.ReactNode;
  className?: string;
  glowOnHover?: boolean;
  delay?: number;
  onClick?: () => void;
  dark?: boolean;
}

export const AnimatedCard: React.FC<AnimatedCardProps> = ({
  children,
  className,
  glowOnHover = true,
  delay = 0,
  onClick,
  dark = true,
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 28, scale: 0.96 }}
      whileInView={{ opacity: 1, y: 0, scale: 1 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{
        duration: 0.6,
        delay,
        ease: [0.16, 1, 0.3, 1], // Buttery cubic-bezier spring curve
      }}
      whileHover={{
        y: -8,
        scale: 1.02,
        transition: { duration: 0.25, ease: 'easeOut' },
      }}
      whileTap={onClick ? { scale: 0.98 } : undefined}
      onClick={onClick}
      className={cn(
        'group relative rounded-3xl p-6 sm:p-7 transition-all duration-300 overflow-hidden select-none',
        'glass-card border-gradient shimmer',
        glowOnHover &&
          'hover:shadow-glow-md hover:border-brand-blue-300/60 dark:hover:border-brand-blue-400/50',
        onClick && 'cursor-pointer',
        className
      )}
    >
      {/* Corner Ambient Glow Blob on Hover */}
      <div className="absolute -top-20 -right-20 w-40 h-40 bg-brand-blue-400/10 rounded-full blur-2xl group-hover:scale-150 group-hover:bg-brand-blue-400/20 transition-all duration-500 pointer-events-none" />

      {/* Top Hairline Light Reflection */}
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-brand-blue-300/50 to-transparent rounded-t-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none" />

      <div className="relative z-10">{children}</div>
    </motion.div>
  );
};
