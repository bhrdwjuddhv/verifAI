'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Badge } from './Badge';

export interface SectionHeadingProps {
  eyebrow?: string;
  title: string;
  subtitle?: string;
  align?: 'left' | 'center' | 'right';
  className?: string;
  dark?: boolean;
}

export const SectionHeading: React.FC<SectionHeadingProps> = ({
  eyebrow,
  title,
  subtitle,
  align = 'center',
  className,
  dark = true,
}) => {
  const alignment =
    align === 'center'
      ? 'text-center mx-auto items-center'
      : align === 'right'
      ? 'text-right ml-auto items-end'
      : 'text-left items-start';

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-40px' }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className={cn('flex flex-col max-w-3xl mb-12 sm:mb-14', alignment, className)}
    >
      {eyebrow && (
        <div className="mb-3">
          <Badge variant="blue" size="sm">
            {eyebrow}
          </Badge>
        </div>
      )}
      <h2
        className={cn(
          'text-2xl sm:text-3xl md:text-4xl font-semibold tracking-tight leading-tight',
          dark ? 'text-white' : 'text-ink-950 dark:text-white'
        )}
      >
        {title}
      </h2>
      {subtitle && (
        <p
          className={cn(
            'mt-3 text-sm sm:text-base leading-relaxed max-w-2xl font-normal',
            dark ? 'text-ink-400' : 'text-ink-600 dark:text-ink-400'
          )}
        >
          {subtitle}
        </p>
      )}
    </motion.div>
  );
};
