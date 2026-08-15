import React from 'react';
import { cn } from '@/lib/utils';
import { VerdictCategory, VERDICT_CONFIG } from '@/lib/verdict';

export interface BadgeProps {
  children?: React.ReactNode;
  variant?: VerdictCategory | 'info' | 'blue' | 'neutral';
  size?: 'sm' | 'md';
  className?: string;
  dot?: boolean;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'neutral',
  size = 'md',
  className,
  dot = true,
}) => {
  let badgeStyles = 'bg-ink-100 text-ink-700 border-ink-200 dark:bg-ink-800 dark:text-ink-300 dark:border-ink-700';
  let dotColor = 'bg-ink-400';

  if (variant === 'genuine' || variant === 'suspicious' || variant === 'uncertain' || variant === 'manipulated') {
    const config = VERDICT_CONFIG[variant];
    badgeStyles = `${config.badgeBg} ${config.badgeText} ${config.badgeBorder}`;
    dotColor =
      variant === 'genuine'
        ? 'bg-emerald-400'
        : variant === 'suspicious'
        ? 'bg-amber-400'
        : variant === 'uncertain'
        ? 'bg-slate-400'
        : 'bg-rose-400';
  } else if (variant === 'blue') {
    badgeStyles = 'bg-brand-blue-100/60 dark:bg-brand-blue-900/30 text-brand-blue-900 dark:text-brand-blue-300 border-brand-blue-300/50';
    dotColor = 'bg-brand-blue-500';
  } else if (variant === 'info') {
    badgeStyles = 'bg-sky-500/10 text-sky-400 border-sky-500/30';
    dotColor = 'bg-sky-400';
  }

  const sizeStyles = size === 'sm' ? 'px-2.5 py-0.5 text-xs' : 'px-3 py-1 text-xs font-medium';

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border shadow-sm transition-colors',
        sizeStyles,
        badgeStyles,
        className
      )}
    >
      {dot && <span className={cn('w-1.5 h-1.5 rounded-full animate-pulse', dotColor)} />}
      <span>{children}</span>
    </span>
  );
};
