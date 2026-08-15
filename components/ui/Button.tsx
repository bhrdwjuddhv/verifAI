'use client';

import React from 'react';
import { motion, HTMLMotionProps } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Loader2 } from 'lucide-react';

export interface ButtonProps extends Omit<HTMLMotionProps<'button'>, 'children'> {
  children?: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  href?: string;
  className?: string;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      children,
      variant = 'primary',
      size = 'md',
      isLoading = false,
      leftIcon,
      rightIcon,
      href,
      className,
      disabled,
      onClick,
      ...props
    },
    ref
  ) => {
    const baseStyles =
      'relative inline-flex items-center justify-center font-medium rounded-2xl transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-blue-400 disabled:opacity-50 disabled:pointer-events-none select-none cursor-pointer overflow-hidden';

    const variants = {
      primary:
        'bg-gradient-to-r from-brand-blue-300 via-brand-blue-400 to-brand-blue-500 text-ink-950 font-bold shadow-glow-sm hover:shadow-glow-lg border border-white/40 hover:scale-[1.03] active:scale-[0.97]',
      secondary:
        'glass-card text-ink-50 hover:text-white border border-brand-blue-400/30 hover:border-brand-blue-400/60 hover:shadow-glow-sm hover:scale-[1.02] active:scale-[0.97]',
      outline:
        'border border-ink-700/80 text-ink-100 hover:bg-ink-800/60 hover:border-brand-blue-400/50 hover:scale-[1.02]',
      ghost:
        'text-ink-300 hover:text-white hover:bg-brand-blue-500/10 hover:border-brand-blue-400/20',
      danger:
        'bg-rose-500/15 text-rose-300 border border-rose-500/40 hover:bg-rose-500/25 hover:shadow-glow-sm',
    };

    const sizes = {
      sm: 'text-xs px-3.5 py-2 gap-1.5 min-h-[38px]',
      md: 'text-sm px-5 py-3 gap-2 min-h-[46px]',
      lg: 'text-base px-7 py-4 gap-2.5 min-h-[52px]',
    };

    const content = (
      <>
        {/* Subtle Shine Reflection */}
        <span className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000 ease-in-out pointer-events-none" />

        {isLoading ? (
          <Loader2 className="w-4 h-4 animate-spin text-current" />
        ) : (
          leftIcon && <span className="flex-shrink-0 transition-transform group-hover:scale-110">{leftIcon}</span>
        )}
        <span className="relative z-10 font-semibold tracking-tight">{children}</span>
        {!isLoading && rightIcon && (
          <span className="flex-shrink-0 transition-transform group-hover:translate-x-1">{rightIcon}</span>
        )}
      </>
    );

    if (href) {
      return (
        <motion.a
          href={href}
          whileHover={{ y: -2 }}
          whileTap={{ scale: 0.96 }}
          className={cn(baseStyles, variants[variant], sizes[size], 'group', className)}
        >
          {content}
        </motion.a>
      );
    }

    return (
      <motion.button
        ref={ref}
        whileHover={{ y: disabled || isLoading ? 0 : -2 }}
        whileTap={{ scale: disabled || isLoading ? 1 : 0.96 }}
        disabled={disabled || isLoading}
        onClick={onClick}
        className={cn(baseStyles, variants[variant], sizes[size], 'group', className)}
        {...props}
      >
        {content}
      </motion.button>
    );
  }
);

Button.displayName = 'Button';