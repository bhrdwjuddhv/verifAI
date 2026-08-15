'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

export interface ToggleOption<T extends string = string> {
  id: T;
  label: string;
  icon?: React.ReactNode;
}

export interface ToggleProps<T extends string = string> {
  options: ToggleOption<T>[];
  value: T;
  onChange: (value: T) => void;
  layoutId?: string;
  className?: string;
}

export function Toggle<T extends string = string>({
  options,
  value,
  onChange,
  layoutId = 'active-toggle-pill',
  className,
}: ToggleProps<T>) {
  return (
    <div
      className={cn(
        'inline-flex items-center p-1.5 rounded-2xl bg-ink-100 dark:bg-ink-900 border border-ink-200 dark:border-ink-800',
        className
      )}
      role="tablist"
    >
      {options.map((option) => {
        const isSelected = value === option.id;

        return (
          <button
            key={option.id}
            type="button"
            role="tab"
            aria-selected={isSelected}
            onClick={() => onChange(option.id)}
            className={cn(
              'relative flex items-center gap-2 px-4 py-2 text-xs sm:text-sm font-medium rounded-xl transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-blue-400 select-none',
              isSelected
                ? 'text-ink-950 dark:text-ink-950 font-semibold'
                : 'text-ink-600 dark:text-ink-400 hover:text-ink-900 dark:hover:text-white'
            )}
          >
            {isSelected && (
              <motion.div
                layoutId={layoutId}
                transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                className="absolute inset-0 bg-brand-blue-300 dark:bg-brand-blue-400 rounded-xl shadow-sm"
              />
            )}
            <span className="relative z-10 flex items-center gap-2">
              {option.icon && <span className="w-4 h-4">{option.icon}</span>}
              {option.label}
            </span>
          </button>
        );
      })}
    </div>
  );
}
