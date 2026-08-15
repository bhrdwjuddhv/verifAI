'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface AccordionItemData {
  id: string;
  title: string;
  subtitle?: string;
  badge?: string;
  content: React.ReactNode;
}

export interface AccordionProps {
  items: AccordionItemData[];
  allowMultiple?: boolean;
  className?: string;
  dark?: boolean;
}

export const Accordion: React.FC<AccordionProps> = ({
  items,
  allowMultiple = false,
  className,
  dark = false,
}) => {
  const [openIds, setOpenIds] = useState<string[]>([items[0]?.id || '']);

  const toggleItem = (id: string) => {
    if (allowMultiple) {
      setOpenIds((prev) =>
        prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
      );
    } else {
      setOpenIds((prev) => (prev.includes(id) ? [] : [id]));
    }
  };

  return (
    <div className={cn('space-y-3 w-full', className)}>
      {items.map((item) => {
        const isOpen = openIds.includes(item.id);

        return (
          <div
            key={item.id}
            className={cn(
              'rounded-2xl border transition-all duration-200 overflow-hidden',
              dark
                ? 'bg-ink-900/70 border-ink-800'
                : 'bg-white/80 dark:bg-ink-900/80 border-ink-200 dark:border-ink-800 shadow-sm'
            )}
          >
            <button
              type="button"
              onClick={() => toggleItem(item.id)}
              aria-expanded={isOpen}
              aria-controls={`accordion-content-${item.id}`}
              className={cn(
                'w-full flex items-center justify-between p-5 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-blue-400',
                dark ? 'hover:bg-ink-800/50 text-white' : 'hover:bg-brand-blue-50/50 dark:hover:bg-ink-800/50 text-ink-900 dark:text-white'
              )}
            >
              <div className="flex items-center gap-3 pr-4">
                <span className="font-semibold text-base sm:text-lg tracking-tight">
                  {item.title}
                </span>
                {item.badge && (
                  <span className="px-2.5 py-0.5 text-xs font-medium rounded-full bg-brand-blue-100 text-brand-blue-900 dark:bg-brand-blue-950 dark:text-brand-blue-300">
                    {item.badge}
                  </span>
                )}
              </div>
              <motion.div
                animate={{ rotate: isOpen ? 180 : 0 }}
                transition={{ duration: 0.2 }}
                className="flex-shrink-0 text-ink-400"
              >
                <ChevronDown className="w-5 h-5" />
              </motion.div>
            </button>

            <AnimatePresence initial={false}>
              {isOpen && (
                <motion.div
                  id={`accordion-content-${item.id}`}
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.3, ease: [0.04, 0.62, 0.23, 0.98] }}
                >
                  <div
                    className={cn(
                      'px-5 pb-5 pt-1 text-sm leading-relaxed border-t border-dashed',
                      dark
                        ? 'border-ink-800 text-ink-300'
                        : 'border-ink-200 dark:border-ink-800 text-ink-600 dark:text-ink-400'
                    )}
                  >
                    {item.content}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        );
      })}
    </div>
  );
};
