'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { NodeTooltipData, useAppStore } from '@/lib/store';
import { Info } from 'lucide-react';

export interface DiagramNodeProps {
  data: NodeTooltipData;
  x: number;
  y: number;
  icon?: React.ReactNode;
  dark?: boolean;
}

export const DiagramNode: React.FC<DiagramNodeProps> = ({
  data,
  icon,
  dark = true,
}) => {
  const { selectedNode, setSelectedNode } = useAppStore();
  const isSelected = selectedNode?.id === data.id;

  return (
    <motion.div
      whileHover={{ scale: 1.03, y: -2 }}
      whileTap={{ scale: 0.97 }}
      onClick={(e) => {
        e.stopPropagation();
        setSelectedNode(isSelected ? null : data);
      }}
      className={cn(
        'group relative flex flex-col p-4 rounded-2xl transition-all duration-300 cursor-pointer select-none glass-card border-gradient shimmer',
        isSelected && 'border-brand-blue-400 ring-2 ring-brand-blue-400/40 shadow-glow-sm'
      )}
    >
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-2">
          {icon && (
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-brand-blue-400/25 via-brand-blue-500/15 to-transparent border border-brand-blue-400/30 flex items-center justify-center text-brand-blue-300 shadow-glow-sm">
              {icon}
            </div>
          )}
          <span className="text-[11px] font-semibold uppercase tracking-wider text-brand-blue-400">
            {data.layer}
          </span>
        </div>
        <div className="p-1 rounded-lg bg-ink-800/80 text-ink-400 group-hover:text-brand-blue-300 transition-colors">
          <Info className="w-3.5 h-3.5" />
        </div>
      </div>

      <h4 className="font-semibold text-sm text-white leading-snug">{data.title}</h4>
      <p className="text-xs text-ink-300 mt-1 line-clamp-2 font-normal">
        {data.tech}
      </p>
    </motion.div>
  );
};
