'use client';

import React from 'react';

export interface TimelineWindow {
  t: number;
  /** P(fake) 0-100, or null when the window was not scored (silence, or no model). */
  fake: number | null;
  speech: boolean;
}

/**
 * One cell per analysed window. Skipped windows are drawn hollow rather than green — "we did
 * not listen" and "we listened and heard nothing wrong" are different claims.
 */
export const CallTimeline: React.FC<{ windows: TimelineWindow[]; max?: number }> = ({
  windows,
  max = 60,
}) => {
  const shown = windows.slice(-max);

  return (
    <div className="rounded-2xl border border-ink-800 bg-ink-900/60 p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-ink-300">Timeline</span>
        <span className="text-[11px] text-ink-500 font-normal">newest on the right</span>
      </div>

      {shown.length === 0 ? (
        <p className="text-xs text-ink-500 font-normal py-4 text-center">
          Nothing analysed yet.
        </p>
      ) : (
        <div className="flex items-end gap-[3px] h-16">
          {shown.map((w, i) => {
            if (!w.speech || w.fake === null) {
              return (
                <div
                  key={i}
                  title={`${w.t.toFixed(0)}s — not scored (no speech)`}
                  className="flex-1 min-w-[3px] h-2 rounded-sm border border-ink-700 bg-transparent self-end"
                />
              );
            }
            const color = w.fake > 70 ? '#EF4444' : w.fake > 30 ? '#F59E0B' : '#10B981';
            return (
              <div
                key={i}
                title={`${w.t.toFixed(0)}s — ${w.fake}% synthetic`}
                className="flex-1 min-w-[3px] rounded-sm"
                style={{ height: `${Math.max(8, w.fake)}%`, backgroundColor: color }}
              />
            );
          })}
        </div>
      )}

      <div className="flex items-center gap-4 mt-3 text-[11px] text-ink-500 font-normal">
        <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-sm bg-emerald-500" /> no signal</span>
        <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-sm bg-amber-500" /> uncertain</span>
        <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-sm bg-rose-500" /> synthetic</span>
        <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-sm border border-ink-600" /> silence, not scored</span>
      </div>
    </div>
  );
};
