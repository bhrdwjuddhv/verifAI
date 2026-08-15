import React from 'react';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'VerifAI Admin — Fusion Evaluation & Tuning Harness',
  description: 'Human-judged evaluation harness and fusion-weight tuning optimizer for VerifAI detection engine.',
};

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-ink-950 text-white font-sans selection:bg-brand-blue-500/30 selection:text-white">
      {children}
    </div>
  );
}
