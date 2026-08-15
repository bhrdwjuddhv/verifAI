import type { Metadata } from 'next';
import React from 'react';

export const metadata: Metadata = {
  title: 'VerifAI Admin Dashboard',
  description: 'Internal admin panel for VerifAI training pipeline and direct model inference.',
  robots: {
    index: false,
    follow: false,
    nocache: true,
    googleBot: {
      index: false,
      follow: false,
    },
  },
};

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-ink-950 text-ink-100 selection:bg-brand-blue-500/30 selection:text-white flex flex-col font-sans">
      {children}
    </div>
  );
}
