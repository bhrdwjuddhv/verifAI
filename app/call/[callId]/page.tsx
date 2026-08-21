import React from 'react';
import type { Metadata } from 'next';
import { CallClient } from './CallClient';

export const dynamic = 'force-dynamic';

export const metadata: Metadata = {
  title: 'Live call — VerifAI',
  description: 'A 1-to-1 call with optional live synthetic-voice monitoring.',
};

export default function CallPage({
  params,
  searchParams,
}: {
  params: { callId: string };
  searchParams: { role?: string };
}) {
  const role = searchParams.role === 'caller' ? 'caller' : 'callee';
  return (
    <main className="min-h-screen bg-ink-950 text-white">
      <CallClient callId={params.callId} role={role} />
    </main>
  );
}
