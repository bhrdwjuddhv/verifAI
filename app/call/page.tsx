'use client';

import React, { Suspense, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { Phone, PhoneIncoming } from 'lucide-react';
import { Button } from '@/components/ui/Button';

/**
 * Lobby. Creating or joining hands back a peerId, which the call page needs — it is kept in
 * sessionStorage rather than the URL so a shared invite link cannot impersonate a participant.
 */
function Lobby() {
  const router = useRouter();
  const params = useSearchParams();
  const [code, setCode] = useState(params.get('join') ?? '');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const go = async (path: string, body: unknown, role: 'caller' | 'callee') => {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const out = await res.json();
      if (!res.ok) throw new Error(out.detail || out.error || 'Could not start the call.');
      sessionStorage.setItem(`verifai-call-${out.callId}`, out.peerId);
      router.push(`/call/${out.callId}?role=${role}`);
    } catch (err: any) {
      setError(err?.message || 'Something went wrong.');
      setBusy(false);
    }
  };

  return (
    <main className="min-h-screen bg-ink-950 text-white">
      <div className="max-w-lg mx-auto py-16 px-4">
        <Link href="/" className="text-xs text-brand-blue-300 hover:underline">← Back to VerifAI</Link>
        <h1 className="text-3xl font-bold mt-4">Live call</h1>
        <p className="text-sm text-ink-300 mt-2 leading-relaxed font-normal">
          A 1-to-1 browser call with optional live synthetic-voice monitoring on the incoming
          audio. You choose whether monitoring runs before it starts.
        </p>

        <div className="mt-8 space-y-4">
          <Button
            variant="primary"
            size="md"
            className="w-full"
            disabled={busy}
            leftIcon={<Phone className="w-4 h-4" />}
            onClick={() => go('/api/call/create', {}, 'caller')}
          >
            Start a call
          </Button>

          <div className="rounded-2xl border border-ink-800 bg-ink-900/60 p-4 space-y-3">
            <label className="text-xs font-medium text-ink-300 block">Or join with a code</label>
            <input
              value={code}
              onChange={(e) => setCode(e.target.value.trim())}
              placeholder="e.g. 4f2a91c3"
              className="w-full px-4 py-2.5 rounded-xl bg-ink-950 border border-ink-700 text-white placeholder-ink-500 text-sm font-mono focus:outline-none focus:border-brand-blue-400"
            />
            <Button
              variant="secondary"
              size="sm"
              disabled={busy || !code}
              leftIcon={<PhoneIncoming className="w-4 h-4" />}
              onClick={() => go('/api/call/join', { callId: code }, 'callee')}
            >
              Join
            </Button>
          </div>

          {error && (
            <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs">
              {error}
            </div>
          )}

          <p className="text-[11px] text-ink-500 leading-relaxed font-normal">
            Calls are peer-to-peer over WebRTC with a public STUN server and no TURN relay, so a
            connection between two restrictive networks may fail. Signalling is held in the
            server's memory for an hour — a restart drops in-progress calls.
          </p>
        </div>
      </div>
    </main>
  );
}

// useSearchParams opts the page out of prerendering unless it sits inside a Suspense boundary.
export default function CallLobbyPage() {
  return (
    <Suspense fallback={<main className="min-h-screen bg-ink-950" />}>
      <Lobby />
    </Suspense>
  );
}
