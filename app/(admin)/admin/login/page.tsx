'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Lock, Mail, ShieldCheck, AlertCircle, ArrowRight, Loader2, Clock } from 'lucide-react';

export default function AdminLoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lockoutInfo, setLockoutInfo] = useState<{ locked: boolean; retryAfter?: string } | null>(null);
  const router = useRouter();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setLockoutInfo(null);

    try {
      const res = await fetch('/api/admin/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      const data = await res.json();

      if (!res.ok) {
        if (data.locked) {
          setLockoutInfo({ locked: true, retryAfter: data.retryAfter });
          setError('Account temporarily locked due to too many failed attempts.');
        } else {
          setError(data.error || 'Login failed');
          if (data.attemptsRemaining !== null && data.attemptsRemaining !== undefined) {
            setError(`${data.error} (${data.attemptsRemaining} attempts remaining)`);
          }
        }
        return;
      }

      router.push('/admin');
      router.refresh();
    } catch (err: any) {
      setError(err.message || 'Connection error');
    } finally {
      setLoading(false);
    }
  };

  const lockoutTimeRemaining = lockoutInfo?.retryAfter
    ? Math.max(0, Math.ceil((new Date(lockoutInfo.retryAfter).getTime() - Date.now()) / 60000))
    : 0;

  return (
    <div className="min-h-screen flex items-center justify-center p-4 relative overflow-hidden bg-ink-950">
      {/* Background Ambient Orbs */}
      <div className="absolute inset-0 pointer-events-none z-0 opacity-30">
        <div className="orb orb-1 top-[-10%] left-[-10%]" />
        <div className="orb orb-2 bottom-[-10%] right-[-10%]" />
        <div className="absolute inset-0 dot-grid opacity-15" />
      </div>

      <div className="relative z-10 w-full max-w-md">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-brand-blue-500/10 border border-brand-blue-500/30 text-brand-blue-400 mb-4 shadow-glow-sm">
            <ShieldCheck className="w-7 h-7" />
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-white">VerifAI Admin Portal</h1>
          <p className="text-sm text-ink-400 mt-1">Authenticated area for team model control & training</p>
        </div>

        <form
          onSubmit={handleLogin}
          className="glass-card p-6 md:p-8 rounded-2xl border border-ink-800 shadow-2xl space-y-5"
        >
          {/* Lockout Warning */}
          {lockoutInfo?.locked && (
            <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-start gap-2.5 text-amber-400 text-sm">
              <Clock className="w-4 h-4 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">Account temporarily locked</p>
                <p className="text-xs text-amber-300/80 mt-0.5">
                  Too many failed login attempts. Try again in ~{lockoutTimeRemaining} minute{lockoutTimeRemaining !== 1 ? 's' : ''}.
                </p>
              </div>
            </div>
          )}

          {/* Error */}
          {error && !lockoutInfo?.locked && (
            <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 flex items-center gap-2.5 text-rose-400 text-sm">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Email */}
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-ink-300 mb-2">
              Email Address
            </label>
            <div className="relative">
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="admin@yourteam.com"
                className="w-full px-4 py-3 bg-ink-900/80 border border-ink-700/80 rounded-xl text-white placeholder-ink-500 focus:outline-none focus:border-brand-blue-400 focus:ring-1 focus:ring-brand-blue-400 transition-all text-sm"
                required
                autoComplete="email"
              />
              <Mail className="w-4 h-4 text-ink-500 absolute right-3.5 top-1/2 -translate-y-1/2" />
            </div>
          </div>

          {/* Password */}
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-ink-300 mb-2">
              Password
            </label>
            <div className="relative">
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password..."
                className="w-full px-4 py-3 bg-ink-900/80 border border-ink-700/80 rounded-xl text-white placeholder-ink-500 focus:outline-none focus:border-brand-blue-400 focus:ring-1 focus:ring-brand-blue-400 transition-all text-sm"
                required
                autoComplete="current-password"
              />
              <Lock className="w-4 h-4 text-ink-500 absolute right-3.5 top-1/2 -translate-y-1/2" />
            </div>
            <p className="text-xs text-ink-500 mt-1.5">
              Contact your team owner if you need an account.
            </p>
          </div>

          <button
            type="submit"
            disabled={loading || !email || !password || lockoutInfo?.locked}
            className="w-full py-3 px-4 rounded-xl bg-brand-blue-600 hover:bg-brand-blue-500 text-white font-medium text-sm transition-all shadow-glow-sm flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Authenticating...</span>
              </>
            ) : (
              <>
                <span>Sign In</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
