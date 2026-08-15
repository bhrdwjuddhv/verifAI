'use client';

// ──────────────────────────────────────────────────────────────
// components/admin/AdminEvalClient.tsx — Client Component for Evaluation Harness
// ──────────────────────────────────────────────────────────────
// NOTE FOR PRODUCTION: Simple password authentication check.
// // PRODUCTION: Replace with real auth (e.g. Firebase Auth / NextAuth session check).

import React, { useState, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  Shield,
  Lock,
  Sliders,
  BarChart3,
  CheckCircle2,
  XCircle,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  RefreshCw,
  Award,
  History,
  Check,
  Zap,
  Sparkles,
  TrendingDown,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  Legend,
} from 'recharts';
import { useEvalStore } from '@/lib/eval/store';
import type { JudgedResult, HumanJudgment, TuningPassResult } from '@/lib/eval/types';
import { Button } from '@/components/ui/Button';
import { ScoreRing } from '@/components/scan/ScoreRing';

export default function AdminEvalClient() {
  // Password auth state for demo
  const [passwordInput, setPasswordInput] = useState('');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [authError, setAuthError] = useState(false);

  // Note text for judgment
  const [reviewerNote, setReviewerNote] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Tuning state
  const [proposedPass, setProposedPass] = useState<TuningPassResult | null>(null);
  const [isTuning, setIsTuning] = useState(false);
  const [isApplying, setIsApplying] = useState(false);
  const [tuningSuccessMsg, setTuningSuccessMsg] = useState<string | null>(null);

  const {
    samples,
    judgedResults,
    weights,
    weightHistory,
    activeSampleIndex,
    setSamples,
    setJudgedResults,
    addJudgment,
    setWeights,
    setWeightHistory,
    addTuningPass,
    setActiveSampleIndex,
    getWorkingMetrics,
    getHeldOutMetrics,
    getOverallMetrics,
  } = useEvalStore();

  // Load initial data from API
  const fetchEvalData = async () => {
    try {
      const res = await fetch('/api/eval/samples');
      if (res.ok) {
        const data = await res.json();
        if (data.samples) setSamples(data.samples);
        if (data.weights) setWeights(data.weights);
        if (data.judgedResults) setJudgedResults(data.judgedResults);
        if (data.weightHistory) setWeightHistory(data.weightHistory);
      }
    } catch {
      // Fall back to client Zustand defaults
    }
  };

  useEffect(() => {
    fetchEvalData();
  }, []);

  const handlePasswordSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (passwordInput === 'verifai-demo' || passwordInput === 'admin' || passwordInput === '1234') {
      setIsAuthenticated(true);
      setAuthError(false);
    } else {
      setAuthError(true);
    }
  };

  const activeSample = samples[activeSampleIndex] || samples[0];

  const currentJudgment = useMemo(() => {
    if (!activeSample) return null;
    return judgedResults.find((j) => j.sampleId === activeSample.id) || null;
  }, [activeSample, judgedResults]);

  useEffect(() => {
    if (currentJudgment?.reviewerNote) {
      setReviewerNote(currentJudgment.reviewerNote);
    } else {
      setReviewerNote('');
    }
  }, [activeSampleIndex, currentJudgment]);

  const handleJudge = async (judgmentType: HumanJudgment) => {
    if (!activeSample) return;
    setIsSubmitting(true);

    const judgmentData: JudgedResult = {
      sampleId: activeSample.id,
      predictedScore: activeSample.predictedScore,
      predictedVerdict: activeSample.predictedVerdict,
      moduleScores: activeSample.moduleScores,
      humanJudgment: judgmentType,
      reviewerNote: reviewerNote.trim() || undefined,
      timestamp: new Date().toISOString(),
    };

    addJudgment(judgmentData);

    try {
      await fetch('/api/eval/judge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(judgmentData),
      });
    } catch (err) {
      console.error('Failed to persist judgment to server:', err);
    } finally {
      setIsSubmitting(false);
      if (activeSampleIndex < samples.length - 1) {
        setActiveSampleIndex(activeSampleIndex + 1);
      }
    }
  };

  const handleRunTuningPass = async () => {
    setIsTuning(true);
    setTuningSuccessMsg(null);
    try {
      const res = await fetch('/api/eval/tune', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        if (data.result) {
          setProposedPass(data.result);
        }
      }
    } catch (err) {
      console.error('Tuning pass failed:', err);
    } finally {
      setIsTuning(false);
    }
  };

  const handleApplyWeights = async () => {
    if (!proposedPass) return;
    setIsApplying(true);
    try {
      const res = await fetch('/api/eval/tune', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ passResult: proposedPass }),
      });

      if (res.ok) {
        setWeights(proposedPass.afterWeights);
        addTuningPass({ ...proposedPass, applied: true });
        setTuningSuccessMsg(`New fusion weights applied successfully! Working Set Acc: ${proposedPass.workingSetAccuracyAfter}%`);
        setProposedPass(null);
      }
    } catch (err) {
      console.error('Failed to apply weights:', err);
    } finally {
      setIsApplying(false);
    }
  };

  const workingMetrics = getWorkingMetrics();
  const heldOutMetrics = getHeldOutMetrics();
  const overallMetrics = getOverallMetrics();

  const judgedCount = judgedResults.length;
  const progressPercent = Math.round((judgedCount / 300) * 100);

  const weightComparisonData = useMemo(() => {
    if (!proposedPass) return [];
    return [
      { name: 'Face Forgery', Current: Math.round(proposedPass.beforeWeights.faceForgery * 100), Proposed: Math.round(proposedPass.afterWeights.faceForgery * 100) },
      { name: '2D FFT Noise', Current: Math.round(proposedPass.beforeWeights.frequencyGan * 100), Proposed: Math.round(proposedPass.afterWeights.frequencyGan * 100) },
      { name: 'Audio Spoof', Current: Math.round(proposedPass.beforeWeights.audioSpoof * 100), Proposed: Math.round(proposedPass.afterWeights.audioSpoof * 100) },
      { name: 'EXIF / ELA', Current: Math.round(proposedPass.beforeWeights.exifEla * 100), Proposed: Math.round(proposedPass.afterWeights.exifEla * 100) },
      { name: 'C2PA Manifest', Current: Math.round(proposedPass.beforeWeights.c2pa * 100), Proposed: Math.round(proposedPass.afterWeights.c2pa * 100) },
    ];
  }, [proposedPass]);

  const activeWeightsData = useMemo(() => {
    return [
      { name: 'Face Forgery', weight: Math.round(weights.faceForgery * 100) },
      { name: '2D FFT Noise', weight: Math.round(weights.frequencyGan * 100) },
      { name: 'Audio Spoof', weight: Math.round(weights.audioSpoof * 100) },
      { name: 'EXIF / ELA', weight: Math.round(weights.exifEla * 100) },
      { name: 'C2PA Manifest', weight: Math.round(weights.c2pa * 100) },
    ];
  }, [weights]);

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4 bg-ink-950">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="w-full max-w-md p-8 rounded-3xl glass-card border border-brand-blue-400/40 shadow-glow-lg text-white space-y-6"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-brand-blue-500/10 border border-brand-blue-400/30 flex items-center justify-center text-brand-blue-400">
              <Lock className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-xl font-bold">VerifAI Evaluation Portal</h2>
              <p className="text-xs text-ink-400">Admin Gated Route • Password Required</p>
            </div>
          </div>

          <form onSubmit={handlePasswordSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-ink-300 mb-1.5">
                Admin Security Password
              </label>
              <input
                type="password"
                value={passwordInput}
                onChange={(e) => setPasswordInput(e.target.value)}
                placeholder="Enter password (e.g. verifai-demo)"
                className="w-full px-4 py-3 rounded-xl bg-ink-900 border border-ink-700 text-sm text-white placeholder-ink-500 focus:outline-none focus:ring-2 focus:ring-brand-blue-400 transition-all"
              />
              {authError && (
                <p className="text-xs text-rose-400 mt-1.5 flex items-center gap-1">
                  <AlertCircle className="w-3.5 h-3.5" /> Invalid password. Try "verifai-demo"
                </p>
              )}
            </div>

            <Button type="submit" variant="primary" className="w-full" rightIcon={<Shield className="w-4 h-4" />}>
              Authenticate & Access Harness
            </Button>
          </form>

          <p className="text-[11px] text-ink-400 text-center leading-relaxed">
            Note: Simple env/password gated for hackathon evaluation. Production requires session auth.
          </p>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-ink-950 text-white p-4 sm:p-6 lg:p-8 space-y-8">
      {/* Header */}
      <header className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-ink-800 pb-6">
        <div>
          <div className="flex items-center gap-2">
            <Shield className="w-6 h-6 text-brand-blue-400" />
            <span className="text-xs font-mono font-bold uppercase tracking-widest text-brand-blue-300">
              VerifAI Admin Engine
            </span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold mt-1 text-gradient">
            300-Sample Human-Judged Evaluation & Fusion Tuner
          </h1>
          <p className="text-xs text-ink-400 mt-1 max-w-2xl leading-relaxed">
            Human evaluation feedback loop. Evaluates 300 ground-truth samples, measures overall accuracy & calibration error, and runs a bounded grid-search optimizer over the 5 detection module fusion weights.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            onClick={fetchEvalData}
            leftIcon={<RefreshCw className="w-4 h-4" />}
          >
            Refresh Dataset
          </Button>
          <a
            href="/"
            className="text-xs font-semibold px-4 py-2 rounded-xl glass border border-ink-700 text-ink-300 hover:text-white transition-all"
          >
            ← Back to Public Sandbox
          </a>
        </div>
      </header>

      {/* Progress & Metrics Summary */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-5 rounded-2xl bg-ink-900/80 border border-ink-800 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-ink-400">Judging Progress</span>
            <span className="text-xs font-mono font-bold text-brand-blue-300">{judgedCount} / 300</span>
          </div>
          <div className="text-2xl font-bold text-white">{progressPercent}% Judged</div>
          <div className="w-full bg-ink-950 h-2 rounded-full overflow-hidden border border-ink-800">
            <div
              className="h-full bg-gradient-to-r from-brand-blue-300 to-brand-blue-500 transition-all duration-300"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>

        <div className="p-5 rounded-2xl bg-ink-900/80 border border-ink-800 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-ink-400">Fusion Accuracy</span>
            <Award className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-emerald-400">{overallMetrics.accuracy}%</div>
          <p className="text-[11px] text-ink-400">
            Working Set: <span className="text-white font-semibold">{workingMetrics.accuracy}%</span> • Held-Out: <span className="text-white font-semibold">{heldOutMetrics.accuracy}%</span>
          </p>
        </div>

        <div className="p-5 rounded-2xl bg-ink-900/80 border border-ink-800 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-ink-400">Mean Calibration Error</span>
            <TrendingDown className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold text-amber-300">
            {overallMetrics.calibrationError} pts
          </div>
          <p className="text-[11px] text-ink-400">
            Average gap between predicted score & ground truth
          </p>
        </div>

        <div className="p-5 rounded-2xl bg-ink-900/80 border border-ink-800 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-ink-400">Fusion Engine Status</span>
            <Sliders className="w-4 h-4 text-brand-blue-400" />
          </div>
          <div className="text-sm font-bold text-white flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            5 Modules Tuned
          </div>
          <p className="text-[11px] text-ink-400">
            {weightHistory.length} weight tuning iterations recorded
          </p>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Sample Judging */}
        <div className="lg:col-span-7 space-y-6">
          <div className="p-6 rounded-3xl glass-card border border-ink-800 space-y-6">
            <div className="flex items-center justify-between border-b border-ink-800 pb-4">
              <div className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-brand-blue-400" />
                <h3 className="text-lg font-bold text-white">Sample Evaluation Harness</h3>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  disabled={activeSampleIndex === 0}
                  onClick={() => setActiveSampleIndex(Math.max(0, activeSampleIndex - 1))}
                  className="p-2 rounded-xl bg-ink-900 border border-ink-800 text-ink-300 hover:text-white disabled:opacity-30 transition-all"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>

                <span className="text-xs font-mono font-semibold px-3 py-1.5 rounded-xl bg-ink-900 border border-ink-800">
                  {activeSampleIndex + 1} / {samples.length}
                </span>

                <button
                  type="button"
                  disabled={activeSampleIndex === samples.length - 1}
                  onClick={() => setActiveSampleIndex(Math.min(samples.length - 1, activeSampleIndex + 1))}
                  className="p-2 rounded-xl bg-ink-900 border border-ink-800 text-ink-300 hover:text-white disabled:opacity-30 transition-all"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>

            {activeSample && (
              <div className="space-y-6">
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-4 rounded-2xl bg-ink-900/60 border border-ink-800">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono text-brand-blue-300 font-semibold">{activeSample.id}</span>
                      <span
                        className={`text-[10px] font-semibold px-2.5 py-0.5 rounded-full border ${
                          activeSample.split === 'working'
                            ? 'bg-brand-blue-500/10 text-brand-blue-300 border-brand-blue-500/30'
                            : 'bg-purple-500/10 text-purple-300 border-purple-500/30'
                        }`}
                      >
                        {activeSample.split === 'working' ? 'Working Set (240)' : 'Held-Out Set (60)'}
                      </span>
                    </div>
                    <h4 className="text-base font-bold text-white mt-1 truncate max-w-sm">
                      {activeSample.mediaRef}
                    </h4>
                  </div>

                  <div className="text-right">
                    <span className="text-[10px] text-ink-400 uppercase font-mono block">Known Ground Truth</span>
                    <span
                      className={`inline-block mt-0.5 text-xs font-bold px-3 py-1 rounded-full border ${
                        activeSample.groundTruthLabel === 'authentic'
                          ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                          : activeSample.groundTruthLabel === 'manipulated'
                          ? 'bg-rose-500/15 text-rose-400 border-rose-500/30'
                          : 'bg-amber-500/15 text-amber-400 border-amber-500/30'
                      }`}
                    >
                      {activeSample.groundTruthLabel.toUpperCase()}
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
                  <div className="md:col-span-5 flex flex-col items-center justify-center p-4 rounded-2xl bg-ink-950/80 border border-ink-800">
                    <ScoreRing score={activeSample.predictedScore} size={150} />
                    <span className="text-xs text-ink-400 mt-2 font-mono">
                      Predicted Score: <strong className="text-white">{activeSample.predictedScore}/100</strong>
                    </span>
                  </div>

                  <div className="md:col-span-7 space-y-2.5">
                    <h5 className="text-xs font-semibold uppercase tracking-wider text-ink-300 mb-2">
                      Modular Signal Scores
                    </h5>

                    {[
                      { label: 'Face Forgery Seams', score: activeSample.moduleScores.faceForgery, weight: weights.faceForgery },
                      { label: '2D FFT Noise Residual', score: activeSample.moduleScores.frequencyGan, weight: weights.frequencyGan },
                      { label: 'Audio Vocoder Spoof', score: activeSample.moduleScores.audioSpoof, weight: weights.audioSpoof },
                      { label: 'EXIF & PRNU Forensics', score: activeSample.moduleScores.exifEla, weight: weights.exifEla },
                      { label: 'C2PA Manifest', score: activeSample.moduleScores.c2pa, weight: weights.c2pa },
                    ].map((mod, idx) => (
                      <div key={idx} className="space-y-1">
                        <div className="flex items-center justify-between text-xs font-mono">
                          <span className="text-ink-300">{mod.label}</span>
                          <span className="text-ink-200">
                            {mod.score}% <span className="text-ink-500">({Math.round(mod.weight * 100)}% w)</span>
                          </span>
                        </div>
                        <div className="w-full bg-ink-900 h-1.5 rounded-full overflow-hidden border border-ink-800">
                          <div
                            className={`h-full rounded-full transition-all duration-300 ${
                              mod.score >= 75 ? 'bg-emerald-400' : mod.score >= 40 ? 'bg-amber-400' : 'bg-rose-400'
                            }`}
                            style={{ width: `${mod.score}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="p-5 rounded-2xl bg-ink-900/80 border border-brand-blue-400/30 space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
                      <Sparkles className="w-4 h-4 text-brand-blue-400" />
                      Judge VerifAI's Prediction
                    </span>

                    {currentJudgment && (
                      <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 flex items-center gap-1">
                        <Check className="w-3 h-3" /> Already Judged ({currentJudgment.humanJudgment})
                      </span>
                    )}
                  </div>

                  <div className="grid grid-cols-3 gap-3">
                    <button
                      type="button"
                      disabled={isSubmitting}
                      onClick={() => handleJudge('correct')}
                      className={`p-3 rounded-xl border flex flex-col items-center justify-center gap-1.5 transition-all text-xs font-bold ${
                        currentJudgment?.humanJudgment === 'correct'
                          ? 'bg-emerald-500/25 border-emerald-400 text-emerald-300 shadow-glow-sm scale-[1.02]'
                          : 'bg-ink-950 border-ink-700 text-emerald-400 hover:bg-emerald-500/10 hover:border-emerald-500/50'
                      }`}
                    >
                      <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                      <span>Correct ✓</span>
                    </button>

                    <button
                      type="button"
                      disabled={isSubmitting}
                      onClick={() => handleJudge('partially_correct')}
                      className={`p-3 rounded-xl border flex flex-col items-center justify-center gap-1.5 transition-all text-xs font-bold ${
                        currentJudgment?.humanJudgment === 'partially_correct'
                          ? 'bg-amber-500/25 border-amber-400 text-amber-300 shadow-glow-sm scale-[1.02]'
                          : 'bg-ink-950 border-ink-700 text-amber-400 hover:bg-amber-500/10 hover:border-amber-500/50'
                      }`}
                    >
                      <AlertCircle className="w-5 h-5 text-amber-400" />
                      <span>Partially Correct ~</span>
                    </button>

                    <button
                      type="button"
                      disabled={isSubmitting}
                      onClick={() => handleJudge('incorrect')}
                      className={`p-3 rounded-xl border flex flex-col items-center justify-center gap-1.5 transition-all text-xs font-bold ${
                        currentJudgment?.humanJudgment === 'incorrect'
                          ? 'bg-rose-500/25 border-rose-400 text-rose-300 shadow-glow-sm scale-[1.02]'
                          : 'bg-ink-950 border-ink-700 text-rose-400 hover:bg-rose-500/10 hover:border-rose-500/50'
                      }`}
                    >
                      <XCircle className="w-5 h-5 text-rose-400" />
                      <span>Incorrect ✗</span>
                    </button>
                  </div>

                  <div>
                    <input
                      type="text"
                      value={reviewerNote}
                      onChange={(e) => setReviewerNote(e.target.value)}
                      placeholder="Optional reviewer notes (e.g. 'False positive due to lighting blur')"
                      className="w-full px-3.5 py-2.5 rounded-xl bg-ink-950 border border-ink-800 text-xs text-white placeholder-ink-500 focus:outline-none focus:border-brand-blue-400 transition-all"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Fusion Weight Optimizer */}
        <div className="lg:col-span-5 space-y-6">
          <div className="p-6 rounded-3xl glass-card border border-ink-800 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Sliders className="w-5 h-5 text-brand-blue-400" />
                <h3 className="text-base font-bold text-white">Active Fusion Weights</h3>
              </div>
              <span className="text-[10px] font-mono text-brand-blue-300 bg-brand-blue-500/10 px-2.5 py-0.5 rounded-full border border-brand-blue-500/20">
                Single Source of Truth
              </span>
            </div>

            <div className="h-44 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={activeWeightsData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                  <XAxis dataKey="name" stroke="#8E949F" fontSize={9} interval={0} />
                  <YAxis domain={[0, 50]} stroke="#5C6370" fontSize={10} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#16181C',
                      borderColor: '#2A2E35',
                      borderRadius: '8px',
                      fontSize: '12px',
                      color: '#FFF',
                    }}
                    formatter={(val: number) => [`${val}%`, 'Fusion Weight']}
                  />
                  <Bar dataKey="weight" fill="#94D0F3" radius={[4, 4, 0, 0]}>
                    {activeWeightsData.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={index === 0 ? '#94D0F3' : index === 1 ? '#5CB4E8' : index === 2 ? '#2A92D7' : index === 3 ? '#10B981' : '#F59E0B'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="p-6 rounded-3xl glass-card border border-brand-blue-400/40 shadow-glow-sm space-y-5">
            <div className="flex items-center justify-between border-b border-ink-800 pb-3">
              <div className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-brand-blue-400" />
                <h3 className="text-base font-bold text-white">Fusion Weight Optimizer</h3>
              </div>

              <Button
                variant="primary"
                size="sm"
                isLoading={isTuning}
                onClick={handleRunTuningPass}
                leftIcon={<Sliders className="w-4 h-4" />}
              >
                Run Tuning Pass
              </Button>
            </div>

            {tuningSuccessMsg && (
              <div className="p-3 rounded-xl bg-emerald-500/15 border border-emerald-500/30 text-xs text-emerald-300 flex items-center gap-2 font-medium">
                <Check className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                <span>{tuningSuccessMsg}</span>
              </div>
            )}

            {proposedPass && (
              <div className="space-y-4 pt-2 border-t border-ink-800/80">
                <div className="flex items-center justify-between text-xs font-semibold">
                  <span className="text-brand-blue-300">Proposed Weight Optimization Result</span>
                  <span className="text-ink-400 font-mono text-[11px]">{new Date(proposedPass.timestamp).toLocaleTimeString()}</span>
                </div>

                <div className="h-44 w-full bg-ink-950/60 p-2 rounded-xl border border-ink-800">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={weightComparisonData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                      <XAxis dataKey="name" stroke="#8E949F" fontSize={9} interval={0} />
                      <YAxis domain={[0, 50]} stroke="#5C6370" fontSize={10} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#16181C',
                          borderColor: '#2A2E35',
                          borderRadius: '8px',
                          fontSize: '11px',
                          color: '#FFF',
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: '10px' }} />
                      <Bar dataKey="Current" fill="#5C6370" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="Proposed" fill="#10B981" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="p-3 rounded-xl bg-ink-950 border border-ink-800 space-y-1">
                    <span className="text-ink-400 block text-[10px]">Working Set Accuracy (240)</span>
                    <div className="flex items-center justify-between font-bold">
                      <span className="text-ink-400">{proposedPass.workingSetAccuracyBefore}%</span>
                      <span className="text-emerald-400 font-mono">→ {proposedPass.workingSetAccuracyAfter}%</span>
                    </div>
                  </div>

                  <div className="p-3 rounded-xl bg-ink-950 border border-ink-800 space-y-1">
                    <span className="text-ink-400 block text-[10px]">Held-Out Set Accuracy (60)</span>
                    <div className="flex items-center justify-between font-bold">
                      <span className="text-ink-400">{proposedPass.heldOutAccuracyBefore}%</span>
                      <span
                        className={
                          proposedPass.heldOutAccuracyAfter < proposedPass.heldOutAccuracyBefore
                            ? 'text-amber-400 font-mono'
                            : 'text-emerald-400 font-mono'
                        }
                      >
                        → {proposedPass.heldOutAccuracyAfter}%
                      </span>
                    </div>
                  </div>
                </div>

                {proposedPass.heldOutAccuracyAfter < proposedPass.heldOutAccuracyBefore && (
                  <div className="p-3 rounded-xl bg-amber-500/15 border border-amber-500/30 text-xs text-amber-300 flex items-start gap-2">
                    <AlertCircle className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <span className="font-bold block">Caution: Held-Out Accuracy Regression</span>
                      <span className="text-[11px] text-amber-200">
                        The proposed weights increase working-set agreement but reduce held-out accuracy from {proposedPass.heldOutAccuracyBefore}% to {proposedPass.heldOutAccuracyAfter}%. Applying may overfit.
                      </span>
                    </div>
                  </div>
                )}

                <Button
                  variant="primary"
                  className="w-full"
                  isLoading={isApplying}
                  onClick={handleApplyWeights}
                  leftIcon={<Check className="w-4 h-4" />}
                >
                  Apply Proposed Fusion Weights to verdict.ts
                </Button>
              </div>
            )}
          </div>

          {weightHistory.length > 0 && (
            <div className="p-6 rounded-3xl glass-card border border-ink-800 space-y-4">
              <div className="flex items-center gap-2">
                <History className="w-5 h-5 text-brand-blue-400" />
                <h3 className="text-base font-bold text-white">Versioned Weight History Log</h3>
              </div>

              <div className="space-y-2.5 max-h-52 overflow-y-auto pr-1">
                {weightHistory.map((entry, idx) => (
                  <div key={idx} className="p-3 rounded-xl bg-ink-950/80 border border-ink-800 text-xs space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-brand-blue-300 font-semibold">{entry.id}</span>
                      <span className="text-[10px] text-ink-400">{new Date(entry.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <div className="flex items-center justify-between text-ink-300">
                      <span>Working Acc: {entry.workingSetAccuracyBefore}% → {entry.workingSetAccuracyAfter}%</span>
                      <span className={entry.applied ? 'text-emerald-400 font-bold' : 'text-ink-500'}>
                        {entry.applied ? 'Applied' : 'Tested'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
