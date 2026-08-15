'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Copy,
  Check,
  ChevronDown,
  ShieldCheck,
  AlertTriangle,
  FileCheck,
  RefreshCw,
  Lock,
  Info,
  Cpu,
} from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';
import { ScanResult, useAppStore } from '@/lib/store';
import { ScoreRing } from './ScoreRing';
import { Button } from '@/components/ui/Button';

export interface ReportPanelProps {
  result: ScanResult;
  onReset: () => void;
}

export const ReportPanel: React.FC<ReportPanelProps> = ({ result, onReset }) => {
  const { showToast } = useAppStore();
  const [copied, setCopied] = useState(false);
  const [reasonsOpen, setReasonsOpen] = useState(true);
  const [modelsOpen, setModelsOpen] = useState(false);

  // A copyable summary of what was measured. There is no hosted badge to link to.
  const handleCopySummary = () => {
    const summary = [
      `VerifAI ${result.id} — ${result.filename}`,
      `Verdict: ${result.verdict.label} (confidence ${result.confidence}%)`,
      `Model: ${result.modelSource}`,
      `P(AI-generated or manipulated): ${result.signals.modelScore ?? 'not measured'}%`,
      `High-frequency energy share: ${result.signals.frequencyScore ?? 'not measured'}%`,
      `Face detected: ${result.signals.faceDetected === null ? 'detector unavailable' : result.signals.faceDetected ? 'yes' : 'no'}`,
      `Scanned ${result.timestamp}`,
    ].join('\n');

    if (navigator.clipboard) {
      navigator.clipboard.writeText(summary);
      setCopied(true);
      showToast('Result summary copied to clipboard');
      setTimeout(() => setCopied(false), 2500);
    }
  };

  // Only measured values get a bar. Nothing is padded out to fill the chart.
  const chartData = [
    result.score !== null && {
      name: 'Likelihood Real',
      score: result.score,
      color: result.verdict.ringColor,
    },
    result.signals.frequencyScore !== null && {
      name: 'High-Freq Energy',
      score: result.signals.frequencyScore,
      color: '#60A5FA',
    },
  ].filter(Boolean) as { name: string; score: number; color: string }[];

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="w-full bg-ink-950/90 text-white rounded-3xl p-6 sm:p-8 border border-ink-800 shadow-dark-glass"
    >
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-ink-800 pb-6 mb-6">
        <div>
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-brand-blue-400" />
            <span className="text-xs font-mono font-semibold text-brand-blue-300 uppercase tracking-widest">
              Verification Result
            </span>
          </div>
          <h3 className="text-xl sm:text-2xl font-semibold mt-1 truncate max-w-md">
            {result.filename}
          </h3>
          <p className="text-xs text-ink-400 mt-0.5 font-normal">
            Scanned {new Date(result.timestamp).toLocaleTimeString()} • File Type: {result.fileType.toUpperCase()} ({result.fileSize})
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={handleCopySummary}
            leftIcon={copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
          >
            {copied ? 'Copied' : 'Copy Result Summary'}
          </Button>
          <Button variant="ghost" size="sm" onClick={onReset} leftIcon={<RefreshCw className="w-4 h-4" />}>
            Scan Another File
          </Button>
        </div>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Score Ring & Plain English Summary */}
        <div className="lg:col-span-5 flex flex-col items-center bg-ink-900/60 p-6 rounded-2xl border border-ink-800">
          <ScoreRing score={result.score} verdict={result.verdict} size={190} />

          {/* Layman English Summary Box */}
          <div className="w-full mt-5 p-4 rounded-xl bg-ink-950/80 border border-brand-blue-400/30 text-xs space-y-2">
            <div className="flex items-center gap-1.5 font-semibold text-brand-blue-300">
              <Info className="w-4 h-4 flex-shrink-0" />
              <span>Summary (Simple Explanation)</span>
            </div>
            <p className="text-ink-300 leading-relaxed font-normal">
              {result.verdict.laymanSummary}
            </p>
            <p className="text-ink-400 leading-relaxed font-normal">
              {result.verdict.recommendation}
            </p>
          </div>

          {/* Grad-CAM overlay, when the active model supports it */}
          {result.heatmap && (
            <div className="w-full mt-4">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={result.heatmap}
                alt="Grad-CAM heatmap of the analyzed region"
                className="w-full rounded-xl border border-ink-800"
              />
              <p className="text-[11px] text-ink-400 mt-1.5 font-normal">
                Grad-CAM: the warmer the region, the more it pushed the model toward its score. It
                shows where the model looked, not what is wrong.
              </p>
            </div>
          )}

          <div className="w-full mt-4 pt-3 border-t border-ink-800/80 space-y-2 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-ink-400 flex items-center gap-1.5 font-normal">
                <Lock className="w-3.5 h-3.5 text-brand-blue-400" />
                C2PA manifest:
              </span>
              <span className="font-semibold px-2 py-0.5 rounded-full bg-ink-800/80 text-ink-200 border border-ink-700">
                {result.metadata.c2paManifestPresent ? 'Present' : 'Absent'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-ink-400 flex items-center gap-1.5 font-normal">
                <Lock className="w-3.5 h-3.5 text-brand-blue-400" />
                EXIF metadata:
              </span>
              <span className="font-semibold px-2 py-0.5 rounded-full bg-ink-800/80 text-ink-200 border border-ink-700">
                {result.metadata.exifPresent ? 'Present' : 'Absent'}
              </span>
            </div>
            <p className="text-[11px] text-ink-500 font-normal leading-relaxed">
              Metadata is a supporting signal only — it is not part of the verdict. Most platforms
              strip it from real photos, and it can be forged.
            </p>
          </div>
        </div>

        {/* Reasons, Model & Signals */}
        <div className="lg:col-span-7 space-y-6">
          {/* Key Findings in Plain English */}
          <div className="bg-ink-900/60 rounded-2xl border border-ink-800 overflow-hidden">
            <button
              type="button"
              onClick={() => setReasonsOpen(!reasonsOpen)}
              className="w-full flex items-center justify-between p-4 text-left hover:bg-ink-800/40 transition-colors"
            >
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                <span className="font-semibold text-sm text-white">
                  What was measured ({result.reasons.length})
                </span>
              </div>
              <motion.div animate={{ rotate: reasonsOpen ? 180 : 0 }}>
                <ChevronDown className="w-4 h-4 text-ink-400" />
              </motion.div>
            </button>

            <AnimatePresence>
              {reasonsOpen && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ opacity: 0, height: 0 }}
                  className="px-4 pb-4 border-t border-ink-800/80 pt-3"
                >
                  <ul className="space-y-2.5 text-xs text-ink-300">
                    {result.reasons.map((reason, idx) => (
                      <li key={idx} className="flex items-start gap-2 leading-relaxed font-normal">
                        <span className="w-1.5 h-1.5 rounded-full bg-brand-blue-400 mt-1.5 flex-shrink-0" />
                        <span>{reason}</span>
                      </li>
                    ))}
                  </ul>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Active model — what actually produced this verdict */}
          <div className="bg-ink-900/60 rounded-2xl border border-ink-800 overflow-hidden">
            <button
              type="button"
              onClick={() => setModelsOpen(!modelsOpen)}
              className="w-full flex items-center justify-between p-4 text-left hover:bg-ink-800/40 transition-colors"
            >
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-brand-blue-400" />
                <span className="font-semibold text-sm text-white">Model used</span>
              </div>
              <motion.div animate={{ rotate: modelsOpen ? 180 : 0 }}>
                <ChevronDown className="w-4 h-4 text-ink-400" />
              </motion.div>
            </button>

            <AnimatePresence>
              {modelsOpen && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ opacity: 0, height: 0 }}
                  className="px-4 pb-4 border-t border-ink-800/80 pt-3 space-y-2.5"
                >
                  <div className="p-3 rounded-xl bg-ink-950/80 border border-ink-800 text-xs space-y-1.5">
                    <div className="font-mono text-brand-blue-300 break-all">{result.modelSource}</div>
                    <p className="text-ink-400 text-[11px] font-normal">
                      Confidence {result.confidence}% in the reported verdict. Accuracy on data
                      like yours is unmeasured — see the{' '}
                      <a href="/model-card" className="text-brand-blue-300 underline">
                        model card
                      </a>{' '}
                      for what is and is not known.
                    </p>
                  </div>
                  {result.notes.map((note, idx) => (
                    <div
                      key={idx}
                      className="p-3 rounded-xl bg-ink-950/80 border border-amber-500/20 text-[11px] text-amber-200/90 font-normal"
                    >
                      {note}
                    </div>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Measured signals */}
          <div className="bg-ink-900/60 p-4 sm:p-5 rounded-2xl border border-ink-800">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-ink-300 mb-3 flex items-center gap-1.5">
              <FileCheck className="w-4 h-4 text-brand-blue-400" />
              Measured signals
            </h4>

            {chartData.length === 0 ? (
              <p className="text-xs text-ink-400 font-normal py-6 text-center">
                Nothing was measured for this file.
              </p>
            ) : (
              <div className="h-44 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={chartData}
                    layout="vertical"
                    margin={{ top: 5, right: 15, left: 30, bottom: 5 }}
                  >
                    <XAxis type="number" domain={[0, 100]} stroke="#5C6370" fontSize={10} />
                    <YAxis
                      dataKey="name"
                      type="category"
                      stroke="#8E949F"
                      fontSize={10}
                      width={110}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#16181C',
                        borderColor: '#2A2E35',
                        borderRadius: '8px',
                        color: '#FFF',
                        fontSize: '12px',
                      }}
                      formatter={(val: number, _n, item: any) => [`${val}%`, item?.payload?.name]}
                    />
                    <Bar dataKey="score" radius={[0, 6, 6, 0]}>
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            <p className="text-[11px] text-ink-500 mt-2 font-normal leading-relaxed">
              “Likelihood Real” is 100 minus the model’s P(AI-generated). “High-Freq Energy” is the
              share of image detail above half-Nyquist — descriptive only; a sharp camera photo
              scores high too, and it does not move the verdict.
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
};
