// ──────────────────────────────────────────────────────────────
// lib/eval/metrics.ts — Evaluation metrics calculation
// ──────────────────────────────────────────────────────────────

import type { JudgedResult, EvalMetrics, EvalSample } from './types';
import type { VerdictCategory } from '@/lib/verdict';

/**
 * Maps human judgment + ground truth or predicted verdict to alignment.
 * A judgment is aligned if:
 * - humanJudgment is 'correct'
 * - or humanJudgment is 'partially_correct' (weighted as 0.5)
 */
export function computeEvalMetrics(
  judgedResults: JudgedResult[],
  samplesById: Record<string, EvalSample>,
  filterSplit?: 'working' | 'heldout'
): EvalMetrics {
  const filtered = judgedResults.filter((j) => {
    if (!filterSplit) return true;
    const sample = samplesById[j.sampleId];
    return sample?.split === filterSplit;
  });

  if (filtered.length === 0) {
    return {
      totalJudged: 0,
      accuracy: 0,
      perClass: {
        genuine: { precision: 0, recall: 0, count: 0 },
        suspicious: { precision: 0, recall: 0, count: 0 },
        manipulated: { precision: 0, recall: 0, count: 0 },
      },
      calibrationError: 0,
    };
  }

  // 1. Overall Accuracy / Agreement
  let totalScore = 0;
  let calibrationSum = 0;

  // Counts for Precision / Recall
  // Matrix: [Predicted][GroundTruthOrHumanTarget]
  const counts: Record<
    VerdictCategory,
    { tp: number; fp: number; fn: number; totalPredicted: number; totalActual: number }
  > = {
    genuine: { tp: 0, fp: 0, fn: 0, totalPredicted: 0, totalActual: 0 },
    suspicious: { tp: 0, fp: 0, fn: 0, totalPredicted: 0, totalActual: 0 },
    uncertain: { tp: 0, fp: 0, fn: 0, totalPredicted: 0, totalActual: 0 },
    manipulated: { tp: 0, fp: 0, fn: 0, totalPredicted: 0, totalActual: 0 },
  };

  for (const j of filtered) {
    const sample = samplesById[j.sampleId];
    const actualLabel = sample ? sample.groundTruthLabel : 'inconclusive';

    // Map actual label to verdict category
    const actualCategory: VerdictCategory =
      actualLabel === 'authentic' ? 'genuine' : actualLabel === 'manipulated' ? 'manipulated' : 'suspicious';

    const predCategory = j.predictedVerdict;

    // Agreement score calculation
    if (j.humanJudgment === 'correct') {
      totalScore += 1;
    } else if (j.humanJudgment === 'partially_correct') {
      totalScore += 0.5;
    }

    // Confusion stats
    counts[predCategory].totalPredicted += 1;
    counts[actualCategory].totalActual += 1;

    if (predCategory === actualCategory) {
      counts[predCategory].tp += 1;
    } else {
      counts[predCategory].fp += 1;
      counts[actualCategory].fn += 1;
    }

    // Calibration error calculation (|predictedScore - actualGroundScore|)
    const actualScore = sample?.groundTruthScore ?? (actualCategory === 'genuine' ? 90 : actualCategory === 'manipulated' ? 15 : 50);
    calibrationSum += Math.abs(j.predictedScore - actualScore);
  }

  const accuracy = Math.round((totalScore / filtered.length) * 100);
  const calibrationError = Number((calibrationSum / filtered.length).toFixed(1));

  const perClass = {
    genuine: {
      precision: counts.genuine.totalPredicted > 0 ? Math.round((counts.genuine.tp / counts.genuine.totalPredicted) * 100) : 0,
      recall: counts.genuine.totalActual > 0 ? Math.round((counts.genuine.tp / counts.genuine.totalActual) * 100) : 0,
      count: counts.genuine.totalActual,
    },
    suspicious: {
      precision: counts.suspicious.totalPredicted > 0 ? Math.round((counts.suspicious.tp / counts.suspicious.totalPredicted) * 100) : 0,
      recall: counts.suspicious.totalActual > 0 ? Math.round((counts.suspicious.tp / counts.suspicious.totalActual) * 100) : 0,
      count: counts.suspicious.totalActual,
    },
    manipulated: {
      precision: counts.manipulated.totalPredicted > 0 ? Math.round((counts.manipulated.tp / counts.manipulated.totalPredicted) * 100) : 0,
      recall: counts.manipulated.totalActual > 0 ? Math.round((counts.manipulated.tp / counts.manipulated.totalActual) * 100) : 0,
      count: counts.manipulated.totalActual,
    },
  };

  return {
    totalJudged: filtered.length,
    accuracy,
    perClass,
    calibrationError,
  };
}
