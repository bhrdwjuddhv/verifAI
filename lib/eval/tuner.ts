// ──────────────────────────────────────────────────────────────
// lib/eval/tuner.ts — Grid Search Optimizer for Fusion Weights
// ──────────────────────────────────────────────────────────────

import type { FusionWeights, JudgedResult, EvalSample, TuningPassResult } from './types';
import { FUSION_WEIGHT_KEYS } from './types';
import { computeWeightedScore, scoreToVerdict } from './sampleGenerator';

/**
 * Calculates human agreement score for a given candidate weight set over a set of judged results.
 */
function evaluateWeightCandidate(
  weights: FusionWeights,
  judgedResults: JudgedResult[],
  samplesById: Record<string, EvalSample>
): number {
  if (judgedResults.length === 0) return 0;

  let score = 0;
  for (const j of judgedResults) {
    const sample = samplesById[j.sampleId];
    if (!sample) continue;

    // Recompute score under candidate weights
    const candidateScore = computeWeightedScore(sample.moduleScores, weights);
    const candidateVerdict = scoreToVerdict(candidateScore);

    // Ground truth target verdict
    const actualLabel = sample.groundTruthLabel;
    const targetVerdict =
      actualLabel === 'authentic' ? 'genuine' : actualLabel === 'manipulated' ? 'manipulated' : 'suspicious';

    if (candidateVerdict === targetVerdict) {
      score += 1;
    } else if (
      (candidateVerdict === 'suspicious' && targetVerdict !== 'suspicious') ||
      (targetVerdict === 'suspicious' && candidateVerdict !== 'suspicious')
    ) {
      score += 0.5; // Partial credit for near misses
    }
  }

  return Math.round((score / judgedResults.length) * 100);
}

/**
 * Grid search optimization over candidate fusion weights.
 * - Explores candidate weights within ±15% relative change of current weights
 * - Normalizes weights so sum = 1.0
 * - Optimizes on working set judged samples (up to 240)
 * - Evaluates before/after accuracy on both working set and held-out set (60)
 */
export function runFusionWeightTuner(
  currentWeights: FusionWeights,
  judgedResults: JudgedResult[],
  samplesById: Record<string, EvalSample>
): TuningPassResult {
  const workingJudged = judgedResults.filter((j) => samplesById[j.sampleId]?.split === 'working');
  const heldOutJudged = judgedResults.filter((j) => samplesById[j.sampleId]?.split === 'heldout');

  const workingSetAccuracyBefore = evaluateWeightCandidate(currentWeights, workingJudged, samplesById);
  const heldOutAccuracyBefore = evaluateWeightCandidate(currentWeights, heldOutJudged, samplesById);

  // If no working set judged samples, return current weights unchanged
  if (workingJudged.length === 0) {
    return {
      id: `TUNE-${Date.now()}`,
      timestamp: new Date().toISOString(),
      beforeWeights: { ...currentWeights },
      afterWeights: { ...currentWeights },
      workingSetAccuracyBefore: 0,
      workingSetAccuracyAfter: 0,
      heldOutAccuracyBefore: 0,
      heldOutAccuracyAfter: 0,
      applied: false,
    };
  }

  // Generate grid search steps for each weight (±15% bounded constraint)
  const MAX_DELTA = 0.05; // Max shift per weight scalar per pass (~15-25% shift)
  const STEPS = [-MAX_DELTA, -MAX_DELTA / 2, 0, MAX_DELTA / 2, MAX_DELTA];

  let bestScore = workingSetAccuracyBefore;
  let bestWeights: FusionWeights = { ...currentWeights };

  // 5-dimensional grid search exploration with normalization
  for (const s1 of STEPS) {
    for (const s2 of STEPS) {
      for (const s3 of STEPS) {
        for (const s4 of STEPS) {
          for (const s5 of STEPS) {
            const rawW: FusionWeights = {
              faceForgery: Math.max(0.05, currentWeights.faceForgery + s1),
              frequencyGan: Math.max(0.05, currentWeights.frequencyGan + s2),
              audioSpoof: Math.max(0.05, currentWeights.audioSpoof + s3),
              exifEla: Math.max(0.05, currentWeights.exifEla + s4),
              c2pa: Math.max(0.05, currentWeights.c2pa + s5),
            };

            // Normalize weights to sum to 1.0
            const sum = rawW.faceForgery + rawW.frequencyGan + rawW.audioSpoof + rawW.exifEla + rawW.c2pa;
            const normalizedW: FusionWeights = {
              faceForgery: Number((rawW.faceForgery / sum).toFixed(4)),
              frequencyGan: Number((rawW.frequencyGan / sum).toFixed(4)),
              audioSpoof: Number((rawW.audioSpoof / sum).toFixed(4)),
              exifEla: Number((rawW.exifEla / sum).toFixed(4)),
              c2pa: Number((rawW.c2pa / sum).toFixed(4)),
            };

            const candScore = evaluateWeightCandidate(normalizedW, workingJudged, samplesById);

            if (candScore > bestScore) {
              bestScore = candScore;
              bestWeights = normalizedW;
            }
          }
        }
      }
    }
  }

  const workingSetAccuracyAfter = evaluateWeightCandidate(bestWeights, workingJudged, samplesById);
  const heldOutAccuracyAfter = evaluateWeightCandidate(bestWeights, heldOutJudged, samplesById);

  return {
    id: `TUNE-${Date.now()}`,
    timestamp: new Date().toISOString(),
    beforeWeights: { ...currentWeights },
    afterWeights: bestWeights,
    workingSetAccuracyBefore,
    workingSetAccuracyAfter,
    heldOutAccuracyBefore,
    heldOutAccuracyAfter,
    applied: false,
  };
}
