// ──────────────────────────────────────────────────────────────
// lib/eval/store.ts — Zustand store for Evaluation Harness
// ──────────────────────────────────────────────────────────────

import { create } from 'zustand';
import type {
  EvalSample,
  JudgedResult,
  FusionWeights,
  TuningPassResult,
  EvalMetrics,
} from './types';
import { DEFAULT_FUSION_WEIGHTS } from './types';
import { generateEvalSamples } from './sampleGenerator';
import { computeEvalMetrics } from './metrics';

interface EvalState {
  // State
  samples: EvalSample[];
  samplesById: Record<string, EvalSample>;
  judgedResults: JudgedResult[];
  weights: FusionWeights;
  weightHistory: TuningPassResult[];
  activeSampleIndex: number;
  isLoading: boolean;
  error: string | null;

  // Actions
  setSamples: (samples: EvalSample[]) => void;
  setJudgedResults: (results: JudgedResult[]) => void;
  addJudgment: (judgment: JudgedResult) => void;
  setWeights: (weights: FusionWeights) => void;
  setWeightHistory: (history: TuningPassResult[]) => void;
  addTuningPass: (pass: TuningPassResult) => void;
  setActiveSampleIndex: (index: number) => void;
  setIsLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;

  // Derived Metrics
  getWorkingMetrics: () => EvalMetrics;
  getHeldOutMetrics: () => EvalMetrics;
  getOverallMetrics: () => EvalMetrics;
}

export const useEvalStore = create<EvalState>((set, get) => {
  const initialSamples = generateEvalSamples(DEFAULT_FUSION_WEIGHTS);
  const initialSamplesById = initialSamples.reduce<Record<string, EvalSample>>((acc, s) => {
    acc[s.id] = s;
    return acc;
  }, {});

  return {
    samples: initialSamples,
    samplesById: initialSamplesById,
    judgedResults: [],
    weights: DEFAULT_FUSION_WEIGHTS,
    weightHistory: [],
    activeSampleIndex: 0,
    isLoading: false,
    error: null,

    setSamples: (samples) => {
      const samplesById = samples.reduce<Record<string, EvalSample>>((acc, s) => {
        acc[s.id] = s;
        return acc;
      }, {});
      set({ samples, samplesById });
    },

    setJudgedResults: (judgedResults) => set({ judgedResults }),

    addJudgment: (judgment) =>
      set((state) => {
        const existingIndex = state.judgedResults.findIndex((j) => j.sampleId === judgment.sampleId);
        let updated: JudgedResult[];
        if (existingIndex >= 0) {
          updated = [...state.judgedResults];
          updated[existingIndex] = judgment;
        } else {
          updated = [...state.judgedResults, judgment];
        }
        return { judgedResults: updated };
      }),

    setWeights: (weights) => {
      // Re-generate sample scores with new weights
      const updatedSamples = generateEvalSamples(weights);
      const samplesById = updatedSamples.reduce<Record<string, EvalSample>>((acc, s) => {
        acc[s.id] = s;
        return acc;
      }, {});
      set({ weights, samples: updatedSamples, samplesById });
    },

    setWeightHistory: (weightHistory) => set({ weightHistory }),

    addTuningPass: (pass) =>
      set((state) => ({
        weightHistory: [pass, ...state.weightHistory],
      })),

    setActiveSampleIndex: (activeSampleIndex) => set({ activeSampleIndex }),
    setIsLoading: (isLoading) => set({ isLoading }),
    setError: (error) => set({ error }),

    getWorkingMetrics: () => {
      const { judgedResults, samplesById } = get();
      return computeEvalMetrics(judgedResults, samplesById, 'working');
    },

    getHeldOutMetrics: () => {
      const { judgedResults, samplesById } = get();
      return computeEvalMetrics(judgedResults, samplesById, 'heldout');
    },

    getOverallMetrics: () => {
      const { judgedResults, samplesById } = get();
      return computeEvalMetrics(judgedResults, samplesById);
    },
  };
});
