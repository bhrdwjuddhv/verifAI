// ──────────────────────────────────────────────────────────────
// app/api/eval/tune/route.ts — POST run tuning pass, PUT apply weights
// ──────────────────────────────────────────────────────────────

export const dynamic = 'force-dynamic';

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { runFusionWeightTuner } from '@/lib/eval/tuner';
import { generateEvalSamples } from '@/lib/eval/sampleGenerator';
import type { FusionWeights, JudgedResult, TuningPassResult } from '@/lib/eval/types';
import { DEFAULT_FUSION_WEIGHTS } from '@/lib/eval/types';

import defaultWeights from '@/data/fusion-weights.json';
import defaultJudged from '@/data/judged-results.json';
import defaultHistory from '@/data/weight-history.json';

const WEIGHTS_FILE = 'data/fusion-weights.json';
const HISTORY_FILE = 'data/weight-history.json';
const JUDGED_FILE = 'data/judged-results.json';

let inMemoryWeights: FusionWeights = (defaultWeights as FusionWeights) || DEFAULT_FUSION_WEIGHTS;
let inMemoryHistory: TuningPassResult[] = (defaultHistory as TuningPassResult[]) || [];

function readJsonFile<T>(relativePath: string, defaultValue: T): T {
  try {
    const filePath = path.join(process.cwd(), relativePath);
    if (fs.existsSync(filePath)) {
      const content = fs.readFileSync(filePath, 'utf8');
      return JSON.parse(content) as T;
    }
  } catch {
    // Ignore fs errors on serverless
  }
  return defaultValue;
}

function writeJsonFile<T>(relativePath: string, data: T) {
  if (relativePath.includes('weights')) inMemoryWeights = data as any;
  if (relativePath.includes('history')) inMemoryHistory = data as any;

  try {
    const filePath = path.join(process.cwd(), relativePath);
    const dirPath = path.dirname(filePath);
    if (!fs.existsSync(dirPath)) {
      fs.mkdirSync(dirPath, { recursive: true });
    }
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf8');
  } catch {
    // Fallback to memory on read-only serverless filesystem
  }
}

export async function POST() {
  try {
    const currentWeights = readJsonFile<FusionWeights>(WEIGHTS_FILE, inMemoryWeights);
    const judgedResults = readJsonFile<JudgedResult[]>(JUDGED_FILE, defaultJudged as JudgedResult[]);

    const samples = generateEvalSamples(currentWeights);
    const samplesById = samples.reduce<Record<string, typeof samples[0]>>((acc, s) => {
      acc[s.id] = s;
      return acc;
    }, {});

    const passResult = runFusionWeightTuner(currentWeights, judgedResults, samplesById);

    return NextResponse.json({
      success: true,
      result: passResult,
    });
  } catch (err: any) {
    return NextResponse.json(
      { error: err?.message || 'Tuning optimization pass failed' },
      { status: 500 }
    );
  }
}

export async function PUT(req: NextRequest) {
  try {
    const body = await req.json();
    const { passResult, weights } = body;

    const newWeights: FusionWeights = weights || passResult?.afterWeights;
    if (!newWeights) {
      return NextResponse.json(
        { error: 'Valid weights payload is required to apply weights.' },
        { status: 400 }
      );
    }

    writeJsonFile(WEIGHTS_FILE, newWeights);

    const history = readJsonFile<TuningPassResult[]>(HISTORY_FILE, inMemoryHistory);
    const passToRecord: TuningPassResult = passResult || {
      id: `TUNE-${Date.now()}`,
      timestamp: new Date().toISOString(),
      beforeWeights: inMemoryWeights,
      afterWeights: newWeights,
      workingSetAccuracyBefore: 0,
      workingSetAccuracyAfter: 0,
      heldOutAccuracyBefore: 0,
      heldOutAccuracyAfter: 0,
      applied: true,
    };
    passToRecord.applied = true;

    history.unshift(passToRecord);
    writeJsonFile(HISTORY_FILE, history);

    return NextResponse.json({
      success: true,
      appliedWeights: newWeights,
      history,
    });
  } catch (err: any) {
    return NextResponse.json(
      { error: err?.message || 'Failed to apply new fusion weights' },
      { status: 500 }
    );
  }
}
