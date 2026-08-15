// ──────────────────────────────────────────────────────────────
// app/api/eval/judge/route.ts — POST human judgments for eval samples
// ──────────────────────────────────────────────────────────────

export const dynamic = 'force-dynamic';

import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import type { JudgedResult } from '@/lib/eval/types';
import defaultJudged from '@/data/judged-results.json';

const DATA_FILE = 'data/judged-results.json';

// In-memory fallback for serverless runtime (Vercel)
let inMemoryResults: JudgedResult[] = (defaultJudged as JudgedResult[]) || [];

function readJudgedResults(): JudgedResult[] {
  try {
    const filePath = path.join(process.cwd(), DATA_FILE);
    if (fs.existsSync(filePath)) {
      const content = fs.readFileSync(filePath, 'utf8');
      const parsed = JSON.parse(content) as JudgedResult[];
      if (Array.isArray(parsed) && parsed.length > 0) {
        inMemoryResults = parsed;
      }
    }
  } catch {
    // Ignore fs errors on serverless
  }
  return inMemoryResults;
}

function writeJudgedResults(results: JudgedResult[]) {
  inMemoryResults = results;
  try {
    const filePath = path.join(process.cwd(), DATA_FILE);
    const dirPath = path.dirname(filePath);
    if (!fs.existsSync(dirPath)) {
      fs.mkdirSync(dirPath, { recursive: true });
    }
    fs.writeFileSync(filePath, JSON.stringify(results, null, 2), 'utf8');
  } catch {
    // Safe fallback to in-memory on Vercel read-only filesystem
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { sampleId, predictedScore, predictedVerdict, moduleScores, humanJudgment, reviewerNote } = body;

    if (!sampleId || !humanJudgment || !['correct', 'incorrect', 'partially_correct'].includes(humanJudgment)) {
      return NextResponse.json(
        { error: 'Invalid payload. sampleId and valid humanJudgment ("correct" | "incorrect" | "partially_correct") are required.' },
        { status: 400 }
      );
    }

    const currentResults = readJudgedResults();

    const judgmentEntry: JudgedResult = {
      sampleId,
      predictedScore: predictedScore ?? 50,
      predictedVerdict: predictedVerdict ?? 'suspicious',
      moduleScores: moduleScores ?? { faceForgery: 50, frequencyGan: 50, audioSpoof: 50, exifEla: 50, c2pa: 50 },
      humanJudgment,
      reviewerNote: reviewerNote || undefined,
      timestamp: new Date().toISOString(),
    };

    const index = currentResults.findIndex((r) => r.sampleId === sampleId);
    if (index >= 0) {
      currentResults[index] = judgmentEntry;
    } else {
      currentResults.push(judgmentEntry);
    }

    writeJudgedResults(currentResults);

    return NextResponse.json({
      success: true,
      judgedCount: currentResults.length,
      judgment: judgmentEntry,
    });
  } catch (err: any) {
    return NextResponse.json(
      { error: err?.message || 'Failed to persist judgment' },
      { status: 500 }
    );
  }
}

export async function GET() {
  const results = readJudgedResults();
  return NextResponse.json({
    totalJudged: results.length,
    results,
  });
}
