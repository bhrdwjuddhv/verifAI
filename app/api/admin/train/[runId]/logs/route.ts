import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET(
  req: NextRequest,
  { params }: { params: { runId: string } }
) {
  try {
    const { runId } = params;
    const runDir = path.resolve('runs', runId);

    if (!fs.existsSync(runDir)) {
      return NextResponse.json({ error: `Run directory '${runId}' not found.` }, { status: 404 });
    }

    const logPath = path.join(runDir, 'log.txt');
    const jsonPath = path.join(runDir, 'run.json');

    let logs = '';
    if (fs.existsSync(logPath)) {
      logs = fs.readFileSync(logPath, 'utf-8');
    }

    let meta = { id: runId, status: 'unknown' };
    if (fs.existsSync(jsonPath)) {
      try {
        meta = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
      } catch {
        // use default meta
      }
    }

    return NextResponse.json({ runId, meta, logs }, { status: 200 });
  } catch (err: any) {
    return NextResponse.json({ error: err?.message || 'Failed to fetch run logs' }, { status: 500 });
  }
}
