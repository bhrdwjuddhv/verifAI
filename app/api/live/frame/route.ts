export const dynamic = 'force-dynamic';
export const maxDuration = 60;

import { NextRequest, NextResponse } from 'next/server';
import { callModelService } from '@/lib/models/model_service';

/**
 * One sampled video frame from a live call -> the model service's /predict (face + NPR).
 *
 * `explain=0`: the heatmap costs ~26 extra forward passes and nobody reads a heatmap that is
 * replaced every few seconds. The file-scan path still asks for it.
 */
export async function POST(req: NextRequest) {
  const form = await req.formData().catch(() => null);
  const file = form?.get('file') as File | null;
  if (!file) {
    return NextResponse.json({ error: 'No frame provided.' }, { status: 400 });
  }
  if (file.size > 4 * 1024 * 1024) {
    return NextResponse.json({ error: 'Frame too large.' }, { status: 413 });
  }

  const service = await callModelService(file, 'frame.jpg', '/predict', { explain: '0' });
  if (!service.ok) {
    const status = [501, 415, 413].includes(service.status ?? 0) ? service.status! : 503;
    return NextResponse.json({ error: 'Analysis unavailable', detail: service.detail }, { status });
  }
  return NextResponse.json(service.result);
}
