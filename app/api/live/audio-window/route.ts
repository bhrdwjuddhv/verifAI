export const dynamic = 'force-dynamic';
// Live windows must be quick; a 60s ceiling exists only so a cold service still answers once.
export const maxDuration = 60;

import { NextRequest, NextResponse } from 'next/server';
import { callAudioWindow } from '@/lib/models/model_service';

/**
 * One live-call audio window -> the model service's /predict-audio-window.
 *
 * A thin proxy on purpose: inference belongs to FastAPI/ONNX, and the browser must not learn
 * MODEL_SERVICE_URL. This route adds no scoring of its own.
 */
export async function POST(req: NextRequest) {
  const form = await req.formData().catch(() => null);
  const file = form?.get('file') as File | null;
  if (!file) {
    return NextResponse.json({ error: 'No audio window provided.' }, { status: 400 });
  }
  // 4 seconds of 16kHz mono PCM is ~128KB. Anything much larger is not a live window.
  if (file.size > 4 * 1024 * 1024) {
    return NextResponse.json({ error: 'Window too large — send 2-4 seconds.' }, { status: 413 });
  }

  const service = await callAudioWindow(file);
  if (!service.ok) {
    const status = [501, 415, 413].includes(service.status ?? 0) ? service.status! : 503;
    return NextResponse.json(
      {
        error: status === 501 ? 'Voice detection is not available' : 'Analysis unavailable',
        detail: service.detail,
      },
      { status }
    );
  }
  return NextResponse.json(service.result);
}
