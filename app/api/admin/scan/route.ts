export const dynamic = 'force-dynamic';

import { NextRequest, NextResponse } from 'next/server';
import { validateFileContent, isFileTooLarge } from '@/lib/admin/file-validation';
import { logError, logAlert, generateRequestId, safeErrorResponse } from '@/lib/admin/logger';
import { VERDICT_CONFIG, VerdictCategory } from '@/lib/verdict';
import { callModelService, readMetadataSignal } from '@/lib/models/model_service';

function isForbiddenUrl(urlString: string): boolean {
  try {
    const parsed = new URL(urlString);
    const hostname = parsed.hostname.toLowerCase();

    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      return true;
    }

    const forbiddenPatterns = [
      'localhost',
      '127.0.0.1',
      '0.0.0.0',
      '::1',
      '169.254.169.254',
      'metadata.google.internal',
      '.internal',
      '.local',
    ];

    if (forbiddenPatterns.some((pattern) => hostname.includes(pattern))) {
      return true;
    }

    if (
      hostname.startsWith('10.') ||
      hostname.startsWith('192.168.') ||
      (hostname.startsWith('172.') && parseInt(hostname.split('.')[1], 10) >= 16 && parseInt(hostname.split('.')[1], 10) <= 31)
    ) {
      return true;
    }

    return false;
  } catch {
    return true;
  }
}

export async function POST(req: NextRequest) {
  const requestId = generateRequestId();
  try {
    let filename = 'admin_scan.jpg';
    let fileSize = '2.4 MB';
    let fileType: 'image' | 'video' | 'audio' | 'url' = 'image';
    let fileBlob: Blob | null = null;
    let fileBuffer: ArrayBuffer | null = null;

    const contentType = req.headers.get('content-type') || '';

    if (contentType.includes('multipart/form-data')) {
      const formData = await req.formData();
      const file = formData.get('file') as File | null;
      if (!file) {
        return NextResponse.json({ error: 'No file provided in form data.' }, { status: 400 });
      }

      // Server-side 50MB enforcement
      if (isFileTooLarge(file.size)) {
        return NextResponse.json(
          { error: 'File exceeds the 50MB size limit.' },
          { status: 413 }
        );
      }

      filename = file.name.replace(/[^a-zA-Z0-9_.-]/g, '_');
      const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
      fileSize = `${sizeMb} MB`;
      if (file.type.startsWith('video/')) fileType = 'video';
      else if (file.type.startsWith('audio/')) fileType = 'audio';
      else fileType = 'image';

      // Magic-byte content validation
      fileBuffer = await file.arrayBuffer();
      const validation = validateFileContent(fileBuffer, fileType);
      if (!validation.valid) {
        return NextResponse.json(
          {
            error: 'File content validation failed',
            detail: validation.reason,
          },
          { status: 400 }
        );
      }

      fileBlob = file;
    } else if (contentType.includes('application/json')) {
      const json = await req.json();
      if (!json.url) {
        return NextResponse.json({ error: 'No URL provided in request payload.' }, { status: 400 });
      }

      if (isForbiddenUrl(json.url)) {
        return NextResponse.json(
          { error: 'Security Blocked: URL points to forbidden internal or invalid network resource.' },
          { status: 400 }
        );
      }

      filename = json.url.replace(/^https?:\/\//, '').split('/')[0] || 'remote_media';
      fileType = 'url';
      fileSize = 'Remote Stream';
    } else {
      return NextResponse.json({ error: 'Unsupported Content-Type header' }, { status: 400 });
    }

    if (!fileBlob) {
      return NextResponse.json(
        { error: 'Direct real-model analysis currently requires file uploads to forward to inference engine.' },
        { status: 400 }
      );
    }

    // Same client the public route uses. No heuristics, no second code path to drift.
    const service = await callModelService(fileBlob, filename);
    if (!service.ok) {
      return NextResponse.json(
        {
          error: 'Model service unavailable',
          detail: service.detail,
          note: 'Admin scan reports the model verbatim and never falls back to a guess. Check MODEL_SERVICE_URL and that the service is running.',
        },
        { status: 503 }
      );
    }

    const model = service.result;
    const category: VerdictCategory =
      model.verdict === 'real' ? 'genuine' : model.verdict === 'fake' ? 'manipulated' : 'uncertain';
    const modelScore = model.signals.modelScore;
    const base = VERDICT_CONFIG[category];

    const laymanSummary =
      modelScore === null
        ? `No verdict — ${model.notes[0] ?? 'the model did not apply to this file'}.`
        : `${model.modelSource}: P(fake) = ${modelScore}%, confidence ${model.confidence}%.`;

    const responseData = {
      id: `ADM-${Math.floor(100000 + Math.random() * 900000)}`,
      filename,
      fileType,
      fileSize,
      score: modelScore === null ? null : 100 - modelScore,
      confidence: model.confidence,
      verdict: { ...base, laymanSummary },
      reasons: [
        `Model: ${model.modelSource}.`,
        modelScore === null
          ? 'No model score — the model did not apply to this file.'
          : `P(AI-generated or manipulated) = ${modelScore}%.`,
        `Face detected: ${model.faceDetected === null ? 'detector unavailable' : model.faceDetected ? 'yes' : 'no'}.`,
        `High-frequency energy share: ${model.signals.frequencyScore ?? 'n/a'}%.`,
        ...model.notes,
      ],
      signals: { ...model.signals, faceDetected: model.faceDetected },
      metadata: fileBuffer
        ? readMetadataSignal(fileBuffer)
        : { c2paManifestPresent: false, exifPresent: false },
      modelSource: model.modelSource,
      heatmap: model.heatmap ?? null,
      notes: model.notes,
      timestamp: new Date().toISOString(),
      rawModelResponse: model,
    };

    return NextResponse.json(responseData, { status: 200 });
  } catch (err) {
    logError('admin-scan', err, requestId);
    return NextResponse.json(safeErrorResponse(requestId), { status: 500 });
  }
}
