export const dynamic = 'force-dynamic';
// CPU inference plus a cold start on the model service can exceed the 10s Vercel default.
// The client below gives up at 30s, so this must stay above that.
export const maxDuration = 60;

import { NextRequest, NextResponse } from 'next/server';
import { validateFileContent, isFileTooLarge } from '@/lib/admin/file-validation';
import { logError, generateRequestId, safeErrorResponse } from '@/lib/admin/logger';
import { VERDICT_CONFIG, VerdictCategory } from '@/lib/verdict';
import { callModelService, describeMetadata, readMetadataSignal } from '@/lib/models/model_service';

// SSRF Firewall: Block requests to internal/private IP ranges
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

const CATEGORY: Record<string, VerdictCategory> = {
  real: 'genuine',
  fake: 'manipulated',
  uncertain: 'uncertain',
};

export async function POST(req: NextRequest) {
  const requestId = generateRequestId();
  try {
    let filename = 'upload';
    let fileSize = '';
    let fileType: 'image' | 'video' | 'audio' | 'url' = 'image';
    let fileBuffer: ArrayBuffer | null = null;
    let fileBlob: Blob | null = null;

    const contentType = req.headers.get('content-type') || '';

    if (contentType.includes('multipart/form-data')) {
      const formData = await req.formData();
      const file = formData.get('file') as File | null;
      if (file) {
        if (isFileTooLarge(file.size)) {
          return NextResponse.json({ error: 'File exceeds the 50MB size limit.' }, { status: 413 });
        }

        filename = file.name.replace(/[^a-zA-Z0-9_.-]/g, '_');
        fileSize = `${(file.size / (1024 * 1024)).toFixed(1)} MB`;
        if (file.type.startsWith('video/')) fileType = 'video';
        else if (file.type.startsWith('audio/')) fileType = 'audio';
        else fileType = 'image';

        fileBlob = file;
        fileBuffer = await file.arrayBuffer();

        const validation = validateFileContent(fileBuffer, fileType);
        if (!validation.valid) {
          return NextResponse.json(
            { error: 'File content validation failed', detail: validation.reason },
            { status: 400 }
          );
        }
      }
    } else if (contentType.includes('application/json')) {
      const json = await req.json();
      if (json.url) {
        if (isForbiddenUrl(json.url)) {
          return NextResponse.json(
            { error: 'Security Blocked: URL points to forbidden internal or invalid network resource.' },
            { status: 400 }
          );
        }
        // TODO(Phase 4): fetch the remote file and analyze it. Until then, say so rather
        // than inventing a verdict from the URL string.
        return NextResponse.json(
          {
            error: 'URL scanning is not implemented yet.',
            detail: 'Download the file and upload it directly. Remote fetching is a later phase.',
          },
          { status: 501 }
        );
      }
    }

    if (!fileBlob || !fileBuffer) {
      return NextResponse.json({ error: 'No file provided.' }, { status: 400 });
    }

    if (fileType !== 'image') {
      // TODO(Phase 4): video = sample frames -> per-frame model -> aggregate;
      // audio = mel-spectrogram -> AASIST/RawNet2. Neither exists, so neither gets a verdict.
      return NextResponse.json(
        {
          error: `${fileType === 'video' ? 'Video' : 'Audio'} analysis is not available yet.`,
          detail: 'Only images are supported in this version. Nothing was analyzed.',
        },
        { status: 501 }
      );
    }

    const service = await callModelService(fileBlob, filename);
    if (!service.ok) {
      return NextResponse.json(
        {
          error: 'Analysis unavailable',
          detail: service.detail,
          unavailable: true,
        },
        { status: 503 }
      );
    }

    const model = service.result;
    const category = CATEGORY[model.verdict] ?? 'uncertain';
    const metadata = readMetadataSignal(fileBuffer);
    const modelScore = model.signals.modelScore;
    const score = modelScore === null ? null : 100 - modelScore;

    // Every reason below is a restatement of something that was actually measured.
    const reasons: string[] = [];
    if (model.faceDetected === true) {
      reasons.push('A face was detected; the model ran on the cropped face region.');
    } else if (model.faceDetected === false) {
      reasons.push('No face was detected in this image.');
    } else {
      reasons.push('Face detection was unavailable on the model service.');
    }
    if (modelScore !== null) {
      reasons.push(
        `The model scored this ${modelScore}% likely AI-generated or manipulated (thresholds: above 70 fake, below 30 real).`
      );
    } else {
      reasons.push('The model was not applied to this image, so there is no model score.');
    }
    if (model.signals.frequencyScore !== null) {
      reasons.push(
        `High-frequency energy share: ${model.signals.frequencyScore}%. A descriptive statistic only — sharp real photos score high too, and it does not affect the verdict.`
      );
    }
    reasons.push(describeMetadata(metadata));
    reasons.push(`Model used: ${model.modelSource}.`);
    reasons.push(...model.notes);

    const base = VERDICT_CONFIG[category];
    const laymanSummary =
      modelScore === null
        ? `No verdict: ${model.notes[0] ?? 'the model did not apply to this file'}.`
        : category === 'manipulated'
        ? `The model scored this ${modelScore}% likely AI-generated or manipulated.`
        : category === 'genuine'
        ? `The model scored this ${100 - modelScore}% likely real.`
        : `The model scored this ${modelScore}% likely AI-generated — inside the uncertain band (30–70%), so it is not calling it either way.`;

    const verdict = { ...base, description: base.description, laymanSummary };

    const responseData = {
      id: `VRF-${Math.floor(100000 + Math.random() * 900000)}`,
      filename,
      fileType,
      fileSize,
      score,
      confidence: model.confidence,
      verdict,
      reasons,
      signals: {
        modelScore,
        frequencyScore: model.signals.frequencyScore,
        faceDetected: model.faceDetected,
      },
      metadata,
      modelSource: model.modelSource,
      heatmap: model.heatmap ?? null,
      notes: model.notes,
      timestamp: new Date().toISOString(),
    };

    return NextResponse.json(responseData, { status: 200 });
  } catch (err) {
    logError('public-scan', err, requestId);
    return NextResponse.json(safeErrorResponse(requestId), { status: 500 });
  }
}
