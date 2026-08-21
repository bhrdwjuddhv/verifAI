import { MetadataSignal } from '@/lib/store';

/** Exactly what scripts/inference_server.py returns. Nothing is filled in on this side. */
export interface ModelServiceResult {
  verdict: 'real' | 'fake' | 'uncertain';
  confidence: number;
  /** Fused P(fake) the verdict is based on. Null when no detector applied. */
  fakeProbability?: number | null;
  modelSource: string;
  faceDetected: boolean | null;
  signals: {
    modelScore: number | null;
    /** NPR whole-image AI-generation detector, P(fake) percent. */
    nprScore?: number | null;
    /** Voice-clone detector, P(fake) percent. Audio uploads only. */
    audioScore?: number | null;
    frequencyScore: number | null;
  };
  /** Which detectors were loaded, and the weights that combined them. */
  detectors?: { face: string | null; npr: string | null };
  fusion?: { weights: Record<string, number>; used: Record<string, number> };
  /** Present only for video: the per-frame breakdown behind the aggregate. */
  video?: VideoSummary;
  notes: string[];
  heatmap?: string | null;
}

export interface VideoFrameScore {
  t: number;
  face: number | null;
  npr: number | null;
  frequency: number | null;
  faceDetected: boolean | null;
  fused: number | null;
}

export interface VideoSummary {
  frames: number;
  durationSeconds: number;
  meanFakeProbability?: number | null;
  maxFakeProbability?: number | null;
  /** Spread of the per-frame scores. Reported as a flicker hint, not fused into the verdict. */
  temporalVariance?: number | null;
  peakFrameSeconds?: number | null;
  perFrame: VideoFrameScore[];
}

export type ModelServiceResponse =
  | { ok: true; result: ModelServiceResult }
  /** `status` is the service's own HTTP status when it answered, absent when unreachable. */
  | { ok: false; detail: string; status?: number };

/**
 * Single door to the model service, for the public and admin routes alike.
 *
 * There is no fallback path on purpose. When this fails the caller reports that analysis
 * is unavailable — a guessed verdict is worse than no verdict.
 */
/** One short live-call window. No verdict band: that is Phase 4's job, over many windows. */
export interface AudioWindowResult {
  speechDetected: boolean;
  fakeProbability: number | null;
  windowSeconds: number;
  vad: { rmsDbfs: number | null; spectralFlatness: number | null; reason: string | null };
  modelSource: string | null;
  notes: string[];
}

export type AudioWindowResponse =
  | { ok: true; result: AudioWindowResult }
  | { ok: false; detail: string; status?: number };

/** Transport only — POST a file to the service and hand back parsed JSON or a reason. */
async function postToService(
  path: string,
  file: Blob,
  filename: string,
  query?: Record<string, string>
): Promise<{ ok: true; json: any } | { ok: false; detail: string; status?: number }> {
  const base = process.env.MODEL_SERVICE_URL?.replace(/\/$/, '');
  if (!base) {
    return { ok: false, detail: 'MODEL_SERVICE_URL is not set — no model service is configured.' };
  }

  const search = query && Object.keys(query).length ? `?${new URLSearchParams(query)}` : '';
  try {
    const form = new FormData();
    form.append('file', file, filename);

    const res = await fetch(`${base}${path}${search}`, {
      method: 'POST',
      headers: {
        // Tunnels (ngrok / localtunnel) serve an interstitial HTML page without these.
        'ngrok-skip-browser-warning': 'true',
        'bypass-tunnel-reminder': 'true',
      },
      body: form,
      // A sleeping free-tier instance takes ~60s to wake, and the lean build needs ~6s after
      // that. Must stay under the route's maxDuration of 60s, or Vercel kills the function
      // first and the user gets a generic error instead of our "unavailable" message.
      signal: AbortSignal.timeout(55000),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      return {
        ok: false,
        detail: body.detail || `Model service returned HTTP ${res.status}.`,
        status: res.status,
      };
    }
    return { ok: true, json: await res.json() };
  } catch (err: any) {
    return { ok: false, detail: `Could not reach the model service: ${err?.message || 'network error'}.` };
  }
}

/**
 * Single door to the model service, for the public and admin routes alike.
 *
 * There is no fallback path on purpose. When this fails the caller reports that analysis
 * is unavailable — a guessed verdict is worse than no verdict.
 */
export async function callModelService(
  file: Blob,
  filename: string,
  path: '/predict' | '/predict-video' | '/predict-audio' = '/predict',
  /**
   * Extra query parameters for the service. Only `explain` is used today: a caller that does
   * not need the heatmap can skip it and save the service 26 forward passes.
   */
  query?: Record<string, string>
): Promise<ModelServiceResponse> {
  const res = await postToService(path, file, filename, query);
  if (!res.ok) return res;
  const result = res.json as ModelServiceResult;
  if (!result || typeof result.verdict !== 'string') {
    return { ok: false, detail: 'Model service returned an unrecognized response.' };
  }
  return { ok: true, result };
}

/** Live-call windows. Validated on `speechDetected`, since a window carries no verdict. */
export async function callAudioWindow(file: Blob, filename = 'window.wav'): Promise<AudioWindowResponse> {
  const res = await postToService('/predict-audio-window', file, filename);
  if (!res.ok) return res;
  const result = res.json as AudioWindowResult;
  if (!result || typeof result.speechDetected !== 'boolean') {
    return { ok: false, detail: 'Model service returned an unrecognized window response.' };
  }
  return { ok: true, result };
}

/**
 * Container metadata only: is a C2PA manifest present, is there an EXIF block. Both are
 * structural facts about the file, not evidence about the pixels — a real photo posted to
 * any social network arrives with both stripped.
 */
export function readMetadataSignal(buffer: ArrayBuffer): MetadataSignal {
  const bytes = new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 256 * 1024));
  const text = new TextDecoder('latin1').decode(bytes);
  return {
    // JUMBF box carrying a C2PA manifest (JPEG APP11 / PNG caBX / ISOBMFF).
    c2paManifestPresent: text.includes('c2pa') || text.includes('jumb'),
    // JPEG APP1 "Exif\0\0" header, or the PNG eXIf chunk.
    exifPresent: text.includes('Exif\u0000\u0000') || text.includes('eXIf'),
  };
}

export function describeMetadata(m: MetadataSignal): string {
  const parts = [
    m.c2paManifestPresent ? 'a C2PA manifest is present' : 'no C2PA manifest',
    m.exifPresent ? 'EXIF metadata is present' : 'no EXIF metadata',
  ];
  return `Metadata signal (supporting only, not part of the verdict): ${parts.join(', ')}. Metadata is stripped by most platforms and can be forged, so neither presence nor absence proves anything.`;
}
