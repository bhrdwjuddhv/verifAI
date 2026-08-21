export const dynamic = 'force-dynamic';

import { NextResponse } from 'next/server';

/**
 * What the browser extension needs to know before it trusts its own numbers.
 *
 * The extension bundles its own copy of the detectors and scores images locally. That only
 * stays honest while its thresholds and fusion weights match the ones the service actually
 * uses — retune FUSION_WEIGHTS on the model service and every installed extension would keep
 * reporting the old blend, confidently and wrongly. So the extension reads this daily and
 * refuses to present an on-device verdict whose settings have drifted.
 *
 * Read-only, no file upload, and it never invents a value: when the model service cannot be
 * reached the fields it owns are omitted rather than guessed.
 */

const API_VERSION = 1;

/**
 * Extensions older than this are told to update. Bump it only when a change makes an older
 * client actively wrong — not for every release.
 */
const MIN_EXTENSION_VERSION = '0.1.0';

/** The model service is the source of truth here; this only stops us hammering it. */
const CACHE_MS = 5 * 60 * 1000;

interface ServiceHealth {
  status?: string;
  fusionWeights?: Record<string, number>;
  thresholds?: { fakeAbove: number; realBelow: number };
  saliencyGrid?: number;
  modelSource?: string;
  faceDetector?: boolean;
  npr?: { available?: boolean; modelSource?: string; reason?: string };
  audio?: { available?: boolean; modelSource?: string; reason?: string };
}

let cached: { at: number; health: ServiceHealth | null } | null = null;

async function readHealth(): Promise<ServiceHealth | null> {
  if (cached && Date.now() - cached.at < CACHE_MS) return cached.health;

  const base = process.env.MODEL_SERVICE_URL?.replace(/\/$/, '');
  let health: ServiceHealth | null = null;

  if (base) {
    try {
      const res = await fetch(`${base}/health`, {
        headers: { 'ngrok-skip-browser-warning': 'true', 'bypass-tunnel-reminder': 'true' },
        signal: AbortSignal.timeout(8000),
        cache: 'no-store',
      });
      if (res.ok) health = (await res.json()) as ServiceHealth;
    } catch {
      health = null; // a cold or absent service is reported as such, not as a failure here
    }
  }

  cached = { at: Date.now(), health };
  return health;
}

export async function GET() {
  const health = await readHealth();

  return NextResponse.json(
    {
      apiVersion: API_VERSION,
      minExtensionVersion: MIN_EXTENSION_VERSION,
      modelService: health ? 'ok' : 'unavailable',
      checkedAt: new Date().toISOString(),
      // Present only when the service answered. An extension comparing against absent values
      // must skip the check rather than assume a default that may not be in force.
      thresholds: health?.thresholds,
      fusionWeights: health?.fusionWeights,
      saliencyGrid: health?.saliencyGrid,
      detectors: health
        ? {
            face: health.modelSource ?? null,
            faceDetector: health.faceDetector ?? null,
            npr: health.npr?.available ? (health.npr.modelSource ?? true) : null,
            audio: health.audio?.available ? (health.audio.modelSource ?? true) : null,
          }
        : undefined,
    },
    {
      status: 200,
      // Cheap for a client to poll daily, and safe to sit in a CDN for a few minutes.
      headers: { 'Cache-Control': 'public, max-age=60, s-maxage=300' },
    }
  );
}
