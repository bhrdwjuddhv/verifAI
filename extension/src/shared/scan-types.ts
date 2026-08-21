/**
 * The scan contract, imported from the app rather than restated here.
 *
 * `/api/scan` assembles exactly `ScanResult`, so importing it means a change to the response
 * shape breaks `npm run typecheck` in this folder instead of silently mis-rendering a verdict
 * in the popup. Every import here is type-only, so nothing from the Next app — zustand
 * included — reaches the bundle.
 */

import type { ScanResult, ScanSignals, MetadataSignal } from '@app/lib/store';
import type { VerdictInfo, VerdictCategory } from '@app/lib/verdict';

export type { ScanResult, ScanSignals, MetadataSignal, VerdictInfo, VerdictCategory };

/** What `/api/scan` returns when it refuses. `unavailable` marks a down model service. */
export interface ScanError {
  error: string;
  detail?: string;
  unavailable?: boolean;
}

/**
 * Why a scan failed, kept separate from the message because the UI treats these differently.
 * "The service is down" and "this file is unusable" are not the same thing, and the API is
 * careful to distinguish them — so we are too.
 */
export type FailureKind =
  | 'unavailable' // 503: model service unreachable
  | 'rejected' // 400/413/415: the file itself
  | 'not-implemented' // 501: no model for this modality yet
  | 'rate-limited' // 429
  | 'network' // never reached the server
  | 'internal'; // 5xx that is not 503

export type ApiOutcome =
  | { ok: true; result: ScanResult }
  | { ok: false; kind: FailureKind; message: string; retryAfterSeconds?: number };
