/**
 * The one door to `/api/scan`.
 *
 * CORS note: this runs in the service worker, which is exempt from CORS for origins the
 * extension holds a host permission for — custom headers included, with no preflight. The
 * same fetch from a content script would be subject to the page's CORS and fail, which is
 * why the POST lives here and nowhere else.
 */

import type { ApiOutcome, ScanError, ScanResult } from '../shared/scan-types';

/** The route allows 60s; its own model client gives up at 30s and answers 503. */
const TIMEOUT_MS = 60_000;

/** middleware.ts rejects anything larger before the route sees it. */
export const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

export async function postScan(
  serverUrl: string,
  bytes: Uint8Array,
  filename: string,
  mime: string,
  version: string
): Promise<ApiOutcome> {
  if (bytes.byteLength > MAX_UPLOAD_BYTES) {
    return { ok: false, kind: 'rejected', message: 'That file is over the 50MB scan limit.' };
  }

  const form = new FormData();
  form.append('file', new Blob([bytes as BlobPart], { type: mime }), filename);

  let res: Response;
  try {
    res = await fetch(`${serverUrl}/api/scan`, {
      method: 'POST',
      body: form,
      // Distinguishes extension traffic from the website's own in the server logs.
      headers: { 'X-VerifAI-Client': `extension/${version}` },
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
  } catch (err) {
    const name = (err as Error)?.name;
    if (name === 'TimeoutError') {
      return {
        ok: false,
        kind: 'network',
        message: 'The scan took longer than a minute and was given up on. The model service may be cold-starting — try again.',
      };
    }
    return {
      ok: false,
      kind: 'network',
      message: `Could not reach ${serverUrl}. Check the server URL in options, and that the app is running.`,
    };
  }

  if (res.ok) {
    try {
      return { ok: true, result: (await res.json()) as ScanResult };
    } catch {
      return {
        ok: false,
        kind: 'internal',
        message: 'The server answered with something that was not a scan result.',
      };
    }
  }

  const body = await readError(res);
  const detail = [body.error, body.detail].filter(Boolean).join(' — ');

  switch (res.status) {
    case 429: {
      const retryAfter = Number(res.headers.get('Retry-After')) || 60;
      return {
        ok: false,
        kind: 'rate-limited',
        message: `The server is rate-limiting scans. Waiting ${retryAfter}s.`,
        retryAfterSeconds: retryAfter,
      };
    }
    case 501:
      return {
        ok: false,
        kind: 'not-implemented',
        message: detail || 'There is no model for this kind of file yet.',
      };
    case 503:
      return {
        ok: false,
        kind: 'unavailable',
        message: detail || 'Analysis is unavailable: the model service did not answer.',
      };
    case 400:
    case 413:
    case 415:
      return { ok: false, kind: 'rejected', message: detail || 'The server rejected this file.' };
    default:
      return {
        ok: false,
        kind: 'internal',
        message: detail || `The server answered ${res.status}.`,
      };
  }
}

async function readError(res: Response): Promise<ScanError> {
  try {
    return (await res.json()) as ScanError;
  } catch {
    return { error: `HTTP ${res.status}` };
  }
}
