/**
 * Presentation helpers shared by the popup and the in-page badge.
 *
 * No colour table lives here on purpose: `/api/scan` already sends `verdict.ringColor` and
 * `verdict.label` straight out of lib/verdict.ts, so the extension shows the same words and
 * the same colours as the website without a second copy to keep in sync.
 */

import type { ScanResult } from './scan-types';

/** Neutral grey for states that have no verdict yet. Matches the app's 'uncertain' ring. */
export const NEUTRAL = '#94A3B8';

/** P(AI-generated or manipulated), 0-100. The API sends the inverse, as "how real". */
export function fakePercent(result: ScanResult): number | null {
  return result.score === null ? null : 100 - result.score;
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/** Enough of a URL to recognise the file, without a 400-character CDN token in the popup. */
export function shortUrl(url: string, max = 46): string {
  if (url.startsWith('data:')) return `data: (${url.length} chars)`;
  if (url.startsWith('blob:')) {
    const host = url.slice(5).replace(/^https?:\/\//, '').split('/')[0];
    return `blob: from ${host}`;
  }
  try {
    const u = new URL(url);
    const name = u.pathname.split('/').filter(Boolean).pop() || u.hostname;
    const label = `${u.hostname}/…/${name}`;
    return label.length > max ? `${label.slice(0, max - 1)}…` : label;
  } catch {
    return url.length > max ? `${url.slice(0, max - 1)}…` : url;
  }
}

export interface DetectorRow {
  key: string;
  label: string;
  /** P(fake) in percent, or null when this detector did not apply. */
  value: number | null;
  /** Fusion weight actually used. 0 means "measured and reported, but it did not vote". */
  weight: number;
  hint: string;
}

/**
 * The per-detector breakdown, in the order the app explains them.
 *
 * A null value is displayed as "did not apply" rather than hidden: which detectors abstained
 * is part of reading the verdict, not noise to tidy away.
 */
export function detectorRows(result: ScanResult): DetectorRow[] {
  const used = result.fusion?.used ?? {};
  const rows: DetectorRow[] = [
    {
      key: 'face',
      label: 'Face classifier',
      value: result.signals.modelScore,
      weight: used.face ?? 0,
      hint: 'Was this face swapped or manipulated? Runs on the cropped face only.',
    },
    {
      key: 'npr',
      label: 'NPR whole-image',
      value: result.signals.nprScore ?? null,
      weight: used.npr ?? 0,
      hint: 'Did a generator make these pixels? Catches fully synthetic images.',
    },
  ];

  if (result.signals.audioScore !== null && result.signals.audioScore !== undefined) {
    rows.push({
      key: 'audio',
      label: 'Voice model',
      value: result.signals.audioScore,
      weight: used.audio ?? 0,
      hint: 'Synthetic or cloned voice, from the clip mel-spectrogram.',
    });
  }

  rows.push({
    key: 'frequency',
    label: 'High-frequency energy',
    value: result.signals.frequencyScore,
    weight: used.frequency ?? 0,
    hint: 'Descriptive only — sharp real photos score high too. Carries no weight.',
  });

  return rows;
}
