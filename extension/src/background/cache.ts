/**
 * Verdict cache, keyed by the SHA-256 of the file bytes.
 *
 * Hashing the bytes rather than the URL is what makes this useful: the same image served from
 * three CDN URLs, or re-uploaded by someone else, is one cache entry — and a URL that now
 * serves different bytes is correctly a miss.
 *
 * Heatmaps live in a separate store with their own budget. A 224px overlay PNG runs 30-80KB,
 * and storage.local is 10MB total, so keeping them next to the verdicts would evict ~150
 * scans' worth of results to hold ~40 pictures.
 */

import type { ScanResult } from '../shared/scan-types';

/** Bump to invalidate every entry — e.g. when the response shape changes. */
const SCHEMA = 1;

const RESULT_LIMIT = 400;
const HEATMAP_BUDGET = 3 * 1024 * 1024;

const rKey = (hash: string) => `r:${hash}`;
const hKey = (hash: string) => `h:${hash}`;
const IDX_RESULTS = 'idx:r';
const IDX_HEATMAPS = 'idx:h';

interface StoredResult {
  v: number;
  at: number;
  /** Stored with `heatmap: null`; the picture is re-attached from the heatmap store on read. */
  result: ScanResult;
}

interface HeatmapIndexEntry {
  hash: string;
  bytes: number;
}

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  // BufferSource wants a plain ArrayBuffer; a Uint8Array view may be a slice of a larger one.
  const buf = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
    ? bytes.buffer
    : bytes.slice().buffer;
  const digest = await crypto.subtle.digest('SHA-256', buf as ArrayBuffer);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

async function readIndex<T>(key: string): Promise<T[]> {
  const got = await chrome.storage.local.get(key);
  const value = got[key];
  return Array.isArray(value) ? (value as T[]) : [];
}

/** Most-recently-used first. */
function touch(index: string[], hash: string): string[] {
  return [hash, ...index.filter((h) => h !== hash)];
}

export async function getCached(hash: string): Promise<ScanResult | null> {
  const got = await chrome.storage.local.get([rKey(hash), hKey(hash)]);
  const entry = got[rKey(hash)] as StoredResult | undefined;
  if (!entry || entry.v !== SCHEMA) return null;

  const index = await readIndex<string>(IDX_RESULTS);
  await chrome.storage.local.set({ [IDX_RESULTS]: touch(index, hash) });

  const heatmap = got[hKey(hash)] as string | undefined;
  return { ...entry.result, heatmap: heatmap ?? null };
}

export async function putCached(hash: string, result: ScanResult): Promise<void> {
  const { heatmap, ...rest } = result;
  const entry: StoredResult = { v: SCHEMA, at: Date.now(), result: { ...rest, heatmap: null } };

  const index = touch(await readIndex<string>(IDX_RESULTS), hash);
  const keep = index.slice(0, RESULT_LIMIT);
  const drop = index.slice(RESULT_LIMIT);

  await chrome.storage.local.set({ [rKey(hash)]: entry, [IDX_RESULTS]: keep });
  if (drop.length) await chrome.storage.local.remove(drop.map(rKey));

  if (heatmap) await putHeatmap(hash, heatmap);
}

async function putHeatmap(hash: string, dataUrl: string): Promise<void> {
  const bytes = dataUrl.length; // UTF-16 chars, but base64 is ASCII so this is the byte count
  if (bytes > HEATMAP_BUDGET) return; // one picture must not evict the whole store

  const index = (await readIndex<HeatmapIndexEntry>(IDX_HEATMAPS)).filter((e) => e.hash !== hash);
  index.unshift({ hash, bytes });

  const keep: HeatmapIndexEntry[] = [];
  const drop: HeatmapIndexEntry[] = [];
  let total = 0;
  for (const entry of index) {
    if (total + entry.bytes <= HEATMAP_BUDGET) {
      keep.push(entry);
      total += entry.bytes;
    } else {
      drop.push(entry);
    }
  }

  await chrome.storage.local.set({ [hKey(hash)]: dataUrl, [IDX_HEATMAPS]: keep });
  if (drop.length) await chrome.storage.local.remove(drop.map((e) => hKey(e.hash)));
}

export async function cacheStats(): Promise<{ entries: number; heatmaps: number; approxBytes: number }> {
  const results = await readIndex<string>(IDX_RESULTS);
  const heatmaps = await readIndex<HeatmapIndexEntry>(IDX_HEATMAPS);
  const heatBytes = heatmaps.reduce((sum, e) => sum + e.bytes, 0);
  // storage.local has no per-key size API in every browser; the verdict JSON is ~1.5KB.
  return {
    entries: results.length,
    heatmaps: heatmaps.length,
    approxBytes: results.length * 1500 + heatBytes,
  };
}

export async function clearCache(): Promise<void> {
  const results = await readIndex<string>(IDX_RESULTS);
  const heatmaps = await readIndex<HeatmapIndexEntry>(IDX_HEATMAPS);
  await chrome.storage.local.remove([
    ...results.map(rKey),
    ...heatmaps.map((e) => hKey(e.hash)),
    IDX_RESULTS,
    IDX_HEATMAPS,
  ]);
}
