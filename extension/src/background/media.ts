/**
 * Turning "the thing you right-clicked" into bytes.
 *
 * Three cases, and only one of them is a plain download:
 *   data:   decoded here.
 *   blob:   only exists inside the page that created it — WhatsApp Web and Instagram serve
 *           images this way, and the service worker cannot fetch one. The page reads it and
 *           hands back base64, because chrome messaging is JSON and an ArrayBuffer would
 *           arrive as `{}`.
 *   http(s) fetched here, where host permissions exempt us from CORS. A content script doing
 *           the same fetch would be subject to the page's CORS and fail on most CDNs.
 */

export interface MediaBytes {
  bytes: Uint8Array;
  mime: string;
  filename: string;
}

export type MediaOutcome =
  | { ok: true; media: MediaBytes }
  | { ok: false; kind: 'permission'; origin: string }
  | { ok: false; kind: 'error'; message: string };

/** Base64 inflates by a third and every hop is JSON, so this is a real limit, not a guess. */
const MAX_PAGE_READ_BYTES = 10 * 1024 * 1024;

export async function resolveMedia(
  url: string,
  kind: 'image' | 'video',
  tabId?: number,
  frameId?: number
): Promise<MediaOutcome> {
  if (url.startsWith('data:')) return fromDataUrl(url, kind);
  if (url.startsWith('blob:')) {
    if (tabId === undefined) {
      return { ok: false, kind: 'error', message: 'This image only exists inside the page, and the page is no longer open.' };
    }
    return readInPage(url, kind, tabId, frameId);
  }
  if (!/^https?:/i.test(url)) {
    return { ok: false, kind: 'error', message: `Cannot read ${url.split(':')[0]}: media.` };
  }

  const direct = await fetchInWorker(url, kind);
  if (direct.ok) return direct;

  // Failed. If it is a permission problem, say which origin — the UI turns that into a
  // one-click grant rather than a dead end.
  const origin = originPatternOf(url);
  if (origin && !(await chrome.permissions.contains({ origins: [origin] }))) {
    // Same-origin media can still be read through the page, which needs no new permission.
    if (tabId !== undefined && (await sameOriginAsTab(url, tabId))) {
      const viaPage = await readInPage(url, kind, tabId, frameId);
      if (viaPage.ok) return viaPage;
    }
    return { ok: false, kind: 'permission', origin };
  }
  return direct;
}

function fromDataUrl(url: string, kind: 'image' | 'video'): MediaOutcome {
  const match = /^data:([^;,]*)[^,]*,(.*)$/s.exec(url);
  if (!match) return { ok: false, kind: 'error', message: 'That data: URL could not be read.' };
  try {
    return {
      ok: true,
      media: {
        bytes: base64ToBytes(match[2]),
        mime: match[1] || defaultMime(kind),
        filename: `pasted.${extensionFor(match[1] || defaultMime(kind))}`,
      },
    };
  } catch {
    return { ok: false, kind: 'error', message: 'That data: URL was not valid base64.' };
  }
}

async function fetchInWorker(url: string, kind: 'image' | 'video'): Promise<MediaOutcome> {
  try {
    const res = await fetch(url, { credentials: 'omit', signal: AbortSignal.timeout(30_000) });
    if (!res.ok) return { ok: false, kind: 'error', message: `The image host answered ${res.status}.` };
    const blob = await res.blob();
    return {
      ok: true,
      media: {
        bytes: new Uint8Array(await blob.arrayBuffer()),
        mime: pickMime(blob.type, url, kind),
        filename: filenameFor(url, pickMime(blob.type, url, kind)),
      },
    };
  } catch {
    return { ok: false, kind: 'error', message: 'Could not download that media.' };
  }
}

/**
 * Runs inside the page. Must stay self-contained — chrome.scripting serialises the function
 * source, so a reference to anything imported at the top of this file would be undefined at
 * the other end, and the failure would look like "nothing happened".
 */
async function readBlobInPage(url: string, limit: number) {
  try {
    const res = await fetch(url);
    if (!res.ok) return { ok: false as const, error: `HTTP ${res.status}` };
    const blob = await res.blob();
    if (blob.size > limit) {
      return { ok: false as const, error: `That file is ${(blob.size / 1048576).toFixed(1)}MB; the page-side reader stops at ${limit / 1048576}MB.` };
    }
    const b64 = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result).split(',')[1] ?? '');
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(blob);
    });
    return { ok: true as const, b64, mime: blob.type };
  } catch (err) {
    return { ok: false as const, error: (err as Error)?.message ?? 'unreadable' };
  }
}

async function readInPage(
  url: string,
  kind: 'image' | 'video',
  tabId: number,
  frameId?: number
): Promise<MediaOutcome> {
  let frames;
  try {
    frames = await chrome.scripting.executeScript({
      // A blob: URL belongs to the frame that minted it, so the read has to happen there.
      target: frameId === undefined ? { tabId } : { tabId, frameIds: [frameId] },
      func: readBlobInPage,
      args: [url, MAX_PAGE_READ_BYTES],
      // Isolated world deliberately: it shares the page's origin, which is all a blob: URL
      // needs, without letting the page shadow fetch or FileReader under us.
    });
  } catch (err) {
    return {
      ok: false,
      kind: 'error',
      message: `Could not read the media from the page (${(err as Error)?.message ?? 'injection failed'}).`,
    };
  }

  const payload = frames?.[0]?.result as Awaited<ReturnType<typeof readBlobInPage>> | undefined;
  if (!payload) return { ok: false, kind: 'error', message: 'The page returned nothing for that media.' };
  if (!payload.ok) return { ok: false, kind: 'error', message: payload.error };

  const mime = pickMime(payload.mime, url, kind);
  return {
    ok: true,
    media: { bytes: base64ToBytes(payload.b64), mime, filename: filenameFor(url, mime) },
  };
}

async function sameOriginAsTab(url: string, tabId: number): Promise<boolean> {
  try {
    const tab = await chrome.tabs.get(tabId);
    return !!tab.url && new URL(tab.url).origin === new URL(url).origin;
  } catch {
    return false;
  }
}

export function originPatternOf(url: string): string | null {
  try {
    return `${new URL(url).origin}/*`;
  } catch {
    return null;
  }
}

export function base64ToBytes(b64: string): Uint8Array {
  const binary = atob(b64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

const defaultMime = (kind: 'image' | 'video') => (kind === 'video' ? 'video/mp4' : 'image/jpeg');

/**
 * The API routes on the MIME type — `video/*` goes to /predict-video, everything else to
 * /predict — and then validates the magic bytes against that choice. An empty or generic
 * type would be routed as an image and rejected, so guess from the extension before falling
 * back to what the context menu said.
 */
function pickMime(blobType: string, url: string, kind: 'image' | 'video'): string {
  if (blobType && blobType !== 'application/octet-stream') return blobType;

  const ext = /\.([a-z0-9]{2,5})(?:[?#]|$)/i.exec(url)?.[1]?.toLowerCase();
  const byExt: Record<string, string> = {
    jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png', webp: 'image/webp',
    gif: 'image/gif', bmp: 'image/bmp', avif: 'image/avif',
    mp4: 'video/mp4', webm: 'video/webm', mov: 'video/quicktime', mkv: 'video/x-matroska',
  };
  return (ext && byExt[ext]) || defaultMime(kind);
}

function extensionFor(mime: string): string {
  return mime.split('/')[1]?.split('+')[0] || 'bin';
}

function filenameFor(url: string, mime: string): string {
  let base = 'media';
  try {
    base = new URL(url).pathname.split('/').filter(Boolean).pop() || 'media';
  } catch {
    /* keep the fallback */
  }
  // The route strips anything outside [A-Za-z0-9_.-] anyway; doing it here keeps the name
  // we display identical to the name the server logs.
  base = base.replace(/[^a-zA-Z0-9_.-]/g, '_').slice(0, 60);
  return /\.[a-z0-9]{2,5}$/i.test(base) ? base : `${base}.${extensionFor(mime)}`;
}
