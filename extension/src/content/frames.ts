/**
 * Sampling frames out of a playing video, and surviving a recycled player.
 *
 * YouTube Shorts and Instagram Reels reuse one <video> element across every clip you swipe
 * through. State therefore lives in a WeakMap keyed by the element itself and resets on
 * `loadstart` / `emptied` / a changed `currentSrc` — never on DOM position, which is stale the
 * moment anything is inserted above.
 *
 * Capture is canvas-first. `drawImage` works for MSE and `blob:` sources because they are
 * same-origin, which covers both target sites; DRM-protected media taints the canvas and
 * `getImageData` throws, which is reported so the caller can fall back to captureVisibleTab
 * rather than silently scoring a black rectangle.
 */

/** One frame when playback starts, then every 3s, four per clip at most. */
const FIRST_DELAY_MS = 400;
const INTERVAL_MS = 3000;
const MAX_PER_CLIP = 4;

/**
 * Longest side of the captured frame.
 *
 * Video-frame scanning is approximate by nature — the service decodes with OpenCV at native
 * resolution and we cannot match that from a canvas — so this trades a little fidelity for a
 * PNG small enough to cross chrome messaging, which is JSON and therefore base64.
 */
const MAX_SIDE = 1440;

export interface Frame {
  b64: string;
  mime: string;
  /** SHA-256 of a 64x64 grayscale reduction: the same clip paused and resumed is one scan. */
  dedupeKey: string;
  /** Playback position, for display. */
  t: number;
  sourceUrl: string;
}

export type OnFrame = (video: HTMLVideoElement, frame: Frame) => void;

/** Called when a player starts a different clip, so a stale verdict can be taken down. */
export type OnClipChange = (video: HTMLVideoElement) => void;

interface ClipState {
  src: string;
  taken: number;
  timer?: number;
  keys: Set<string>;
  tainted: boolean;
}

const clips = new WeakMap<HTMLVideoElement, ClipState>();

/**
 * Every pending timer, so they can all be cancelled at once.
 *
 * A WeakMap cannot be iterated, and after the extension is reloaded there is no other handle
 * on the timer chain a recycled player leaves behind.
 */
const timers = new Set<number>();
let stopped = false;

/** Cancels all sampling. Called when the extension is reloaded away from under us. */
export function stopWatching(): void {
  stopped = true;
  for (const timer of timers) clearTimeout(timer);
  timers.clear();
}

/**
 * Grabs a frame from whatever is playing, right now, ignoring the sampling policy.
 *
 * This is the manual path, and on YouTube it is the *only* path: the player swallows
 * right-click and shows its own menu, so the context-menu item never appears over a Short.
 * Asking the page directly sidesteps that entirely.
 */
export async function captureVisible(): Promise<Frame | { error: string }> {
  const videos = [...document.querySelectorAll('video')].filter((v) => v.videoWidth > 0);
  if (!videos.length) return { error: 'No video is loaded on this page.' };

  // Prefer what is playing and largest on screen — on a feed that is the one you are watching.
  const best = videos
    .map((video) => {
      const rect = video.getBoundingClientRect();
      const onScreen = rect.bottom > 0 && rect.top < innerHeight;
      return { video, score: (onScreen ? 1e9 : 0) + (video.paused ? 0 : 1e8) + rect.width * rect.height };
    })
    .sort((a, b) => b.score - a.score)[0].video;

  const frame = await capture(best);
  if (frame === 'tainted') {
    return { error: 'This video is DRM-protected, so its frames cannot be read.' };
  }
  if (!frame) return { error: 'The video has not decoded a frame yet — let it play for a moment.' };
  return frame;
}

export function watchVideo(video: HTMLVideoElement, onFrame: OnFrame, onClipChange?: OnClipChange): void {
  const reset = () => {
    // Whatever was on screen described the previous clip.
    onClipChange?.(video);
    const existing = clips.get(video);
    if (existing?.timer) clearTimeout(existing.timer);
    clips.set(video, { src: video.currentSrc || video.src, taken: 0, keys: new Set(), tainted: false });

    // Re-arm if the element is already playing. A recycled player can swap its source through
    // MSE without ever firing `playing` again, and waiting for an event that never comes is
    // how a Shorts feed silently stops being scanned after the first clip.
    if (!video.paused) schedule(video, onFrame, FIRST_DELAY_MS);
  };

  // A recycled player fires these on every new clip. Treat each as "this is a different
  // video now", because it is.
  video.addEventListener('loadstart', reset);
  video.addEventListener('emptied', reset);
  video.addEventListener('playing', () => schedule(video, onFrame, FIRST_DELAY_MS));
  video.addEventListener('pause', () => {
    const state = clips.get(video);
    if (state?.timer) clearTimeout(state.timer);
  });

  if (!clips.has(video)) reset();
  if (!video.paused) schedule(video, onFrame, FIRST_DELAY_MS);
}

function schedule(video: HTMLVideoElement, onFrame: OnFrame, delay: number): void {
  if (stopped) return;
  let state = clips.get(video);
  // currentSrc changing without an event still means a new clip.
  if (!state || state.src !== (video.currentSrc || video.src)) {
    if (state?.timer) clearTimeout(state.timer);
    state = { src: video.currentSrc || video.src, taken: 0, keys: new Set(), tainted: false };
    clips.set(video, state);
  }
  if (state.taken >= MAX_PER_CLIP || state.tainted) return;
  if (state.timer) {
    clearTimeout(state.timer);
    timers.delete(state.timer);
  }

  const timer = setTimeout(() => {
    timers.delete(timer);
    void tick(video, onFrame);
  }, delay) as unknown as number;
  state.timer = timer;
  timers.add(timer);
}

async function tick(video: HTMLVideoElement, onFrame: OnFrame): Promise<void> {
  if (stopped) return;
  const state = clips.get(video);
  if (!state || video.paused || state.taken >= MAX_PER_CLIP) return;

  const frame = await capture(video);
  if (frame === 'tainted') {
    // Nothing readable will ever come off this element; stop rather than retry every 3s.
    state.tainted = true;
    return;
  }

  if (frame && !state.keys.has(frame.dedupeKey)) {
    state.keys.add(frame.dedupeKey);
    state.taken += 1;
    onFrame(video, frame);
  }

  schedule(video, onFrame, INTERVAL_MS);
}

async function capture(video: HTMLVideoElement): Promise<Frame | 'tainted' | null> {
  if (!video.videoWidth || !video.videoHeight || video.readyState < 2) return null;

  const scale = Math.min(1, MAX_SIDE / Math.max(video.videoWidth, video.videoHeight));
  const width = Math.max(1, Math.round(video.videoWidth * scale));
  const height = Math.max(1, Math.round(video.videoHeight * scale));

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  try {
    ctx.drawImage(video, 0, 0, width, height);
  } catch {
    return 'tainted';
  }

  const dedupeKey = await reduceToKey(video);
  if (dedupeKey === null) return 'tainted';

  // PNG, not JPEG: the frame is already lossy from the codec, and a second lossy pass would
  // overwrite the resampling fingerprint NPR is built to read.
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'));
  if (!blob) return null;

  return {
    b64: await toBase64(blob),
    mime: 'image/png',
    dedupeKey,
    t: Math.round(video.currentTime * 10) / 10,
    sourceUrl: video.currentSrc || video.src,
  };
}

/**
 * A 64x64 grayscale fingerprint of the frame.
 *
 * Cheap enough to run every tick, and stable across the sub-pixel noise between two frames of
 * a static shot — so a paused or still video is scanned once, not four times. Null means the
 * canvas is tainted and nothing can be read back.
 */
async function reduceToKey(video: HTMLVideoElement): Promise<string | null> {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;

  try {
    ctx.drawImage(video, 0, 0, 64, 64);
    const { data } = ctx.getImageData(0, 0, 64, 64);
    const gray = new Uint8Array(64 * 64);
    for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
      // Coarse on purpose: quantising to 16 levels absorbs codec noise between frames that
      // are visually the same shot.
      gray[i] = ((data[p] * 19595 + data[p + 1] * 38470 + data[p + 2] * 7471 + 0x8000) >> 16) & 0xf0;
    }
    const digest = await crypto.subtle.digest('SHA-256', gray);
    return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
  } catch {
    return null; // SecurityError: DRM-protected media
  }
}

function toBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(',')[1] ?? '');
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}
