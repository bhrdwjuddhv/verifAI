/**
 * One scan at a time, under a token bucket.
 *
 * middleware.ts caps /api/* at 20 requests per minute per IP — a budget shared with whoever
 * is browsing the website from the same address. Scanning at the ceiling would rate-limit the
 * user out of their own site, so this runs at 12/min and tells the UI where it is in line
 * rather than failing quietly.
 */

const CAPACITY = 12;
const WINDOW_MS = 60_000;

class TokenBucket {
  private tokens = CAPACITY;
  private last = Date.now();

  /** Milliseconds to wait before a token is free. 0 means "go now" and spends one. */
  take(): number {
    const now = Date.now();
    this.tokens = Math.min(CAPACITY, this.tokens + ((now - this.last) * CAPACITY) / WINDOW_MS);
    this.last = now;

    if (this.tokens >= 1) {
      this.tokens -= 1;
      return 0;
    }
    return Math.ceil(((1 - this.tokens) * WINDOW_MS) / CAPACITY);
  }

  /** Called after a 429: stop spending until the server says we may. */
  penalise(seconds: number): void {
    this.tokens = 0;
    this.last = Date.now() + seconds * 1000 - WINDOW_MS / CAPACITY;
  }
}

const bucket = new TokenBucket();

interface Job {
  id: string;
  run: () => Promise<void>;
  onPosition: (position: number) => void;
  /**
   * Whether this job hits the server. On-device scans still queue — one heavy inference at a
   * time is the point — but they must not spend a budget that exists to protect an API they
   * never touch.
   */
  costsToken: boolean;
}

const pending: Job[] = [];
let running = false;

/**
 * A service worker with nothing in flight is killed after ~30s, which would strand a job
 * that is only waiting on the token bucket. An open port to ourselves keeps it alive for as
 * long as there is queued work — and is released the moment there is not.
 */
let keepalive: number | undefined;

function holdWorkerAlive(): void {
  if (keepalive !== undefined) return;
  keepalive = setInterval(() => void chrome.runtime.getPlatformInfo(), 20_000) as unknown as number;
}

function releaseWorker(): void {
  if (keepalive === undefined) return;
  clearInterval(keepalive);
  keepalive = undefined;
}

function announcePositions(): void {
  pending.forEach((job, i) => job.onPosition(running ? i + 1 : i));
}

export function enqueue(job: Job): void {
  pending.push(job);
  holdWorkerAlive();
  announcePositions();
  void drain();
}

export function cancel(id: string): void {
  const i = pending.findIndex((j) => j.id === id);
  if (i >= 0) pending.splice(i, 1);
  announcePositions();
}

/** Told by the API layer when the server pushed back, so the bucket can respect Retry-After. */
export function backOff(seconds: number): void {
  bucket.penalise(seconds);
}

async function drain(): Promise<void> {
  if (running) return;
  running = true;

  try {
    while (pending.length) {
      if (pending[0].costsToken) {
        const wait = bucket.take();
        if (wait > 0) {
          announcePositions();
          await new Promise((resolve) => setTimeout(resolve, wait));
          continue;
        }
      }

      const job = pending.shift()!;
      announcePositions();
      try {
        await job.run();
      } catch (err) {
        // A job is responsible for reporting its own failure; reaching here means it threw
        // on the way to doing that, and swallowing it would stall every scan behind it.
        console.error('[verifai] job failed outside its own error handling', err);
      }
    }
  } finally {
    running = false;
    releaseWorker();
  }
}
