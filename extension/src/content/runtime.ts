/**
 * Surviving the extension being reloaded out from under us.
 *
 * A content script already injected into a page keeps running after its extension is
 * reloaded, updated or disabled — but its connection to that extension is gone. Two things
 * then happen, and both showed up in the error log:
 *
 *   - `chrome.runtime.sendMessage` throws `Extension context invalidated`, *synchronously*,
 *     so a `.catch()` on the returned promise never sees it;
 *   - `chrome.runtime` itself becomes undefined, giving
 *     `Cannot read properties of undefined (reading 'sendMessage')`.
 *
 * Catching those is the small half. The important half is stopping: an orphaned script keeps
 * its observers and its 3-second frame timer running for as long as the tab is open, sampling
 * video into a void. One reload of the extension while a few Shorts tabs are open leaves
 * several of those behind.
 */

const teardowns: (() => void)[] = [];
let dead = false;
let heartbeat: number | undefined;

/**
 * How often to check we are still attached.
 *
 * Waiting until the next send is not enough: a script whose video is paused, or whose frames
 * keep deduplicating, never sends anything again — and would sit there with stale badges on
 * the page for as long as the tab stayed open.
 */
const HEARTBEAT_MS = 5000;

/** Register work to undo when the extension goes away. */
export function onTeardown(fn: () => void): void {
  teardowns.push(fn);
  if (heartbeat === undefined) {
    heartbeat = setInterval(() => {
      if (!extensionAlive()) teardown();
    }, HEARTBEAT_MS) as unknown as number;
  }
}

export function extensionAlive(): boolean {
  if (dead) return false;
  try {
    // `chrome.runtime.id` is the documented liveness check: it is undefined once orphaned.
    return Boolean(chrome.runtime?.id);
  } catch {
    return false;
  }
}

/** Idempotent: shuts down everything this script started, once. */
export function teardown(): void {
  if (dead) return;
  dead = true;
  if (heartbeat !== undefined) {
    clearInterval(heartbeat);
    heartbeat = undefined;
  }
  for (const fn of teardowns) {
    try {
      fn();
    } catch {
      // Nothing useful to do — the extension is already gone.
    }
  }
  teardowns.length = 0;
}

const ORPHANED = /context invalidated|Extension context|message port closed|receiving end does not exist/i;

/**
 * Sends to the service worker, or gives up quietly and stops the script.
 *
 * Returns null rather than throwing: every caller here is fire-and-forget, and a page should
 * never see an exception from an extension that is no longer installed.
 */
export async function send<T = unknown>(message: unknown): Promise<T | null> {
  if (!extensionAlive()) {
    teardown();
    return null;
  }
  try {
    return (await chrome.runtime.sendMessage(message)) as T;
  } catch (err) {
    if (ORPHANED.test(String((err as Error)?.message ?? err))) teardown();
    return null;
  }
}
