/**
 * Auto-scan: watches a feed and asks the worker to score what scrolls past.
 *
 * The rule that shapes this file, from PLAN.md: **auto mode is on-device only**. Uploading
 * whatever a user scrolls past is a privacy problem, and at 20 requests/min/IP it would also
 * rate-limit them out of their own website within seconds. The worker enforces it; this side
 * simply never handles bytes except to hand them straight over.
 *
 * Video frames are captured in the page, so they need no permission beyond the site itself.
 * Feed images live on a CDN the extension usually has no access to — those are attempted and
 * reported as skipped rather than prompting mid-scroll.
 */

import { discover, type Discovery } from './discover';
import { watchVideo, stopWatching, type Frame } from './frames';
import { clearFor } from './index';
import { onTeardown, send } from './runtime';

let running: Discovery | null = null;

// An extension reload leaves this script running with no way home. Stop rather than keep
// observing and sampling for the life of the tab.
onTeardown(() => stopAuto());

export function startAuto(): void {
  if (running) return;
  running = discover(
    (_element, url) => {
      void send({ type: 'auto:image', url });
    },
    (video) => {
      watchVideo(
        video,
        (_element, frame: Frame) => {
          void send({
            type: 'auto:frame',
            url: frame.sourceUrl,
            b64: frame.b64,
            mime: frame.mime,
            dedupeKey: frame.dedupeKey,
            t: frame.t,
          });
        },
        (element) => clearFor(element)
      );
    }
  );
}

export function stopAuto(): void {
  running?.stop();
  running = null;
  stopWatching();
}
