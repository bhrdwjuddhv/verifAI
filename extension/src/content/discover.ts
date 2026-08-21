/**
 * Finding media in a feed, generically.
 *
 * There are no class-name selectors anywhere in this file, and that is the point. Instagram's
 * and YouTube's class names are obfuscated and change without notice; an adapter built on them
 * is broken by someone else's deploy. What does not change is that a post contains an <img> or
 * a <video>, that it is large, and that it enters the viewport — so that is what we key on.
 *
 * Everything here is cheap: an IntersectionObserver callback and a getBoundingClientRect. No
 * decoding, no hashing, no inference — the ship gate is no main-thread frame over 16ms.
 */

/** Below this on either side it is an avatar, an icon or a tracking pixel, not a post. */
const MIN_SIDE = 100;

/** Half in view before it is worth scanning; a strip at the edge of the screen is not. */
const VISIBILITY = 0.5;

/**
 * URL shapes that are never post content. Deliberately about the *role* of the asset rather
 * than any one site's paths, so this does not become a per-site list by another name.
 */
const NOT_CONTENT = /(^data:image\/svg|\.svg([?#]|$)|\/sprite|\/sprites\/|\/emoji\/|\/favicon|\/icons?\/|\/badge)/i;

export type OnImage = (element: HTMLImageElement, url: string) => void;
export type OnVideo = (element: HTMLVideoElement) => void;

export interface Discovery {
  stop: () => void;
}

export function discover(onImage: OnImage, onVideo: OnVideo): Discovery {
  // Element identity, not DOM position: a feed recycles nodes, and an index would go stale
  // the moment anything is inserted above.
  const seenImages = new WeakSet<HTMLImageElement>();
  const knownVideos = new WeakSet<HTMLVideoElement>();

  /**
   * Evaluate an image, or wait until there is something to evaluate.
   *
   * A feed puts the <img> in the DOM first and sets `src` later, so the intersection callback
   * usually arrives while the element is still empty. Dropping it there is the bug that makes
   * a discoverer look like it works on a test page and find nothing on a real feed: the
   * element is already intersecting, so the observer never fires for it again.
   */
  function tryImage(element: HTMLImageElement): void {
    if (seenImages.has(element)) return;

    if (!element.complete || !element.naturalWidth) {
      element.addEventListener('load', () => tryImage(element), { once: true });
      return;
    }

    const url = element.currentSrc || element.src;
    if (!url || !isContent(element, url)) return;

    seenImages.add(element);
    onImage(element, url);
  }

  const visible = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const element = entry.target;

        if (element instanceof HTMLImageElement) {
          tryImage(element);
        } else if (element instanceof HTMLVideoElement) {
          if (!bigEnough(element)) continue;
          onVideo(element);
        }
      }
    },
    { threshold: VISIBILITY }
  );

  function consider(root: ParentNode): void {
    for (const element of root.querySelectorAll<HTMLImageElement>('img')) {
      if (!seenImages.has(element)) visible.observe(element);
    }
    for (const element of root.querySelectorAll<HTMLVideoElement>('video')) {
      // Videos stay observed for their whole life: one element plays many clips, and each
      // clip has to be noticed again.
      if (knownVideos.has(element)) continue;
      knownVideos.add(element);
      visible.observe(element);
      onVideo(element);
    }
  }

  consider(document);

  // Feeds are built by script after load, and a Shorts swipe replaces the subtree rather than
  // navigating. Watching added nodes survives both — and watching `src` survives the third
  // case, where the feed keeps the element and swaps the picture inside it.
  const mutations = new MutationObserver((records) => {
    for (const record of records) {
      if (record.type === 'attributes') {
        const element = record.target;
        if (element instanceof HTMLImageElement) {
          seenImages.delete(element); // a new picture in an old element is a new candidate
          tryImage(element);
        }
        continue;
      }
      for (const node of record.addedNodes) {
        if (!(node instanceof Element)) continue;
        if (node instanceof HTMLImageElement || node instanceof HTMLVideoElement) {
          consider(node.parentNode ?? document);
        } else {
          consider(node);
        }
      }
    }
  });
  mutations.observe(document.documentElement, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['src', 'srcset'],
  });

  return {
    stop: () => {
      visible.disconnect();
      mutations.disconnect();
    },
  };
}

function bigEnough(element: Element): boolean {
  const rect = element.getBoundingClientRect();
  return rect.width >= MIN_SIDE && rect.height >= MIN_SIDE;
}

/**
 * Rendered size *and* natural size both have to clear the bar.
 *
 * An avatar blown up to 200px is still a 32px asset, and scoring an upscaled thumbnail tells
 * you about the upscaler rather than about the image.
 */
function isContent(image: HTMLImageElement, url: string): boolean {
  if (NOT_CONTENT.test(url)) return false;
  if (image.naturalWidth && (image.naturalWidth < MIN_SIDE || image.naturalHeight < MIN_SIDE)) return false;
  return bigEnough(image);
}
