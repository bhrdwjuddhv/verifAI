/**
 * The sites auto-scan can watch.
 *
 * Pure data plus one matcher, so the options page can render the list without pulling the
 * service worker's registration logic into its bundle.
 */

export interface AutoSite {
  id: string;
  label: string;
  /** What the feed actually is, in the user's words — this is the permission's context. */
  description: string;
  origins: string[];
}

export const AUTO_SITES: AutoSite[] = [
  {
    id: 'youtube',
    label: 'YouTube',
    description: 'Shorts and video thumbnails',
    origins: ['https://*.youtube.com/*'],
  },
  {
    id: 'instagram',
    label: 'Instagram',
    description: 'Reels and feed images',
    origins: ['https://*.instagram.com/*'],
  },
];

/** `https://*.youtube.com/*` vs `www.youtube.com` — subdomain wildcards included. */
export function matchesHost(pattern: string, host: string): boolean {
  const bare = pattern.replace(/^https?:\/\//, '').replace(/\/\*$/, '');
  if (bare.startsWith('*.')) {
    const domain = bare.slice(2);
    return host === domain || host.endsWith(`.${domain}`);
  }
  return host === bare;
}
