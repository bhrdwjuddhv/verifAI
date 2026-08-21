/**
 * Call platforms Live Guard can watch.
 *
 * Browser tabs only. A native Zoom/Teams/WhatsApp desktop app and a cellular call are both
 * invisible to an extension — no API exposes their audio — and pretending otherwise would be
 * the most damaging kind of dishonesty here: someone trusting a guard that is not running.
 */

export interface CallSite {
  id: string;
  label: string;
  /** Where the call actually happens, in the user's words. */
  description: string;
  origins: string[];
  /** Matched against location.hostname + pathname to tell "in a call" from "on the home page". */
  inCall: (url: URL) => boolean;
}

export const CALL_SITES: CallSite[] = [
  {
    id: 'meet',
    label: 'Google Meet',
    description: 'Meetings at meet.google.com',
    origins: ['https://meet.google.com/*'],
    // Meeting codes look like abc-defg-hij; the landing page is just "/".
    inCall: (url) => /^\/[a-z]{3}-[a-z]{4}-[a-z]{3}/i.test(url.pathname),
  },
  {
    id: 'discord',
    label: 'Discord (web)',
    description: 'Voice and video channels in the browser app',
    origins: ['https://discord.com/*'],
    inCall: (url) => url.pathname.startsWith('/channels/'),
  },
  {
    id: 'teams',
    label: 'Microsoft Teams (web)',
    description: 'Calls and meetings in the browser app',
    origins: ['https://teams.microsoft.com/*', 'https://teams.live.com/*'],
    inCall: (url) => /(_#\/|\/)(call|meetup-join|modern-calling|pre-join)/i.test(url.href),
  },
  {
    id: 'zoom',
    label: 'Zoom (web client)',
    description: 'Meetings joined in the browser, not the desktop app',
    origins: ['https://*.zoom.us/*'],
    inCall: (url) => /\/(wc|j|s)\//.test(url.pathname),
  },
];

export function siteForUrl(rawUrl: string | undefined): CallSite | null {
  if (!rawUrl) return null;
  let url: URL;
  try {
    url = new URL(rawUrl);
  } catch {
    return null;
  }
  return (
    CALL_SITES.find((site) =>
      site.origins.some((pattern) => matchOrigin(pattern, url))
    ) ?? null
  );
}

/** True when the tab is not merely on the platform but plausibly inside a call. */
export function isInCall(rawUrl: string | undefined): boolean {
  const site = siteForUrl(rawUrl);
  if (!site) return false;
  try {
    return site.inCall(new URL(rawUrl!));
  } catch {
    return false;
  }
}

/** Minimal match-pattern check: scheme + host (with one leading wildcard) is enough here. */
export function matchOrigin(pattern: string, url: URL): boolean {
  const match = /^(https?):\/\/([^/]+)\//.exec(pattern);
  if (!match) return false;
  const [, scheme, host] = match;
  if (`${scheme}:` !== url.protocol) return false;
  if (host.startsWith('*.')) {
    const base = host.slice(2);
    return url.hostname === base || url.hostname.endsWith(`.${base}`);
  }
  return url.hostname === host;
}

export const CALL_ORIGINS = CALL_SITES.flatMap((site) => site.origins);
