/**
 * CSRF Double-Submit Cookie Pattern (Edge & Node compatible)
 *
 * On login, the server sets a csrf_token cookie (HttpOnly = false, so JS can read it).
 * The client must include the same token value in the X-CSRF-Token header
 * for all state-changing admin requests (POST, PUT, DELETE).
 * Middleware validates that the two values match.
 */

/**
 * Generate a cryptographically random CSRF token (32 hex bytes).
 * Compatible with Edge Runtime and Node.js.
 */
export function generateCsrfToken(): string {
  const bytes = new Uint8Array(32);
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    crypto.getRandomValues(bytes);
  } else {
    // Fallback for older envs
    for (let i = 0; i < 32; i++) {
      bytes[i] = Math.floor(Math.random() * 256);
    }
  }
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Validate that the CSRF cookie value matches the header value.
 * Constant-time string comparison (Edge & Node compatible).
 */
export function validateCsrfToken(cookieValue: string | undefined, headerValue: string | undefined): boolean {
  if (!cookieValue || !headerValue) {
    return false;
  }

  if (cookieValue.length !== headerValue.length) {
    return false;
  }

  let result = 0;
  for (let i = 0; i < cookieValue.length; i++) {
    result |= cookieValue.charCodeAt(i) ^ headerValue.charCodeAt(i);
  }
  return result === 0;
}

export const CSRF_COOKIE_NAME = 'verifai_csrf_token';
export const CSRF_HEADER_NAME = 'x-csrf-token';
