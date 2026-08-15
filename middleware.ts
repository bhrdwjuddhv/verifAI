import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { validateCsrfToken, CSRF_COOKIE_NAME, CSRF_HEADER_NAME } from './lib/admin/csrf';

const COOKIE_NAME = 'verifai_admin_session';

// In-memory rate limiting stores for Edge Middleware
const publicRateLimitMap = new Map<string, { count: number; resetTime: number }>();
const adminRateLimitMap = new Map<string, { count: number; resetTime: number }>();

const RATE_LIMIT_WINDOW = 60 * 1000; // 1 minute
const PUBLIC_MAX_REQUESTS = 20;       // 20 req/min for public API
const ADMIN_MAX_REQUESTS = 60;        // 60 req/min for admin routes

const BLOCKED_USER_AGENTS = [
  'sqlmap',
  'nikto',
  'nmap',
  'masscan',
  'zgrab',
  'gospider',
];

export async function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;
  const userAgent = (req.headers.get('user-agent') || '').toLowerCase();

  // 1. Bot & Vulnerability Scanner Firewall (Applies globally)
  if (BLOCKED_USER_AGENTS.some((bot) => userAgent.includes(bot))) {
    return new NextResponse('Access Denied (WAF Blocked)', { status: 403 });
  }

  // 2. Admin Protection & Auth Check (/admin/* & /api/admin/*)
  const isAdminRoute = pathname.startsWith('/admin');
  const isAdminApiRoute = pathname.startsWith('/api/admin');

  if (isAdminRoute || isAdminApiRoute) {
    const isLoginPath =
      pathname === '/admin/login' ||
      pathname === '/api/admin/auth/login';

    if (!isLoginPath) {
      const token = req.cookies.get(COOKIE_NAME)?.value;

      if (!token) {
        if (isAdminApiRoute) {
          return new NextResponse(
            JSON.stringify({ error: 'Unauthorized', message: 'Admin authentication required.' }),
            { status: 401, headers: { 'Content-Type': 'application/json' } }
          );
        }
        const loginUrl = new URL('/admin/login', req.url);
        return NextResponse.redirect(loginUrl);
      }

      // --- CSRF Validation for state-changing admin API requests ---
      if (
        isAdminApiRoute &&
        ['POST', 'PUT', 'DELETE'].includes(req.method) &&
        pathname !== '/api/admin/auth/logout' // Allow logout without CSRF
      ) {
        const csrfCookie = req.cookies.get(CSRF_COOKIE_NAME)?.value;
        const csrfHeader = req.headers.get(CSRF_HEADER_NAME);

        if (!validateCsrfToken(csrfCookie, csrfHeader || undefined)) {
          return new NextResponse(
            JSON.stringify({
              error: 'CSRF validation failed',
              message: 'Missing or invalid CSRF token. Include the X-CSRF-Token header.',
            }),
            { status: 403, headers: { 'Content-Type': 'application/json' } }
          );
        }
      }

      // Admin rate limiter
      const ip = req.headers.get('x-forwarded-for')?.split(',')[0] || '127.0.0.1';
      const now = Date.now();
      const currentRecord = adminRateLimitMap.get(ip);

      if (!currentRecord || now > currentRecord.resetTime) {
        adminRateLimitMap.set(ip, { count: 1, resetTime: now + RATE_LIMIT_WINDOW });
      } else {
        currentRecord.count += 1;
        if (currentRecord.count > ADMIN_MAX_REQUESTS) {
          return new NextResponse(
            JSON.stringify({ error: 'Too Many Requests', message: 'Admin rate limit exceeded.' }),
            { status: 429, headers: { 'Content-Type': 'application/json', 'Retry-After': '60' } }
          );
        }
      }
    }
  }

  // 3. Protect Public API Routes (/api/* excluding /api/admin/*)
  if (pathname.startsWith('/api/') && !isAdminApiRoute) {
    const ip = req.headers.get('x-forwarded-for')?.split(',')[0] || '127.0.0.1';
    const now = Date.now();
    const currentRecord = publicRateLimitMap.get(ip);

    if (!currentRecord || now > currentRecord.resetTime) {
      publicRateLimitMap.set(ip, { count: 1, resetTime: now + RATE_LIMIT_WINDOW });
    } else {
      currentRecord.count += 1;
      if (currentRecord.count > PUBLIC_MAX_REQUESTS) {
        return new NextResponse(
          JSON.stringify({
            error: 'Too Many Requests',
            message: 'API rate limit exceeded. Please wait 60 seconds before scanning again.',
          }),
          {
            status: 429,
            headers: {
              'Content-Type': 'application/json',
              'Retry-After': '60',
            },
          }
        );
      }
    }

    // Payload Size Guard (50MB Limit)
    const contentLength = parseInt(req.headers.get('content-length') || '0', 10);
    if (contentLength > 50 * 1024 * 1024) {
      return new NextResponse(
        JSON.stringify({
          error: 'Payload Too Large',
          message: 'Uploaded file size exceeds 50MB security limit.',
        }),
        {
          status: 413,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/api/:path*', '/admin/:path*'],
};
