import { NextRequest, NextResponse } from 'next/server';
import { COOKIE_NAME, CSRF_COOKIE_NAME, verifySessionToken, logoutSession } from '@/lib/admin/auth';
import { logError, generateRequestId, safeErrorResponse } from '@/lib/admin/logger';

export async function POST(req: NextRequest) {
  const requestId = generateRequestId();

  try {
    const token = req.cookies.get(COOKIE_NAME)?.value;
    const ip = req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() || '127.0.0.1';

    if (token) {
      const session = await verifySessionToken(token);
      if (session.valid && session.email) {
        logoutSession(token, session.email, ip);
      }
    }

    const response = NextResponse.json(
      { success: true, message: 'Logged out successfully.' },
      { status: 200 }
    );

    // Clear both cookies
    response.cookies.delete(COOKIE_NAME);
    response.cookies.delete(CSRF_COOKIE_NAME);

    return response;
  } catch (err) {
    logError('logout', err, requestId);
    return NextResponse.json(safeErrorResponse(requestId), { status: 500 });
  }
}
