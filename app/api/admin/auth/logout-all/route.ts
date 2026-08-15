import { NextRequest, NextResponse } from 'next/server';
import { COOKIE_NAME, CSRF_COOKIE_NAME, verifySessionToken, logoutAllSessions } from '@/lib/admin/auth';
import { logError, generateRequestId, safeErrorResponse } from '@/lib/admin/logger';

/**
 * POST /api/admin/auth/logout-all
 * Invalidate ALL sessions for the currently authenticated user.
 * Use case: lost/stolen laptop, suspected session compromise.
 */
export async function POST(req: NextRequest) {
  const requestId = generateRequestId();

  try {
    const token = req.cookies.get(COOKIE_NAME)?.value;
    const ip = req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() || '127.0.0.1';

    if (!token) {
      return NextResponse.json({ error: 'Not authenticated.' }, { status: 401 });
    }

    const session = await verifySessionToken(token);
    if (!session.valid || !session.userId || !session.email) {
      return NextResponse.json({ error: 'Invalid session.' }, { status: 401 });
    }

    const count = logoutAllSessions(session.userId, session.email, ip);

    const response = NextResponse.json(
      {
        success: true,
        message: `All sessions invalidated. ${count} session(s) were removed.`,
        sessionsRemoved: count,
      },
      { status: 200 }
    );

    // Clear cookies on this response too
    response.cookies.delete(COOKIE_NAME);
    response.cookies.delete(CSRF_COOKIE_NAME);

    return response;
  } catch (err) {
    logError('logout-all', err, requestId);
    return NextResponse.json(safeErrorResponse(requestId), { status: 500 });
  }
}
