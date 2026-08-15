import { NextRequest, NextResponse } from 'next/server';
import { authenticateUser, COOKIE_NAME, SESSION_EXPIRY_SECONDS, CSRF_COOKIE_NAME } from '@/lib/admin/auth';
import { logError, generateRequestId, safeErrorResponse } from '@/lib/admin/logger';

export async function POST(req: NextRequest) {
  const requestId = generateRequestId();

  try {
    const body = await req.json();
    const { email, password } = body;

    if (!email || !password) {
      return NextResponse.json(
        { error: 'Email and password are required.' },
        { status: 400 }
      );
    }

    const ip = req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() || '127.0.0.1';

    const result = await authenticateUser(email, password, ip);

    if (!result.success) {
      const status = result.lockout?.locked ? 423 : 401;
      return NextResponse.json(
        {
          error: result.error,
          locked: result.lockout?.locked || false,
          retryAfter: result.lockout?.retryAfter || null,
          attemptsRemaining: result.lockout?.attemptsRemaining ?? null,
        },
        { status }
      );
    }

    // Success — set session cookie + CSRF cookie
    const response = NextResponse.json(
      {
        success: true,
        message: 'Authenticated successfully.',
        user: {
          id: result.user!.id,
          email: result.user!.email,
          role: result.user!.role,
        },
      },
      { status: 200 }
    );

    // Session cookie: HttpOnly, Secure, SameSite=Strict
    response.cookies.set({
      name: COOKIE_NAME,
      value: result.token!,
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      path: '/',
      maxAge: SESSION_EXPIRY_SECONDS,
    });

    // CSRF cookie: NOT HttpOnly (client JS needs to read it), Secure, SameSite=Strict
    response.cookies.set({
      name: CSRF_COOKIE_NAME,
      value: result.csrfToken!,
      httpOnly: false,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      path: '/',
      maxAge: SESSION_EXPIRY_SECONDS,
    });

    return response;
  } catch (err) {
    logError('login', err, requestId);
    return NextResponse.json(safeErrorResponse(requestId), { status: 500 });
  }
}
