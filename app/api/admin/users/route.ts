export const dynamic = 'force-dynamic';

import { NextRequest, NextResponse } from 'next/server';
import { verifySessionToken, COOKIE_NAME } from '@/lib/admin/auth';
import { listUsers, createUser, type UserRole } from '@/lib/admin/users';
import { logAuditEvent, AUDIT_ACTIONS } from '@/lib/admin/audit';
import { logError, generateRequestId, safeErrorResponse } from '@/lib/admin/logger';

/**
 * GET /api/admin/users — List all users (owner only). Passwords are never returned.
 */
export async function GET(req: NextRequest) {
  const requestId = generateRequestId();

  try {
    const token = req.cookies.get(COOKIE_NAME)?.value;
    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const session = await verifySessionToken(token);
    if (!session.valid || session.role !== 'owner') {
      return NextResponse.json(
        { error: 'Forbidden. Owner role required to manage accounts.' },
        { status: 403 }
      );
    }

    const users = listUsers();
    return NextResponse.json({ users }, { status: 200 });
  } catch (err) {
    logError('list-users', err, requestId);
    return NextResponse.json(safeErrorResponse(requestId), { status: 500 });
  }
}

/**
 * POST /api/admin/users — Create a new user (owner only).
 * Body: { email: string, password: string, role: 'owner' | 'trainer' | 'viewer' }
 */
export async function POST(req: NextRequest) {
  const requestId = generateRequestId();

  try {
    const token = req.cookies.get(COOKIE_NAME)?.value;
    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const session = await verifySessionToken(token);
    if (!session.valid || session.role !== 'owner') {
      return NextResponse.json(
        { error: 'Forbidden. Owner role required to create accounts.' },
        { status: 403 }
      );
    }

    const body = await req.json();
    const { email, password, role } = body;

    if (!email || !password || !role) {
      return NextResponse.json(
        { error: 'email, password, and role are required.' },
        { status: 400 }
      );
    }

    const validRoles: UserRole[] = ['owner', 'trainer', 'viewer'];
    if (!validRoles.includes(role)) {
      return NextResponse.json(
        { error: `Invalid role. Must be one of: ${validRoles.join(', ')}` },
        { status: 400 }
      );
    }

    if (password.length < 8) {
      return NextResponse.json(
        { error: 'Password must be at least 8 characters.' },
        { status: 400 }
      );
    }

    const ip = req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() || '127.0.0.1';

    const newUser = await createUser(email, password, role as UserRole);
    logAuditEvent(session.email!, AUDIT_ACTIONS.ACCOUNT_CREATED, ip, {
      createdUser: newUser.email,
      role: newUser.role,
    });

    return NextResponse.json(
      { success: true, user: newUser },
      { status: 201 }
    );
  } catch (err: any) {
    if (err?.message?.includes('already exists')) {
      return NextResponse.json({ error: err.message }, { status: 409 });
    }
    logError('create-user', err, requestId);
    return NextResponse.json(safeErrorResponse(requestId), { status: 500 });
  }
}
