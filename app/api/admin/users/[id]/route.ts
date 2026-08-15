export const dynamic = 'force-dynamic';

import { NextRequest, NextResponse } from 'next/server';
import { verifySessionToken, COOKIE_NAME } from '@/lib/admin/auth';
import { updateUserRole, deleteUser, getUserById, type UserRole } from '@/lib/admin/users';
import { deleteAllUserSessions } from '@/lib/admin/sessions';
import { logAuditEvent, AUDIT_ACTIONS } from '@/lib/admin/audit';
import { logError, generateRequestId, safeErrorResponse } from '@/lib/admin/logger';

/**
 * PUT /api/admin/users/[id] — Update a user's role (owner only).
 * Body: { role: 'owner' | 'trainer' | 'viewer' }
 */
export async function PUT(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  const requestId = generateRequestId();

  try {
    const token = req.cookies.get(COOKIE_NAME)?.value;
    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const session = await verifySessionToken(token);
    if (!session.valid || session.role !== 'owner') {
      return NextResponse.json(
        { error: 'Forbidden. Owner role required.' },
        { status: 403 }
      );
    }

    const { id } = params;
    const body = await req.json();
    const { role } = body;

    const validRoles: UserRole[] = ['owner', 'trainer', 'viewer'];
    if (!role || !validRoles.includes(role)) {
      return NextResponse.json(
        { error: `Invalid role. Must be one of: ${validRoles.join(', ')}` },
        { status: 400 }
      );
    }

    const targetUser = getUserById(id);
    if (!targetUser) {
      return NextResponse.json({ error: 'User not found.' }, { status: 404 });
    }

    const ip = req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() || '127.0.0.1';
    const oldRole = targetUser.role;

    const updated = updateUserRole(id, role as UserRole);
    if (!updated) {
      return NextResponse.json({ error: 'Failed to update user.' }, { status: 500 });
    }

    // Invalidate all sessions for the user whose role changed
    deleteAllUserSessions(id);

    logAuditEvent(session.email!, AUDIT_ACTIONS.ROLE_CHANGED, ip, {
      targetUser: targetUser.email,
      oldRole,
      newRole: role,
    });

    return NextResponse.json({ success: true, user: updated }, { status: 200 });
  } catch (err) {
    logError('update-user-role', err, requestId);
    return NextResponse.json(safeErrorResponse(requestId), { status: 500 });
  }
}

/**
 * DELETE /api/admin/users/[id] — Delete a user (owner only). Cannot delete self.
 */
export async function DELETE(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  const requestId = generateRequestId();

  try {
    const token = req.cookies.get(COOKIE_NAME)?.value;
    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const session = await verifySessionToken(token);
    if (!session.valid || session.role !== 'owner') {
      return NextResponse.json(
        { error: 'Forbidden. Owner role required.' },
        { status: 403 }
      );
    }

    const { id } = params;

    // Prevent self-deletion
    if (id === session.userId) {
      return NextResponse.json(
        { error: 'Cannot delete your own account.' },
        { status: 400 }
      );
    }

    const targetUser = getUserById(id);
    if (!targetUser) {
      return NextResponse.json({ error: 'User not found.' }, { status: 404 });
    }

    const ip = req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() || '127.0.0.1';

    // Delete all sessions first, then the user
    deleteAllUserSessions(id);
    const deleted = deleteUser(id);

    if (!deleted) {
      return NextResponse.json({ error: 'Failed to delete user.' }, { status: 500 });
    }

    logAuditEvent(session.email!, AUDIT_ACTIONS.ACCOUNT_DELETED, ip, {
      deletedUser: targetUser.email,
      deletedRole: targetUser.role,
    });

    return NextResponse.json(
      { success: true, message: `User ${targetUser.email} deleted.` },
      { status: 200 }
    );
  } catch (err) {
    logError('delete-user', err, requestId);
    return NextResponse.json(safeErrorResponse(requestId), { status: 500 });
  }
}
