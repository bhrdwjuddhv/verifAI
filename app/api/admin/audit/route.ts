export const dynamic = 'force-dynamic';

import { NextRequest, NextResponse } from 'next/server';
import { verifySessionToken, COOKIE_NAME } from '@/lib/admin/auth';
import { getRecentAuditEvents } from '@/lib/admin/audit';
import { logError, generateRequestId, safeErrorResponse } from '@/lib/admin/logger';

/**
 * GET /api/admin/audit — Read-only audit log view (owner only).
 * Query params: ?limit=100 (default 100, max 500)
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
        { error: 'Forbidden. Owner role required to view audit log.' },
        { status: 403 }
      );
    }

    const limitParam = req.nextUrl.searchParams.get('limit');
    const limit = Math.min(Math.max(parseInt(limitParam || '100', 10) || 100, 1), 500);

    const events = getRecentAuditEvents(limit);

    return NextResponse.json({ events, count: events.length }, { status: 200 });
  } catch (err) {
    logError('audit-log', err, requestId);
    return NextResponse.json(safeErrorResponse(requestId), { status: 500 });
  }
}
