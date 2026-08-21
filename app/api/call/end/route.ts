export const dynamic = 'force-dynamic';

import { NextRequest, NextResponse } from 'next/server';
import { endRoom } from '@/lib/call/rooms';

export async function POST(req: NextRequest) {
  const { callId } = await req.json().catch(() => ({ callId: null }));
  if (!callId || typeof callId !== 'string') {
    return NextResponse.json({ error: 'callId is required.' }, { status: 400 });
  }
  // Ending is best-effort: the media connection is peer-to-peer and already torn down by the
  // client. This only drops the mailbox.
  return NextResponse.json({ ended: endRoom(callId) });
}
