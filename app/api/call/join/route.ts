export const dynamic = 'force-dynamic';

import { NextRequest, NextResponse } from 'next/server';
import { joinRoom, peerCount, roomExists } from '@/lib/call/rooms';

export async function POST(req: NextRequest) {
  const { callId } = await req.json().catch(() => ({ callId: null }));
  if (!callId || typeof callId !== 'string') {
    return NextResponse.json({ error: 'callId is required.' }, { status: 400 });
  }

  if (!roomExists(callId)) {
    return NextResponse.json(
      {
        error: 'No such call.',
        detail: 'The link may have expired (calls live for one hour), or the server restarted.',
      },
      { status: 404 }
    );
  }

  const joined = joinRoom(callId);
  if (!joined) {
    return NextResponse.json(
      { error: 'This call is full.', detail: `It already has ${peerCount(callId)} participants; 1-to-1 only.` },
      { status: 409 }
    );
  }
  return NextResponse.json({ callId, ...joined });
}
