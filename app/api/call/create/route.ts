export const dynamic = 'force-dynamic';

import { NextResponse } from 'next/server';
import { createRoom } from '@/lib/call/rooms';

export async function POST() {
  const { callId, peerId } = createRoom();
  return NextResponse.json({ callId, peerId, role: 'caller' });
}
