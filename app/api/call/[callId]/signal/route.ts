export const dynamic = 'force-dynamic';

import { NextRequest, NextResponse } from 'next/server';
import { postSignal, readSignals } from '@/lib/call/rooms';

/** Poll for whatever the other peer has said since `since`. */
export async function GET(req: NextRequest, { params }: { params: { callId: string } }) {
  const peerId = req.nextUrl.searchParams.get('peerId') ?? '';
  const since = Number(req.nextUrl.searchParams.get('since') ?? '0') || 0;

  const messages = readSignals(params.callId, peerId, since);
  if (messages === null) {
    return NextResponse.json({ error: 'No such call.' }, { status: 404 });
  }
  const last = messages.length ? messages[messages.length - 1].seq : since;
  return NextResponse.json({ messages, last });
}

/** Put an SDP offer/answer or an ICE candidate in the other peer's mailbox. */
export async function POST(req: NextRequest, { params }: { params: { callId: string } }) {
  const body = await req.json().catch(() => null);
  if (!body?.peerId || body.data === undefined) {
    return NextResponse.json({ error: 'peerId and data are required.' }, { status: 400 });
  }

  const seq = postSignal(params.callId, body.peerId, body.data);
  if (seq === null) {
    return NextResponse.json(
      { error: 'No such call, or you are not a participant.' },
      { status: 404 }
    );
  }
  return NextResponse.json({ seq });
}
