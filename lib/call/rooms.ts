/**
 * Signalling for 1-to-1 calls: an in-memory mailbox that peers poll.
 *
 * No WebSocket server, because Next.js route handlers cannot hold one — and no Socket.IO,
 * because it needs the same long-lived process. Polling every 700ms costs a few requests and
 * works on the runtime we already deploy to.
 *
 * KNOWN LIMIT, stated rather than papered over: this Map lives in one Node process. It works
 * for `next dev`, `next start`, and any single container. On multi-instance serverless the two
 * peers can land on different instances and never see each other's offers. Swap this module
 * for Redis/Upstash before running calls on Vercel — the interface is four functions.
 */

export interface SignalMessage {
  seq: number;
  from: string;
  /** Opaque to the server: an SDP offer/answer or an ICE candidate. */
  data: unknown;
}

interface Room {
  id: string;
  createdAt: number;
  peers: string[];
  messages: SignalMessage[];
  seq: number;
}

const ROOM_TTL_MS = 60 * 60 * 1000;
const MAX_MESSAGES = 200;

// `globalThis` so a dev-server hot reload does not drop calls that are in progress.
const store: Map<string, Room> = (globalThis as any).__verifaiCallRooms ?? new Map();
(globalThis as any).__verifaiCallRooms = store;

function sweep() {
  const cutoff = Date.now() - ROOM_TTL_MS;
  for (const [id, room] of store) if (room.createdAt < cutoff) store.delete(id);
}

function id(bytes = 6) {
  const a = new Uint8Array(bytes);
  crypto.getRandomValues(a);
  return Array.from(a, (b) => b.toString(16).padStart(2, '0')).join('');
}

export function createRoom(): { callId: string; peerId: string } {
  sweep();
  const callId = id(4);
  const peerId = id(6);
  store.set(callId, { id: callId, createdAt: Date.now(), peers: [peerId], messages: [], seq: 0 });
  return { callId, peerId };
}

export function joinRoom(callId: string): { peerId: string; role: 'caller' | 'callee' } | null {
  sweep();
  const room = store.get(callId);
  if (!room) return null;
  // Two peers only. A third would silently break the peer connection, so it is refused.
  if (room.peers.length >= 2) return null;
  const peerId = id(6);
  room.peers.push(peerId);
  return { peerId, role: 'callee' };
}

export function roomExists(callId: string): boolean {
  sweep();
  return store.has(callId);
}

export function postSignal(callId: string, from: string, data: unknown): number | null {
  const room = store.get(callId);
  if (!room || !room.peers.includes(from)) return null;
  room.seq += 1;
  room.messages.push({ seq: room.seq, from, data });
  if (room.messages.length > MAX_MESSAGES) room.messages.splice(0, room.messages.length - MAX_MESSAGES);
  return room.seq;
}

/** Everything the *other* peer has said since `since`. */
export function readSignals(callId: string, self: string, since: number): SignalMessage[] | null {
  const room = store.get(callId);
  if (!room) return null;
  return room.messages.filter((m) => m.seq > since && m.from !== self);
}

export function endRoom(callId: string): boolean {
  return store.delete(callId);
}

export function peerCount(callId: string): number {
  return store.get(callId)?.peers.length ?? 0;
}
