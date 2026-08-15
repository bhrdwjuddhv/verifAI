import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

export interface Session {
  token: string;
  userId: string;
  email: string;
  role: string;
  createdAt: string;
  expiresAt: string;
  lastRenewedAt: string;
}

const SESSION_EXPIRY_MS = 30 * 60 * 1000; // 30 minutes
const SESSIONS_FILE = path.resolve('data', 'sessions.json');

function ensureDataDir(): void {
  const dir = path.dirname(SESSIONS_FILE);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

function readSessions(): Session[] {
  ensureDataDir();
  if (!fs.existsSync(SESSIONS_FILE)) {
    return [];
  }
  try {
    const raw = fs.readFileSync(SESSIONS_FILE, 'utf-8');
    return JSON.parse(raw) as Session[];
  } catch {
    return [];
  }
}

function writeSessions(sessions: Session[]): void {
  ensureDataDir();
  fs.writeFileSync(SESSIONS_FILE, JSON.stringify(sessions, null, 2), 'utf-8');
}

/** Purge expired sessions from the store (housekeeping). */
function purgeExpired(sessions: Session[]): Session[] {
  const now = Date.now();
  return sessions.filter((s) => new Date(s.expiresAt).getTime() > now);
}

/**
 * Create a new session for a user.
 * Returns a cryptographically random 48-byte hex token.
 */
export function createSession(userId: string, email: string, role: string): Session {
  const sessions = purgeExpired(readSessions());
  const now = new Date();

  const session: Session = {
    token: crypto.randomBytes(48).toString('hex'),
    userId,
    email,
    role,
    createdAt: now.toISOString(),
    expiresAt: new Date(now.getTime() + SESSION_EXPIRY_MS).toISOString(),
    lastRenewedAt: now.toISOString(),
  };

  sessions.push(session);
  writeSessions(sessions);
  return session;
}

/**
 * Look up a session by its token. Returns null if expired or not found.
 */
export function getSession(token: string): Session | null {
  const sessions = readSessions();
  const session = sessions.find((s) => s.token === token);
  if (!session) return null;

  // Check expiry
  if (new Date(session.expiresAt).getTime() < Date.now()) {
    // Expired — remove it
    deleteSession(token);
    return null;
  }

  return session;
}

/**
 * Extend session expiry by SESSION_EXPIRY_MS from now (sliding renewal).
 */
export function renewSession(token: string): Session | null {
  const sessions = readSessions();
  const idx = sessions.findIndex((s) => s.token === token);
  if (idx === -1) return null;

  const now = new Date();
  sessions[idx].expiresAt = new Date(now.getTime() + SESSION_EXPIRY_MS).toISOString();
  sessions[idx].lastRenewedAt = now.toISOString();
  writeSessions(sessions);
  return sessions[idx];
}

/**
 * Delete a single session (logout).
 */
export function deleteSession(token: string): boolean {
  const sessions = readSessions();
  const filtered = sessions.filter((s) => s.token !== token);
  if (filtered.length === sessions.length) return false;
  writeSessions(filtered);
  return true;
}

/**
 * Delete ALL sessions for a user (lost/stolen laptop, password change, etc.).
 */
export function deleteAllUserSessions(userId: string): number {
  const sessions = readSessions();
  const filtered = sessions.filter((s) => s.userId !== userId);
  const deletedCount = sessions.length - filtered.length;
  writeSessions(filtered);
  return deletedCount;
}

export const SESSION_EXPIRY_SECONDS = SESSION_EXPIRY_MS / 1000;
