import fs from 'fs';
import path from 'path';

interface LockoutRecord {
  email: string;
  failedAttempts: number;
  lastFailedAt: string;
  lockedUntil: string | null;
  lockoutCount: number; // How many times this account has been locked out (for exponential backoff)
}

const MAX_ATTEMPTS = 5;
const BASE_LOCKOUT_MS = 5 * 60 * 1000; // 5 minutes
const LOCKOUTS_FILE = path.resolve('data', 'lockouts.json');

function ensureDataDir(): void {
  const dir = path.dirname(LOCKOUTS_FILE);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

function readLockouts(): LockoutRecord[] {
  ensureDataDir();
  if (!fs.existsSync(LOCKOUTS_FILE)) {
    return [];
  }
  try {
    const raw = fs.readFileSync(LOCKOUTS_FILE, 'utf-8');
    return JSON.parse(raw) as LockoutRecord[];
  } catch {
    return [];
  }
}

function writeLockouts(records: LockoutRecord[]): void {
  ensureDataDir();
  fs.writeFileSync(LOCKOUTS_FILE, JSON.stringify(records, null, 2), 'utf-8');
}

function getRecord(email: string, records: LockoutRecord[]): LockoutRecord | undefined {
  return records.find((r) => r.email.toLowerCase() === email.toLowerCase());
}

/**
 * Check if an account is currently locked out.
 * Returns { locked: true, retryAfter: Date } if locked, or { locked: false }.
 */
export function isLockedOut(email: string): { locked: boolean; retryAfterMs?: number; retryAfter?: string } {
  const records = readLockouts();
  const record = getRecord(email, records);

  if (!record || !record.lockedUntil) {
    return { locked: false };
  }

  const lockedUntilMs = new Date(record.lockedUntil).getTime();
  const now = Date.now();

  if (now < lockedUntilMs) {
    return {
      locked: true,
      retryAfterMs: lockedUntilMs - now,
      retryAfter: record.lockedUntil,
    };
  }

  // Lockout has expired — don't reset attempts yet, let the next login attempt do that
  return { locked: false };
}

/**
 * Record a failed login attempt. If MAX_ATTEMPTS is reached, lock the account.
 * Uses exponential backoff: 5m, 15m, 1h, ...
 */
export function recordFailedAttempt(email: string): { locked: boolean; attemptsRemaining: number; retryAfter?: string } {
  const records = readLockouts();
  let record = getRecord(email, records);
  const now = new Date();

  if (!record) {
    record = {
      email: email.toLowerCase(),
      failedAttempts: 0,
      lastFailedAt: now.toISOString(),
      lockedUntil: null,
      lockoutCount: 0,
    };
    records.push(record);
  }

  // If lockout has expired, reset attempts but keep lockoutCount for backoff
  if (record.lockedUntil && new Date(record.lockedUntil).getTime() <= Date.now()) {
    record.failedAttempts = 0;
    record.lockedUntil = null;
  }

  record.failedAttempts += 1;
  record.lastFailedAt = now.toISOString();

  if (record.failedAttempts >= MAX_ATTEMPTS) {
    // Exponential backoff: 5m * 3^lockoutCount, capped at 1 hour
    const lockoutMs = Math.min(
      BASE_LOCKOUT_MS * Math.pow(3, record.lockoutCount),
      60 * 60 * 1000 // Cap at 1 hour
    );
    record.lockedUntil = new Date(now.getTime() + lockoutMs).toISOString();
    record.lockoutCount += 1;

    writeLockouts(records);
    return {
      locked: true,
      attemptsRemaining: 0,
      retryAfter: record.lockedUntil,
    };
  }

  writeLockouts(records);
  return {
    locked: false,
    attemptsRemaining: MAX_ATTEMPTS - record.failedAttempts,
  };
}

/**
 * Reset lockout state for an account (on successful login).
 */
export function resetLockout(email: string): void {
  const records = readLockouts();
  const filtered = records.filter((r) => r.email.toLowerCase() !== email.toLowerCase());
  writeLockouts(filtered);
}
