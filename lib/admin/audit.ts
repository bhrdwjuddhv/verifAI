import fs from 'fs';
import path from 'path';

export interface AuditEvent {
  timestamp: string;
  actor: string; // email or 'system'
  action: string;
  ip: string;
  details?: Record<string, unknown>;
}

const AUDIT_LOG_FILE = path.resolve('data', 'audit_log.jsonl');

function ensureDataDir(): void {
  const dir = path.dirname(AUDIT_LOG_FILE);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

/**
 * Append an audit event to the log. This is append-only by design.
 * The file is never truncated or edited from the application.
 */
export function logAuditEvent(
  actor: string,
  action: string,
  ip: string,
  details?: Record<string, unknown>
): void {
  ensureDataDir();

  const event: AuditEvent = {
    timestamp: new Date().toISOString(),
    actor,
    action,
    ip,
    ...(details ? { details } : {}),
  };

  const line = JSON.stringify(event) + '\n';

  try {
    fs.appendFileSync(AUDIT_LOG_FILE, line, 'utf-8');
  } catch (err) {
    // If we can't write to the audit log, log to console as a fallback
    // Never silently swallow audit log failures
    console.error('[ALERT] Failed to write audit log entry:', err);
    console.error('[AUDIT_FALLBACK]', JSON.stringify(event));
  }
}

/**
 * Read the most recent audit events (up to `limit`).
 * Returns them in reverse chronological order (most recent first).
 */
export function getRecentAuditEvents(limit: number = 100): AuditEvent[] {
  ensureDataDir();

  if (!fs.existsSync(AUDIT_LOG_FILE)) {
    return [];
  }

  try {
    const raw = fs.readFileSync(AUDIT_LOG_FILE, 'utf-8');
    const lines = raw.trim().split('\n').filter(Boolean);

    // Take last `limit` lines and parse
    const recentLines = lines.slice(-limit);
    const events: AuditEvent[] = [];

    for (const line of recentLines) {
      try {
        events.push(JSON.parse(line));
      } catch {
        // Skip malformed lines
      }
    }

    // Reverse to get most recent first
    return events.reverse();
  } catch {
    return [];
  }
}

// Well-known audit actions
export const AUDIT_ACTIONS = {
  LOGIN_SUCCESS: 'login_success',
  LOGIN_FAILED: 'login_failed',
  LOGOUT: 'logout',
  LOGOUT_ALL: 'logout_all_sessions',
  ACCOUNT_CREATED: 'account_created',
  ACCOUNT_DELETED: 'account_deleted',
  ROLE_CHANGED: 'role_changed',
  TRAINING_STARTED: 'training_started',
  CHECKPOINT_PROMOTED: 'checkpoint_promoted',
  LOCKOUT_TRIGGERED: 'lockout_triggered',
} as const;
