import { getUserByEmail, verifyPassword, type UserRole } from './users';
import { createSession, getSession, renewSession, deleteSession, deleteAllUserSessions, SESSION_EXPIRY_SECONDS } from './sessions';
import { isLockedOut, recordFailedAttempt, resetLockout } from './lockout';
import { logAuditEvent, AUDIT_ACTIONS } from './audit';
import { generateCsrfToken, CSRF_COOKIE_NAME } from './csrf';
import { logAlert } from './logger';

// --- Configuration ---

const JWT_SECRET = process.env.JWT_SECRET || 'verifai-default-dev-secret-key-2026-must-change';

export const COOKIE_NAME = 'verifai_admin_session';

// --- Types ---

export interface AuthResult {
  success: boolean;
  error?: string;
  token?: string;
  csrfToken?: string;
  user?: {
    id: string;
    email: string;
    role: UserRole;
  };
  lockout?: {
    locked: boolean;
    retryAfter?: string;
    attemptsRemaining?: number;
  };
}

export interface SessionVerification {
  valid: boolean;
  userId?: string;
  email?: string;
  role?: UserRole;
}

// --- Authentication ---

/**
 * Authenticate a user with email and password.
 * Handles lockout checking, password verification, session creation, and audit logging.
 */
export async function authenticateUser(
  email: string,
  password: string,
  ip: string
): Promise<AuthResult> {
  // 1. Check lockout
  const lockoutStatus = isLockedOut(email);
  if (lockoutStatus.locked) {
    logAuditEvent(email, AUDIT_ACTIONS.LOGIN_FAILED, ip, {
      reason: 'Account locked out',
      retryAfter: lockoutStatus.retryAfter,
    });
    return {
      success: false,
      error: 'Account is temporarily locked due to too many failed attempts.',
      lockout: {
        locked: true,
        retryAfter: lockoutStatus.retryAfter,
      },
    };
  }

  // 2. Look up user
  const user = getUserByEmail(email);
  if (!user) {
    // Record failed attempt even for non-existent users (prevents user enumeration timing attacks)
    const result = recordFailedAttempt(email);
    logAuditEvent(email, AUDIT_ACTIONS.LOGIN_FAILED, ip, { reason: 'User not found' });

    if (result.locked) {
      logAlert(`Account lockout triggered for ${email} after repeated failed attempts`, { ip });
      logAuditEvent(email, AUDIT_ACTIONS.LOCKOUT_TRIGGERED, ip);
    }

    return {
      success: false,
      error: 'Invalid email or password.',
      lockout: {
        locked: result.locked,
        attemptsRemaining: result.attemptsRemaining,
        retryAfter: result.retryAfter,
      },
    };
  }

  // 3. Verify password
  const passwordValid = await verifyPassword(user, password);
  if (!passwordValid) {
    const result = recordFailedAttempt(email);
    logAuditEvent(email, AUDIT_ACTIONS.LOGIN_FAILED, ip, { reason: 'Invalid password' });

    if (result.locked) {
      logAlert(`Account lockout triggered for ${email} after repeated failed attempts`, { ip });
      logAuditEvent(email, AUDIT_ACTIONS.LOCKOUT_TRIGGERED, ip);
    }

    return {
      success: false,
      error: 'Invalid email or password.',
      lockout: {
        locked: result.locked,
        attemptsRemaining: result.attemptsRemaining,
        retryAfter: result.retryAfter,
      },
    };
  }

  // 4. Success — reset lockout, create session
  resetLockout(email);
  const session = createSession(user.id, user.email, user.role);
  const csrfToken = generateCsrfToken();

  logAuditEvent(user.email, AUDIT_ACTIONS.LOGIN_SUCCESS, ip, { role: user.role });

  return {
    success: true,
    token: session.token,
    csrfToken,
    user: {
      id: user.id,
      email: user.email,
      role: user.role as UserRole,
    },
  };
}

// --- Session Verification ---

/**
 * Verify a session token and return user info.
 * Used by API route handlers.
 */
export async function verifySessionToken(token: string): Promise<SessionVerification> {
  if (!token) return { valid: false };

  const session = getSession(token);
  if (!session) return { valid: false };

  return {
    valid: true,
    userId: session.userId,
    email: session.email,
    role: session.role as UserRole,
  };
}

/**
 * Renew (slide) a session's expiry.
 */
export function slideSession(token: string) {
  return renewSession(token);
}

/**
 * Logout: delete a single session.
 */
export function logoutSession(token: string, email: string, ip: string): void {
  deleteSession(token);
  logAuditEvent(email, AUDIT_ACTIONS.LOGOUT, ip);
}

/**
 * Logout all sessions for a user.
 */
export function logoutAllSessions(userId: string, email: string, ip: string): number {
  const count = deleteAllUserSessions(userId);
  logAuditEvent(email, AUDIT_ACTIONS.LOGOUT_ALL, ip, { sessionsInvalidated: count });
  return count;
}

// --- Exports ---

export { SESSION_EXPIRY_SECONDS };
export { CSRF_COOKIE_NAME };
