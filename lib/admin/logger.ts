/**
 * Structured server-side logging utilities.
 * - Never logs API keys, session tokens, or file contents.
 * - Uses [ALERT] tag for critical events that need attention.
 * - Uses [ERROR] tag for unexpected failures.
 */

export function logError(context: string, error: unknown, requestId?: string): void {
  const message = error instanceof Error ? error.message : String(error);
  const stack = error instanceof Error ? error.stack : undefined;

  // Never log full stack traces that contain file paths in production
  const safeStack = process.env.NODE_ENV === 'production' ? undefined : stack;

  console.error(
    JSON.stringify({
      level: 'ERROR',
      tag: '[ERROR]',
      context,
      message,
      requestId,
      stack: safeStack,
      timestamp: new Date().toISOString(),
    })
  );
}

export function logAlert(message: string, details?: Record<string, unknown>): void {
  console.error(
    JSON.stringify({
      level: 'ALERT',
      tag: '[ALERT]',
      message,
      details,
      timestamp: new Date().toISOString(),
    })
  );
}

export function logInfo(context: string, message: string, details?: Record<string, unknown>): void {
  console.log(
    JSON.stringify({
      level: 'INFO',
      tag: '[INFO]',
      context,
      message,
      details,
      timestamp: new Date().toISOString(),
    })
  );
}

/**
 * Generate a short request ID for correlating logs with client error responses.
 */
export function generateRequestId(): string {
  return `req_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 8)}`;
}

/**
 * Create a safe error response that never leaks internal details to the client.
 */
export function safeErrorResponse(requestId: string): { error: string; requestId: string } {
  return {
    error: 'An internal error occurred. Please try again or contact an administrator.',
    requestId,
  };
}
