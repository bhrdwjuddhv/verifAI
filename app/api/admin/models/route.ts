import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { verifySessionToken, COOKIE_NAME } from '@/lib/admin/auth';
import { hasPermission, type UserRole } from '@/lib/admin/users';
import { logAuditEvent, AUDIT_ACTIONS } from '@/lib/admin/audit';
import { logError, generateRequestId, safeErrorResponse } from '@/lib/admin/logger';

interface ModelCheckpointInfo {
  filename: string;
  path?: string; // Only included for owner role
  sizeBytes: number;
  updatedAt: string;
  type: 'pth' | 'onnx' | 'other';
  metadata?: any;
}

export async function GET(req: NextRequest) {
  const requestId = generateRequestId();

  try {
    // Determine caller's role for field-level access control
    const token = req.cookies.get(COOKIE_NAME)?.value;
    let callerRole: UserRole = 'viewer';
    if (token) {
      const session = await verifySessionToken(token);
      if (session.valid && session.role) {
        callerRole = session.role;
      }
    }

    const searchDirs = [
      path.resolve('models'),
      path.resolve('public/models'),
      path.resolve('runs'),
    ];

    const checkpoints: ModelCheckpointInfo[] = [];

    for (const dir of searchDirs) {
      if (!fs.existsSync(dir)) continue;

      const files = fs.readdirSync(dir, { recursive: true }) as string[];
      for (const relFile of files) {
        const fullPath = path.join(dir, relFile);
        if (!fs.existsSync(fullPath) || fs.statSync(fullPath).isDirectory()) continue;

        const ext = path.extname(relFile).toLowerCase();
        if (ext === '.pth' || ext === '.onnx') {
          const stats = fs.statSync(fullPath);
          let metadata: any = null;

          if (ext === '.pth') {
            try {
              const pyCode = `import torch, json, sys; sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None; ckpt=torch.load(r'''${fullPath}''', map_location='cpu', weights_only=False); print(json.dumps({k: (v if not hasattr(v, 'tolist') else v.tolist()) for k,v in ckpt.items() if k != 'state_dict'}))`;
              const { execFileSync } = require('child_process');
              const output = execFileSync('python', ['-c', pyCode], { timeout: 4000, encoding: 'utf-8' }).trim();
              metadata = JSON.parse(output);
            } catch {
              metadata = { note: 'No metadata or python torch unavailable to inspect checkpoint' };
            }
          }

          const entry: ModelCheckpointInfo = {
            filename: path.basename(relFile),
            sizeBytes: stats.size,
            updatedAt: stats.mtime.toISOString(),
            type: ext === '.pth' ? 'pth' : 'onnx',
            metadata,
          };

          // Only expose full filesystem paths to owner role
          if (callerRole === 'owner') {
            entry.path = fullPath.replace(/\\/g, '/');
          }

          // Strip potentially identifying metadata for non-owner roles
          if (callerRole !== 'owner' && metadata) {
            // Remove any fields that might contain file paths or personal info
            delete metadata.data_dir;
            delete metadata.output_dir;
            delete metadata.dataset_path;
          }

          checkpoints.push(entry);
        }
      }
    }

    return NextResponse.json({ checkpoints }, { status: 200 });
  } catch (err) {
    logError('list-models', err, requestId);
    return NextResponse.json(safeErrorResponse(requestId), { status: 500 });
  }
}

/**
 * POST /api/admin/models — Promote a checkpoint (owner only).
 * Body: { filename: string }
 */
export async function POST(req: NextRequest) {
  const requestId = generateRequestId();

  try {
    const token = req.cookies.get(COOKIE_NAME)?.value;
    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const session = await verifySessionToken(token);
    if (!session.valid || session.role !== 'owner') {
      return NextResponse.json(
        { error: 'Forbidden. Owner role required to promote checkpoints.' },
        { status: 403 }
      );
    }

    const body = await req.json();
    const { filename } = body;

    if (!filename) {
      return NextResponse.json({ error: 'filename is required.' }, { status: 400 });
    }

    // Sanitize filename to prevent path traversal
    const safeFilename = path.basename(filename);
    if (safeFilename !== filename) {
      return NextResponse.json(
        { error: 'Invalid filename. Path traversal is not allowed.' },
        { status: 400 }
      );
    }

    const ip = req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() || '127.0.0.1';

    logAuditEvent(session.email!, AUDIT_ACTIONS.CHECKPOINT_PROMOTED, ip, {
      filename: safeFilename,
    });

    return NextResponse.json(
      {
        success: true,
        message: `Checkpoint "${safeFilename}" marked for promotion. Manual deployment required.`,
      },
      { status: 200 }
    );
  } catch (err) {
    logError('promote-checkpoint', err, requestId);
    return NextResponse.json(safeErrorResponse(requestId), { status: 500 });
  }
}
