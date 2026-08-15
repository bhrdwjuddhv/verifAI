import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { verifySessionToken, COOKIE_NAME } from '@/lib/admin/auth';
import { hasPermission } from '@/lib/admin/users';
import { logAuditEvent, AUDIT_ACTIONS } from '@/lib/admin/audit';
import { logError, logAlert, generateRequestId, safeErrorResponse } from '@/lib/admin/logger';

export async function POST(req: NextRequest) {
  const requestId = generateRequestId();

  try {
    // --- RBAC: require 'run_training' permission (owner or trainer) ---
    const token = req.cookies.get(COOKIE_NAME)?.value;
    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const session = await verifySessionToken(token);
    if (!session.valid || !session.role) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    if (!hasPermission(session.role, 'run_training')) {
      return NextResponse.json(
        { error: 'Forbidden. Your role does not have permission to start training runs.' },
        { status: 403 }
      );
    }

    const body = await req.json().catch(() => ({}));
    const scriptType = body.scriptType || 'image'; // 'image' or 'audio'
    const customArgs: string[] = body.args || [];

    const runId = `run_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
    const runDir = path.resolve('runs', runId);
    fs.mkdirSync(runDir, { recursive: true });

    const logPath = path.join(runDir, 'log.txt');
    const runJsonPath = path.join(runDir, 'run.json');

    const runData: {
      id: string;
      scriptType: string;
      status: string;
      startTime: string;
      endTime: string | null;
      exitCode: number | null;
      args: string[];
      startedBy: string;
    } = {
      id: runId,
      scriptType,
      status: 'running',
      startTime: new Date().toISOString(),
      endTime: null,
      exitCode: null,
      args: customArgs,
      startedBy: session.email || 'unknown',
    };

    fs.writeFileSync(runJsonPath, JSON.stringify(runData, null, 2));

    const scriptFile = scriptType === 'audio' ? 'scripts/train_audio_detector.py' : 'scripts/train_deepfake_detector.py';
    const logStream = fs.createWriteStream(logPath, { flags: 'a' });

    logStream.write(`[SYSTEM] Starting ${scriptType} training run ${runId}...\n`);
    logStream.write(`[SYSTEM] Started by: ${session.email}\n`);
    logStream.write(`[SYSTEM] Command: python ${scriptFile} ${customArgs.join(' ')}\n\n`);

    const child = spawn('python', [scriptFile, ...customArgs], {
      cwd: path.resolve('.'),
      detached: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    child.stdout.on('data', (data) => {
      logStream.write(data);
    });

    child.stderr.on('data', (data) => {
      logStream.write(data);
    });

    child.on('close', (code) => {
      logStream.write(`\n[SYSTEM] Process finished with exit code ${code}\n`);
      logStream.end();

      runData.status = code === 0 ? 'completed' : 'failed';
      runData.endTime = new Date().toISOString();
      (runData as any).exitCode = code;
      fs.writeFileSync(runJsonPath, JSON.stringify(runData, null, 2));

      if (code !== 0) {
        logAlert(`Training run ${runId} (${scriptType}) crashed with exit code ${code}`, {
          runId,
          scriptType,
          startedBy: session.email,
        });
      }
    });

    child.unref();

    // Audit log
    const ip = req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() || '127.0.0.1';
    logAuditEvent(session.email!, AUDIT_ACTIONS.TRAINING_STARTED, ip, {
      runId,
      scriptType,
      args: customArgs,
    });

    return NextResponse.json({ success: true, runId, runDir: `runs/${runId}` }, { status: 200 });
  } catch (err) {
    logError('start-training', err, requestId);
    return NextResponse.json(safeErrorResponse(requestId), { status: 500 });
  }
}

export async function GET(req: NextRequest) {
  const requestId = generateRequestId();

  try {
    // All authenticated users can view runs
    const runsDir = path.resolve('runs');
    if (!fs.existsSync(runsDir)) {
      return NextResponse.json({ runs: [] });
    }

    const entries = fs.readdirSync(runsDir);
    const runs: any[] = [];

    for (const entry of entries) {
      const jsonPath = path.join(runsDir, entry, 'run.json');
      if (fs.existsSync(jsonPath)) {
        try {
          const runJson = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
          runs.push(runJson);
        } catch {
          // Ignore invalid json
        }
      }
    }

    runs.sort((a, b) => new Date(b.startTime).getTime() - new Date(a.startTime).getTime());
    return NextResponse.json({ runs }, { status: 200 });
  } catch (err) {
    logError('list-runs', err, requestId);
    return NextResponse.json(safeErrorResponse(requestId), { status: 500 });
  }
}
