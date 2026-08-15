'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import {
  ShieldCheck,
  LogOut,
  Cpu,
  Play,
  AlertTriangle,
  UploadCloud,
  CheckCircle2,
  RefreshCw,
  Terminal,
  Activity,
  Server,
  Layers,
  Database,
  Users,
  ScrollText,
  Crown,
  Wrench,
  Eye,
  Plus,
  Trash2,
  AlertCircle,
  LogOutIcon,
} from 'lucide-react';
import { ScoreRing } from '@/components/scan/ScoreRing';
import { ReportPanel } from '@/components/scan/ReportPanel';

/** Read the CSRF token from the cookie for double-submit pattern */
function getCsrfToken(): string {
  if (typeof document === 'undefined') return '';
  const match = document.cookie.match(/(?:^|;\s*)verifai_csrf_token=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : '';
}

/** Fetch wrapper that auto-attaches CSRF token header */
async function secureFetch(url: string, options: RequestInit = {}): Promise<Response> {
  const csrfToken = getCsrfToken();
  const headers = new Headers(options.headers || {});
  if (['POST', 'PUT', 'DELETE'].includes((options.method || 'GET').toUpperCase())) {
    headers.set('X-CSRF-Token', csrfToken);
  }
  return fetch(url, { ...options, headers });
}

type UserRole = 'owner' | 'trainer' | 'viewer';

const ROLE_CONFIG: Record<UserRole, { label: string; icon: typeof Crown; color: string }> = {
  owner: { label: 'Owner', icon: Crown, color: 'text-amber-400 bg-amber-500/10 border-amber-500/30' },
  trainer: { label: 'Trainer', icon: Wrench, color: 'text-blue-400 bg-blue-500/10 border-blue-500/30' },
  viewer: { label: 'Viewer', icon: Eye, color: 'text-ink-400 bg-ink-500/10 border-ink-500/30' },
};

export default function AdminDashboardPage() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<'scan' | 'models' | 'train' | 'users' | 'audit'>('scan');
  const [currentUser, setCurrentUser] = useState<{ id: string; email: string; role: UserRole } | null>(null);

  // Real Scan state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [scanning, setScanning] = useState(false);
  const [scanResult, setScanResult] = useState<any | null>(null);
  const [scanError, setScanError] = useState<{ error: string; detail?: string; note?: string } | null>(null);

  // Models state
  const [models, setModels] = useState<any[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);

  // Training state
  const [runs, setRuns] = useState<any[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [runLogs, setRunLogs] = useState<string>('');
  const [startingTrain, setStartingTrain] = useState(false);
  const [trainForm, setTrainForm] = useState({
    scriptType: 'image',
    arch: 'b0',
    epochs: 12,
    batchSize: 32,
    lr: 0.0003,
  });

  // Users state (owner only)
  const [users, setUsers] = useState<any[]>([]);
  const [loadingUsers, setLoadingUsers] = useState(false);
  const [newUserForm, setNewUserForm] = useState({ email: '', password: '', role: 'viewer' as UserRole });
  const [userError, setUserError] = useState<string | null>(null);

  // Audit state (owner only)
  const [auditEvents, setAuditEvents] = useState<any[]>([]);
  const [loadingAudit, setLoadingAudit] = useState(false);

  // Load current user info from cookie/session on mount
  useEffect(() => {
    // We'll fetch user info from a lightweight endpoint or parse from login response stored in sessionStorage
    const stored = sessionStorage.getItem('verifai_user');
    if (stored) {
      try { setCurrentUser(JSON.parse(stored)); } catch { /* ignore */ }
    }
  }, []);

  const handleLogout = async () => {
    await secureFetch('/api/admin/auth/logout', { method: 'POST' });
    sessionStorage.removeItem('verifai_user');
    router.push('/admin/login');
    router.refresh();
  };

  const handleLogoutAll = async () => {
    if (!confirm('This will invalidate ALL your sessions (including other devices). Continue?')) return;
    await secureFetch('/api/admin/auth/logout-all', { method: 'POST' });
    sessionStorage.removeItem('verifai_user');
    router.push('/admin/login');
    router.refresh();
  };

  const fetchModels = async () => {
    setLoadingModels(true);
    try {
      const res = await fetch('/api/admin/models');
      if (res.ok) {
        const data = await res.json();
        setModels(data.checkpoints || []);
      }
    } catch {
      // silently handle
    } finally {
      setLoadingModels(false);
    }
  };

  const fetchRuns = async () => {
    try {
      const res = await fetch('/api/admin/train');
      if (res.ok) {
        const data = await res.json();
        setRuns(data.runs || []);
      }
    } catch {
      // silently handle
    }
  };

  const fetchLogs = async (runId: string) => {
    try {
      const res = await fetch(`/api/admin/train/${runId}/logs`);
      if (res.ok) {
        const data = await res.json();
        setRunLogs(data.logs || 'No logs available.');
      }
    } catch {
      setRunLogs('Failed to load logs.');
    }
  };

  const fetchUsers = async () => {
    setLoadingUsers(true);
    try {
      const res = await fetch('/api/admin/users');
      if (res.ok) {
        const data = await res.json();
        setUsers(data.users || []);
      }
    } catch {
      // silently handle
    } finally {
      setLoadingUsers(false);
    }
  };

  const fetchAudit = async () => {
    setLoadingAudit(true);
    try {
      const res = await fetch('/api/admin/audit?limit=100');
      if (res.ok) {
        const data = await res.json();
        setAuditEvents(data.events || []);
      }
    } catch {
      // silently handle
    } finally {
      setLoadingAudit(false);
    }
  };

  useEffect(() => {
    fetchModels();
    fetchRuns();
  }, []);

  useEffect(() => {
    if (activeTab === 'users' && currentUser?.role === 'owner') fetchUsers();
    if (activeTab === 'audit' && currentUser?.role === 'owner') fetchAudit();
  }, [activeTab, currentUser?.role]);

  useEffect(() => {
    if (selectedRunId) {
      fetchLogs(selectedRunId);
      const interval = setInterval(() => fetchLogs(selectedRunId), 2000);
      return () => clearInterval(interval);
    }
  }, [selectedRunId]);

  const handleDirectScan = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    setScanning(true);
    setScanResult(null);
    setScanError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const res = await secureFetch('/api/admin/scan', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        setScanError({
          error: data.error || 'Direct Scan Failed',
          detail: data.detail || 'The direct model server responded with an error.',
          note: data.note,
        });
      } else {
        setScanResult(data);
      }
    } catch (err: any) {
      setScanError({
        error: 'Network Failure',
        detail: err.message || 'Could not connect to /api/admin/scan',
      });
    } finally {
      setScanning(false);
    }
  };

  const handleStartTraining = async (e: React.FormEvent) => {
    e.preventDefault();
    setStartingTrain(true);

    try {
      const args = [
        '--arch', trainForm.arch,
        '--epochs', trainForm.epochs.toString(),
        '--batch-size', trainForm.batchSize.toString(),
        '--lr', trainForm.lr.toString(),
      ];

      const res = await secureFetch('/api/admin/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scriptType: trainForm.scriptType,
          args,
        }),
      });

      const data = await res.json();
      if (res.ok && data.runId) {
        setSelectedRunId(data.runId);
        fetchRuns();
      }
    } catch (err) {
      console.error(err);
    } finally {
      setStartingTrain(false);
    }
  };

  const handleCreateUser = async (e: React.FormEvent) => {
    e.preventDefault();
    setUserError(null);

    try {
      const res = await secureFetch('/api/admin/users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newUserForm),
      });

      const data = await res.json();
      if (!res.ok) {
        setUserError(data.error || 'Failed to create user');
        return;
      }

      setNewUserForm({ email: '', password: '', role: 'viewer' });
      fetchUsers();
    } catch (err: any) {
      setUserError(err.message || 'Network error');
    }
  };

  const handleDeleteUser = async (id: string, userEmail: string) => {
    if (!confirm(`Delete user ${userEmail}? This cannot be undone.`)) return;

    try {
      const res = await secureFetch(`/api/admin/users/${id}`, { method: 'DELETE' });
      if (res.ok) fetchUsers();
    } catch {
      // silently handle
    }
  };

  const handleChangeRole = async (id: string, newRole: UserRole) => {
    try {
      const res = await secureFetch(`/api/admin/users/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role: newRole }),
      });
      if (res.ok) fetchUsers();
    } catch {
      // silently handle
    }
  };

  const isOwner = currentUser?.role === 'owner';
  const canTrain = currentUser?.role === 'owner' || currentUser?.role === 'trainer';

  const RoleIcon = currentUser?.role ? ROLE_CONFIG[currentUser.role]?.icon || Eye : Eye;
  const roleColor = currentUser?.role ? ROLE_CONFIG[currentUser.role]?.color || '' : '';

  return (
    <div className="min-h-screen flex flex-col bg-ink-950 text-ink-100">
      {/* Admin Top Header */}
      <header className="border-b border-ink-800 bg-ink-900/60 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-brand-blue-500/10 border border-brand-blue-500/30 flex items-center justify-center text-brand-blue-400">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-bold text-white tracking-wide">VerifAI Admin Portal</h1>
                {currentUser?.role && (
                  <span className={`px-2 py-0.5 text-[10px] uppercase font-bold tracking-wider rounded-md border ${roleColor}`}>
                    <RoleIcon className="w-3 h-3 inline-block mr-1" />
                    {currentUser.role}
                  </span>
                )}
              </div>
              <p className="text-xs text-ink-400">
                {currentUser?.email || 'Authenticated'} &mdash; Direct PyTorch Inference & Training Pipeline
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleLogoutAll}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-ink-700 bg-ink-800/60 hover:bg-rose-900/40 hover:border-rose-500/30 text-xs font-medium text-ink-300 hover:text-rose-300 transition-all"
              title="Invalidate all sessions (all devices)"
            >
              <LogOutIcon className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">All Sessions</span>
            </button>
            <button
              onClick={handleLogout}
              className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg border border-ink-700 bg-ink-800/60 hover:bg-ink-700 text-xs font-medium text-ink-200 hover:text-white transition-all"
            >
              <LogOut className="w-3.5 h-3.5" />
              <span>Sign Out</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Dashboard Body */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-grow w-full">
        {/* Nav Tabs */}
        <div className="flex items-center gap-2 border-b border-ink-800 pb-4 mb-8 flex-wrap">
          <button
            onClick={() => setActiveTab('scan')}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all ${
              activeTab === 'scan'
                ? 'bg-brand-blue-600 text-white shadow-glow-sm'
                : 'text-ink-400 hover:text-white hover:bg-ink-900/60'
            }`}
          >
            <Cpu className="w-4 h-4" />
            <span>Direct Scan</span>
          </button>

          <button
            onClick={() => setActiveTab('models')}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all ${
              activeTab === 'models'
                ? 'bg-brand-blue-600 text-white shadow-glow-sm'
                : 'text-ink-400 hover:text-white hover:bg-ink-900/60'
            }`}
          >
            <Layers className="w-4 h-4" />
            <span>Models</span>
          </button>

          <button
            onClick={() => setActiveTab('train')}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all ${
              activeTab === 'train'
                ? 'bg-brand-blue-600 text-white shadow-glow-sm'
                : 'text-ink-400 hover:text-white hover:bg-ink-900/60'
            }`}
          >
            <Activity className="w-4 h-4" />
            <span>Training</span>
          </button>

          {isOwner && (
            <>
              <button
                onClick={() => setActiveTab('users')}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                  activeTab === 'users'
                    ? 'bg-brand-blue-600 text-white shadow-glow-sm'
                    : 'text-ink-400 hover:text-white hover:bg-ink-900/60'
                }`}
              >
                <Users className="w-4 h-4" />
                <span>Team</span>
              </button>

              <button
                onClick={() => setActiveTab('audit')}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                  activeTab === 'audit'
                    ? 'bg-brand-blue-600 text-white shadow-glow-sm'
                    : 'text-ink-400 hover:text-white hover:bg-ink-900/60'
                }`}
              >
                <ScrollText className="w-4 h-4" />
                <span>Audit Log</span>
              </button>
            </>
          )}
        </div>

        {/* TAB 1: Direct Real Model Scan */}
        {activeTab === 'scan' && (
          <div className="space-y-8">
            <div className="glass-card p-6 rounded-2xl border border-ink-800">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <Server className="w-5 h-5 text-brand-blue-400" />
                    Bypass Heuristic Fallback — Direct FastAPI Model Test
                  </h2>
                  <p className="text-xs text-ink-400 mt-1">
                    Posts to <code className="text-brand-blue-300">$MODEL_SERVICE_URL/predict</code> via <code className="text-brand-blue-300">/api/admin/scan</code>.
                    If the Python server is offline, this mode returns an explicit error instead of falling back to heuristic predictions.
                  </p>
                </div>
                <span className="px-3 py-1 text-xs rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-400 font-medium">
                  Strict Real Inference
                </span>
              </div>

              <form onSubmit={handleDirectScan} className="space-y-4">
                <div className="border-2 border-dashed border-ink-700 rounded-xl p-6 text-center hover:border-brand-blue-500/50 transition-colors bg-ink-900/40">
                  <UploadCloud className="w-8 h-8 text-ink-400 mx-auto mb-2" />
                  <p className="text-sm font-medium text-white">Select sample image for real deepfake inference</p>
                  <p className="text-xs text-ink-500 mt-1 mb-3">Accepts PNG, JPG, JPEG, WEBP up to 50MB</p>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                    className="hidden"
                    id="admin-file-input"
                  />
                  <label
                    htmlFor="admin-file-input"
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-ink-800 hover:bg-ink-700 text-xs font-medium text-white cursor-pointer border border-ink-700 transition-all"
                  >
                    {selectedFile ? selectedFile.name : 'Choose File'}
                  </label>
                </div>

                <button
                  type="submit"
                  disabled={!selectedFile || scanning}
                  className="w-full py-3 rounded-xl bg-brand-blue-600 hover:bg-brand-blue-500 text-white text-sm font-semibold transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {scanning ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span>Running Real PyTorch Model Inference...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" />
                      <span>Execute Direct Model Scan</span>
                    </>
                  )}
                </button>
              </form>
            </div>

            {/* Error Display */}
            {scanError && (
              <div className="p-6 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-300 space-y-3">
                <div className="flex items-center gap-3 text-rose-400 font-bold">
                  <AlertTriangle className="w-6 h-6 flex-shrink-0" />
                  <span className="text-base">{scanError.error}</span>
                </div>
                <p className="text-sm text-rose-200">{scanError.detail}</p>
                {scanError.note && (
                  <p className="text-xs text-rose-300/80 bg-rose-950/40 p-3 rounded-lg border border-rose-800/40">
                    {scanError.note}
                  </p>
                )}
                <div className="pt-2 text-xs text-ink-300">
                  <p className="font-semibold text-white mb-1">To launch the inference server locally:</p>
                  <code className="block bg-ink-950 p-2.5 rounded-lg border border-ink-800 text-emerald-400 font-mono">
                    python scripts/inference_server.py
                  </code>
                </div>
              </div>
            )}

            {/* Direct Model Result Display */}
            {scanResult && (
              <div className="space-y-6">
                <div className="flex items-center justify-between p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
                  <div className="flex items-center gap-2 text-sm font-medium">
                    <CheckCircle2 className="w-4 h-4" />
                    <span>Model response received verbatim</span>
                  </div>
                  <span className="text-xs font-mono text-emerald-300">Model: {scanResult.modelSource}</span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="md:col-span-1 glass-card p-6 rounded-2xl border border-ink-800 flex flex-col items-center justify-center">
                    <ScoreRing score={scanResult.score} verdict={scanResult.verdict} />
                  </div>
                  <div className="md:col-span-2">
                    <ReportPanel result={scanResult} onReset={() => setScanResult(null)} />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 2: Model Checkpoints */}
        {activeTab === 'models' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-bold text-white">Stored Model Checkpoints & Artifacts</h2>
                <p className="text-xs text-ink-400">Trained PyTorch (`.pth`) and ONNX (`.onnx`) models found in codebase</p>
              </div>
              <button
                onClick={fetchModels}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-ink-800 hover:bg-ink-700 text-xs font-medium text-ink-300 hover:text-white border border-ink-700 transition-all"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${loadingModels ? 'animate-spin' : ''}`} />
                <span>Refresh Models</span>
              </button>
            </div>

            {models.length === 0 ? (
              <div className="glass-card p-8 rounded-2xl border border-ink-800 text-center text-ink-400">
                <Database className="w-8 h-8 text-ink-500 mx-auto mb-2" />
                <p className="text-sm font-medium">No checkpoint files found in `models/` or `public/models/`</p>
                <p className="text-xs text-ink-500 mt-1">Run a training job under the &quot;Training&quot; tab to create one.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-4">
                {models.map((ckpt: any, idx: number) => (
                  <div key={idx} className="glass-card p-5 rounded-2xl border border-ink-800 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-bold text-white">{ckpt.filename}</span>
                        <span className={`px-2 py-0.5 text-[10px] uppercase font-bold rounded ${ckpt.type === 'pth' ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30' : 'bg-blue-500/20 text-blue-300 border border-blue-500/30'}`}>
                          {ckpt.type}
                        </span>
                      </div>
                      {ckpt.path && <p className="text-xs font-mono text-ink-400">{ckpt.path}</p>}
                      <p className="text-xs text-ink-500">
                        Size: {(ckpt.sizeBytes / (1024 * 1024)).toFixed(2)} MB | Last Modified: {new Date(ckpt.updatedAt).toLocaleString()}
                      </p>
                    </div>

                    {ckpt.metadata && (
                      <div className="bg-ink-900/80 p-3 rounded-xl border border-ink-700/60 text-xs space-y-1 min-w-[280px]">
                        <p className="font-semibold text-brand-blue-300">Checkpoint Metadata:</p>
                        {ckpt.metadata.arch && <p className="text-ink-300">Arch: <span className="text-white">EfficientNet-{ckpt.metadata.arch.toUpperCase()}</span></p>}
                        {ckpt.metadata.classes && <p className="text-ink-300">Classes: <span className="text-white">{JSON.stringify(ckpt.metadata.classes)}</span></p>}
                        {ckpt.metadata.temperature && <p className="text-ink-300">Temperature (T): <span className="text-white">{ckpt.metadata.temperature.toFixed(3)}</span></p>}
                        {ckpt.metadata.val_metrics?.balanced_acc && (
                          <p className="text-ink-300">Val Balanced Acc: <span className="text-emerald-400 font-bold">{(ckpt.metadata.val_metrics.balanced_acc * 100).toFixed(2)}%</span></p>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* TAB 3: Training & Pipeline Runs */}
        {activeTab === 'train' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Left: Start Training Form */}
            <div className="lg:col-span-1 space-y-6">
              {canTrain ? (
                <div className="glass-card p-6 rounded-2xl border border-ink-800 space-y-4">
                  <h2 className="text-base font-bold text-white flex items-center gap-2">
                    <Play className="w-4 h-4 text-brand-blue-400" />
                    Launch Training Job
                  </h2>

                  <form onSubmit={handleStartTraining} className="space-y-4 text-xs">
                    <div>
                      <label className="block text-ink-300 font-semibold mb-1">Detector Target</label>
                      <select
                        value={trainForm.scriptType}
                        onChange={(e) => setTrainForm({ ...trainForm, scriptType: e.target.value })}
                        className="w-full px-3 py-2 bg-ink-900 border border-ink-700 rounded-lg text-white"
                      >
                        <option value="image">Image Deepfake Detector (EfficientNet)</option>
                        <option value="audio">Audio Deepfake Detector (Spectrogram CNN)</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-ink-300 font-semibold mb-1">Architecture</label>
                      <select
                        value={trainForm.arch}
                        onChange={(e) => setTrainForm({ ...trainForm, arch: e.target.value })}
                        className="w-full px-3 py-2 bg-ink-900 border border-ink-700 rounded-lg text-white"
                      >
                        <option value="b0">EfficientNet-B0 (Fast, 224px)</option>
                        <option value="b4">EfficientNet-B4 (High-res, 380px)</option>
                      </select>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="block text-ink-300 font-semibold mb-1">Epochs</label>
                        <input
                          type="number"
                          value={trainForm.epochs}
                          onChange={(e) => setTrainForm({ ...trainForm, epochs: parseInt(e.target.value, 10) })}
                          className="w-full px-3 py-2 bg-ink-900 border border-ink-700 rounded-lg text-white"
                        />
                      </div>
                      <div>
                        <label className="block text-ink-300 font-semibold mb-1">Batch Size</label>
                        <input
                          type="number"
                          value={trainForm.batchSize}
                          onChange={(e) => setTrainForm({ ...trainForm, batchSize: parseInt(e.target.value, 10) })}
                          className="w-full px-3 py-2 bg-ink-900 border border-ink-700 rounded-lg text-white"
                        />
                      </div>
                    </div>

                    <button
                      type="submit"
                      disabled={startingTrain}
                      className="w-full py-2.5 rounded-xl bg-brand-blue-600 hover:bg-brand-blue-500 text-white font-semibold transition-all disabled:opacity-50 flex items-center justify-center gap-2 text-xs mt-2"
                    >
                      {startingTrain ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                      <span>Spawn Background Training Job</span>
                    </button>
                  </form>
                </div>
              ) : (
                <div className="glass-card p-6 rounded-2xl border border-ink-800 text-center text-ink-400">
                  <Eye className="w-8 h-8 mx-auto mb-2 text-ink-500" />
                  <p className="text-sm font-medium">View-Only Access</p>
                  <p className="text-xs text-ink-500 mt-1">Your role does not allow starting training runs.</p>
                </div>
              )}

              {/* Past Runs List */}
              <div className="glass-card p-5 rounded-2xl border border-ink-800 space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-white">Pipeline Execution History</h3>
                  <button onClick={fetchRuns} className="text-ink-400 hover:text-white">
                    <RefreshCw className="w-3.5 h-3.5" />
                  </button>
                </div>

                <div className="space-y-2 max-h-[300px] overflow-y-auto pr-1">
                  {runs.map((r: any) => (
                    <button
                      key={r.id}
                      onClick={() => setSelectedRunId(r.id)}
                      className={`w-full text-left p-3 rounded-xl border transition-all ${
                        selectedRunId === r.id
                          ? 'bg-brand-blue-500/20 border-brand-blue-400 text-white'
                          : 'bg-ink-900/60 border-ink-800 text-ink-300 hover:bg-ink-800/60'
                      }`}
                    >
                      <div className="flex items-center justify-between text-xs mb-1">
                        <span className="font-mono font-bold">{r.id}</span>
                        <span
                          className={`px-1.5 py-0.5 text-[10px] rounded uppercase font-bold ${
                            r.status === 'running'
                              ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30 animate-pulse'
                              : r.status === 'completed'
                              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                              : 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                          }`}
                        >
                          {r.status}
                        </span>
                      </div>
                      <p className="text-[11px] text-ink-400">
                        Target: {r.scriptType} | Started: {new Date(r.startTime).toLocaleTimeString()}
                        {r.startedBy && ` | By: ${r.startedBy}`}
                      </p>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Right: Live Log Viewer */}
            <div className="lg:col-span-2 glass-card p-6 rounded-2xl border border-ink-800 flex flex-col min-h-[500px]">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Terminal className="w-5 h-5 text-brand-blue-400" />
                  <h3 className="text-base font-bold text-white">
                    {selectedRunId ? `Execution Stream Logs: ${selectedRunId}` : 'Execution Stream Logs'}
                  </h3>
                </div>
                {selectedRunId && (
                  <span className="text-xs font-mono text-ink-400">Polling active</span>
                )}
              </div>

              <div className="flex-grow bg-black/80 rounded-xl p-4 border border-ink-800 font-mono text-xs text-emerald-400 overflow-y-auto max-h-[550px] whitespace-pre-wrap">
                {selectedRunId ? runLogs : 'Select a run from the history list on the left to monitor stdout/stderr logs.'}
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: Team / User Management (Owner Only) */}
        {activeTab === 'users' && isOwner && (
          <div className="space-y-8">
            {/* Create User Form */}
            <div className="glass-card p-6 rounded-2xl border border-ink-800">
              <h2 className="text-lg font-bold text-white flex items-center gap-2 mb-4">
                <Plus className="w-5 h-5 text-brand-blue-400" />
                Add Team Member
              </h2>

              {userError && (
                <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 flex items-center gap-2 text-rose-400 text-sm mb-4">
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  <span>{userError}</span>
                </div>
              )}

              <form onSubmit={handleCreateUser} className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
                <div>
                  <label className="block text-xs text-ink-300 font-semibold mb-1">Email</label>
                  <input
                    type="email"
                    value={newUserForm.email}
                    onChange={(e) => setNewUserForm({ ...newUserForm, email: e.target.value })}
                    placeholder="teammate@company.com"
                    className="w-full px-3 py-2 bg-ink-900 border border-ink-700 rounded-lg text-white text-sm"
                    required
                  />
                </div>
                <div>
                  <label className="block text-xs text-ink-300 font-semibold mb-1">Password</label>
                  <input
                    type="password"
                    value={newUserForm.password}
                    onChange={(e) => setNewUserForm({ ...newUserForm, password: e.target.value })}
                    placeholder="Min 8 characters"
                    className="w-full px-3 py-2 bg-ink-900 border border-ink-700 rounded-lg text-white text-sm"
                    required
                    minLength={8}
                  />
                </div>
                <div>
                  <label className="block text-xs text-ink-300 font-semibold mb-1">Role</label>
                  <select
                    value={newUserForm.role}
                    onChange={(e) => setNewUserForm({ ...newUserForm, role: e.target.value as UserRole })}
                    className="w-full px-3 py-2 bg-ink-900 border border-ink-700 rounded-lg text-white text-sm"
                  >
                    <option value="viewer">Viewer (read-only)</option>
                    <option value="trainer">Trainer (can run training)</option>
                    <option value="owner">Owner (full access)</option>
                  </select>
                </div>
                <button
                  type="submit"
                  className="px-4 py-2 rounded-lg bg-brand-blue-600 hover:bg-brand-blue-500 text-white text-sm font-semibold transition-all"
                >
                  Create Account
                </button>
              </form>
            </div>

            {/* Users List */}
            <div className="glass-card p-6 rounded-2xl border border-ink-800">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <Users className="w-5 h-5 text-brand-blue-400" />
                  Team Members
                </h2>
                <button onClick={fetchUsers} className="text-ink-400 hover:text-white">
                  <RefreshCw className={`w-4 h-4 ${loadingUsers ? 'animate-spin' : ''}`} />
                </button>
              </div>

              <div className="space-y-3">
                {users.map((u: any) => {
                  const rc = ROLE_CONFIG[u.role as UserRole] || ROLE_CONFIG.viewer;
                  const Ic = rc.icon;
                  return (
                    <div key={u.id} className="flex items-center justify-between p-4 rounded-xl bg-ink-900/60 border border-ink-800">
                      <div className="flex items-center gap-3">
                        <div className={`w-8 h-8 rounded-lg flex items-center justify-center border ${rc.color}`}>
                          <Ic className="w-4 h-4" />
                        </div>
                        <div>
                          <p className="text-sm font-medium text-white">{u.email}</p>
                          <p className="text-xs text-ink-500">Created: {new Date(u.createdAt).toLocaleDateString()}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <select
                          value={u.role}
                          onChange={(e) => handleChangeRole(u.id, e.target.value as UserRole)}
                          className="px-2 py-1 bg-ink-900 border border-ink-700 rounded-lg text-xs text-white"
                          disabled={u.id === currentUser?.id}
                        >
                          <option value="viewer">Viewer</option>
                          <option value="trainer">Trainer</option>
                          <option value="owner">Owner</option>
                        </select>
                        <button
                          onClick={() => handleDeleteUser(u.id, u.email)}
                          disabled={u.id === currentUser?.id}
                          className="p-1.5 rounded-lg text-ink-500 hover:text-rose-400 hover:bg-rose-500/10 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
                          title={u.id === currentUser?.id ? 'Cannot delete yourself' : `Delete ${u.email}`}
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* TAB 5: Audit Log (Owner Only) */}
        {activeTab === 'audit' && isOwner && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <ScrollText className="w-5 h-5 text-brand-blue-400" />
                  Recent Activity (Audit Log)
                </h2>
                <p className="text-xs text-ink-400">Append-only log of all admin actions. Not editable from this UI.</p>
              </div>
              <button
                onClick={fetchAudit}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-ink-800 hover:bg-ink-700 text-xs font-medium text-ink-300 hover:text-white border border-ink-700 transition-all"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${loadingAudit ? 'animate-spin' : ''}`} />
                <span>Refresh</span>
              </button>
            </div>

            <div className="glass-card rounded-2xl border border-ink-800 overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="bg-ink-900/80 text-ink-400">
                      <th className="text-left px-4 py-3 font-semibold">Timestamp</th>
                      <th className="text-left px-4 py-3 font-semibold">Actor</th>
                      <th className="text-left px-4 py-3 font-semibold">Action</th>
                      <th className="text-left px-4 py-3 font-semibold">IP</th>
                      <th className="text-left px-4 py-3 font-semibold">Details</th>
                    </tr>
                  </thead>
                  <tbody>
                    {auditEvents.map((event: any, idx: number) => (
                      <tr key={idx} className="border-t border-ink-800/60 hover:bg-ink-900/40">
                        <td className="px-4 py-2.5 text-ink-300 font-mono whitespace-nowrap">
                          {new Date(event.timestamp).toLocaleString()}
                        </td>
                        <td className="px-4 py-2.5 text-white">{event.actor}</td>
                        <td className="px-4 py-2.5">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                            event.action.includes('failed') || event.action.includes('lockout')
                              ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                              : event.action.includes('success') || event.action.includes('created')
                              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                              : 'bg-blue-500/20 text-blue-300 border border-blue-500/30'
                          }`}>
                            {event.action}
                          </span>
                        </td>
                        <td className="px-4 py-2.5 text-ink-400 font-mono">{event.ip}</td>
                        <td className="px-4 py-2.5 text-ink-400 max-w-[300px] truncate">
                          {event.details ? JSON.stringify(event.details) : '—'}
                        </td>
                      </tr>
                    ))}
                    {auditEvents.length === 0 && (
                      <tr>
                        <td colSpan={5} className="px-4 py-8 text-center text-ink-500">No audit events recorded yet.</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
