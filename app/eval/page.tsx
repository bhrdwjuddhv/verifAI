// ──────────────────────────────────────────────────────────────
// app/eval/page.tsx — Direct & Alias Route for Evaluation Portal
// ──────────────────────────────────────────────────────────────

export const dynamic = 'force-dynamic';

import AdminEvalClient from '@/components/admin/AdminEvalClient';

export default function EvalPage() {
  return <AdminEvalClient />;
}
