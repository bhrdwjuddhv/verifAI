// ──────────────────────────────────────────────────────────────
// app/admin/eval/page.tsx — Server Component Entry for Evaluation Portal
// ──────────────────────────────────────────────────────────────

export const dynamic = 'force-dynamic';

import AdminEvalClient from '@/components/admin/AdminEvalClient';

export default function AdminEvalPage() {
  return <AdminEvalClient />;
}
