// 'suspicious' is kept for the admin evaluation harness (lib/eval/*), which labels
// human-judged samples. The live scanner only ever returns genuine / uncertain / manipulated.
export type VerdictCategory = 'genuine' | 'suspicious' | 'uncertain' | 'manipulated';

export interface VerdictInfo {
  category: VerdictCategory;
  label: string;
  shortLabel: string;
  badgeBg: string;
  badgeText: string;
  badgeBorder: string;
  ringColor: string;
  glowColor: string;
  description: string;
  recommendation: string;
  laymanSummary: string;
}

export const VERDICT_CONFIG: Record<VerdictCategory, VerdictInfo> = {
  genuine: {
    category: 'genuine',
    label: 'Real',
    shortLabel: 'Real',
    badgeBg: 'bg-emerald-500/10',
    badgeText: 'text-emerald-400',
    badgeBorder: 'border-emerald-500/30',
    ringColor: '#10B981',
    glowColor: 'rgba(16, 185, 129, 0.25)',
    description: 'The detector scored this below the AI-generated threshold. That is one model’s opinion on one image, not proof of authenticity.',
    recommendation: 'No sign of manipulation was found. A detector missing something is always possible.',
    laymanSummary: 'The model scored this as likely real.',
  },
  suspicious: {
    category: 'suspicious',
    label: 'Edited or Modified',
    shortLabel: 'Edited',
    badgeBg: 'bg-amber-500/10',
    badgeText: 'text-amber-400',
    badgeBorder: 'border-amber-500/30',
    ringColor: '#F59E0B',
    glowColor: 'rgba(245, 158, 11, 0.25)',
    description: 'Some parts of this image look modified or re-saved using photo editing tools like Photoshop. Exercise caution.',
    recommendation: 'Be careful before sharing. While it might not be a total deepfake, it shows clear signs of digital editing or missing camera information.',
    laymanSummary: 'This image appears to have been edited or retouched. We found signs of image re-compression and software editing tags (like Photoshop), but it might not be 100% fake.',
  },
  uncertain: {
    category: 'uncertain',
    label: 'Uncertain',
    shortLabel: 'Uncertain',
    badgeBg: 'bg-slate-500/10',
    badgeText: 'text-slate-300',
    badgeBorder: 'border-slate-400/30',
    ringColor: '#94A3B8',
    glowColor: 'rgba(148, 163, 184, 0.25)',
    description: 'The detector’s score landed between the two thresholds, or the model did not apply to this file. There is not enough signal to call it either way.',
    recommendation: 'Treat this as “unknown”, not as “probably fine”. Verify the source another way.',
    laymanSummary: 'The model could not decide. This is an honest “don’t know”, not a pass.',
  },
  manipulated: {
    category: 'manipulated',
    label: 'AI-Generated or Manipulated',
    shortLabel: 'AI / Manipulated',
    badgeBg: 'bg-rose-500/10',
    badgeText: 'text-rose-400',
    badgeBorder: 'border-rose-500/30',
    ringColor: '#EF4444',
    glowColor: 'rgba(239, 68, 68, 0.25)',
    description: 'The detector scored this above the AI-generated threshold. It cannot tell you which tool made it, or which region was altered beyond the heatmap below.',
    recommendation: 'Do not treat this file as authentic without a second, independent check.',
    laymanSummary: 'The model scored this as likely AI-generated or manipulated.',
  },
};

/** Bands mirror the model service: >=70 likely real, <=30 likely fake, middle = uncertain. */
export function getVerdict(score: number): VerdictInfo {
  if (score >= 70) {
    return VERDICT_CONFIG.genuine;
  }
  if (score > 30) {
    return VERDICT_CONFIG.uncertain;
  }
  return VERDICT_CONFIG.manipulated;
}

export interface ScoringLegendItem {
  range: string;
  category: VerdictCategory;
  label: string;
  badgeStyle: string;
}

export const SCORING_LEGEND: ScoringLegendItem[] = [
  {
    range: '70 – 100',
    category: 'genuine',
    label: 'Real',
    badgeStyle: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
  },
  {
    range: '31 – 69',
    category: 'uncertain',
    label: 'Uncertain',
    badgeStyle: 'bg-slate-500/10 text-slate-300 border-slate-400/30',
  },
  {
    range: '0 – 30',
    category: 'manipulated',
    label: 'AI-Generated or Manipulated',
    badgeStyle: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
  },
];