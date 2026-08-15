import { create } from 'zustand';
import { VerdictInfo } from './verdict';

export type ScanStatus = 'idle' | 'uploading' | 'analyzing' | 'complete' | 'error';

/** Everything here is measured. A field is null when it was not measured — never filled in. */
export interface ScanSignals {
  /** Model P(AI-generated or manipulated), 0-100. Null when the model did not apply. */
  modelScore: number | null;
  /** Share of FFT energy above half-Nyquist, 0-100. Descriptive statistic, not a probability. */
  frequencyScore: number | null;
  faceDetected: boolean | null;
}

/**
 * Container metadata. A SUPPORTING SIGNAL shown next to the verdict — never part of it.
 * Metadata is trivially stripped or forged, and every social network strips it from real
 * photos too, so its absence means nothing.
 */
export interface MetadataSignal {
  c2paManifestPresent: boolean;
  exifPresent: boolean;
}

export interface ScanResult {
  id: string;
  filename: string;
  fileType: 'image' | 'video' | 'audio' | 'url';
  fileSize: string;
  /** Likelihood the file is real, 0-100 (100 - modelScore). Null when the model did not apply. */
  score: number | null;
  /** Model confidence in the reported verdict, 0-100. */
  confidence: number;
  verdict: VerdictInfo;
  reasons: string[];
  signals: ScanSignals;
  metadata: MetadataSignal;
  /** "trained_checkpoint" or "hf_fallback:<id>" — which model actually produced this. */
  modelSource: string;
  /** Grad-CAM overlay as a data URL, when the active model supports it. */
  heatmap: string | null;
  /** Caveats from the model service. Display them. */
  notes: string[];
  timestamp: string;
}

export interface NodeTooltipData {
  id: string;
  title: string;
  layer: string;
  tech: string;
  description: string;
  metrics?: string;
}

interface AppState {
  // Scan State
  scanStatus: ScanStatus;
  progress: number;
  activeFile: File | null;
  activeUrl: string;
  scanResult: ScanResult | null;
  scanError: string | null;
  
  // Handlers
  setScanStatus: (status: ScanStatus) => void;
  setProgress: (progress: number) => void;
  setActiveFile: (file: File | null) => void;
  setActiveUrl: (url: string) => void;
  setScanResult: (result: ScanResult | null) => void;
  setScanError: (error: string | null) => void;
  resetScan: () => void;
  
  // UI State
  activeNavSection: string;
  setActiveNavSection: (section: string) => void;
  mobileMenuOpen: boolean;
  setMobileMenuOpen: (open: boolean) => void;
  
  // Diagram Tooltips State
  selectedNode: NodeTooltipData | null;
  setSelectedNode: (node: NodeTooltipData | null) => void;
  
  // Persona Toggle State (§5.9)
  persona: 'users' | 'newsrooms';
  setPersona: (persona: 'users' | 'newsrooms') => void;
  
  // Tech Stack Filter State (§5.7)
  techFilter: string;
  setTechFilter: (filter: string) => void;
  
  // Toast State
  toastMessage: string | null;
  showToast: (msg: string) => void;
  hideToast: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  // Scan State
  scanStatus: 'idle',
  progress: 0,
  activeFile: null,
  activeUrl: '',
  scanResult: null,
  scanError: null,

  setScanStatus: (scanStatus) => set({ scanStatus }),
  setProgress: (progress) => set({ progress }),
  setActiveFile: (activeFile) => set({ activeFile, activeUrl: '' }),
  setActiveUrl: (activeUrl) => set({ activeUrl, activeFile: null }),
  setScanResult: (scanResult) => set({ scanResult }),
  setScanError: (scanError) => set({ scanError }),
  resetScan: () => set({ scanStatus: 'idle', progress: 0, activeFile: null, activeUrl: '', scanResult: null, scanError: null }),

  // UI State
  activeNavSection: 'hero',
  setActiveNavSection: (activeNavSection) => set({ activeNavSection }),
  mobileMenuOpen: false,
  setMobileMenuOpen: (mobileMenuOpen) => set({ mobileMenuOpen }),

  // Diagram Tooltips
  selectedNode: null,
  setSelectedNode: (selectedNode) => set({ selectedNode }),

  // Persona Toggle
  persona: 'users',
  setPersona: (persona) => set({ persona }),

  // Tech Stack Filter
  techFilter: 'all',
  setTechFilter: (techFilter) => set({ techFilter }),

  // Toast
  toastMessage: null,
  showToast: (msg) => {
    set({ toastMessage: msg });
    setTimeout(() => {
      set({ toastMessage: null });
    }, 3000);
  },
  hideToast: () => set({ toastMessage: null }),
}));