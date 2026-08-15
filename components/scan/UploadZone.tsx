'use client';

import React, { useState, useRef } from 'react';
import { UploadCloud, Link as LinkIcon, AlertCircle, FileText } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { useAppStore } from '@/lib/store';
import { cn } from '@/lib/utils';

export interface UploadZoneProps {
  onStartScan: (file: File | null, url: string) => void;
  className?: string;
}

export const UploadZone: React.FC<UploadZoneProps> = ({ onStartScan, className }) => {
  const { activeFile, setActiveFile, activeUrl, setActiveUrl } = useAppStore();
  const [isDragging, setIsDragging] = useState(false);
  const [mode, setMode] = useState<'file' | 'url'>('file');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndSetFile = (file: File) => {
    setErrorMsg(null);
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/gif', 'video/mp4', 'video/webm', 'audio/mpeg', 'audio/wav', 'audio/mp3'];
    if (!validTypes.some((t) => file.type.startsWith(t.split('/')[0]))) {
      setErrorMsg('Unsupported format. Please upload an image, video, or audio file.');
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      setErrorMsg('File size exceeds 50MB limit.');
      return;
    }
    setActiveFile(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (mode === 'file' && activeFile) {
      onStartScan(activeFile, '');
    } else if (mode === 'url' && activeUrl.trim()) {
      if (!activeUrl.startsWith('http://') && !activeUrl.startsWith('https://')) {
        setErrorMsg('Please enter a valid URL starting with http:// or https://');
        return;
      }
      onStartScan(null, activeUrl.trim());
    } else {
      setErrorMsg('Please select a file or enter a media URL.');
    }
  };

  return (
    <div className={cn('w-full', className)}>
      {/* Mode Switcher */}
      <div className="flex items-center gap-2 mb-4">
        <button
          type="button"
          onClick={() => { setMode('file'); setErrorMsg(null); }}
          className={cn(
            'px-3.5 py-1.5 text-xs font-semibold rounded-xl transition-all flex items-center gap-1.5 select-none',
            mode === 'file'
              ? 'bg-brand-blue-400 text-ink-950 shadow-sm'
              : 'glass-card text-ink-300 hover:text-white'
          )}
        >
          <UploadCloud className="w-3.5 h-3.5 fill-brand-blue-400/20 stroke-[2.2]" />
          File Upload
        </button>
        <button
          type="button"
          onClick={() => { setMode('url'); setErrorMsg(null); }}
          className={cn(
            'px-3.5 py-1.5 text-xs font-semibold rounded-xl transition-all flex items-center gap-1.5 select-none',
            mode === 'url'
              ? 'bg-brand-blue-400 text-ink-950 shadow-sm'
              : 'glass-card text-ink-300 hover:text-white'
          )}
        >
          <LinkIcon className="w-3.5 h-3.5 stroke-[2.2]" />
          Paste URL
        </button>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {mode === 'file' ? (
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={cn(
              'relative rounded-2xl border-2 border-dashed p-8 text-center transition-all cursor-pointer select-none flex flex-col items-center justify-center min-h-[200px]',
              isDragging
                ? 'border-brand-blue-400 bg-brand-blue-500/10 scale-[1.01]'
                : 'border-ink-700/80 bg-ink-900/60 hover:border-brand-blue-400/60 hover:bg-ink-900/90'
            )}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,video/*,audio/*"
              className="hidden"
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  validateAndSetFile(e.target.files[0]);
                }
              }}
            />

            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-brand-blue-400/35 via-brand-blue-500/20 to-transparent border border-brand-blue-400/40 flex items-center justify-center text-brand-blue-300 mb-3 shadow-glow-sm">
              <UploadCloud className="w-6 h-6 fill-brand-blue-400/20 stroke-[2]" />
            </div>

            {activeFile ? (
              <div className="flex items-center gap-2 text-sm font-semibold text-brand-blue-300">
                <FileText className="w-4 h-4 stroke-[2]" />
                <span>{activeFile.name}</span>
                <span className="text-xs text-ink-400 font-normal">
                  ({(activeFile.size / (1024 * 1024)).toFixed(2)} MB)
                </span>
              </div>
            ) : (
              <>
                <p className="text-sm font-semibold text-white">
                  Drag & drop media file here, or{' '}
                  <span className="text-brand-blue-300 underline">browse</span>
                </p>
                <p className="text-xs text-ink-400 mt-1 font-normal">
                  Supports JPG, PNG, WEBP, MP4, WEBM, MP3, WAV (Max 50MB)
                </p>
              </>
            )}
          </div>
        ) : (
          <div className="space-y-2">
            <label className="text-xs font-medium text-ink-300 block">
              Enter Public Media URL (Image/Video/Audio)
            </label>
            <div className="relative">
              <input
                type="url"
                value={activeUrl}
                onChange={(e) => setActiveUrl(e.target.value)}
                placeholder="https://example.com/suspicious_media.mp4"
                className="w-full px-4 py-3 rounded-xl bg-ink-900 border border-ink-700 text-white placeholder-ink-500 text-sm focus:outline-none focus:border-brand-blue-400 focus:ring-1 focus:ring-brand-blue-400"
              />
              <LinkIcon className="absolute right-3.5 top-3.5 w-4 h-4 text-ink-400 stroke-[2]" />
            </div>
          </div>
        )}

        {/* Consent & Privacy Retention Notice */}
        <p className="text-[11px] text-ink-400 bg-ink-900/40 p-2.5 rounded-xl border border-ink-800/60 leading-normal">
          <span className="font-semibold text-brand-blue-300">Privacy & Acceptable Use:</span> Files are processed in memory only and are <strong className="text-white">not stored</strong> on our servers. Please ensure you have consent from any depicted individuals before scanning media.
        </p>

        {errorMsg && (
          <div className="flex items-center gap-2 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs">
            <AlertCircle className="w-4 h-4 flex-shrink-0 stroke-[2]" />
            <span>{errorMsg}</span>
          </div>
        )}

        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-2">
          {/* The sample buttons that used to sit here shipped fixed scores with fabricated
              filenames — they only "worked" because detection was filename matching. */}
          <p className="text-[11px] text-ink-400 font-normal">
            Upload a photo containing a face. Video and audio are not supported yet.
          </p>

          <Button
            type="submit"
            variant="primary"
            size="md"
            className="w-full sm:w-auto"
          >
            Start Scan
          </Button>
        </div>
      </form>
    </div>
  );
};
