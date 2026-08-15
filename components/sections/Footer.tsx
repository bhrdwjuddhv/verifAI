'use client';

import React from 'react';
import { Shield, Github, Twitter, Disc as Discord, BookOpen } from 'lucide-react';

export const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-ink-950 text-white border-t border-ink-800/80 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-8 pb-10 border-b border-ink-800/60">
          {/* Brand */}
          <div className="md:col-span-5 space-y-3">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-lg bg-brand-blue-400 text-ink-950 flex items-center justify-center">
                <Shield className="w-4 h-4 fill-current" />
              </div>
              <span className="font-semibold text-lg tracking-tight">
                Verif<span className="text-brand-blue-400">AI</span>
              </span>
            </div>
            <p className="text-xs text-ink-400 max-w-sm leading-relaxed font-normal">
              Open-source multi-modal AI deepfake detection and digital forensics platform for images, video, and audio.
            </p>
            <p className="text-[11px] text-brand-blue-300 font-mono">
              100% free & open-source software stack.
            </p>
          </div>

          {/* Quick Nav */}
          <div className="md:col-span-4 grid grid-cols-2 gap-6 text-xs">
            <div>
              <h4 className="font-semibold text-white uppercase tracking-wider text-[11px] mb-2.5">Platform</h4>
              <ul className="space-y-2 text-ink-400">
                <li><a href="#how-it-works" className="hover:text-brand-blue-300 transition-colors">How It Works</a></li>
                <li><a href="#architecture" className="hover:text-brand-blue-300 transition-colors">Architecture</a></li>
                <li><a href="#pipeline" className="hover:text-brand-blue-300 transition-colors">Model Pipeline</a></li>
                <li><a href="/model-card" className="hover:text-brand-blue-300 transition-colors">Model Card</a></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-white uppercase tracking-wider text-[11px] mb-2.5">Ecosystem</h4>
              <ul className="space-y-2 text-ink-400">
                <li><a href="#tech-stack" className="hover:text-brand-blue-300 transition-colors">Tech Stack</a></li>
                <li><a href="#three-ways-in" className="hover:text-brand-blue-300 transition-colors">For Teams</a></li>
                <li><a href="https://c2pa.org" target="_blank" rel="noreferrer" className="hover:text-brand-blue-300 transition-colors">C2PA Specs</a></li>
              </ul>
            </div>
          </div>

          {/* Connect */}
          <div className="md:col-span-3 space-y-3">
            <h4 className="font-semibold text-[11px] text-white uppercase tracking-wider">Connect & Source</h4>
            <div className="flex items-center gap-2.5">
              <a
                href="https://github.com"
                target="_blank"
                rel="noreferrer"
                aria-label="GitHub Repository"
                className="p-2 rounded-xl glass-card text-ink-300 hover:text-white hover:border-brand-blue-400 transition-colors"
              >
                <Github className="w-4 h-4" />
              </a>
              <a
                href="https://twitter.com"
                target="_blank"
                rel="noreferrer"
                aria-label="Twitter X Account"
                className="p-2 rounded-xl glass-card text-ink-300 hover:text-white hover:border-brand-blue-400 transition-colors"
              >
                <Twitter className="w-4 h-4" />
              </a>
              <a
                href="https://discord.com"
                target="_blank"
                rel="noreferrer"
                aria-label="Discord Community"
                className="p-2 rounded-xl glass-card text-ink-300 hover:text-white hover:border-brand-blue-400 transition-colors"
              >
                <Discord className="w-4 h-4" />
              </a>
              <a
                href="https://c2pa.org"
                target="_blank"
                rel="noreferrer"
                aria-label="C2PA Specification Docs"
                className="p-2 rounded-xl glass-card text-ink-300 hover:text-white hover:border-brand-blue-400 transition-colors"
              >
                <BookOpen className="w-4 h-4" />
              </a>
            </div>
          </div>
        </div>

        {/* Copyright */}
        <div className="pt-6 flex flex-col sm:flex-row items-center justify-between text-xs text-ink-400 gap-3 font-normal">
          <p>© {currentYear} VerifAI Open Source Project. MIT License.</p>
          <p className="font-mono text-ink-400 text-[11px]">VerifAI Core v1.4.0-release</p>
        </div>
      </div>
    </footer>
  );
};
