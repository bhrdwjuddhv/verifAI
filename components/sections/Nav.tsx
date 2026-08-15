'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Shield, Menu, X, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { useAppStore } from '@/lib/store';
import { cn } from '@/lib/utils';

export interface NavLink {
  label: string;
  href: string;
}

const NAV_LINKS: NavLink[] = [
  { label: 'How It Works', href: '#how-it-works' },
  { label: 'Architecture', href: '#architecture' },
  { label: 'Pipeline', href: '#pipeline' },
  { label: 'Tech Stack', href: '#tech-stack' },
  { label: 'For Teams', href: '#three-ways-in' },
];

export const Nav: React.FC = () => {
  const [scrolled, setScrolled] = useState(false);
  const { mobileMenuOpen, setMobileMenuOpen, activeNavSection, setActiveNavSection } = useAppStore();

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);

      const sections = NAV_LINKS.map((link) => link.href.substring(1));
      for (const sectionId of sections) {
        const el = document.getElementById(sectionId);
        if (el) {
          const rect = el.getBoundingClientRect();
          if (rect.top <= 200 && rect.bottom >= 100) {
            setActiveNavSection(sectionId);
            break;
          }
        }
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [setActiveNavSection]);

  const scrollToDemo = () => {
    setMobileMenuOpen(false);
    const demoEl = document.getElementById('live-scan-demo');
    if (demoEl) {
      demoEl.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <header
      className={cn(
        'fixed top-0 inset-x-0 z-50 transition-all duration-300',
        scrolled
          ? 'glass py-3 shadow-dark-glass'
          : 'bg-transparent py-5'
      )}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-between">
        {/* Logo */}
        <a
          href="#"
          onClick={(e) => {
            e.preventDefault();
            window.scrollTo({ top: 0, behavior: 'smooth' });
          }}
          className="flex items-center gap-2 group focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-blue-400 rounded-lg p-1"
        >
          <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-brand-blue-300 to-brand-blue-500 flex items-center justify-center text-ink-950 shadow-glow-sm group-hover:scale-105 transition-transform">
            <Shield className="w-4 h-4 fill-current" />
          </div>
          <span className="font-semibold text-lg tracking-tight text-white flex items-center gap-0.5">
            Verif<span className="text-brand-blue-400">AI</span>
          </span>
        </a>

        {/* Desktop Links */}
        <nav className="hidden md:flex items-center gap-1 glass-card p-1 rounded-full border border-ink-800">
          {NAV_LINKS.map((link) => {
            const isActive = activeNavSection === link.href.substring(1);
            return (
              <a
                key={link.label}
                href={link.href}
                className={cn(
                  'px-3.5 py-1 text-xs font-medium rounded-full transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-blue-400 select-none',
                  isActive
                    ? 'bg-ink-800 text-brand-blue-300 shadow-sm font-semibold'
                    : 'text-ink-300 hover:text-white'
                )}
              >
                {link.label}
              </a>
            );
          })}
        </nav>

        {/* Action Button */}
        <div className="hidden md:flex items-center gap-3">
          <Button
            variant="secondary"
            size="sm"
            href="/eval"
          >
            Eval Harness
          </Button>
          <Button
            variant="primary"
            size="sm"
            onClick={scrollToDemo}
            rightIcon={<ArrowRight className="w-3.5 h-3.5" />}
          >
            Try Demo
          </Button>
        </div>

        {/* Mobile Hamburger Toggle */}
        <button
          type="button"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          aria-label="Toggle Navigation Menu"
          aria-expanded={mobileMenuOpen}
          className="md:hidden p-2 rounded-xl text-ink-300 hover:bg-ink-900 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-blue-400 min-h-[44px] min-w-[44px] flex items-center justify-center"
        >
          {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {/* Mobile Drawer */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden glass border-b border-ink-800 px-6 py-5"
          >
            <div className="flex flex-col space-y-2">
              {NAV_LINKS.map((link) => (
                <a
                  key={link.label}
                  href={link.href}
                  onClick={() => setMobileMenuOpen(false)}
                  className="px-4 py-2.5 rounded-xl text-sm font-medium text-ink-200 hover:bg-ink-900 transition-colors"
                >
                  {link.label}
                </a>
              ))}
              <div className="pt-3 border-t border-ink-800">
                <Button
                  variant="primary"
                  size="md"
                  onClick={scrollToDemo}
                  className="w-full"
                >
                  Try Live Scan Demo
                </Button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
};
