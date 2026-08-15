import type { Metadata } from 'next';
import './globals.css';
import React from 'react';

export const metadata: Metadata = {
  title: 'VerifAI — Open-Source AI & Deepfake Verification Platform',
  description:
    'A zero-cost, open-source AI + digital-forensics platform that detects manipulated or AI-generated images, video, and voice, giving every piece of content a transparent, explainable trust score.',
  keywords: [
    'deepfake detection',
    'AI image verification',
    'voice clone detector',
    'digital forensics',
    'C2PA provenance',
    'open source AI',
    'Grad-CAM heatmaps',
  ],
  authors: [{ name: 'VerifAI Open Source Team' }],
  openGraph: {
    title: 'VerifAI — Know what is real before you hit share.',
    description:
      'Detect synthetic media in seconds with multi-modal neural networks and digital forensics.',
    type: 'website',
    url: 'https://verifai.open',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark scroll-smooth">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen flex flex-col font-sans bg-ink-950 text-ink-50 selection:bg-brand-blue-300 selection:text-ink-950">
        {children}
      </body>
    </html>
  );
}
