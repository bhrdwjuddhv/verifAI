'use client';

import React from 'react';
import { SectionHeading } from '@/components/ui/SectionHeading';
import { Accordion, AccordionItemData } from '@/components/ui/Accordion';

const RESPONSIBILITIES_DATA: AccordionItemData[] = [
  {
    id: 'comp-next',
    title: '1. Next.js Web Application',
    badge: 'Frontend',
    content: (
      <div className="space-y-2">
        <p>
          Serves the user interface, file drag-and-drop zone, live interactive scan sandbox, and dynamic score charts.
        </p>
        <ul className="list-disc list-inside space-y-1 text-xs text-ink-500 dark:text-ink-400">
          <li>Handles client-side media validation & SHA-256 hash computation</li>
          <li>Renders R3F 3D WebGL background and animated SVG node graphs</li>
          <li>Communicates with the FastAPI backend over REST/WebSockets</li>
        </ul>
      </div>
    ),
  },
  {
    id: 'comp-ext',
    title: '2. Chrome Extension (Manifest V3)',
    badge: 'Browser Client',
    content: (
      <div className="space-y-2">
        <p>
          Integrates directly into Google Chrome, allowing users to right-click any image or video on any website to verify authenticity without leaving the page.
        </p>
        <ul className="list-disc list-inside space-y-1 text-xs text-ink-500 dark:text-ink-400">
          <li>Adds context menu action item "Verify with VerifAI"</li>
          <li>Injects lightweight floating trust score pill on top of verified media</li>
          <li>Utilizes background Service Worker for non-blocking API calls</li>
        </ul>
      </div>
    ),
  },
  {
    id: 'comp-expo',
    title: '3. Mobile App & Native Share Sheet (Expo)',
    badge: 'Mobile Client',
    content: (
      <div className="space-y-2">
        <p>
          Cross-platform React Native app for iOS and Android, providing one-tap verification directly from social apps via the native OS share sheet.
        </p>
        <ul className="list-disc list-inside space-y-1 text-xs text-ink-500 dark:text-ink-400">
          <li>Hooks into iOS Activity View & Android Intent Filter</li>
          <li>Offline hash database check for instant cached verdicts</li>
          <li>Camera capture mode for instant document & photo verification</li>
        </ul>
      </div>
    ),
  },
  {
    id: 'comp-fastapi',
    title: '4. FastAPI REST Gateway',
    badge: 'Backend API',
    content: (
      <div className="space-y-2">
        <p>
          High-performance asynchronous Python REST server that handles endpoint routing, JWT authentication, rate limiting, and request validation.
        </p>
        <ul className="list-disc list-inside space-y-1 text-xs text-ink-500 dark:text-ink-400">
          <li>Validates incoming multipart media payloads and public URL streams</li>
          <li>Dispatches heavy forensic processing tasks to Celery task queues</li>
          <li>Exposes public REST API endpoints for enterprise newsroom integrations</li>
        </ul>
      </div>
    ),
  },
  {
    id: 'comp-celery',
    title: '5. Celery & Redis Async Task Workers',
    badge: 'Queue Worker',
    content: (
      <div className="space-y-2">
        <p>
          Background distributed worker queue system that manages long-running video and audio inference pipelines without blocking API threads.
        </p>
        <ul className="list-disc list-inside space-y-1 text-xs text-ink-500 dark:text-ink-400">
          <li>Scales GPU worker instances horizontally based on queue depth</li>
          <li>Implements task retry logic and priority scheduling</li>
          <li>Pushes progress updates back via Redis pub/sub WebSockets</li>
        </ul>
      </div>
    ),
  },
  {
    id: 'comp-ffmpeg',
    title: '6. OpenCV & FFmpeg Ingestion Engine',
    badge: 'Preprocessing',
    content: (
      <div className="space-y-2">
        <p>
          Media decoding pipeline responsible for demuxing containers, extracting video keyframes, cropping faces, and normalizing audio tracks.
        </p>
        <ul className="list-disc list-inside space-y-1 text-xs text-ink-500 dark:text-ink-400">
          <li>Extracts 2 frames per second from long video streams</li>
          <li>Executes MediaPipe facial detection for 224x224 bounding box crops</li>
          <li>Converts audio streams to 16kHz mono WAV for acoustic analysis</li>
        </ul>
      </div>
    ),
  },
  {
    id: 'comp-xception',
    title: '7. XceptionNet Spatial Deepfake CNN',
    badge: 'Deep Learning',
    content: (
      <div className="space-y-2">
        <p>
          Convolutional Neural Network fine-tuned on FaceForensics++ and DFDC datasets to detect subtle spatial facial swapping artifacts.
        </p>
        <ul className="list-disc list-inside space-y-1 text-xs text-ink-500 dark:text-ink-400">
          <li>Analyzes facial landmark boundary blurring and eye glint inconsistencies</li>
          <li>Evaluates skin texture noise distribution across consecutive frames</li>
          <li>Outputs spatial forgery probability score from 0.0 to 1.0</li>
        </ul>
      </div>
    ),
  },
  {
    id: 'comp-rawnet',
    title: '8. RawNet2 Acoustic Voice Spoof Detector',
    badge: 'Audio ML',
    content: (
      <div className="space-y-2">
        <p>
          Deep acoustic neural network trained on ASVspoof datasets that processes raw audio waveforms directly to spot voice cloning and TTS synthesis.
        </p>
        <ul className="list-disc list-inside space-y-1 text-xs text-ink-500 dark:text-ink-400">
          <li>Detects high-frequency acoustic phase distortions from neural vocoders</li>
          <li>Identifies unnatural lack of human breath pauses and micro-pitch shifts</li>
          <li>Flags AI voice clones generated by ElevenLabs or Bark models</li>
        </ul>
      </div>
    ),
  },
  {
    id: 'comp-ela',
    title: '9. ELA & Frequency Residual Analyzer',
    badge: 'Signal Forensics',
    content: (
      <div className="space-y-2">
        <p>
          Classical digital forensics engine performing Error Level Analysis (ELA) and 2D Fast Fourier Transform (FFT) noise residual checks.
        </p>
        <ul className="list-disc list-inside space-y-1 text-xs text-ink-500 dark:text-ink-400">
          <li>Uncovers localized image re-compression differences across regions</li>
          <li>Spots periodic grid artifacts characteristic of GAN and diffusion upsamplers</li>
          <li>Operates deterministically without neural network model hallucination</li>
        </ul>
      </div>
    ),
  },
  {
    id: 'comp-c2pa',
    title: '10. C2PA Provenance & EXIF Module',
    badge: 'Metadata',
    content: (
      <div className="space-y-2">
        <p>
          Cryptographic metadata parser verifying C2PA manifest signatures, hardware camera EXIF tags, and digital watermarks.
        </p>
        <ul className="list-disc list-inside space-y-1 text-xs text-ink-500 dark:text-ink-400">
          <li>Validates C2PA certificates against trusted Certificate Authorities (CAs)</li>
          <li>Inspects EXIF camera maker notes and software editing tags</li>
          <li>Detects SynthID and invisible digital watermark signals</li>
        </ul>
      </div>
    ),
  },
  {
    id: 'comp-fusion',
    title: '11. Ensemble Fusion & Explainability Layer',
    badge: 'Scoring Engine',
    content: (
      <div className="space-y-2">
        <p>
          Combines all multi-modal model outputs into a unified 0-100 trust score and generates Grad-CAM highlight maps for explainable transparency.
        </p>
        <ul className="list-disc list-inside space-y-1 text-xs text-ink-500 dark:text-ink-400">
          <li>Applies learned modal weights based on payload type (image vs video vs audio)</li>
          <li>Generates spatial heatmap highlights over manipulated image regions</li>
          <li>Produces human-readable bullet list explaining why content was flagged</li>
        </ul>
      </div>
    ),
  },
];

export const ComponentResponsibilities: React.FC = () => {
  return (
    <section id="responsibilities" className="py-24 bg-white dark:bg-ink-900 relative">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="System Breakdown"
          title="Component Responsibilities & Technical Specs"
          subtitle="Explore the exact technical duties executed by each of VerifAI's 11 system components."
        />

        <Accordion items={RESPONSIBILITIES_DATA} allowMultiple={false} />
      </div>
    </section>
  );
};
