'use client';

import React from 'react';
import { SectionHeading } from '@/components/ui/SectionHeading';
import { DiagramCanvas } from '@/components/diagrams/DiagramCanvas';
import { DiagramNode } from '@/components/diagrams/DiagramNode';
import { NodeTooltipData } from '@/lib/store';
import { Monitor, Server, Cpu, Layers, Sparkles, Database } from 'lucide-react';

const ARCHITECTURE_NODES: { data: NodeTooltipData; icon: React.ReactNode }[] = [
  {
    data: {
      id: 'arch-client',
      title: 'Client Ingestion Layer',
      layer: 'Ingestion Layer',
      tech: 'Next.js App Router, Chrome Extension MV3, Expo Mobile',
      description: 'Accepts drag-and-drop uploads, URL streams, and browser right-click context menu triggers.',
      metrics: 'Target: latency < 50ms',
    },
    icon: <Monitor className="w-4 h-4" />,
  },
  {
    data: {
      id: 'arch-gateway',
      title: 'API Gateway & Auth',
      layer: 'Gateway Layer',
      tech: 'FastAPI, JWT, Rate Limiter',
      description: 'Asynchronous Python REST API gateway enforcing rate limits, CORS policies, and JWT auth tokens.',
      metrics: 'Target: 10,000 req/sec',
    },
    icon: <Server className="w-4 h-4" />,
  },
  {
    data: {
      id: 'arch-preprocess',
      title: 'Preprocessing & Frame Extract',
      layer: 'Pipeline Layer',
      tech: 'OpenCV, FFmpeg, MediaPipe',
      description: 'Extracts keyframes from video streams, crops facial bounding boxes via MediaPipe, and normalizes audio.',
      metrics: 'Target: 60 FPS extraction',
    },
    icon: <Cpu className="w-4 h-4" />,
  },
  {
    data: {
      id: 'arch-engines',
      title: 'Parallel Detection Engine',
      layer: 'ML Engine Layer',
      tech: 'PyTorch, XceptionNet, RawNet2, ELA/PRNU',
      description: 'Runs parallel GPU/ONNX inference across spatial face-forgery CNNs and audio acoustic spoof detectors.',
      metrics: 'Target: infer < 1.2s',
    },
    icon: <Layers className="w-4 h-4" />,
  },
  {
    data: {
      id: 'arch-fusion',
      title: 'Weighted Ensemble Fusion',
      layer: 'Scoring Layer',
      tech: 'Ensemble Logistic Fusion',
      description: 'Combines outputs from independent classifiers using learned modality weights for a 0-100 score.',
      metrics: 'Accuracy: TBD — pending evaluation',
    },
    icon: <Sparkles className="w-4 h-4" />,
  },
  {
    data: {
      id: 'arch-storage',
      title: 'Data & Caching Layer',
      layer: 'Storage Layer',
      tech: 'PostgreSQL, Redis, MinIO',
      description: 'Caches hash lookups in Redis for instant repeat-scan results. Persists audit trails in PostgreSQL.',
      metrics: 'Target: cache hit < 5ms',
    },
    icon: <Database className="w-4 h-4" />,
  },
];

export const ArchitectureDiagram: React.FC = () => {
  return (
    <section id="architecture" className="py-20 bg-ink-950 text-white relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <SectionHeading
          eyebrow="System Blueprint"
          title="Modular Microservices Architecture"
          subtitle="Designed for horizontal scalability, zero vendor lock-in, and real-time payload processing."
          dark
        />

        <DiagramCanvas
          title="VerifAI Enterprise System Topology"
          subtitle="Click any microservice node to inspect technical specifications and performance benchmarks."
          dark
        >
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 pt-2">
            {ARCHITECTURE_NODES.map((node) => (
              <DiagramNode key={node.data.id} data={node.data} icon={node.icon} x={0} y={0} dark />
            ))}
          </div>
        </DiagramCanvas>
      </div>
    </section>
  );
};
