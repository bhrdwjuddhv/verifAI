'use client';

import React, { useRef, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

// Individual Interconnected Blockchain Node Cube
function BlockNode({ position, rotation, scale, delay }: {
  position: [number, number, number];
  rotation: [number, number, number];
  scale: number;
  delay: number;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const wireRef = useRef<THREE.LineSegments>(null);
  const clock = useRef(delay);

  useFrame((_, delta) => {
    clock.current += delta * 0.8;
    if (meshRef.current) {
      meshRef.current.position.y = position[1] + Math.sin(clock.current) * 0.12;
      meshRef.current.rotation.x += delta * 0.2;
      meshRef.current.rotation.y += delta * 0.25;
    }
  });

  return (
    <group position={position}>
      {/* Solid Inner Glass Block */}
      <mesh ref={meshRef} rotation={rotation} scale={scale}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial
          color="#1D6EB0"
          roughness={0.1}
          metalness={0.8}
          transparent
          opacity={0.35}
        />
      </mesh>

      {/* Outer Glowing Wireframe Border */}
      <mesh rotation={rotation} scale={scale * 1.05}>
        <boxGeometry args={[1, 1, 1]} />
        <meshBasicMaterial
          color="#A8D8F0"
          wireframe
          transparent
          opacity={0.6}
        />
      </mesh>

      {/* Corner Node Glow Dots */}
      <mesh position={[scale * 0.55, scale * 0.55, scale * 0.55]}>
        <sphereGeometry args={[0.06, 12, 12]} />
        <meshBasicMaterial color="#5CB4E8" />
      </mesh>
    </group>
  );
}

// Connector Link Laser Line between Blockchain Nodes
function BlockConnection({ start, end }: { start: [number, number, number]; end: [number, number, number] }) {
  const lineRef = useRef<THREE.Line>(null);

  const points = [
    new THREE.Vector3(...start),
    new THREE.Vector3(...end),
  ];
  const geometry = new THREE.BufferGeometry().setFromPoints(points);

  return (
    <primitive object={new THREE.Line(
      geometry,
      new THREE.LineBasicMaterial({
        color: '#5CB4E8',
        transparent: true,
        opacity: 0.3,
        linewidth: 1,
      })
    )} />
  );
}

// 3D Floating Cryptographic Hash Packet Particle
function HashParticle({ position, speed }: { position: [number, number, number]; speed: number }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const clock = useRef(Math.random() * 10);

  useFrame((_, delta) => {
    if (!meshRef.current) return;
    clock.current += delta * speed;
    meshRef.current.position.y = position[1] + Math.sin(clock.current) * 0.25;
    meshRef.current.position.x = position[0] + Math.cos(clock.current * 0.7) * 0.15;
  });

  return (
    <mesh ref={meshRef} position={position}>
      <octahedronGeometry args={[0.08, 0]} />
      <meshBasicMaterial color="#A8D8F0" wireframe />
    </mesh>
  );
}

function BlockchainNetworkGroup() {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (!groupRef.current) return;
    // Mouse subtle parallax tracking
    const targetX = (state.pointer.x * Math.PI) / 24;
    const targetY = (state.pointer.y * Math.PI) / 24;
    groupRef.current.rotation.x = THREE.MathUtils.lerp(groupRef.current.rotation.x, targetY, 0.04);
    groupRef.current.rotation.y = THREE.MathUtils.lerp(groupRef.current.rotation.y, targetX, 0.04);
  });

  // Node Positions across the right side of the canvas
  const nodes: [number, number, number][] = [
    [1.8, 1.2, 0],
    [3.2, -0.4, -0.8],
    [0.8, -1.4, -0.5],
    [2.6, 2.2, -1.2],
    [3.8, 0.8, -1.5],
  ];

  return (
    <group ref={groupRef}>
      {/* Blockchain Nodes */}
      <BlockNode position={nodes[0]} rotation={[0.4, 0.2, 0]} scale={0.9} delay={0} />
      <BlockNode position={nodes[1]} rotation={[0.2, 0.6, 0.1]} scale={1.1} delay={1.2} />
      <BlockNode position={nodes[2]} rotation={[0.5, 0.1, 0.3]} scale={0.8} delay={2.4} />
      <BlockNode position={nodes[3]} rotation={[0.1, 0.8, 0.2]} scale={0.7} delay={3.6} />
      <BlockNode position={nodes[4]} rotation={[0.3, 0.4, 0.5]} scale={0.65} delay={4.8} />

      {/* Network Interconnection Laser Lines */}
      <BlockConnection start={nodes[0]} end={nodes[1]} />
      <BlockConnection start={nodes[1]} end={nodes[2]} />
      <BlockConnection start={nodes[0]} end={nodes[3]} />
      <BlockConnection start={nodes[1]} end={nodes[4]} />
      <BlockConnection start={nodes[3]} end={nodes[4]} />

      {/* Floating Hash Data Packets */}
      <HashParticle position={[2.3, 0.4, -0.4]} speed={1.2} />
      <HashParticle position={[1.3, -0.7, -0.2]} speed={0.9} />
      <HashParticle position={[3.2, 1.5, -1]} speed={1.5} />
    </group>
  );
}

export default function HeroScene() {
  const [reducedMotion, setReducedMotion] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReducedMotion(mq.matches);
    const listener = (e: MediaQueryListEvent) => setReducedMotion(e.matches);
    mq.addEventListener('change', listener);
    return () => mq.removeEventListener('change', listener);
  }, []);

  if (!mounted || reducedMotion) {
    return (
      <div className="w-full h-full min-h-[400px] bg-gradient-to-tr from-brand-blue-400/15 via-brand-blue-500/5 to-transparent rounded-full blur-3xl" />
    );
  }

  return (
    <div className="w-full h-full min-h-[400px] sm:min-h-[500px] pointer-events-none">
      <Canvas
        camera={{ position: [0, 0, 5.5], fov: 45 }}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        style={{ background: 'transparent' }}
      >
        <ambientLight intensity={0.8} />
        <directionalLight position={[10, 10, 5]} intensity={1.5} color="#FFFFFF" />
        <pointLight position={[-5, -5, -3]} intensity={1.2} color="#A8D8F0" />
        <BlockchainNetworkGroup />
      </Canvas>
    </div>
  );
}
