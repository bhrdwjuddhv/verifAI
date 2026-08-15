'use client';

import React, { useRef, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

// Individual Non-Overlapping 3D Blockchain Node Cube
function BlockNode({ position, rotation, scale, delay }: {
  position: [number, number, number];
  rotation: [number, number, number];
  scale: number;
  delay: number;
}) {
  const meshRef = useRef<THREE.Group>(null);
  const clock = useRef(delay);

  useFrame((_, delta) => {
    clock.current += delta;
    if (!meshRef.current) return;

    // Smooth hover float
    meshRef.current.position.y = position[1] + Math.sin(clock.current * 0.7 + delay) * 0.14;
    meshRef.current.rotation.x += delta * 0.18;
    meshRef.current.rotation.y += delta * 0.22;
  });

  return (
    <group ref={meshRef} position={position} scale={scale}>
      {/* Translucent Core */}
      <mesh rotation={rotation}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial
          color="#1D6EB0"
          roughness={0.25}
          metalness={0.6}
          transparent
          opacity={0.3}
        />
      </mesh>

      {/* Wireframe Outline */}
      <mesh rotation={rotation} scale={1.05}>
        <boxGeometry args={[1, 1, 1]} />
        <meshBasicMaterial
          color="#A8D8F0"
          wireframe
          transparent
          opacity={0.45}
        />
      </mesh>

      {/* Corner Node Light Dot */}
      <mesh position={[0.55, 0.55, 0.55]}>
        <sphereGeometry args={[0.05, 8, 8]} />
        <meshBasicMaterial color="#5CB4E8" />
      </mesh>
    </group>
  );
}

// Continuous Staggered Line Animation (Draws 0% -> 100% over 2s on repeating cycle with varied delays)
function ContinuousAnimatedConnection({ start, end, staggerOffset = 0, baseOpacity = 0.35 }: {
  start: [number, number, number];
  end: [number, number, number];
  staggerOffset?: number;
  baseOpacity?: number;
}) {
  const lineRef = useRef<THREE.Line>(null);
  const startVec = new THREE.Vector3(...start);
  const endVec = new THREE.Vector3(...end);
  const currentEndVec = useRef(new THREE.Vector3(...start));
  const clock = useRef(staggerOffset);

  const CYCLE_DURATION = 4.2; // Total cycle length (2s draw + 1.2s hold + 1.0s reset)

  useFrame((_, delta) => {
    clock.current += delta;
    if (!lineRef.current) return;

    const cycleTime = (clock.current + staggerOffset) % CYCLE_DURATION;
    let progress = 0;
    let currentOpacity = baseOpacity;

    if (cycleTime <= 2.0) {
      // 0s to 2.0s: Draw line from start (0%) to end (100%)
      progress = cycleTime / 2.0;
      // Ease-out quad
      progress = 1 - Math.pow(1 - progress, 2);
    } else if (cycleTime <= 3.2) {
      // 2.0s to 3.2s: Hold line at 100% full connection with slight pulse
      progress = 1.0;
      currentOpacity = baseOpacity + Math.sin((cycleTime - 2.0) * Math.PI * 2) * 0.1;
    } else {
      // 3.2s to 4.2s: Smooth fade out before next loop
      progress = 1.0;
      const fadeProgress = (cycleTime - 3.2) / 1.0;
      currentOpacity = baseOpacity * (1 - fadeProgress);
    }

    currentEndVec.current.lerpVectors(startVec, endVec, progress);

    const positions = lineRef.current.geometry.attributes.position;
    positions.setXYZ(1, currentEndVec.current.x, currentEndVec.current.y, currentEndVec.current.z);
    positions.needsUpdate = true;

    if (lineRef.current.material) {
      (lineRef.current.material as THREE.LineBasicMaterial).opacity = Math.max(0, currentOpacity);
    }
  });

  const initialPoints = [startVec, startVec];
  const geometry = new THREE.BufferGeometry().setFromPoints(initialPoints);

  return (
    <primitive object={new THREE.Line(
      geometry,
      new THREE.LineBasicMaterial({
        color: '#5CB4E8',
        transparent: true,
        opacity: baseOpacity,
        linewidth: 1.5,
      })
    )} ref={lineRef} />
  );
}

// Floating Data Packet
function HashDataPacket({ position, speed }: { position: [number, number, number]; speed: number }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const clock = useRef(Math.random() * 10);

  useFrame((_, delta) => {
    if (!meshRef.current) return;
    clock.current += delta * speed;
    meshRef.current.position.y = position[1] + Math.sin(clock.current) * 0.16;
    meshRef.current.position.x = position[0] + Math.cos(clock.current * 0.5) * 0.12;
  });

  return (
    <mesh ref={meshRef} position={position}>
      <octahedronGeometry args={[0.06, 0]} />
      <meshBasicMaterial color="#A8D8F0" wireframe transparent opacity={0.5} />
    </mesh>
  );
}

function NonOverlappingBlockchainScene() {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (!groupRef.current) return;
    const targetX = (state.pointer.x * Math.PI) / 36;
    const targetY = (state.pointer.y * Math.PI) / 36;
    groupRef.current.rotation.x = THREE.MathUtils.lerp(groupRef.current.rotation.x, targetY, 0.03);
    groupRef.current.rotation.y = THREE.MathUtils.lerp(groupRef.current.rotation.y, targetX, 0.03);
  });

  // NON-OVERLAPPING Position Grid (Well-spaced coordinates across viewport)
  const nodes: [number, number, number][] = [
    // Top Row
    [-6.2, 3.8, -2.5],  // 0: Far Top Left
    [-2.6, 4.2, -2.0],  // 1: Top Center Left
    [2.6, 4.2, -2.0],   // 2: Top Center Right
    [6.2, 3.8, -2.5],   // 3: Far Top Right

    // Mid Row
    [-6.4, 0.2, -2.2],  // 4: Far Mid Left
    [6.4, 0.2, -2.2],   // 5: Far Mid Right

    // Bottom Row
    [-5.2, -3.4, -2.2], // 6: Bottom Left
    [-1.8, -4.4, -2.0], // 7: Bottom Center Left
    [1.8, -4.4, -2.0],  // 8: Bottom Center Right
    [5.2, -3.4, -2.2],  // 9: Bottom Right
  ];

  return (
    <group ref={groupRef}>
      {/* Well-Spaced Non-Overlapping Nodes */}
      {nodes.map((pos, i) => (
        <BlockNode
          key={i}
          position={pos}
          rotation={[0.25 * i, 0.35 * i, 0.15 * i]}
          scale={0.7 + (i % 3) * 0.1}
          delay={i * 0.4}
        />
      ))}

      {/* Varied 2-Second Line Connection Animations with Staggered Time Delays */}
      <ContinuousAnimatedConnection start={nodes[0]} end={nodes[1]} staggerOffset={0.0} baseOpacity={0.35} />
      <ContinuousAnimatedConnection start={nodes[1]} end={nodes[2]} staggerOffset={0.7} baseOpacity={0.3} />
      <ContinuousAnimatedConnection start={nodes[2]} end={nodes[3]} staggerOffset={1.4} baseOpacity={0.35} />
      <ContinuousAnimatedConnection start={nodes[0]} end={nodes[4]} staggerOffset={2.1} baseOpacity={0.3} />
      <ContinuousAnimatedConnection start={nodes[3]} end={nodes[5]} staggerOffset={2.8} baseOpacity={0.3} />
      <ContinuousAnimatedConnection start={nodes[4]} end={nodes[6]} staggerOffset={0.5} baseOpacity={0.35} />
      <ContinuousAnimatedConnection start={nodes[6]} end={nodes[7]} staggerOffset={1.2} baseOpacity={0.3} />
      <ContinuousAnimatedConnection start={nodes[7]} end={nodes[8]} staggerOffset={1.9} baseOpacity={0.35} />
      <ContinuousAnimatedConnection start={nodes[8]} end={nodes[9]} staggerOffset={2.6} baseOpacity={0.3} />
      <ContinuousAnimatedConnection start={nodes[5]} end={nodes[9]} staggerOffset={3.3} baseOpacity={0.35} />

      {/* Floating Data Packets */}
      <HashDataPacket position={[-4.5, 2.8, -1.8]} speed={1.1} />
      <HashDataPacket position={[4.5, 2.5, -1.5]} speed={1.4} />
      <HashDataPacket position={[4.2, -1.5, -1.5]} speed={0.9} />
      <HashDataPacket position={[-4.0, -1.8, -1.5]} speed={1.2} />
    </group>
  );
}

export default function BlockchainBackground() {
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
      <div className="fixed inset-0 bg-gradient-to-b from-ink-950 via-ink-900 to-ink-950 pointer-events-none z-0" />
    );
  }

  return (
    <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
      <Canvas
        camera={{ position: [0, 0, 7.5], fov: 48 }}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        style={{ background: 'transparent' }}
      >
        <ambientLight intensity={0.7} />
        <directionalLight position={[10, 10, 5]} intensity={1.3} color="#FFFFFF" />
        <pointLight position={[-5, -5, -3]} intensity={1} color="#A8D8F0" />
        <NonOverlappingBlockchainScene />
      </Canvas>
    </div>
  );
}
