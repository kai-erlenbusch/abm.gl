'use client';
import { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
// @ts-ignore
import { WebGPURenderer, StorageInstancedBufferAttribute, MeshBasicNodeMaterial } from 'three/webgpu';
// @ts-ignore
import { uniform, storage, positionLocal, color, instanceIndex } from 'three/tsl';
import { useSimulationBridge } from '@/hooks/useSimulationBridge';
import { flockingBehavior } from './TslPrimitives';

// The massive scale WebGPU is capable of
const AGENT_COUNT = 10000;

function MicroEngine() {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const { currentPolicy, isMacroThinking, sendAggregateStats } = useSimulationBridge();
  
  const [computeNode, setComputeNode] = useState<any>(null);
  const [material, setMaterial] = useState<any>(null);
  const [speedUniform] = useState(() => uniform(0.1));
  const [velocityAttribute, setVelocityAttribute] = useState<any>(null);
  const lastReadbackRef = useRef(0);

  useEffect(() => {
    // 1. StorageBuffers Setup
    const posArray = new Float32Array(AGENT_COUNT * 3);
    const velArray = new Float32Array(AGENT_COUNT * 3);
    for (let i = 0; i < AGENT_COUNT; i++) {
      posArray[i * 3] = (Math.random() - 0.5) * 50;
      posArray[i * 3 + 1] = (Math.random() - 0.5) * 50;
      posArray[i * 3 + 2] = 0;
      
      velArray[i * 3] = (Math.random() - 0.5) * 2;
      velArray[i * 3 + 1] = (Math.random() - 0.5) * 2;
      velArray[i * 3 + 2] = 0;
    }

    const posAttr = new StorageInstancedBufferAttribute(posArray, 3);
    const velAttr = new StorageInstancedBufferAttribute(velArray, 3);
    setVelocityAttribute(velAttr);

    const positionsNode = storage(posAttr, 'vec3', AGENT_COUNT);
    const velocitiesNode = storage(velAttr, 'vec3', AGENT_COUNT);
    
    // 2. TSL ComputeNode Implementation
    const behaviorNode = flockingBehavior(positionsNode, velocitiesNode, speedUniform);
    setComputeNode(behaviorNode.compute(AGENT_COUNT));

    // 3. WebGPU Material Binding
    const mat = new MeshBasicNodeMaterial();
    mat.positionNode = positionLocal.add(positionsNode.element(instanceIndex)); 
    setMaterial(mat);
  }, [speedUniform]);

  useEffect(() => {
    if (material) {
      material.colorNode = color(isMacroThinking ? "#ff3366" : "#00ff88");
    }
  }, [material, isMacroThinking]);

  useEffect(() => {
    if (currentPolicy?.movement_speed !== undefined) {
      speedUniform.value = currentPolicy.movement_speed;
    }
  }, [currentPolicy, speedUniform]);

  useFrame(async (state, delta) => {
    if (!meshRef.current || !computeNode || !velocityAttribute) return;
    
    // Check if our WebGPU hack has initialized
    if (!(state.gl as any).__initialized) return;

    // --- LOCKSTEP TIME SYNCHRONIZATION ---
    if (isMacroThinking) return;

    // Execute Micro Engine physics via WebGPU Compute Shader natively!
    await (state.gl as any).computeAsync(computeNode);
    
    // CPU Aggregation (Readback API) every 5 seconds
    const time = state.clock.getElapsedTime();
    if (time - lastReadbackRef.current > 5) {
      lastReadbackRef.current = time;
      
      try {
        const buffer = await (state.gl as any).backend.getArrayBufferAsync(velocityAttribute);
        const floatArray = new Float32Array(buffer);
        let totalSpeed = 0;
        for (let i = 0; i < AGENT_COUNT; i++) {
           const vx = floatArray[i*3];
           const vy = floatArray[i*3+1];
           totalSpeed += Math.sqrt(vx*vx + vy*vy);
        }
        const avgSpeed = totalSpeed / AGENT_COUNT;
        
        sendAggregateStats({
          active_agents: AGENT_COUNT,
          average_speed: avgSpeed,
          tick: time,
          system_status: "Awaiting LLM Instructions"
        });
      } catch (e) {
        console.error("Readback failed", e);
      }
    }
  });

  if (!material) return null;

  return (
    <instancedMesh ref={meshRef} args={[undefined, material, AGENT_COUNT]}>
      <circleGeometry args={[0.15, 8]} />
    </instancedMesh>
  );
}

export default function SimulationCanvas() {
  return (
    <div className="w-full h-screen bg-neutral-950 relative">
      {/* UI Overlay */}
      <div className="absolute top-4 left-4 z-10 text-white font-mono text-sm bg-black/50 p-4 rounded-lg border border-neutral-800 pointer-events-none">
        <h1 className="text-xl font-bold mb-2">abm.gl</h1>
        <p>Micro Agents (WebGPU): 10,000</p>
        <p>Macro Engine (Python): Active</p>
      </div>

      {/* WebGPURenderer Injection with Async Init Hack */}
      <Canvas 
        gl={(props) => {
          const renderer = new WebGPURenderer(props) as any;
          renderer.__initialized = false;
          
          // R3F calls render() synchronously, but WebGPU requires async init
          const originalRender = renderer.render.bind(renderer);
          renderer.render = (...args: any[]) => {
            if (!renderer.__initialized) return;
            originalRender(...args);
          };

          renderer.init().then(() => {
            renderer.__initialized = true;
          });

          return renderer;
        }} 
        camera={{ position: [0, 0, 50], fov: 75 }}
      >
        <MicroEngine />
      </Canvas>
    </div>
  );
}
