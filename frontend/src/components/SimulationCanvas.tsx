'use client';
import { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
// @ts-ignore
import { WebGPURenderer, StorageInstancedBufferAttribute, StorageBufferAttribute, MeshBasicNodeMaterial } from 'three/webgpu';
// @ts-ignore
import { uniform, storage, positionLocal, color, instanceIndex } from 'three/tsl';
import { useSimulationBridge } from '@/hooks/useSimulationBridge';
import { useSimulationStore } from '@/store/simulationStore';
import { flockingBehavior, resetAggregate } from './TslPrimitives';
import DashboardOverlay from './DashboardOverlay';

// The massive scale WebGPU is capable of
const AGENT_COUNT = 1000000;

function MicroEngine() {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const { currentPolicy, isMacroThinking, sendAggregateStats } = useSimulationBridge();
  
  const [computeNode, setComputeNode] = useState<any>(null);
  const [resetComputeNode, setResetComputeNode] = useState<any>(null);
  const [material, setMaterial] = useState<any>(null);
  const [speedUniform] = useState(() => uniform(0.1));
  const [aggregateAttribute, setAggregateAttribute] = useState<any>(null);
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
    
    const aggArray = new Uint32Array(200); // 10x10 grid * 2 stats (speed, count)
    const aggAttr = new StorageBufferAttribute(aggArray, 1);
    setAggregateAttribute(aggAttr);

    const positionsNode = storage(posAttr, 'vec3', AGENT_COUNT);
    const velocitiesNode = storage(velAttr, 'vec3', AGENT_COUNT);
    const aggregateNode = storage(aggAttr, 'uint', 200).toAtomic();
    
    // 2. TSL ComputeNode Implementation
    const behaviorNode = flockingBehavior(positionsNode, velocitiesNode, speedUniform, aggregateNode);
    setComputeNode(behaviorNode.compute(AGENT_COUNT));

    const resetNode = resetAggregate(aggregateNode.toAtomic());
    setResetComputeNode(resetNode.compute(200)); // 200 threads for 200 cells

    // 3. WebGPU Material Binding
    const mat = new MeshBasicNodeMaterial();
    mat.positionNode = positionLocal.add(positionsNode.element(instanceIndex)); 
    mat.colorNode = color("#00ff88");
    setMaterial(mat);
  }, [speedUniform]);

  useEffect(() => {
    if (material) {
      material.colorNode = color(isMacroThinking ? "#ff3366" : "#00ff88");
      material.needsUpdate = true;
    }
  }, [material, isMacroThinking]);

  useEffect(() => {
    if (currentPolicy?.movement_speed !== undefined) {
      speedUniform.value = currentPolicy.movement_speed;
    }
  }, [currentPolicy, speedUniform]);

  useFrame(async (state, delta) => {
    if (!meshRef.current || !computeNode || !resetComputeNode || !aggregateAttribute) return;
    
    // Check if our WebGPU hack has initialized
    if (!(state.gl as any).__initialized) return;

    // --- ASYNC PHYSICS ---
    const { isPaused, lastLlmSend, setLastLlmSend } = useSimulationStore.getState();
    if (isPaused) return;

    // Execute Micro Engine physics via WebGPU Compute Shader natively!
    await (state.gl as any).computeAsync(resetComputeNode);
    await (state.gl as any).computeAsync(computeNode);
    
    // CPU Aggregation (Readback API)
    const time = state.clock.getElapsedTime();
    if (time - lastReadbackRef.current > 0.1) {
      lastReadbackRef.current = time;
      
      try {
        const buffer = await (state.gl as any).backend.getArrayBufferAsync(aggregateAttribute);
        const uintArray = new Uint32Array(buffer);
        
        // Parse 200-element flat array into 10x10 spatial grid
        const grid = [];
        let totalGlobalSpeed = 0;
        let totalGlobalAgents = 0;

        for (let r = 0; r < 10; r++) {
          const row = [];
          for (let c = 0; c < 10; c++) {
            const index = r * 10 + c;
            const totalSpeedScaled = uintArray[index * 2];
            const count = uintArray[index * 2 + 1];
            
            const avgSpeed = count > 0 ? (totalSpeedScaled / 100.0) / count : 0;
            row.push({
              density: count,
              average_speed: avgSpeed
            });
            
            totalGlobalSpeed += (totalSpeedScaled / 100.0);
            totalGlobalAgents += count;
          }
          grid.push(row);
        }

        const globalAvgSpeed = totalGlobalAgents > 0 ? totalGlobalSpeed / totalGlobalAgents : 0;
        
        // 1. High-frequency telemetry for ChartGPU dashboard
        window.dispatchEvent(new CustomEvent('abm-telemetry', {
          detail: {
            timestamp: Date.now(),
            actual_speed: globalAvgSpeed,
            policy_speed: currentPolicy?.movement_speed || 0.1,
            grid: grid
          }
        }));

        // 2. Low-frequency payload for LLM backend (every 5 seconds)
        if (time - lastLlmSend > 5) {
          setLastLlmSend(time);
          sendAggregateStats({
            active_agents: totalGlobalAgents,
            average_speed: globalAvgSpeed,
            tick: time,
            system_status: "Awaiting LLM Instructions",
            grid: grid
          });
        }
      } catch (e) {
        console.warn("Readback error (usually occurs if canvas unmounts mid-frame)", e);
      }
    }
  });

  if (!material) return null;

  return (
    <instancedMesh ref={meshRef} args={[undefined, material, AGENT_COUNT]}>
      <planeGeometry args={[0.05, 0.05]} />
    </instancedMesh>
  );
}

export default function SimulationCanvas() {
  return (
    <div className="w-full h-screen bg-neutral-950 relative">
      <DashboardOverlay />

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
