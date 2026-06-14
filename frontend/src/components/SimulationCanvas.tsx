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
import { flockingBehavior, resetAggregate, spatialResetNode, spatialCountNode, spatialPrefixSumNode, spatialScatterNode, spatialCollisionNode } from './TslPrimitives';
import DashboardOverlay from './DashboardOverlay';

// The massive scale WebGPU is capable of
const AGENT_COUNT = 1000000;

function MicroEngine() {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const { currentPolicy, isMacroThinking, sendAggregateStats } = useSimulationBridge();
  
  const [computeNode, setComputeNode] = useState<any>(null);
  const [resetComputeNode, setResetComputeNode] = useState<any>(null);
  
  // Spatial Grid Nodes
  const [pass0Node, setPass0Node] = useState<any>(null);
  const [pass1Node, setPass1Node] = useState<any>(null);
  const [pass2Node, setPass2Node] = useState<any>(null);
  const [pass3Node, setPass3Node] = useState<any>(null);
  const [pass4Node, setPass4Node] = useState<any>(null);

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

    // 3. Spatial Grid Buffers (Point D)
    const cellCountArray = new Uint32Array(100);
    const cellOffsetArray = new Uint32Array(100);
    const cellOffsetAtomicArray = new Uint32Array(100);
    const sortedAgentArray = new Uint32Array(AGENT_COUNT);
    
    const countAttr = new StorageBufferAttribute(cellCountArray, 1);
    const offsetAttr = new StorageBufferAttribute(cellOffsetArray, 1);
    const offsetAtomicAttr = new StorageBufferAttribute(cellOffsetAtomicArray, 1);
    const sortedAttr = new StorageBufferAttribute(sortedAgentArray, 1);
    
    const countNode = storage(countAttr, 'uint', 100);
    const offsetNode = storage(offsetAttr, 'uint', 100);
    const offsetAtomicNode = storage(offsetAtomicAttr, 'uint', 100);
    const sortedNode = storage(sortedAttr, 'uint', AGENT_COUNT);

    setPass0Node(spatialResetNode(countNode.toAtomic(), offsetAtomicNode.toAtomic()).compute(100));
    setPass1Node(spatialCountNode(positionsNode, countNode.toAtomic()).compute(AGENT_COUNT));
    setPass2Node(spatialPrefixSumNode(countNode, offsetNode, offsetAtomicNode.toAtomic()).compute(1));
    setPass3Node(spatialScatterNode(positionsNode, offsetAtomicNode.toAtomic(), sortedNode).compute(AGENT_COUNT));
    setPass4Node(spatialCollisionNode(positionsNode, velocitiesNode, countNode, offsetNode, sortedNode).compute(AGENT_COUNT));

    // 4. WebGPU Material Binding
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
    if (!pass0Node || !pass1Node || !pass2Node || !pass3Node || !pass4Node) return;
    
    // Check if our WebGPU hack has initialized
    if (!(state.gl as any).__initialized) return;

    // --- ASYNC PHYSICS ---
    const { isPaused, lastLlmSend, setLastLlmSend } = useSimulationStore.getState();
    if (isPaused) return;

    // Execute Micro Engine physics via WebGPU Compute Shader natively!
    await (state.gl as any).computeAsync(resetComputeNode);
    await (state.gl as any).computeAsync(computeNode);
    
    // Execute Point D: 5-Pass Spatial Hash Grid
    await (state.gl as any).computeAsync(pass0Node);
    await (state.gl as any).computeAsync(pass1Node);
    await (state.gl as any).computeAsync(pass2Node);
    await (state.gl as any).computeAsync(pass3Node);
    await (state.gl as any).computeAsync(pass4Node);
    
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
            
            const avgSpeed = count > 0 ? (totalSpeedScaled / 10.0) / count : 0;
            row.push({
              density: count,
              average_speed: avgSpeed
            });
            
            totalGlobalSpeed += (totalSpeedScaled / 10.0);
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
