'use client';
import { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
// @ts-ignore
import { WebGPURenderer, StorageInstancedBufferAttribute, StorageBufferAttribute, MeshBasicNodeMaterial } from 'three/webgpu';
// @ts-ignore
import { uniform, storage, positionLocal, color, instanceIndex, texture } from 'three/tsl';
import { useSimulationBridge } from '@/hooks/useSimulationBridge';
import { useSimulationStore } from '@/store/simulationStore';
import { flockingBehavior, resetAggregate, spatialResetNode, spatialCountNode, spatialPrefixSum_LocalScanNode, spatialPrefixSum_BlockScanNode, spatialPrefixSum_AddNode, spatialScatterNode, spatialCollisionNode } from './TslPrimitives';
import DashboardOverlay from './DashboardOverlay';

// The massive scale WebGPU is capable of
const AGENT_COUNT = 10000;

function MicroEngine() {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const { currentPolicy, isMacroThinking, sendAggregateStats } = useSimulationBridge();
  
  const [computeNode, setComputeNode] = useState<any>(null);
  const [resetComputeNode, setResetComputeNode] = useState<any>(null);
  
  // Spatial Grid Nodes
  const [pass0Node, setPass0Node] = useState<any>(null);
  const [pass1Node, setPass1Node] = useState<any>(null);
  const [pass2aNode, setPass2aNode] = useState<any>(null);
  const [pass2bNode, setPass2bNode] = useState<any>(null);
  const [pass2cNode, setPass2cNode] = useState<any>(null);
  const [pass3Node, setPass3Node] = useState<any>(null);
  const [pass4Node, setPass4Node] = useState<any>(null);

  const [material, setMaterial] = useState<any>(null);
  const [policyTexture] = useState(() => {
    const data = new Float32Array(100 * 4);
    for (let i = 0; i < 400; i++) data[i] = 0.1;
    const tex = new THREE.DataTexture(data, 10, 10, THREE.RGBAFormat, THREE.FloatType);
    tex.magFilter = THREE.LinearFilter;
    tex.minFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    return tex;
  });
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
    const policyMapTextureNode = texture(policyTexture);
    
    // 2. TSL ComputeNode Implementation
    const behaviorNode = flockingBehavior(positionsNode, velocitiesNode, policyMapTextureNode, aggregateNode);
    setComputeNode(behaviorNode.compute(AGENT_COUNT));

    const resetNode = resetAggregate(aggregateNode.toAtomic());
    setResetComputeNode(resetNode.compute(200)); // 200 threads for 200 cells

    // 3. Spatial Grid Buffers (Point D & V2 Decoupling)
    // Pad to 10240 (40 chunks of 256) for Blelloch Scan alignment
    const cellCountArray = new Uint32Array(10240);
    const cellOffsetArray = new Uint32Array(10240);
    const cellOffsetAtomicArray = new Uint32Array(10240);
    const chunkSumsArray = new Uint32Array(64);
    const sortedAgentArray = new Uint32Array(AGENT_COUNT);
    
    const countAttr = new StorageBufferAttribute(cellCountArray, 1);
    const offsetAttr = new StorageBufferAttribute(cellOffsetArray, 1);
    const offsetAtomicAttr = new StorageBufferAttribute(cellOffsetAtomicArray, 1);
    const chunkSumsAttr = new StorageBufferAttribute(chunkSumsArray, 1);
    const sortedAttr = new StorageBufferAttribute(sortedAgentArray, 1);
    
    const countNode = storage(countAttr, 'uint', 10240);
    const offsetNode = storage(offsetAttr, 'uint', 10240);
    const offsetAtomicNode = storage(offsetAtomicAttr, 'uint', 10240);
    const chunkSumsNode = storage(chunkSumsAttr, 'uint', 64);
    const sortedNode = storage(sortedAttr, 'uint', AGENT_COUNT);

    setPass0Node(spatialResetNode(countNode.toAtomic(), offsetAtomicNode.toAtomic()).compute(10240));
    setPass1Node(spatialCountNode(positionsNode, countNode.toAtomic()).compute(AGENT_COUNT));
    
    // 3-Pass Parallel Blelloch Scan
    const pass2a = spatialPrefixSum_LocalScanNode(countNode, offsetNode, chunkSumsNode).compute(10240);
    pass2a.workgroupSize = [256, 1, 1];
    setPass2aNode(pass2a);
    
    const pass2b = spatialPrefixSum_BlockScanNode(chunkSumsNode).compute(64);
    pass2b.workgroupSize = [64, 1, 1];
    setPass2bNode(pass2b);
    
    const pass2c = spatialPrefixSum_AddNode(offsetNode, offsetAtomicNode.toAtomic(), chunkSumsNode).compute(10240);
    pass2c.workgroupSize = [256, 1, 1];
    setPass2cNode(pass2c);
    
    setPass3Node(spatialScatterNode(positionsNode, offsetAtomicNode.toAtomic(), sortedNode).compute(AGENT_COUNT));
    setPass4Node(spatialCollisionNode(positionsNode, velocitiesNode, countNode, offsetNode, sortedNode).compute(AGENT_COUNT));

    // 4. WebGPU Material Binding
    const mat = new MeshBasicNodeMaterial();
    mat.positionNode = positionLocal.add(positionsNode.element(instanceIndex)); 
    mat.colorNode = color("#00ff88");
    setMaterial(mat);
  }, [policyTexture]);

  useEffect(() => {
    if (material) {
      material.colorNode = color(isMacroThinking ? "#ff3366" : "#00ff88");
      material.needsUpdate = true;
    }
  }, [material, isMacroThinking]);

  useEffect(() => {
    if (currentPolicy?.policy_speed_map) {
      const map = currentPolicy.policy_speed_map;
      const data = policyTexture.image.data;
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          const texR = 9 - r; // Invert Y for DataTexture (v=0 is bottom)
          const i = (texR * 10 + c) * 4;
          data[i] = map[r][c]; // R channel
          data[i+1] = 0;
          data[i+2] = 0;
          data[i+3] = 1;
        }
      }
      policyTexture.needsUpdate = true;
      policyTexture.needsUpdate = true;
    }
  }, [currentPolicy, policyTexture]);

  useFrame(async (state, delta) => {
    if (!meshRef.current || !computeNode || !resetComputeNode || !aggregateAttribute) return;
    if (!pass0Node || !pass1Node || !pass2aNode || !pass2bNode || !pass2cNode || !pass3Node || !pass4Node) return;
    
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
    await (state.gl as any).computeAsync(pass2aNode);
    await (state.gl as any).computeAsync(pass2bNode);
    await (state.gl as any).computeAsync(pass2cNode);
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
        
        let policyScalar = 0.1;
        if (currentPolicy?.policy_speed_map) {
          let sum = 0;
          for(let r=0; r<10; r++) {
            for(let c=0; c<10; c++) sum += currentPolicy.policy_speed_map[r][c];
          }
          policyScalar = sum / 100;
        }

        // 1. High-frequency telemetry for ChartGPU dashboard
        window.dispatchEvent(new CustomEvent('abm-telemetry', {
          detail: {
            timestamp: Date.now(),
            actual_speed: globalAvgSpeed,
            policy_speed: policyScalar,
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
