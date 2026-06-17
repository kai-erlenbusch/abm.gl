'use client';
import dynamic from 'next/dynamic';
import DashboardOverlay from '@/components/DashboardOverlay';
import { useSimulationStore } from '@/store/simulationStore';
import { useMemo, useEffect, useState, useRef } from 'react';
import * as THREE from 'three';
import { PointsNodeMaterial, StorageBufferAttribute } from 'three/webgpu';
// @ts-ignore
import { uniform, positionLocal, vec3, select, uint, vertexIndex, color, storage } from 'three/tsl';

import { AgentDataStore } from '@/engine/AgentDataStore';
import { SpatialGrid } from '@/engine/SpatialGrid';
import { 
  setupEpidemicNode, 
  flockingBehavior, 
  telemetryAggregateNode, 
  resetAggregate, 
  spatialCollisionNode 
} from '@/components/TslPrimitives';

const AbmCanvas = dynamic(() => import('@/components/AbmCanvas'), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center w-full h-screen bg-neutral-950 text-emerald-400 font-mono text-sm">
      <div className="flex flex-col items-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-emerald-500 mb-4"></div>
        <div>Initializing WebGPU Compute Pipeline...</div>
      </div>
    </div>
  )
});

export default function Home() {
  const agentCount = useSimulationStore(state => state.dynamicParams.agent_count ?? 100000);
  const dynamicParams = useSimulationStore(state => state.dynamicParams);

  // Uniforms
  const infectionRadiusUniform = useMemo(() => uniform(0.2), []);
  const initialInfectedUniform = useMemo(() => uniform(2.0), []);
  const transmissionProbUniform = useMemo(() => uniform(1.0), []);
  const recoveryTimeUniform = useMemo(() => uniform(600.0), []);
  const deltaUniform = useMemo(() => uniform(1.0), []);
  const seedUniform = useMemo(() => uniform(0.0), []);

  useEffect(() => {
    infectionRadiusUniform.value = dynamicParams.infection_radius ?? 0.2;
    transmissionProbUniform.value = dynamicParams.transmission_probability ?? 1.0;
    recoveryTimeUniform.value = dynamicParams.recovery_time ?? 60.0;
    const n = dynamicParams.initial_infected ?? 100;
    const density = agentCount / 2500.0;
    initialInfectedUniform.value = Math.sqrt(n / (density * Math.PI));
  }, [dynamicParams, agentCount, infectionRadiusUniform, transmissionProbUniform, recoveryTimeUniform, initialInfectedUniform]);

  const { store, grid, material, setupPass, passes, resetComputeNode, telemetryNode, aggregateAttribute } = useMemo(() => {
    const store = new AgentDataStore(agentCount);
    store.addProperty('position', 2, 'vec2');
    store.addProperty('velocity', 2, 'vec2');
    store.addProperty('infection', 1, 'uint');
    store.addProperty('timer', 1, 'float');

    const grid = new SpatialGrid(100, 100, agentCount);
    const agentCountUniform = uniform(agentCount);
    
    // Setup Pass
    // @ts-ignore
    const setupPass = setupEpidemicNode(store.getNode('position'), store.getNode('velocity'), store.getNode('infection'), store.getNode('timer'), seedUniform, initialInfectedUniform, recoveryTimeUniform, agentCountUniform).compute(agentCount);

    // Dummy Policy Texture (for legacy flocking logic)
    const policyData = new Uint8Array(100 * 4);
    for (let i = 0; i < policyData.length; i += 4) { policyData[i] = 25; policyData[i+3] = 1; }
    const policyTex = new THREE.DataTexture(policyData, 10, 10, THREE.RGBAFormat);
    policyTex.needsUpdate = true;
    
    const positions = store.getNode('position');
    const velocities = store.getNode('velocity');
    const infection = store.getNode('infection');
    const timer = store.getNode('timer');
    
    const positions_next = store.getNextNode('position');
    const velocities_next = store.getNextNode('velocity');
    const infection_next = store.getNextNode('infection');
    const timer_next = store.getNextNode('timer');
    
    // Node passes
    // @ts-ignore
    const flockNode = flockingBehavior(positions, positions_next, velocities_next, infection_next, timer_next, deltaUniform, agentCountUniform).compute(agentCount);

    const passes = [
       grid.getResetNode(),
       grid.getCountNode(positions),
       grid.getPrefixSumChunkNode(),
       grid.getPrefixSumBlockNode(),
       grid.getPrefixSumScatterNode(),
       grid.getAgentScatterNode(positions, velocities),
       // @ts-ignore
       spatialCollisionNode(velocities, velocities_next, infection, infection_next, timer, timer_next, grid.nodes.count, grid.nodes.offset, grid.nodes.sortedIndices, grid.nodes.sortedPositions, infectionRadiusUniform, transmissionProbUniform, recoveryTimeUniform, seedUniform, agentCountUniform, deltaUniform).compute(agentCount),
       flockNode,
       // @ts-ignore
       store.getCopyPassNode(agentCountUniform).compute(agentCount)
    ];
    
    const mat = new PointsNodeMaterial();
    mat.size = 2.0;
    // @ts-ignore
    mat.positionNode = positionLocal.add(vec3(positions.element(vertexIndex), 0.0));
    
    const stateVal = infection.element(vertexIndex);
    // @ts-ignore
    mat.colorNode = select(
      stateVal.equal(uint(1)), 
      color(0xff0000), 
      select(
        stateVal.equal(uint(2)),
        color(0x0080ff), 
        color(0x00ff80)
      )
    );
    
    // --- Telemetry & Aggregation ---
    const aggregateData = new Uint32Array(400);
    // @ts-ignore
    const aggregateAttribute = new StorageBufferAttribute(aggregateData, 1);
    // @ts-ignore
    const aggregateBuffer = storage(aggregateAttribute, 'uint', aggregateData.length).toAtomic();
    
    // @ts-ignore
    const resetComputeNode = resetAggregate(aggregateBuffer).compute(400);
    // @ts-ignore
    const telemetryNode = telemetryAggregateNode(positions, velocities, policyTex, aggregateBuffer, infection, agentCountUniform).compute(agentCount);

    return { 
       store, grid, material: mat, setupPass, passes,
       resetComputeNode, telemetryNode, aggregateAttribute
    };
  }, [agentCount]);

  // Prevent Massive GPU Memory Leaks when agentCount changes
  useEffect(() => {
    return () => {
      store.dispose();
      grid.dispose();
      aggregateAttribute.dispose();
    };
  }, [store, grid, aggregateAttribute]);

  // Readback references
  const lastReadbackRef = useRef(0);
  const readbackPendingRef = useRef(false);
  const aggregateDataRef = useRef(new Uint32Array(400));
  const gridRef = useRef(Array.from({ length: 10 }, () => Array.from({ length: 10 }, () => ({ density: 0, average_speed: 0, infected_count: 0, recovered_count: 0 }))));
  const lastFrameDuration = useRef(16);
  const diagnosticStepPendingRef = useRef(false);

  useEffect(() => {
      const handleDiag = () => {
          diagnosticStepPendingRef.current = true;
          useSimulationStore.getState().setIsPaused(true);
      };
      window.addEventListener('abm-diagnostic-step', handleDiag);
      return () => window.removeEventListener('abm-diagnostic-step', handleDiag);
  }, []);

  const renderCallback = async (gl: any, delta: number) => {
    deltaUniform.value = delta;
    const frameStart = performance.now();
    
    // Throttle telemetry dynamically
    const readbackInterval = lastFrameDuration.current > 33 ? 500 : 100;
    
    if (diagnosticStepPendingRef.current) {
        diagnosticStepPendingRef.current = false;
        console.log("=== EXECUTING MATHEMATICAL DIAGNOSTIC STEP ===");
        
        try {
            // Read BEFORE state
            const posBufferBefore = await gl.backend.getArrayBufferAsync(store.attributes['position']);
            const velBufferBefore = await gl.backend.getArrayBufferAsync(store.attributes['velocity']);
            const infBufferBefore = await gl.backend.getArrayBufferAsync(store.attributes['infection']);
            
            const posA = new Float32Array(posBufferBefore);
            const velA = new Float32Array(velBufferBefore);
            const infA = new Uint32Array(infBufferBefore);
            
            // Execute ONE frame explicitly
            for (const pass of passes) {
                 if (pass) gl.compute(pass);
            }
            
            // Read AFTER state (from _next buffers)
            const posBufferAfter = await gl.backend.getArrayBufferAsync(store.attributes['position_next']);
            const velBufferAfter = await gl.backend.getArrayBufferAsync(store.attributes['velocity_next']);
            const infBufferAfter = await gl.backend.getArrayBufferAsync(store.attributes['infection_next']);
            
            const posB = new Float32Array(posBufferAfter);
            const velB = new Float32Array(velBufferAfter);
            const infB = new Uint32Array(infBufferAfter);
            
            // Read Copied state (from Primary buffers to ensure copyPass worked)
            const posBufferCopied = await gl.backend.getArrayBufferAsync(store.attributes['position']);
            
            const posC = new Float32Array(posBufferCopied);
            
            const diagnosticData = [];
            for (let i = 0; i < 10; i++) {
                diagnosticData.push({
                    AgentID: i,
                    PosX_Before: posA[i*2].toFixed(4),
                    PosY_Before: posA[i*2+1].toFixed(4),
                    VelX_Before: velA[i*2].toFixed(4),
                    VelY_Before: velA[i*2+1].toFixed(4),
                    Inf_Before: infA[i],
                    
                    PosX_Next: posB[i*2].toFixed(4),
                    PosY_Next: posB[i*2+1].toFixed(4),
                    VelX_Next: velB[i*2].toFixed(4),
                    VelY_Next: velB[i*2+1].toFixed(4),
                    Inf_Next: infB[i],
                    
                    PosX_Copied: posC[i*2].toFixed(4),
                    PosY_Copied: posC[i*2+1].toFixed(4)
                });
            }
            console.log(JSON.stringify(diagnosticData, null, 2));
            console.log("=== DIAGNOSTIC COMPLETE ===");
            
        } catch (e) {
            console.error("Diagnostic Readback Error:", e);
        }
    } else if (frameStart - lastReadbackRef.current > readbackInterval && !readbackPendingRef.current) {
        lastReadbackRef.current = frameStart;
        readbackPendingRef.current = true;
        
        try {
            gl.compute(resetComputeNode);
            gl.compute(telemetryNode);
            
            const buffer = await gl.backend.getArrayBufferAsync(aggregateAttribute);
            aggregateDataRef.current.set(new Uint32Array(buffer));
            const aggregateData = aggregateDataRef.current;
            
            let totalSpeed = 0;
            let totalCount = 0;
            let totalInfected = 0;
            let totalRecovered = 0;
            const grid = gridRef.current;
            
            for (let r = 0; r < 10; r++) {
               for (let c = 0; c < 10; c++) {
                 const idx = (r * 10 + c) * 4;
                 const speed = aggregateData[idx] / 100.0;
                 const count = aggregateData[idx + 1];
                 const infected = aggregateData[idx + 2];
                 const recovered = aggregateData[idx + 3];
                 
                 const cell = grid[r][c];
                 cell.density = count;
                 cell.average_speed = count > 0 ? speed / count : 0;
                 cell.infected_count = infected;
                 cell.recovered_count = recovered;
                 
                 totalSpeed += speed;
                 totalCount += count;
                 totalInfected += infected;
                 totalRecovered += recovered;
               }
            }
            
            const globalAvgSpeed = totalCount > 0 ? totalSpeed / totalCount : 0;
            let policyScalar = 0.1; // Hardcoded default for the moment since policy maps are simplified
            
            window.dispatchEvent(new CustomEvent('abm-telemetry', {
                detail: {
                    timestamp: Date.now(),
                    actual_speed: globalAvgSpeed,
                    policy_speed: policyScalar,
                    grid: grid,
                    total_count: totalCount,
                    total_infected: totalInfected,
                    total_recovered: totalRecovered,
                    total_healthy: totalCount - totalInfected - totalRecovered
                }
            }));
            
        } catch (e) {
            console.warn("Readback error", e);
        } finally {
            readbackPendingRef.current = false;
        }
    }
    lastFrameDuration.current = performance.now() - frameStart;
  };

  return (
    <main className="flex min-h-screen flex-col">
      <DashboardOverlay />
      <AbmCanvas 
         agentCount={agentCount} 
         material={material} 
         setupPass={setupPass} 
         computePasses={passes} 
         renderCallback={renderCallback}
      />
    </main>
  );
}
