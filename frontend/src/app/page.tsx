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
import { flockingBehavior } from '@/engine/physics/Boids';
import { setupEpidemicNode, epidemicCollisionNode } from '@/models/Epidemiology';
import { telemetryAggregateNode, resetAggregate } from '@/engine/Telemetry';

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
  const worldSize = useSimulationStore(state => state.dynamicParams.world_size ?? 50.0);
  
  // Phase 4: Automate Spatial Cell Size (Max Interaction Radius)
  const infectionRadius = useSimulationStore(state => state.dynamicParams.infection_radius ?? 0.2);
  const cellSize = Math.max(0.5, infectionRadius * 1.5);

  const gridDimX = Math.ceil(worldSize / cellSize);
  const gridDimY = Math.ceil(worldSize / cellSize);

  // Uniforms
  const infectionRadiusUniform = useMemo(() => uniform(0.2), []);
  const initialInfectedUniform = useMemo(() => uniform(2.0), []);
  const transmissionProbUniform = useMemo(() => uniform(1.0), []);
  const recoveryTimeUniform = useMemo(() => uniform(600.0), []);
  const deltaUniform = useMemo(() => uniform(1.0), []);
  const seedUniform = useMemo(() => uniform(0.0), []);
  
  // Spatial Scale Uniforms
  const worldSizeUniform = useMemo(() => uniform(worldSize), []);
  const worldOffsetUniform = useMemo(() => uniform(worldSize / 2.0), []);
  const cellSizeUniform = useMemo(() => uniform(cellSize), []);
  const gridDimXUniform = useMemo(() => uniform(gridDimX), []);
  const gridDimYUniform = useMemo(() => uniform(gridDimY), []);

  const { store, grid, material, setupPass, passes, resetComputeNodeA, telemetryNodeA, aggregateAttributeA, resetComputeNodeB, telemetryNodeB, aggregateAttributeB } = useMemo(() => {
    const store = new AgentDataStore(agentCount);
    store.addProperty('position', 2, 'vec2');
    store.addProperty('velocity', 2, 'vec2');
    store.addProperty('infection', 1, 'uint');
    store.addProperty('timer', 1, 'float');

    const grid = new SpatialGrid(gridDimX, gridDimY, agentCount, cellSize);
    const agentCountUniform = uniform(agentCount);
    
    // Setup Pass
    // @ts-ignore
    const setupPass = setupEpidemicNode(store.getNode('position'), store.getNode('velocity'), store.getNode('infection'), store.getNode('timer'), seedUniform, initialInfectedUniform, recoveryTimeUniform, agentCountUniform, worldSizeUniform).compute(agentCount);

    // Dummy Policy Texture
    const policyData = new Uint8Array(100 * 4);
    for (let i = 0; i < policyData.length; i += 4) { policyData[i] = 25; policyData[i+3] = 1; }
    const policyTex = new THREE.DataTexture(policyData, 10, 10, THREE.RGBAFormat);
    policyTex.needsUpdate = true;
    
    const positions = store.getNode('position');
    const velocities = store.getNode('velocity');
    const infection = store.getNode('infection');
    const timer = store.getNode('timer');
    
    // Node passes
    // @ts-ignore
    const flockNode = flockingBehavior(positions, velocities, infection, timer, deltaUniform, agentCountUniform, worldSizeUniform, worldOffsetUniform).compute(agentCount);

    const passes = [
       grid.getResetNode(),
       grid.getCountNode(positions, worldOffsetUniform, gridDimXUniform, gridDimYUniform, cellSizeUniform),
       grid.getPrefixSumChunkNode(),
       grid.getPrefixSumBlockNode(),
       grid.getPrefixSumScatterNode(),
       grid.getAgentScatterNode(positions, velocities, worldOffsetUniform, gridDimXUniform, gridDimYUniform, cellSizeUniform),
       // @ts-ignore
       epidemicCollisionNode(velocities, infection, timer, grid.nodes.count, grid.nodes.offset, grid.nodes.sortedIndices, grid.nodes.sortedPositions, infectionRadiusUniform, transmissionProbUniform, recoveryTimeUniform, seedUniform, agentCountUniform, deltaUniform, worldOffsetUniform, cellSizeUniform, gridDimXUniform, gridDimYUniform).compute(agentCount),
       flockNode
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
    // @ts-ignore
    const aggregateAttributeA = new StorageBufferAttribute(new Uint32Array(400), 1);
    // @ts-ignore
    const aggregateBufferA = storage(aggregateAttributeA, 'uint', 400).toAtomic();
    // @ts-ignore
    const resetComputeNodeA = resetAggregate(aggregateBufferA).compute(400);
    // @ts-ignore
    const telemetryNodeA = telemetryAggregateNode(positions, velocities, policyTex, aggregateBufferA, infection, agentCountUniform, worldSizeUniform, worldOffsetUniform).compute(agentCount);
    telemetryNodeA.workgroupSize = [256, 1, 1];

    // @ts-ignore
    const aggregateAttributeB = new StorageBufferAttribute(new Uint32Array(400), 1);
    // @ts-ignore
    const aggregateBufferB = storage(aggregateAttributeB, 'uint', 400).toAtomic();
    // @ts-ignore
    const resetComputeNodeB = resetAggregate(aggregateBufferB).compute(400);
    // @ts-ignore
    const telemetryNodeB = telemetryAggregateNode(positions, velocities, policyTex, aggregateBufferB, infection, agentCountUniform, worldSizeUniform, worldOffsetUniform).compute(agentCount);
    telemetryNodeB.workgroupSize = [256, 1, 1];

    return { 
       store, grid, material: mat, setupPass, passes,
       resetComputeNodeA, telemetryNodeA, aggregateAttributeA,
       resetComputeNodeB, telemetryNodeB, aggregateAttributeB
    };
  }, [agentCount, worldSize, cellSize]);

  // Prevent Massive GPU Memory Leaks when agentCount changes
  useEffect(() => {
    return () => {
      store.dispose();
      grid.dispose();
      aggregateAttributeA.dispose();
      aggregateAttributeB.dispose();
      material.dispose();
    };
  }, [store, grid, aggregateAttributeA, aggregateAttributeB, material]);

  // Readback references
  const frameIndexRef = useRef(0);
  const readbackPendingRef = useRef(false);
  const aggregateDataRef = useRef(new Uint32Array(400));
  const gridRef = useRef(Array.from({ length: 10 }, () => Array.from({ length: 10 }, () => ({ density: 0, average_speed: 0, infected_count: 0, recovered_count: 0 }))));
  const lastFrameDuration = useRef(16);

  const renderCallback = async (gl: any, delta: number) => {
    const state = useSimulationStore.getState();
    const params = state.dynamicParams;
    
    // Phase 3: Concurrent Sync & Dynamic Scale Sync
    infectionRadiusUniform.value = params.infection_radius ?? 0.2;
    transmissionProbUniform.value = params.transmission_probability ?? 1.0;
    recoveryTimeUniform.value = params.recovery_time ?? 60.0;
    
    const wSize = params.world_size ?? 50.0;
    
    // Phase 4: Automate Spatial Cell Size
    const currentInfectionRadius = params.infection_radius ?? 0.2;
    const cSize = Math.max(0.5, currentInfectionRadius * 1.5);
    
    const density = agentCount / (wSize * wSize);
    const n = params.initial_infected ?? 100;
    initialInfectedUniform.value = Math.sqrt(n / (density * Math.PI));
    
    worldSizeUniform.value = wSize;
    worldOffsetUniform.value = wSize / 2.0;
    cellSizeUniform.value = cSize;
    gridDimXUniform.value = Math.ceil(wSize / cSize);
    gridDimYUniform.value = Math.ceil(wSize / cSize);

    deltaUniform.value = delta;
    seedUniform.value = Math.random(); // Phase 2: Fix PRNG Stagnation

    const frameStart = performance.now();
    
    const currentFrame = frameIndexRef.current++;
    const isEven = currentFrame % 2 === 0;

    // Dispatch compute for the CURRENT frame
    if (isEven) {
        gl.compute(resetComputeNodeA);
        gl.compute(telemetryNodeA);
    } else {
        gl.compute(resetComputeNodeB);
        gl.compute(telemetryNodeB);
    }
    
    // Readback the PREVIOUS frame's buffer
    if (currentFrame > 0 && !readbackPendingRef.current) {
        readbackPendingRef.current = true;
        
        try {
            const attrToRead = isEven ? aggregateAttributeB : aggregateAttributeA;
            const buffer = await gl.backend.getArrayBufferAsync(attrToRead);
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
            let policyScalar = 0.1;
            
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
