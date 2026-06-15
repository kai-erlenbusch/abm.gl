'use client';
import { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
// @ts-ignore
import { WebGPURenderer, StorageInstancedBufferAttribute, StorageBufferAttribute, PointsNodeMaterial } from 'three/webgpu';
// @ts-ignore
import { uniform, storage, positionLocal, color, instanceIndex, vertexIndex, texture, select, uint, vec2, vec3 } from 'three/tsl';
import { useSimulationBridge } from '@/hooks/useSimulationBridge';
import { useSimulationStore } from '@/store/simulationStore';
import { flockingBehavior, resetAggregate, spatialResetNode, spatialCountNode, spatialPrefixSum_ChunkNode, spatialPrefixSum_BlockNode, spatialPrefixSum_ScatterNode, spatialScatterNode, spatialCollisionNode, setupEpidemicNode, telemetryAggregateNode } from './TslPrimitives';
import DashboardOverlay from './DashboardOverlay';

// The massive scale WebGPU is capable of

function MicroEngine({ agentCount }: { agentCount: number }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const { currentPolicy, sendAggregateStats } = useSimulationBridge();
  
  const [computeNode, setComputeNode] = useState<any>(null);
  const [resetComputeNode, setResetComputeNode] = useState<any>(null);
  const [telemetryNode, setTelemetryNode] = useState<any>(null);
  
  const [setupNodePass, setSetupNodePass] = useState<any>(null);
  const needsSetupRef = useRef(true);
  const lastSetupTrigger = useRef(0);
  const lastFrameDuration = useRef(0);

  const dynamicParams = useSimulationStore(state => state.dynamicParams);
  const setupTrigger = useSimulationStore(state => state.setupTrigger);

  const infectionRadiusUniform = useMemo(() => uniform(0.2), []);
  const initialInfectedUniform = useMemo(() => uniform(2.0), []);
  const transmissionProbUniform = useMemo(() => uniform(1.0), []);
  const recoveryTimeUniform = useMemo(() => uniform(600.0), []);
  const collisionFidelityUniform = useMemo(() => uniform(8), []);
  const deltaUniform = useMemo(() => uniform(1.0), []); // frame delta
  const seedUniform = useMemo(() => uniform(0.0), []);

  useEffect(() => {
    infectionRadiusUniform.value = dynamicParams.infection_radius ?? 0.2;
    transmissionProbUniform.value = dynamicParams.transmission_probability ?? 1.0;
    recoveryTimeUniform.value = (dynamicParams.recovery_time ?? 60.0) * 60.0; // Assuming 60 fps frames
    
    // Convert N initial agents into a physical radius 
    // Area = 50x50 = 2500. Density = agentCount / 2500.
    // N = pi * r^2 * density => r = sqrt(N / (density * pi))
    const n = dynamicParams.initial_infected ?? 100;
    const density = agentCount / 2500.0;
    initialInfectedUniform.value = Math.sqrt(n / (density * Math.PI));
    collisionFidelityUniform.value = dynamicParams.collision_fidelity ?? 8;
  }, [dynamicParams, infectionRadiusUniform, initialInfectedUniform, transmissionProbUniform, recoveryTimeUniform, collisionFidelityUniform]);

  useEffect(() => {
    if (setupTrigger > lastSetupTrigger.current) {
      needsSetupRef.current = true;
      lastSetupTrigger.current = setupTrigger;
    }
  }, [setupTrigger]);

  const [aggregateAttribute, setAggregateAttribute] = useState<any>(null);
  const [material, setMaterial] = useState<any>(null);
  const [pass0Node, setPass0Node] = useState<any>(null);
  const [pass1Node, setPass1Node] = useState<any>(null);
  const [pass2aNode, setPass2aNode] = useState<any>(null);
  const [pass2bNode, setPass2bNode] = useState<any>(null);
  const [pass2cNode, setPass2cNode] = useState<any>(null);
  const [pass3Node, setPass3Node] = useState<any>(null);
  const [pass4Node, setPass4Node] = useState<any>(null);

  const policyTexture = useMemo(() => {
    const data = new Uint8Array(10 * 10 * 4);
    for (let i = 0; i < data.length; i += 4) {
      data[i] = 25;   // R
      data[i+1] = 0;  // G
      data[i+2] = 0;  // B
      data[i+3] = 1;  // A
    }
    const tex = new THREE.DataTexture(data, 10, 10, THREE.RGBAFormat);
    tex.needsUpdate = true;
    return tex;
  }, []);
  const lastReadbackRef = useRef(0);
  const readbackPendingRef = useRef(false);
  const aggregateDataRef = useRef(new Uint32Array(400));
  const gridRef = useRef(
    Array.from({ length: 10 }, () =>
      Array.from({ length: 10 }, () => ({
        density: 0,
        average_speed: 0,
        infected_count: 0,
        recovered_count: 0
      }))
    )
  );

  useEffect(() => {
    // 1. Data Buffers
    const posArray = new Float32Array(agentCount * 2);
    const velArray = new Float32Array(agentCount * 2);
    const infectionArray = new Uint32Array(agentCount);
    for (let i = 0; i < agentCount; i++) {
      posArray[i * 2] = (Math.random() * 50) - 25;
      posArray[i * 2 + 1] = (Math.random() * 50) - 25;
      
      const angle = Math.random() * Math.PI * 2;
      velArray[i * 2] = Math.cos(angle);
      velArray[i * 2 + 1] = Math.sin(angle);
    }
    
    const posAttr = new StorageBufferAttribute(posArray, 2);
    const velAttr = new StorageBufferAttribute(velArray, 2);
    const infectionAttr = new StorageBufferAttribute(infectionArray, 1);
    
    // Recovery timers
    const timerArray = new Float32Array(agentCount);
    const timerAttr = new StorageBufferAttribute(timerArray, 1);
    
    const aggArray = new Uint32Array(400); // 10x10 grid * 4 stats (speed, count, infected, recovered)
    const aggAttr = new StorageBufferAttribute(aggArray, 1);
    setAggregateAttribute(aggAttr);

    const positionsNode = storage(posAttr, 'vec2', agentCount);
    const velocitiesNode = storage(velAttr, 'vec2', agentCount);
    const infectionNode = storage(infectionAttr, 'uint', agentCount);
    const timerNode = storage(timerAttr, 'float', agentCount);
    const aggregateNode = storage(aggAttr, 'uint', 400).toAtomic();
    const policyMapTextureNode = texture(policyTexture);
    
    // 2. TSL ComputeNode Implementation
    // @ts-ignore
    setSetupNodePass(setupEpidemicNode(positionsNode, velocitiesNode, infectionNode, timerNode, seedUniform, initialInfectedUniform, recoveryTimeUniform).compute(agentCount));

    // @ts-ignore
    const behaviorNode = flockingBehavior(positionsNode, velocitiesNode, policyMapTextureNode, infectionNode, timerNode, deltaUniform);
    setComputeNode(behaviorNode.compute(agentCount));

    // @ts-ignore
    const resetNode = resetAggregate(aggregateNode.toAtomic());
    setResetComputeNode(resetNode.compute(400)); // 400 threads for 400 cells

    // @ts-ignore
    const telNode = telemetryAggregateNode(positionsNode, velocitiesNode, policyMapTextureNode, aggregateNode.toAtomic(), infectionNode);
    setTelemetryNode(telNode.compute(agentCount));

    // 3. Spatial Grid Buffers
    const cellCountArray = new Uint32Array(10240);
    const cellOffsetArray = new Uint32Array(10240);
    const cellOffsetAtomicArray = new Uint32Array(10240);
    const chunkSumsArray = new Uint32Array(40); // 10240 / 256 = 40 chunks
    const sortedAgentArray = new Uint32Array(agentCount);
    const sortedPosArray = new Float32Array(agentCount * 2);
    const sortedVelArray = new Float32Array(agentCount * 2);
    
    const countAttr = new StorageBufferAttribute(cellCountArray, 1);
    const offsetAttr = new StorageBufferAttribute(cellOffsetArray, 1);
    const offsetAtomicAttr = new StorageBufferAttribute(cellOffsetAtomicArray, 1);
    const chunkSumsAttr = new StorageBufferAttribute(chunkSumsArray, 1);
    const sortedAttr = new StorageBufferAttribute(sortedAgentArray, 1);
    const sortedPosAttr = new StorageBufferAttribute(sortedPosArray, 2);
    const sortedVelAttr = new StorageBufferAttribute(sortedVelArray, 2);
    
    const countAtomicNode = storage(countAttr, 'uint', 10240).toAtomic();
    const countNode = storage(countAttr, 'uint', 10240);
    const offsetNode = storage(offsetAttr, 'uint', 10240);
    const offsetAtomicNode = storage(offsetAtomicAttr, 'uint', 10240).toAtomic();
    const chunkSumsNode = storage(chunkSumsAttr, 'uint', 40);
    const sortedNode = storage(sortedAttr, 'uint', agentCount);
    const sortedPosNode = storage(sortedPosAttr, 'vec2', agentCount);
    const sortedVelNode = storage(sortedVelAttr, 'vec2', agentCount);

    // @ts-ignore
    setPass0Node(spatialResetNode(countAtomicNode, offsetAtomicNode).compute(10240));
    
    // @ts-ignore
    const pass1 = spatialCountNode(positionsNode, countAtomicNode, uint(agentCount)).compute(agentCount);
    pass1.workgroupSize = [256, 1, 1];
    setPass1Node(pass1);
    
    // @ts-ignore
    const pass2a = spatialPrefixSum_ChunkNode(countNode, offsetNode, chunkSumsNode).compute(10240);
    pass2a.workgroupSize = [256, 1, 1];
    setPass2aNode(pass2a);
    
    // @ts-ignore
    const pass2b = spatialPrefixSum_BlockNode(chunkSumsNode).compute(1);
    setPass2bNode(pass2b);
    
    // @ts-ignore
    const pass2c = spatialPrefixSum_ScatterNode(offsetNode, offsetAtomicNode, chunkSumsNode).compute(10240);
    pass2c.workgroupSize = [256, 1, 1];
    setPass2cNode(pass2c);
    
    // @ts-ignore
    setPass3Node(spatialScatterNode(positionsNode, velocitiesNode, offsetAtomicNode, sortedNode, sortedPosNode, sortedVelNode, uint(agentCount)).compute(agentCount));
    // @ts-ignore
    setPass4Node(spatialCollisionNode(positionsNode, velocitiesNode, countNode, offsetNode, sortedNode, sortedPosNode, sortedVelNode, infectionNode, timerNode, infectionRadiusUniform, transmissionProbUniform, recoveryTimeUniform, collisionFidelityUniform, seedUniform, uint(agentCount)).compute(agentCount));

    // 4. WebGPU Material Binding
    const mat = new PointsNodeMaterial();
    mat.size = 2.0;
    mat.positionNode = positionLocal.add(vec3(positionsNode.element(vertexIndex), 0.0));
    
    const stateVal = infectionNode.element(vertexIndex);
    mat.colorNode = select(
      stateVal.equal(uint(1)), 
      vec3(1.0, 0.0, 0.0), // Red (Infected)
      select(
        stateVal.equal(uint(2)),
        vec3(0.0, 0.5, 1.0), // Blue (Recovered)
        vec3(0.0, 1.0, 0.5)  // Green (Susceptible)
      )
    );
    
    setMaterial(mat);

    return () => {
      posAttr.dispose();
      velAttr.dispose();
      infectionAttr.dispose();
      timerAttr.dispose();
      aggAttr.dispose();
      countAttr.dispose();
      offsetAttr.dispose();
      offsetAtomicAttr.dispose();
      chunkSumsAttr.dispose();
      sortedAttr.dispose();
      sortedPosAttr.dispose();
      sortedVelAttr.dispose();
      mat.dispose();
      policyTexture.dispose();
    };
  }, [policyTexture, seedUniform, initialInfectedUniform, recoveryTimeUniform, infectionRadiusUniform, transmissionProbUniform]);

  useEffect(() => {
    if (currentPolicy?.policy_speed_map) {
      const map = currentPolicy.policy_speed_map;
      const data = policyTexture.image.data;
      if (!data) return;
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          const texR = 9 - r; 
          const i = (texR * 10 + c) * 4;
          data[i] = map[r][c]; 
        }
      }
      policyTexture.needsUpdate = true;
    }
  }, [currentPolicy, policyTexture]);

  useFrame(async (state, delta) => {
    if (!meshRef.current || !computeNode || !resetComputeNode || !telemetryNode || !aggregateAttribute) return;
    if (!pass0Node || !pass1Node || !pass2aNode || !pass2bNode || !pass2cNode || !pass3Node || !pass4Node) return;
    
    if (!(state.gl as any).__initialized) return;

    const frameStart = performance.now();

    // Dispatch real frame event for FPS meter regardless of pause state
    window.dispatchEvent(new CustomEvent('abm-frame'));

    if (needsSetupRef.current && setupNodePass) {
        (state.gl as any).compute(setupNodePass);
        needsSetupRef.current = false;
    }

    const { isPaused, isMacroThinking, lastLlmSend, setLastLlmSend } = useSimulationStore.getState();
    if (!isPaused && !isMacroThinking) {
        try {
          (state.gl as any).compute(pass0Node);
          (state.gl as any).compute(pass1Node);
          
          // Pass 2: 3-Pass Parallel Prefix Sum
          (state.gl as any).compute(pass2aNode);
          (state.gl as any).compute(pass2bNode);
          (state.gl as any).compute(pass2cNode);
          
          // Pass 3: Scatter agents into sorted array
          (state.gl as any).compute(pass3Node);
          
          // Pass 4: Resolve Collisions & Viral Transmission
          (state.gl as any).compute(pass4Node);
          
          // Pass 5: Resolve Flocking/Steering Physics
          (state.gl as any).compute(computeNode);
        } catch (e) {
          console.warn("Compute pass error:", e);
        }
        
        // CPU Aggregation (Readback API)
        const time = performance.now();
        // Throttle telemetry readback dynamically based on frame drops. If last frame was > 33ms, scale to 500ms
        const readbackInterval = lastFrameDuration.current > 33 ? 500 : 100;
        
        if (time - lastReadbackRef.current > readbackInterval && !readbackPendingRef.current) {
          lastReadbackRef.current = time;
          readbackPendingRef.current = true;
          
          try {
            (state.gl as any).compute(resetComputeNode);
            (state.gl as any).compute(telemetryNode);

            const buffer = await (state.gl as any).backend.getArrayBufferAsync(aggregateAttribute);
            aggregateDataRef.current.set(new Uint32Array(buffer));
            const aggregateData = aggregateDataRef.current;

            let totalSpeed = 0;
            let totalCount = 0;
            let totalInfected = 0;
            let totalRecovered = 0;
            const grid = gridRef.current;
            
            if (Math.random() < 0.1) {
              console.log(`Buffer size: ${buffer.byteLength}, speed0: ${aggregateData[0]}, count0: ${aggregateData[1]}`);
            }

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
            const totalGlobalAgents = totalCount;
            
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
          } finally {
            readbackPendingRef.current = false;
          }
        }
    }
    
    lastFrameDuration.current = performance.now() - frameStart;
  });

  const dummyPositions = useMemo(() => new Float32Array(agentCount * 3), [agentCount]);

  if (!material) return null;

  return (
    <points ref={meshRef as any}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[dummyPositions, 3]} />
      </bufferGeometry>
      <primitive object={material} attach="material" />
    </points>
  );
}

export default function SimulationCanvas() {
  const dynamicParams = useSimulationStore(state => state.dynamicParams);
  const agentCount = dynamicParams.agent_count ?? 100000;

  return (
    <div className="w-full h-screen bg-neutral-950 relative">
      <DashboardOverlay />

      {/* WebGPURenderer Injection with Async Init Hack */}
      <Canvas 
        gl={(props) => {
          const renderer = new WebGPURenderer(props as any) as any;
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
        <MicroEngine key={`agent-count-${agentCount}`} agentCount={agentCount} />
        <ambientLight intensity={0.5} />
      </Canvas>
    </div>
  );
}
