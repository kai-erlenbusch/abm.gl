'use client';
import { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
// @ts-ignore
import { WebGPURenderer, StorageInstancedBufferAttribute, StorageBufferAttribute, MeshBasicNodeMaterial } from 'three/webgpu';
// @ts-ignore
import { uniform, storage, positionLocal, color, instanceIndex, texture, select, uint, vec3 } from 'three/tsl';
import { useSimulationBridge } from '@/hooks/useSimulationBridge';
import { useSimulationStore } from '@/store/simulationStore';
import { flockingBehavior, resetAggregate, spatialResetNode, spatialCountNode, spatialPrefixSum_SequentialNode, spatialScatterNode, spatialCollisionNode, setupEpidemicNode } from './TslPrimitives';
import DashboardOverlay from './DashboardOverlay';

// The massive scale WebGPU is capable of
const AGENT_COUNT = 100000;

function MicroEngine() {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const { currentPolicy, isMacroThinking, sendAggregateStats } = useSimulationBridge();
  
  const [computeNode, setComputeNode] = useState<any>(null);
  const [resetComputeNode, setResetComputeNode] = useState<any>(null);
  
  const [setupNodePass, setSetupNodePass] = useState<any>(null);
  const needsSetupRef = useRef(true);
  const lastSetupTrigger = useRef(0);

  const dynamicParams = useSimulationStore(state => state.dynamicParams);
  const setupTrigger = useSimulationStore(state => state.setupTrigger);

  const infectionRadiusUniform = useMemo(() => uniform(0.2), []);
  const initialInfectedUniform = useMemo(() => uniform(2.0), []);
  const transmissionProbUniform = useMemo(() => uniform(1.0), []);
  const recoveryTimeUniform = useMemo(() => uniform(600.0), []);
  const deltaUniform = useMemo(() => uniform(1.0), []); // frame delta
  const seedUniform = useMemo(() => uniform(0.0), []);

  useEffect(() => {
    infectionRadiusUniform.value = dynamicParams.infection_radius ?? 0.2;
    transmissionProbUniform.value = dynamicParams.transmission_probability ?? 1.0;
    recoveryTimeUniform.value = (dynamicParams.recovery_time ?? 60.0) * 60.0; // Assuming 60 fps frames
    
    // Convert N initial agents into a physical radius 
    // Area = 50x50 = 2500. Density = 100k / 2500 = 40 agents per unit area.
    // N = pi * r^2 * 40 => r = sqrt(N / (40 * pi))
    const n = dynamicParams.initial_infected ?? 100;
    initialInfectedUniform.value = Math.sqrt(n / (40 * Math.PI));
  }, [dynamicParams, infectionRadiusUniform, initialInfectedUniform, transmissionProbUniform, recoveryTimeUniform]);

  useEffect(() => {
    if (setupTrigger > lastSetupTrigger.current) {
        lastSetupTrigger.current = setupTrigger;
        seedUniform.value = Math.random();
        needsSetupRef.current = true;
    }
  }, [setupTrigger, seedUniform]);
  
  // Spatial Grid Nodes
  const [pass0Node, setPass0Node] = useState<any>(null);
  const [pass1Node, setPass1Node] = useState<any>(null);
  const [pass2aNode, setPass2aNode] = useState<any>(null);
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
    
    // CPU initializes as 0, GPU setupNodePass handles the Ground Zero seeding
    const infectionArray = new Uint32Array(AGENT_COUNT);
    const infectionAttr = new StorageInstancedBufferAttribute(infectionArray, 1);
    
    const timerArray = new Float32Array(AGENT_COUNT);
    const timerAttr = new StorageInstancedBufferAttribute(timerArray, 1);
    
    const aggArray = new Uint32Array(400); // 10x10 grid * 4 stats (speed, count, infected, recovered)
    const aggAttr = new StorageBufferAttribute(aggArray, 1);
    setAggregateAttribute(aggAttr);

    const positionsNode = storage(posAttr, 'vec3', AGENT_COUNT);
    const velocitiesNode = storage(velAttr, 'vec3', AGENT_COUNT);
    const infectionNode = storage(infectionAttr, 'uint', AGENT_COUNT);
    const timerNode = storage(timerAttr, 'float', AGENT_COUNT);
    const aggregateNode = storage(aggAttr, 'uint', 400).toAtomic();
    const policyMapTextureNode = texture(policyTexture);
    
    // 2. TSL ComputeNode Implementation
    // @ts-ignore
    setSetupNodePass(setupEpidemicNode(positionsNode, velocitiesNode, infectionNode, timerNode, seedUniform, initialInfectedUniform, recoveryTimeUniform).compute(AGENT_COUNT));

    // @ts-ignore
    const behaviorNode = flockingBehavior(positionsNode, velocitiesNode, policyMapTextureNode, aggregateNode, infectionNode, timerNode, deltaUniform);
    setComputeNode(behaviorNode.compute(AGENT_COUNT));

    // @ts-ignore
    const resetNode = resetAggregate(aggregateNode.toAtomic());
    setResetComputeNode(resetNode.compute(400)); // 400 threads for 400 cells

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
    const countAtomicNode = storage(countAttr, 'uint', 10240).toAtomic();
    const offsetNode = storage(offsetAttr, 'uint', 10240);
    const offsetAtomicNode = storage(offsetAtomicAttr, 'uint', 10240).toAtomic();
    const chunkSumsNode = storage(chunkSumsAttr, 'uint', 64);
    const sortedNode = storage(sortedAttr, 'uint', AGENT_COUNT);

    // @ts-ignore
    setPass0Node(spatialResetNode(countAtomicNode, offsetAtomicNode).compute(10240));
    
    // @ts-ignore
    const pass1 = spatialCountNode(positionsNode, countAtomicNode).compute(100096);
    pass1.workgroupSize = [256, 1, 1];
    setPass1Node(pass1);
    
    // --------------------------------------------------------
    // PASS 2: Prefix Sum
    // Calculates global offsets for each cell
    // Single-threaded pass (O(N)) because 10,000 cells is tiny for GPU
    // --------------------------------------------------------
    // @ts-ignore
    const pass2 = spatialPrefixSum_SequentialNode(countNode, offsetNode, offsetAtomicNode).compute(1);
    setPass2aNode(pass2);
    
    // --------------------------------------------------------
    // PASS 3: Scatter
    // Sorts agent indices into the sorted array based on grid cells
    // --------------------------------------------------------
    // @ts-ignore
    setPass3Node(spatialScatterNode(positionsNode, offsetAtomicNode, sortedNode).compute(AGENT_COUNT));
    // Here we use the standard nodes to read the populated data without atomic locks
    // @ts-ignore
    setPass4Node(spatialCollisionNode(positionsNode, velocitiesNode, countNode, offsetNode, sortedNode, infectionNode, timerNode, infectionRadiusUniform, transmissionProbUniform, recoveryTimeUniform, seedUniform).compute(AGENT_COUNT));

    // 4. WebGPU Material Binding
    const mat = new MeshBasicNodeMaterial();
    mat.positionNode = positionLocal.add(positionsNode.element(instanceIndex)); 
    
    const stateVal = infectionNode.element(instanceIndex);
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
  }, [policyTexture]);

  useEffect(() => {
    if (currentPolicy?.policy_speed_map) {
      const map = currentPolicy.policy_speed_map;
      const data = policyTexture.image.data;
      if (!data) return;
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
    }
  }, [currentPolicy, policyTexture]);

  useFrame(async (state, delta) => {
    if (!meshRef.current || !computeNode || !resetComputeNode || !aggregateAttribute) return;
    if (!pass0Node || !pass1Node || !pass2aNode || !pass3Node || !pass4Node) return;
    
    // Check if our WebGPU hack has initialized
    if (!(state.gl as any).__initialized) return;

    // Phase 8: Dynamic Setup Trigger (moved above pause check so you can preview setup)
    if (needsSetupRef.current && setupNodePass) {
        await (state.gl as any).computeAsync(setupNodePass);
        needsSetupRef.current = false;
    }

    const { isPaused, lastLlmSend, setLastLlmSend } = useSimulationStore.getState();
    if (!isPaused) {
        try {
          // Pass 0: Reset cell counts and offsets
          await (state.gl as any).computeAsync(pass0Node);
          
          // Pass 1: Count agents per cell
          await (state.gl as any).computeAsync(pass1Node);
          
          // Pass 2: Sequential Prefix Sum
          await (state.gl as any).computeAsync(pass2aNode);
          
          // Pass 3: Scatter agents into sorted array
          await (state.gl as any).computeAsync(pass3Node);
          
          // Pass 4: Resolve Collisions & Viral Transmission
          await (state.gl as any).computeAsync(pass4Node);
          
          // Pass 5: Resolve Flocking/Steering Physics
          await (state.gl as any).computeAsync(resetComputeNode);
          await (state.gl as any).computeAsync(computeNode);
        } catch (e) {
          console.warn("Compute pass error:", e);
        }
        
        // CPU Aggregation (Readback API)
        const time = performance.now();
        if (time - lastReadbackRef.current > 100) {
          lastReadbackRef.current = time;
          
          try {
            const buffer = await (state.gl as any).backend.getArrayBufferAsync(aggregateAttribute);
            const aggregateData = new Uint32Array(buffer);

            let totalSpeed = 0;
            let totalCount = 0;
            let totalInfected = 0;
            let totalRecovered = 0;
            const grid = [];

            for (let r = 0; r < 10; r++) {
              const row = [];
              for (let c = 0; c < 10; c++) {
                const idx = (r * 10 + c) * 4;
                const speed = aggregateData[idx] / 100.0;
                const count = aggregateData[idx + 1];
                const infected = aggregateData[idx + 2];
                const recovered = aggregateData[idx + 3];

                row.push({
                  density: count,
                  avg_speed: count > 0 ? speed / count : 0,
                  infected_count: infected,
                  recovered_count: recovered,
                });

                totalSpeed += speed;
                totalCount += count;
                totalInfected += infected;
                totalRecovered += recovered;
              }
              grid.push(row);
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
          }
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
        <MicroEngine />
      </Canvas>
    </div>
  );
}
