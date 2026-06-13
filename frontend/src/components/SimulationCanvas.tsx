'use client';
import { useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useSimulationBridge } from '@/hooks/useSimulationBridge';

// The massive scale WebGPU is capable of
const AGENT_COUNT = 10000;

function MicroEngine() {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const { currentPolicy, isMacroThinking, sendAggregateStats } = useSimulationBridge();
  
  // Random starting positions for our 'Turtles'
  const dummyPositions = useMemo(() => {
    const array = new Float32Array(AGENT_COUNT * 3);
    for (let i = 0; i < AGENT_COUNT; i++) {
      array[i * 3] = (Math.random() - 0.5) * 50;
      array[i * 3 + 1] = (Math.random() - 0.5) * 50;
      array[i * 3 + 2] = 0;
    }
    return array;
  }, []);

  const dummyObject = useMemo(() => new THREE.Object3D(), []);

  useFrame((state, delta) => {
    if (!meshRef.current) return;
    
    // --- LOCKSTEP TIME SYNCHRONIZATION ---
    // If the Shachi Macro Agents are thinking, the entire physical world freezes
    if (isMacroThinking) {
      return; 
    }

    // Every 5 real-time seconds, trigger the LLMs
    if (state.clock.elapsedTime % 5 < delta) { 
      sendAggregateStats({
        active_agents: AGENT_COUNT,
        average_speed: currentPolicy?.movement_speed || 0.5,
        tick: state.clock.getElapsedTime(),
        system_status: "Awaiting LLM Instructions"
      });
    }

    // Execute Micro Engine physics
    // Note: In the final WebGPU build, this is handled via TSL ComputeNode
    // For this WebGL React fallback, we apply uniform rotation based on policy
    meshRef.current.rotation.z += delta * (currentPolicy?.movement_speed || 0.1);
  });

  useEffect(() => {
    if (meshRef.current) {
      for (let i = 0; i < AGENT_COUNT; i++) {
        dummyObject.position.set(dummyPositions[i*3], dummyPositions[i*3+1], dummyPositions[i*3+2]);
        dummyObject.updateMatrix();
        meshRef.current.setMatrixAt(i, dummyObject.matrix);
      }
      meshRef.current.instanceMatrix.needsUpdate = true;
    }
  }, [dummyPositions, dummyObject]);

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, AGENT_COUNT]}>
      <circleGeometry args={[0.15, 8]} />
      {/* Visual cue: turn red if waiting for Macro Agents, green if running */}
      <meshBasicMaterial color={isMacroThinking ? "#ff3366" : "#00ff88"} />
    </instancedMesh>
  );
}

export default function SimulationCanvas() {
  return (
    <div className="w-full h-screen bg-neutral-950 relative">
      {/* UI Overlay */}
      <div className="absolute top-4 left-4 z-10 text-white font-mono text-sm bg-black/50 p-4 rounded-lg border border-neutral-800">
        <h1 className="text-xl font-bold mb-2">abm.gl</h1>
        <p>Micro Agents (GPU): 10,000</p>
        <p>Macro Engine (Python): Active</p>
      </div>

      <Canvas camera={{ position: [0, 0, 50], fov: 75 }}>
        <MicroEngine />
      </Canvas>
    </div>
  );
}
