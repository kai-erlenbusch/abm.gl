'use client';
import { useRef, useEffect, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { WebGPURenderer, Node } from 'three/webgpu';
Node.captureStackTrace = true;
import { useSimulationStore } from '@/store/simulationStore';

export interface AbmCanvasProps {
  agentCount: number;
  material: any; // PointsNodeMaterial
  setupPass: any; // Setup compute node pass
  computePasses: any[]; // Array of compute node passes to run every frame
  renderCallback?: (gl: any, delta: number) => void;
}

function AbmEngine({ agentCount, material, setupPass, computePasses, renderCallback }: AbmCanvasProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const needsSetupRef = useRef(true);
  const lastSetupTrigger = useRef(0);
  
  const setupTrigger = useSimulationStore(state => state.setupTrigger);
  const isPaused = useSimulationStore(state => state.isPaused);

  useEffect(() => {
    if (setupTrigger > lastSetupTrigger.current) {
      needsSetupRef.current = true;
      lastSetupTrigger.current = setupTrigger;
    }
  }, [setupTrigger]);

  useFrame(async (state, delta) => {
    if (!meshRef.current) return;
    if (!(state.gl as any).__initialized) return;

    // Trigger FPS telemetry event
    window.dispatchEvent(new CustomEvent('abm-frame'));

    const gl = state.gl as any;

    if (needsSetupRef.current && setupPass) {
        gl.compute(setupPass);
        needsSetupRef.current = false;
    }

    if (!isPaused && computePasses.length > 0) {
        try {
          for (const pass of computePasses) {
             if (pass) gl.compute(pass);
          }
        } catch (e) {
            console.error("Compute Pass Error:", e);
        }
    }

    if (renderCallback && !isPaused) {
       renderCallback(gl, delta);
    }
  });

  const geometry = useMemo(() => {
      const geo = new THREE.BufferGeometry();
      // Allocate max possible vertices, but we will clamp draw range
      const dummyPos = new Float32Array(1000000 * 3);
      geo.setAttribute('position', new THREE.BufferAttribute(dummyPos, 3));
      return geo;
  }, []);

  useEffect(() => {
      geometry.setDrawRange(0, agentCount);
  }, [geometry, agentCount]);

  return (
    <points ref={meshRef} args={[geometry, material]}>
      {material && <primitive object={material} attach="material" />}
    </points>
  );
}

// Global renderer instance trick to work with R3F
let globalRenderer: any = null;

export default function AbmCanvas(props: AbmCanvasProps) {
  return (
    <div className="w-full h-screen absolute inset-0 z-0 bg-neutral-950">
      <Canvas
        camera={{ position: [0, 0, 50], fov: 60 }}
        gl={(canvasProp) => {
            const actualCanvas = (canvasProp && (canvasProp as any).canvas) ? (canvasProp as any).canvas : canvasProp;
            const renderer = new WebGPURenderer({ 
                canvas: actualCanvas as HTMLCanvasElement, 
                antialias: false, 
                powerPreference: 'high-performance',
                // @ts-ignore
                requiredLimits: { maxStorageBuffersPerShaderStage: 16 }
            });
            
            renderer.init().then(() => {
                renderer.__initialized = true;
            }).catch((e: any) => {
                console.error("WebGPU Initialization Failed", e);
            });
            
            // WebGPURenderer requires async init(), but R3F calls render() synchronously.
            // We must mock the render and compute methods until it's ready.
            const originalRender = renderer.render.bind(renderer);
            const originalCompute = renderer.compute.bind(renderer);
            
            renderer.render = (...args: any[]) => {
               if (renderer.__initialized) {
                   if (args.length > 2) console.log("RENDER ARGS HAS TARGET?", args[2]);
                   originalRender(args[0], args[1]); // Force only scene and camera
               }
            };
            
            renderer.compute = (...args: any[]) => {
                if (renderer.__initialized) {
                    try {
                        const result = originalCompute(...args);
                        if (result && typeof result.catch === 'function') {
                            result.catch((e: any) => {
                                // Ignore async rejections
                            });
                        }
                    } catch (e) {
                        throw e;
                    }
               }
            };

            return renderer;
        }}
      >
        <color attach="background" args={['#0a0a0a']} />
        <AbmEngine {...props} />
      </Canvas>
    </div>
  );
}
