'use client';
import { useEffect, useRef, useState } from 'react';
import { useSimulationStore } from '@/store/simulationStore';
import modelSchema from '../config/modelSchema.json';

function SliderWidget({ control }: { control: any }) {
  const value = useSimulationStore(state => state.dynamicParams[control.id] ?? control.min);
  const setDynamicParam = useSimulationStore(state => state.setDynamicParam);

  return (
    <div className="mb-3">
      <div className="flex justify-between items-center text-xs mb-1 text-neutral-400">
        <span>{control.label}</span>
        <input 
          type="number" 
          value={value} 
          min={control.min} 
          max={control.max} 
          step={control.step || 1}
          onChange={(e) => setDynamicParam(control.id, parseFloat(e.target.value) || 0)}
          className="bg-neutral-800 text-right w-16 px-1 rounded border border-neutral-700 outline-none focus:border-emerald-500"
        />
      </div>
      <input 
        type="range" 
        min={control.min} 
        max={control.max} 
        step={control.step || 1}
        value={value}
        onChange={(e) => setDynamicParam(control.id, parseFloat(e.target.value))}
        className="w-full h-1 bg-neutral-800 rounded-lg appearance-none cursor-pointer"
      />
    </div>
  );
}

function NumberWidget({ control }: { control: any }) {
  const value = useSimulationStore(state => state.dynamicParams[control.id] ?? control.default ?? 100000);
  const setDynamicParam = useSimulationStore(state => state.setDynamicParam);

  return (
    <div className="mb-3">
      <div className="flex justify-between items-center text-xs mb-1 text-neutral-400">
        <span>{control.label}</span>
        <input 
          type="number" 
          value={value} 
          min={control.min} 
          max={control.max} 
          step={control.step || 1}
          onChange={(e) => {
            const val = parseFloat(e.target.value);
            if (!isNaN(val)) {
               setDynamicParam(control.id, Math.max(control.min, Math.min(control.max, val)));
            }
          }}
          className="bg-neutral-800 text-right w-24 px-2 py-1 rounded border border-neutral-700 outline-none focus:border-emerald-500"
        />
      </div>
    </div>
  );
}

function ToggleWidget({ control }: { control: any }) {
  const isPaused = useSimulationStore(state => state.isPaused);
  const setIsPaused = useSimulationStore(state => state.setIsPaused);

  return (
    <button 
      onClick={() => setIsPaused(!isPaused)}
      className={`flex-1 py-2 rounded-lg text-xs font-bold transition-colors ${
        isPaused ? 'bg-amber-500/20 text-amber-500 hover:bg-amber-500/30' : 'bg-neutral-800 hover:bg-neutral-700'
      }`}
    >
      {isPaused ? `▶ ${control.label || 'RESUME'}` : `⏸ PAUSE`}
    </button>
  );
}

function SetupButtonWidget({ control }: { control: any }) {
  const triggerSetup = useSimulationStore(state => state.triggerSetup);
  return (
    <button 
      onClick={() => triggerSetup()}
      className="flex-1 py-2 rounded-lg text-xs font-bold transition-colors bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 border border-emerald-500/30"
    >
      ↺ {control.label || 'SETUP'}
    </button>
  );
}

function FPSMeter() {
  const [fps, setFps] = useState(0);
  const frameCountRef = useRef(0);
  const lastFpsTimeRef = useRef(Date.now());

  useEffect(() => {
    const handleTelemetry = (e: any) => {
      frameCountRef.current++;
      const now = Date.now();
      if (now - lastFpsTimeRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        lastFpsTimeRef.current = now;
      }
    };
    window.addEventListener('abm-frame', handleTelemetry);
    return () => window.removeEventListener('abm-frame', handleTelemetry);
  }, []);

  return (
    <div className="flex justify-between items-center mb-2">
      <div className="text-xs text-neutral-400">FPS: <span className="text-emerald-400 font-bold">{fps}</span></div>
    </div>
  );
}

function TelemetryChart() {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<any>(null);
  const startTime = useRef(Date.now());

  useEffect(() => {
    let isMounted = true;
    
    async function initChart() {
      if (!containerRef.current) return;
      
      try {
        const { ChartGPU } = await import('chartgpu');
        const chart = await ChartGPU.create(containerRef.current, {
          series: [
            { 
              name: 'Actual Speed',
              type: 'line', 
              data: [], 
              sampling: 'none',
              // @ts-ignore
              style: { color: '#00ffcc', lineWidth: 2 } 
            },
            { 
              name: 'Policy Speed',
              type: 'line', 
              data: [], 
              sampling: 'none',
              // @ts-ignore
              style: { color: '#ff0055', lineWidth: 2, lineDash: [5, 5] } 
            }
          ],
        });
        
        if (isMounted) {
          chartInstanceRef.current = chart;
        } else {
          if (chart && typeof chart.dispose === 'function') {
            chart.dispose();
          }
        }
      } catch (err) {
        console.warn("ChartGPU failed to initialize (likely due to WebGPU resource limits at 500k+ agents).", err);
      }
    }
    
    initChart();

    return () => {
      isMounted = false;
      if (chartInstanceRef.current && typeof chartInstanceRef.current.dispose === 'function') {
        chartInstanceRef.current.dispose();
      }
      chartInstanceRef.current = null;
    };
  }, []);

  useEffect(() => {
    const handleTelemetry = (e: any) => {
      if (!chartInstanceRef.current) return;
      
      const { timestamp, actual_speed, policy_speed } = e.detail;
      const elapsed = (timestamp - startTime.current) / 1000; // seconds

      const currentIsPaused = useSimulationStore.getState().isPaused;

      if (!currentIsPaused) {
        if (typeof chartInstanceRef.current.appendData === 'function') {
          chartInstanceRef.current.appendData(0, [[elapsed, actual_speed]]);
          chartInstanceRef.current.appendData(1, [[elapsed, policy_speed]]);
        }
      }
    };

    window.addEventListener('abm-telemetry', handleTelemetry);
    
    return () => {
      window.removeEventListener('abm-telemetry', handleTelemetry);
    };
  }, []);

  return (
    <>
      <FPSMeter />
      <div ref={containerRef} className="w-full h-48 mt-2" />
    </>
  );
}

// Phase 6: High-performance raw DOM Heatmap (Zero React Re-renders)
function SpatialHeatmap() {
  const cellsRef = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    const handleTelemetry = (e: any) => {
      const { grid } = e.detail;
      if (!grid) return;

      let maxDensity = 0;
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          if (grid[r][c].density > maxDensity) {
            maxDensity = grid[r][c].density;
          }
        }
      }

      // Dynamic max to adapt to actual density (so colors are always visible)
      // Cap minimum at 100 to prevent division by zero or extreme noise
      const MAX_THEORETICAL = Math.max(100, maxDensity * 1.2, grid[0][0]?.density * 1.5 || 100);

      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          const index = r * 10 + c;
          const cellNode = cellsRef.current[index];
          if (!cellNode) continue;

          const cell = grid[r][c];
          const count = cell.density;
          
          if (count === 0) {
            cellNode.style.backgroundColor = 'transparent';
            continue;
          }

          const infected = cell.infected_count;
          const recovered = cell.recovered_count ?? 0;
          const susceptible = Math.max(0, count - infected - recovered);

          // Ratios S/I/R
          const pS = susceptible / count;
          const pI = infected / count;
          const pR = recovered / count;

          // RGB blend S=(0, 255, 136) I=(255, 51, 102) R=(0, 128, 255)
          const rColor = Math.round(pS * 0 + pI * 255 + pR * 0);
          const gColor = Math.round(pS * 255 + pI * 51 + pR * 128);
          const bColor = Math.round(pS * 136 + pI * 102 + pR * 255);

          // Overall brightness based on physical density
          const opacity = Math.min(count / MAX_THEORETICAL, 1.0);

          cellNode.style.backgroundColor = `rgba(${rColor}, ${gColor}, ${bColor}, ${Math.max(opacity, 0.2)})`;
        }
      }
    };

    window.addEventListener('abm-telemetry', handleTelemetry);
    return () => window.removeEventListener('abm-telemetry', handleTelemetry);
  }, []);

  return (
    <div className="w-full aspect-square bg-neutral-900 border border-neutral-800 rounded-lg overflow-hidden grid grid-cols-10 grid-rows-10 gap-[1px]">
      {Array.from({ length: 100 }).map((_, i) => (
        <div 
          key={i} 
          ref={(el) => { cellsRef.current[i] = el; }}
          className="w-full h-full bg-transparent"
        />
      ))}
    </div>
  );
}

export default function DashboardOverlay() {
  const isPaused = useSimulationStore(state => state.isPaused);
  const setIsPaused = useSimulationStore(state => state.setIsPaused);

  return (
    <div className="absolute top-4 left-4 z-10 w-96 text-white font-mono text-sm bg-black/50 backdrop-blur-md p-6 rounded-xl border border-neutral-800 shadow-2xl pointer-events-auto">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">
            abm.gl
          </h1>
          <p className="text-neutral-400 text-xs tracking-wider uppercase mt-1">Next.js Command Center</p>
        </div>
        
        {/* Status indicator */}
        <div className="flex items-center space-x-2">
          <span className="relative flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
          </span>
          <span className="text-xs text-emerald-400">Live</span>
        </div>
      </div>
      
      <div className="space-y-4">
        <div className="bg-neutral-900/50 p-4 rounded-lg border border-neutral-800">
          {modelSchema.monitors.map((m: any) => (
            <div key={m.id} className="flex justify-between text-xs mb-1 text-neutral-400">
              <span>{m.label}</span>
              <span>{m.value}</span>
            </div>
          ))}
        </div>

        <div className="pt-4 border-t border-neutral-800">
          <h2 className="text-xs uppercase tracking-widest text-neutral-500 mb-2">Controls</h2>
          <div className="flex space-x-2 mb-4">
            {modelSchema.controls.filter((c: any) => c.type === 'button' || c.type === 'toggle').map((c: any) => {
              if (c.type === 'button') return <SetupButtonWidget key={c.id} control={c} />;
              if (c.type === 'toggle') return <ToggleWidget key={c.id} control={c} />;
              return null;
            })}
          </div>
          <div className="space-y-2">
            {modelSchema.controls.filter((c: any) => c.type === 'slider' || c.type === 'number').map((c: any) => {
              if (c.type === 'slider') return <SliderWidget key={c.id} control={c} />;
              if (c.type === 'number') return <NumberWidget key={c.id} control={c} />;
              return null;
            })}
          </div>
        </div>

        <div>
          <h2 className="text-xs uppercase tracking-widest text-neutral-500 mb-2">Real-time Telemetry</h2>
          <TelemetryChart />
        </div>

        <div className="pt-4 border-t border-neutral-800">
          <h2 className="text-xs uppercase tracking-widest text-neutral-500 mb-2">Spatial Density Grid</h2>
          <SpatialHeatmap />
        </div>
      </div>
    </div>
  );
}
