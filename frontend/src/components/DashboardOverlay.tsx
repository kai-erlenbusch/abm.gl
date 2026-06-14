'use client';
import { useEffect, useRef } from 'react';
import { ChartGPU } from 'chartgpu';
import { useSimulationStore } from '@/store/simulationStore';

// Custom wrapper for ChartGPU
function TelemetryChart() {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<any>(null);
  
  // Data history stored outside React state to prevent re-renders
  const telemetryHistory = useRef<{x: number, y: number}[]>([]);
  const policyHistory = useRef<{x: number, y: number}[]>([]);
  
  const startTime = useRef(Date.now());

  const initStarted = useRef(false);

  useEffect(() => {
    if (initStarted.current) return;
    initStarted.current = true;

    let isMounted = true;
    
    async function initChart() {
      if (!containerRef.current) return;
      
      const chart = await ChartGPU.create(containerRef.current, {
        series: [
          { 
            name: 'Actual Speed',
            type: 'line', 
            data: [], 
            style: { color: '#00ffcc', lineWidth: 2 } 
          },
          { 
            name: 'Policy Speed',
            type: 'line', 
            data: [], 
            style: { color: '#ff0055', lineWidth: 2, lineDash: [5, 5] } 
          }
        ],
      });
      
      if (isMounted) {
        chartInstanceRef.current = chart;
      }
    }
    
    initChart();

    return () => {
      isMounted = false;
      if (chartInstanceRef.current && chartInstanceRef.current.destroy) {
        chartInstanceRef.current.destroy();
      }
    };
  }, []);

  useEffect(() => {
    const handleTelemetry = (e: any) => {
      if (!chartInstanceRef.current) return;
      
      const { timestamp, actual_speed, policy_speed } = e.detail;
      const elapsed = (timestamp - startTime.current) / 1000; // seconds

      // Use ChartGPU's native streaming appendData API
      if (typeof chartInstanceRef.current.appendData === 'function') {
        chartInstanceRef.current.appendData(0, [[elapsed, actual_speed]]);
        chartInstanceRef.current.appendData(1, [[elapsed, policy_speed]]);
      } else if (typeof chartInstanceRef.current.setOption === 'function') {
        // Fallback just in case
        telemetryHistory.current.push([elapsed, actual_speed] as any);
        policyHistory.current.push([elapsed, policy_speed] as any);
        if (telemetryHistory.current.length > 200) {
           telemetryHistory.current.shift();
           policyHistory.current.shift();
        }
        chartInstanceRef.current.setOption({
          series: [
            { type: 'line', data: telemetryHistory.current },
            { type: 'line', data: policyHistory.current }
          ]
        });
      }
    };

    window.addEventListener('abm-telemetry', handleTelemetry);
    
    return () => {
      window.removeEventListener('abm-telemetry', handleTelemetry);
    };
  }, []);

  return <div ref={containerRef} className="w-full h-48 mt-4" />;
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

      // Hardcoded max to prevent flickering (e.g. 15,000 agents)
      const MAX_THEORETICAL = 15000;

      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          const index = r * 10 + c;
          const cellNode = cellsRef.current[index];
          if (!cellNode) continue;

          const density = grid[r][c].density;
          const opacity = Math.min(density / MAX_THEORETICAL, 1.0);

          if (density === maxDensity && density > 0) {
            // Hotspot tinting: Pink/Red
            cellNode.style.backgroundColor = `rgba(255, 51, 102, ${Math.max(opacity, 0.5)})`;
          } else {
            // Base color: Neon Emerald
            cellNode.style.backgroundColor = `rgba(0, 255, 136, ${opacity})`;
          }
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
          <div className="flex justify-between text-xs mb-1 text-neutral-400">
            <span>Agents</span>
            <span>1,000,000 (WebGPU)</span>
          </div>
          <div className="flex justify-between text-xs text-neutral-400">
            <span>Macro Engine</span>
            <span>Shachi (Python LLM)</span>
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

        <div className="pt-4 border-t border-neutral-800 flex space-x-2">
           <button 
             onClick={() => {
               setIsPaused(!isPaused);
             }}
             className={`flex-1 py-2 rounded-lg text-xs font-bold transition-colors ${
               isPaused ? 'bg-amber-500/20 text-amber-500 hover:bg-amber-500/30' : 'bg-neutral-800 hover:bg-neutral-700'
             }`}
           >
             {isPaused ? '▶ RESUME' : '⏸ PAUSE LOCKSTEP'}
           </button>
        </div>
      </div>
    </div>
  );
}
