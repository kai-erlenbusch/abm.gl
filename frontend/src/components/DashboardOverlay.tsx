'use client';
import { useEffect, useRef, useState } from 'react';
import { ChartGPU } from 'chartgpu';

// Custom wrapper for ChartGPU
function TelemetryChart() {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<any>(null);
  
  // Data history stored outside React state to prevent re-renders
  const telemetryHistory = useRef<{x: number, y: number}[]>([]);
  const policyHistory = useRef<{x: number, y: number}[]>([]);
  
  const startTime = useRef(Date.now());

  useEffect(() => {
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

export default function DashboardOverlay() {
  const [isPaused, setIsPaused] = useState(false);

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

        <div className="pt-4 border-t border-neutral-800 flex space-x-2">
           <button 
             onClick={() => {
               const newPaused = !isPaused;
               setIsPaused(newPaused);
               // @ts-ignore
               window.abmPaused = newPaused;
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
