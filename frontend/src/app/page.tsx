'use client';
import dynamic from 'next/dynamic';

const SimulationCanvas = dynamic(() => import('@/components/SimulationCanvas'), {
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
  return (
    <main className="flex min-h-screen flex-col">
      <SimulationCanvas />
    </main>
  );
}
