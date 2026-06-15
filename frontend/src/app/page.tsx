'use client';
import dynamic from 'next/dynamic';

const SimulationCanvas = dynamic(() => import('@/components/SimulationCanvas'), {
  ssr: false,
});

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col">
      <SimulationCanvas />
    </main>
  );
}
