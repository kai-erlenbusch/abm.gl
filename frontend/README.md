# abm.gl

abm.gl is a high-performance Agent-Based Modeling (ABM) simulation engine built for the browser. It leverages **WebGPU** via Three.js (TSL - Three Shading Language) to simulate and render up to **1,000,000 agents** simultaneously at 60 FPS.

## Features

- **Massive Scale**: Simulates 1,000,000 agents entirely on the GPU.
- **WebGPU Compute**: Utilizes Compute Shaders for physics, spatial partitioning, and behavioral logic.
- **Strict Determinism**: Uses double-buffering (ping-pong state) to ensure all physics calculations are strictly deterministic and free of race conditions.
- **Spatial Grid Partitioning**: Implements a highly optimized 3-pass parallel prefix sum (Blelloch scan variant) for spatial neighbor lookups, eliminating O(N^2) collision checks.
- **SIR Epidemic Model**: Real-time simulation of Susceptible, Infected, and Recovered states with configurable transmission probabilities and recovery times.
- **Flocking & Boids**: GPU-accelerated separation and movement logic.
- **Real-time Telemetry**: Aggregates population statistics (e.g., infected counts, healthy counts) dynamically via atomic operations on the GPU and bridges them back to React for rendering charts.

## Architecture

- **Frontend**: Next.js 14, React 19, Tailwind CSS.
- **Graphics & Compute**: Three.js WebGPU renderer (`@react-three/fiber`, `@react-three/drei`), utilizing TSL (`three/tsl`) for compute shader graph definition.
- **Data Structure**: Structure of Arrays (SoA) layout. All agent data (position, velocity, infection state, timer) are stored in individual `StorageBuffer`s.

## Getting Started

First, install the dependencies:

```bash
npm install
```

Run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the simulation. Use the UI panel to tweak transmission probability, infection radius, and agent counts in real-time.
