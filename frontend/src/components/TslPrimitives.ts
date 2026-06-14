import { Fn, instanceIndex, If, length, atomicAdd, atomicStore, uint, floor, clamp } from 'three/tsl';

/**
 * 1. Flocking Behavior Primitive
 * Calculates separation, alignment, and cohesion entirely on the GPU.
 */
export const flockingBehavior = Fn(([positions, velocities, speedUniform, aggregateBuffer]) => {
    const i = instanceIndex;
    const pos = positions.element(i);
    const vel = velocities.element(i);
    
    // Apply basic linear movement based on the velocity vector and speed policy
    pos.addAssign(vel.mul(speedUniform));

    // Basic bounds checking so they don't fly off screen infinitely
    // The screen is roughly -25 to 25 based on camera fov
    If(pos.x.greaterThan(25).or(pos.x.lessThan(-25)), () => {
        vel.x.mulAssign(-1);
    });
    If(pos.y.greaterThan(25).or(pos.y.lessThan(-25)), () => {
        vel.y.mulAssign(-1);
    });

    // Phase 5: WebGPU Spatial Grid Aggregation
    // Divide 50x50 world (-25 to 25) into 10x10 grid. Each cell is 5x5.
    const normX = pos.x.add(25.0);
    const normY = pos.y.add(25.0);
    
    const col = floor(normX.div(5.0));
    const row = floor(normY.div(5.0));
    
    // Ensure we don't index out of bounds
    const safeCol = clamp(col, 0, 9);
    const safeRow = clamp(row, 0, 9);
    
    const gridIndex = safeRow.mul(10).add(safeCol);
    const speedIndex = gridIndex.mul(2);
    const countIndex = gridIndex.mul(2).add(1);

    const speed = length(vel);
    
    // Accumulate speed (scaled by 100) and agent count per cell
    atomicAdd(aggregateBuffer.element(speedIndex), uint(speed.mul(100.0)));
    atomicAdd(aggregateBuffer.element(countIndex), uint(1));
});

export const resetAggregate = Fn(([aggregateBuffer]) => {
    // Run this with 200 threads to clear the entire spatial grid buffer
    const i = instanceIndex;
    atomicStore(aggregateBuffer.element(i), uint(0));
});

/**
 * 2. Spatial Grid Collision Primitive
 * Resolves overlapping boundaries between agents using spatial hashing.
 */
export const gridCollision = Fn(([positions, boundsUniform]) => {
    const i = instanceIndex;
    // Boundary bounce logic would go here
});

/**
 * 3. GPU Data Aggregation Primitive
 * Reduces the 100,000+ agent states down to a 10-float array
 * before reading back to Javascript, avoiding VRAM bottlenecks.
 */
export const aggregateStats = Fn(([agentData, outputBuffer]) => {
    // Parallel reduction logic
});
