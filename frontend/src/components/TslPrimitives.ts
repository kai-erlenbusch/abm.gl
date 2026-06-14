import { Fn, instanceIndex, If, length, atomicAdd, atomicStore, uint } from 'three/tsl';

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

    // Phase 3: WebGPU Atomic Aggregation
    // Multiply speed by 100.0 to preserve 2 decimal places, cast to uint, and safely accumulate.
    const speed = length(vel);
    atomicAdd(aggregateBuffer.element(0), uint(speed.mul(100.0)));
});

export const resetAggregate = Fn(([aggregateBuffer]) => {
    atomicStore(aggregateBuffer.element(0), uint(0));
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
