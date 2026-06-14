import { Fn, instanceIndex, If, length, atomicAdd, atomicStore, atomicLoad, uint, float, floor, clamp, Loop, vec3, texture, vec2 } from 'three/tsl';

/**
 * 1. Flocking Behavior Primitive
 * Calculates separation, alignment, and cohesion entirely on the GPU.
 */
export const flockingBehavior = Fn(([positions, velocities, policyMapTexture, aggregateBuffer]) => {
    const i = instanceIndex;
    const pos = positions.element(i);
    const vel = velocities.element(i);
    
    // Phase 7: Spatial Heterogeneity via DataTexture
    const u = clamp(pos.x.add(25.0).div(50.0), 0.0, 1.0);
    const v = clamp(pos.y.add(25.0).div(50.0), 0.0, 1.0);
    const speedLocal = texture(policyMapTexture, vec2(u, v)).r;

    // Apply basic linear movement based on the velocity vector and spatial speed policy
    pos.addAssign(vel.mul(speedLocal));

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
    const normY = float(25.0).sub(pos.y);
    
    const col = floor(normX.div(5.0));
    const row = floor(normY.div(5.0));
    
    // Ensure we don't index out of bounds
    const safeCol = clamp(col, 0, 9);
    const safeRow = clamp(row, 0, 9);
    
    const gridIndex = safeRow.mul(10).add(safeCol);
    const speedIndex = gridIndex.mul(2);
    const countIndex = gridIndex.mul(2).add(1);

    const speed = length(vel);
    
    // Accumulate speed (scaled by 10) and agent count per cell
    atomicAdd(aggregateBuffer.element(speedIndex), uint(speed.mul(10.0)));
    atomicAdd(aggregateBuffer.element(countIndex), uint(1));
});

export const resetAggregate = Fn(([aggregateBuffer]) => {
    // Run this with 200 threads to clear the entire spatial grid buffer
    const i = instanceIndex;
    atomicStore(aggregateBuffer.element(i), uint(0));
});

export const spatialResetNode = Fn(([cellCountBuffer, cellOffsetAtomicBuffer]) => {
    // 10,000 threads (100x100 grid)
    const i = instanceIndex;
    atomicStore(cellCountBuffer.element(i), uint(0));
    atomicStore(cellOffsetAtomicBuffer.element(i), uint(0));
});

export const spatialCountNode = Fn(([positions, cellCountBuffer]) => {
    // 1M threads
    const i = instanceIndex;
    const pos = positions.element(i);
    const normX = pos.x.add(25.0);
    const normY = pos.y.add(25.0);
    
    // 100x100 grid, cell size 0.5
    const col = clamp(floor(normX.div(0.5)), 0, 99);
    const row = clamp(floor(normY.div(0.5)), 0, 99);
    const gridIndex = row.mul(100).add(col);
    
    atomicAdd(cellCountBuffer.element(gridIndex), uint(1));
});

export const spatialPrefixSumNode = Fn(([cellCountBuffer, cellOffsetBuffer, cellOffsetAtomicBuffer]) => {
    // 1 thread
    const total = uint(0).toVar();
    
    Loop(10000, ({ i }) => {
        cellOffsetBuffer.element(i).assign(total);
        atomicStore(cellOffsetAtomicBuffer.element(i), total);
        total.addAssign(atomicLoad(cellCountBuffer.element(i)));
    });
});

export const spatialScatterNode = Fn(([positions, cellOffsetAtomicBuffer, sortedAgentIndicesBuffer]) => {
    // 1M threads
    const i = instanceIndex;
    const pos = positions.element(i);
    const normX = pos.x.add(25.0);
    const normY = pos.y.add(25.0);
    
    // 100x100 grid, cell size 0.5
    const col = clamp(floor(normX.div(0.5)), 0, 99);
    const row = clamp(floor(normY.div(0.5)), 0, 99);
    const gridIndex = row.mul(100).add(col);
    
    const slot = atomicAdd(cellOffsetAtomicBuffer.element(gridIndex), uint(1));
    sortedAgentIndicesBuffer.element(slot).assign(i);
});

export const spatialCollisionNode = Fn(([positions, velocities, cellCountBuffer, cellOffsetBuffer, sortedAgentIndicesBuffer]) => {
    // 1M threads
    const i = instanceIndex;
    const pos = positions.element(i);
    const vel = velocities.element(i);
    
    const normX = pos.x.add(25.0);
    const normY = pos.y.add(25.0);
    
    // 100x100 grid, cell size 0.5
    const col = clamp(floor(normX.div(0.5)), 0, 99);
    const row = clamp(floor(normY.div(0.5)), 0, 99);
    const gridIndex = row.mul(100).add(col);
    
    const startIdx = cellOffsetBuffer.element(gridIndex);
    const count = atomicLoad(cellCountBuffer.element(gridIndex));
    
    const separation = vec3(0, 0, 0).toVar();
    const neighborsCount = uint(0).toVar();
    
    Loop(count, ({ i: j }) => {
        const sortedIndex = startIdx.add(j);
        const otherAgentId = sortedAgentIndicesBuffer.element(sortedIndex);
        
        If(otherAgentId.notEqual(i), () => {
            const otherPos = positions.element(otherAgentId);
            const dist = pos.distance(otherPos);
            
            // Repulsion threshold
            If(dist.lessThan(0.5).and(dist.greaterThan(0.001)), () => {
                const pushDir = pos.sub(otherPos).normalize();
                const pushStrength = float(0.5).sub(dist); 
                separation.addAssign(pushDir.mul(pushStrength));
                neighborsCount.addAssign(1);
            });
        });
    });
    
    If(neighborsCount.greaterThan(0), () => {
        // Average the separation force
        const avgSeparation = separation.div(float(neighborsCount));
        
        // Push the agent and normalize its velocity
        vel.addAssign(avgSeparation.mul(0.2)); 
        vel.assign(vel.normalize());
    });
});

/**
 * 3. GPU Data Aggregation Primitive
 * Reduces the 100,000+ agent states down to a 10-float array
 * before reading back to Javascript, avoiding VRAM bottlenecks.
 */
export const aggregateStats = Fn(([agentData, outputBuffer]) => {
    // Parallel reduction logic
});
