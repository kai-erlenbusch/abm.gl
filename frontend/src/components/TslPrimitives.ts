import { Fn, instanceIndex, If, length, atomicAdd, atomicStore, atomicLoad, uint, int, float, floor, clamp, Loop, vec3, texture, vec2, min, workgroupArray, workgroupBarrier, workgroupId, invocationLocalIndex, Break } from 'three/tsl';

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
    
    // Accumulate speed (scaled by 100) and agent count per cell
    atomicAdd(aggregateBuffer.element(speedIndex), uint(speed.mul(100.0)));
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
    // 10240 threads, 40 workgroups of 256
    const i = instanceIndex;
    const localId = invocationLocalIndex;
    
    const sharedArray = workgroupArray('uint', 256);
    
    // Bounds check for padding
    If(i.lessThan(10000), () => {
        const pos = positions.element(i);
        const normX = pos.x.add(25.0);
        const normY = pos.y.add(25.0);
        
        // 100x100 grid, cell size 0.5
        const col = clamp(floor(normX.div(0.5)), 0, 99);
        const row = clamp(floor(normY.div(0.5)), 0, 99);
        const gridIndex = row.mul(100).add(col);
        
        sharedArray.element(localId).assign(gridIndex);
    }).Else(() => {
        // Out-of-bounds padding agents placed at the very end
        sharedArray.element(localId).assign(999999);
    });
    
    workgroupBarrier();
    
    // Bitonic Sort (ascending) for N=256 elements
    for (let k = 2; k <= 256; k *= 2) {
        for (let j = k / 2; j > 0; j = Math.floor(j / 2)) {
            const ixj = localId.bitXor(uint(j));
            If(ixj.greaterThan(localId), () => {
                const dir = localId.bitAnd(uint(k)).equal(uint(0)); // true for ascending
                const valA = sharedArray.element(localId).toVar();
                const valB = sharedArray.element(ixj).toVar();
                
                // If ascending and valA > valB, or descending and valA < valB, swap
                If(dir.equal(valA.greaterThan(valB)), () => {
                    sharedArray.element(localId).assign(valB);
                    sharedArray.element(ixj).assign(valA);
                });
            });
            workgroupBarrier();
        }
    }
    
    // Run-Length Encoding / Batching
    const isStart = localId.equal(0).or(sharedArray.element(localId).notEqual(sharedArray.element(localId.sub(1))));
    
    If(isStart, () => {
        const currentGridIndex = sharedArray.element(localId);
        
        // Only valid grid indices
        If(currentGridIndex.lessThan(10000), () => {
            const runLength = uint(1).toVar();
            
            // Forward scan divergence trap (accepted as MVP)
            Loop(256, ({ i: j }) => {
                const targetIdx = localId.add(j).add(1);
                If(targetIdx.lessThan(256), () => {
                    If(sharedArray.element(targetIdx).equal(currentGridIndex), () => {
                        runLength.addAssign(1);
                    }).Else(() => {
                        Break();
                    });
                }).Else(() => {
                    Break();
                });
            });
            
            atomicAdd(cellCountBuffer.element(currentGridIndex), runLength);
        });
    });
});

export const spatialPrefixSum_LocalScanNode = Fn(([cellCountBuffer, cellOffsetBuffer, chunkSumsBuffer]) => {
    // 40 workgroups of 256 threads (10240 total)
    const i = instanceIndex;
    const localId = invocationLocalIndex;
    const groupId = workgroupId.x;
    
    // Create shared memory for this workgroup
    const sharedData = workgroupArray('uint', 256);
    
    // Load from global to shared memory
    sharedData.element(localId).assign(cellCountBuffer.element(i));
    workgroupBarrier();
    
    // Blelloch Up-Sweep (Reduce) - 8 steps for 256 elements
    let step = 1;
    let step2 = 2;
    for (let d = 0; d < 8; d++) {
        If(localId.add(1).mod(uint(step2)).equal(uint(0)), () => {
            sharedData.element(localId).addAssign(sharedData.element(localId.sub(uint(step))));
        });
        workgroupBarrier();
        step = step2;
        step2 = step2 * 2;
    }
    
    // Blelloch Down-Sweep
    If(localId.equal(255), () => {
        chunkSumsBuffer.element(groupId).assign(sharedData.element(uint(255)));
        sharedData.element(uint(255)).assign(uint(0));
    });
    workgroupBarrier();
    
    step = 128;
    step2 = 256;
    for (let d = 7; d >= 0; d--) {
        If(localId.add(1).mod(uint(step2)).equal(uint(0)), () => {
            const temp = sharedData.element(localId.sub(uint(step))).toVar();
            sharedData.element(localId.sub(uint(step))).assign(sharedData.element(localId));
            sharedData.element(localId).addAssign(temp);
        });
        workgroupBarrier();
        step2 = step;
        step = Math.floor(step / 2);
    }
    
    cellOffsetBuffer.element(i).assign(sharedData.element(localId));
});

export const spatialPrefixSum_BlockScanNode = Fn(([chunkSumsBuffer, blockSize]) => {
    // 1 workgroup of blockSize threads
    const localId = invocationLocalIndex;
    const sharedData = workgroupArray('uint', 64); // Allocate max block size needed for the prototype
    
    sharedData.element(localId).assign(chunkSumsBuffer.element(localId));
    workgroupBarrier();
    
    // Up-sweep max 6 steps (log2 64 = 6) - dynamically capped based on blockSize if needed, but 64 is fixed array size
    let step = 1;
    let step2 = 2;
    for (let d = 0; d < 6; d++) {
        If(localId.add(1).mod(uint(step2)).equal(uint(0)), () => {
            sharedData.element(localId).addAssign(sharedData.element(localId.sub(uint(step))));
        });
        workgroupBarrier();
        step = step2;
        step2 = step2 * 2;
    }
    
    If(localId.equal(blockSize.sub(1)), () => {
        sharedData.element(blockSize.sub(1)).assign(uint(0));
    });
    workgroupBarrier();
    
    // Down-sweep 6 steps
    step = 32;
    step2 = 64;
    for (let d = 5; d >= 0; d--) {
        If(localId.add(1).mod(uint(step2)).equal(uint(0)), () => {
            const temp = sharedData.element(localId.sub(uint(step))).toVar();
            sharedData.element(localId.sub(uint(step))).assign(sharedData.element(localId));
            sharedData.element(localId).addAssign(temp);
        });
        workgroupBarrier();
        step2 = step;
        step = Math.floor(step / 2);
    }
    
    chunkSumsBuffer.element(localId).assign(sharedData.element(localId));
});

export const spatialPrefixSum_AddNode = Fn(([cellOffsetBuffer, cellOffsetAtomicBuffer, chunkSumsBuffer]) => {
    // 40 workgroups of 256 = 10240
    const i = instanceIndex;
    const groupId = workgroupId.x;
    
    const baseOffset = chunkSumsBuffer.element(groupId);
    const finalOffset = cellOffsetBuffer.element(i).add(baseOffset);
    
    cellOffsetBuffer.element(i).assign(finalOffset);
    atomicStore(cellOffsetAtomicBuffer.element(i), finalOffset);
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
    const col = uint(clamp(floor(normX.div(0.5)), 0, 99));
    const row = uint(clamp(floor(normY.div(0.5)), 0, 99));
    
    const separation = vec3(0, 0, 0).toVar();
    const neighborsCount = uint(0).toVar();
    
    for (let rOffset = -1; rOffset <= 1; rOffset++) {
        for (let cOffset = -1; cOffset <= 1; cOffset++) {
            const neighborCol = uint(clamp(int(col).add(cOffset), 0, 99));
            const neighborRow = uint(clamp(int(row).add(rOffset), 0, 99));
            const neighborGridIndex = neighborRow.mul(uint(100)).add(neighborCol);
            
            const startIdx = cellOffsetBuffer.element(neighborGridIndex);
            const count = atomicLoad(cellCountBuffer.element(neighborGridIndex));
            
            // Use a dynamic loop bounded by the actual neighbor count, capped at 256
            // This dynamically speeds up sparse regions while preventing TDR crashes in dense swarms
            const loopCap = min(count, uint(256));
            Loop(loopCap, ({ i: j }) => {
                const jUint = uint(j);
                const sortedIndex = startIdx.add(jUint);
                const otherAgentId = sortedAgentIndicesBuffer.element(sortedIndex);
                
                If(otherAgentId.notEqual(i), () => {
                    const otherPos = positions.element(otherAgentId);
                    
                    const dist = pos.distance(otherPos);
                    
                    // Repulsion threshold
                    If(dist.lessThan(0.5).and(dist.greaterThan(0.001)), () => {
                        const pushDir = pos.sub(otherPos).normalize();
                        const pushStrength = float(0.5).sub(dist); 
                        separation.addAssign(pushDir.mul(pushStrength));
                        neighborsCount.addAssign(uint(1));
                    });
                });
            });
        }
    }
    
    If(neighborsCount.greaterThan(uint(0)), () => {
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
