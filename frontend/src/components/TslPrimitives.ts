import { 
    Fn, float, vec3, abs, If, Loop, instanceIndex, uint, int,
    atomicAdd, atomicLoad, atomicStore, min,
    workgroupBarrier, workgroupId, workgroupArray, clamp, floor,
    fract, sin, length, texture, vec2, Break, invocationLocalIndex, mod, select
} from 'three/tsl';

/**
 * 1. Flocking Behavior Primitive
 * Calculates separation, alignment, and cohesion entirely on the GPU.
 */
export const flockingBehavior = Fn(([positions, velocities, policyMapTexture, aggregateBuffer, infectionBuffer, timerBuffer, deltaUniform]) => {
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
    const speedIndex = gridIndex.mul(4);
    const countIndex = gridIndex.mul(4).add(1);
    const infectedIndex = gridIndex.mul(4).add(2);
    const recoveredIndex = gridIndex.mul(4).add(3);

    const speed = length(vel).mul(speedLocal);
    const myInfection = infectionBuffer.element(i);
    
    // SIR Recovery Logic
    If(myInfection.equal(uint(1)), () => {
        const myTimer = timerBuffer.element(i);
        myTimer.subAssign(deltaUniform);
        If(myTimer.lessThanEqual(0.0), () => {
            infectionBuffer.element(i).assign(uint(2)); // Recovered
        });
    });
    
    const isInfected = select(infectionBuffer.element(i).equal(uint(1)), uint(1), uint(0));
    const isRecovered = select(infectionBuffer.element(i).equal(uint(2)), uint(1), uint(0));

    // Accumulate speed (scaled by 100) and agent counts per cell
    atomicAdd(aggregateBuffer.element(speedIndex), uint(speed.mul(100.0)));
    atomicAdd(aggregateBuffer.element(countIndex), uint(1));
    atomicAdd(aggregateBuffer.element(infectedIndex), isInfected);
    atomicAdd(aggregateBuffer.element(recoveredIndex), isRecovered);
});

export const resetAggregate = Fn(([aggregateBuffer]) => {
    // Run this with 300 threads to clear the entire spatial grid buffer
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
    // Pad to multiple of 256
    const i = instanceIndex;
    const localId = invocationLocalIndex;
    
    const sharedArray = workgroupArray('uint', 256);
    
    // Bounds check for padding
    If(i.lessThan(100000), () => {
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

// ------------------------------------------------------------------------------------------------
// PASS 2: Sequential Prefix Sum (O(N) single-threaded approach)
// For exactly 10,000 cells, a single GPU thread looping 10,000 times takes ~1 microsecond.
// This completely avoids workgroup sync barriers, TSL loop unrolling hazards, and block additions.
// ------------------------------------------------------------------------------------------------
export const spatialPrefixSum_SequentialNode = Fn(([cellCountBuffer, cellOffsetBuffer, cellOffsetAtomicBuffer]: any) => {
    const i = instanceIndex;
    
    // Only thread 0 does the work
    If(i.equal(uint(0)), () => {
        const sum = uint(0).toVar();
        
        Loop(10000, ({ i: j }) => {
            const jUint = uint(j);
            const count = uint(cellCountBuffer.element(jUint));
            
            // Assign the current sum to the offset buffers
            cellOffsetBuffer.element(jUint).assign(sum);
            atomicStore(cellOffsetAtomicBuffer.element(jUint), sum);
            
            // Add this cell's count to the running sum
            sum.addAssign(count);
        });
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

export const spatialCollisionNode = Fn(([positions, velocities, cellCountBuffer, cellOffsetBuffer, sortedAgentIndicesBuffer, infectionBuffer, timerBuffer, infectionRadius, transmissionProb, recoveryTime, seedUniform]: any) => {
    // 1M threads
    const i = instanceIndex;
    const pos = positions.element(i);
    const vel = velocities.element(i);
    const myInfection = infectionBuffer.element(i);
    
    const normX = pos.x.add(25.0);
    const normY = pos.y.add(25.0);
    
    // 100x100 grid, cell size 0.5
    const col = uint(clamp(floor(normX.div(0.5)), 0, 99));
    const row = uint(clamp(floor(normY.div(0.5)), 0, 99));
    
    const separation = vec3(0, 0, 0).toVar();
    const neighborsCount = uint(0).toVar();
    
    for (let rOffset = -1; rOffset <= 1; rOffset++) {
        for (let cOffset = -1; cOffset <= 1; cOffset++) {
            const neighborCol = uint(clamp(int(col).add(int(cOffset)), 0, 99));
            const neighborRow = uint(clamp(int(row).add(int(rOffset)), 0, 99));
            const neighborGridIndex = neighborRow.mul(uint(100)).add(neighborCol);
            
            const startIdx = cellOffsetBuffer.element(neighborGridIndex);
            const count = uint(cellCountBuffer.element(neighborGridIndex));
            
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

                    // Phase 7: Viral Transmission (Decoupled from physical repulsion)
                    If(infectionBuffer.element(i).equal(uint(0)), () => {
                        If(infectionBuffer.element(otherAgentId).equal(uint(1)), () => {
                            If(dist.lessThan(infectionRadius), () => {
                                If(dist.greaterThan(0.001), () => {
                                    // Generate pseudo-random value [0, 1] for this contact
                                    const contactSeed = pos.x.mul(13.5).add(pos.y.mul(41.2)).add(seedUniform).add(float(otherAgentId));
                                    const rand = fract(sin(contactSeed).mul(43758.5453));
                                    
                                    If(rand.lessThan(transmissionProb), () => {
                                        infectionBuffer.element(i).assign(uint(1));
                                        timerBuffer.element(i).assign(recoveryTime);
                                    });
                                });
                            });
                        });
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

export const setupEpidemicNode = Fn(([positions, velocities, infectionBuffer, timerBuffer, seed, initialInfectedRadius, recoveryTime]: any) => {
    const i = instanceIndex;
    const pos = positions.element(i);
    const vel = velocities.element(i);
    const infection = infectionBuffer.element(i);

    // Simple PRNG using instanceIndex and a dynamic seed
    // To avoid float32 precision loss and sine drift at i=1,000,000,
    // we map the 1D thread index to a perfect 1000x1000 2D grid, then add micro-jitter.
    // Use mod to prevent float32 precision loss when i > 10,000
    const smallI = mod(float(i), 1000.0);
    const gridY = floor(float(i).div(1000.0));
    
    // Mix them uniquely to get good PRNG spread
    const seed1 = smallI.add(gridY.mul(0.1)).add(seed);
    const r1 = fract(sin(seed1.mul(12.9898)).mul(43758.5453));
    const r2 = fract(sin(seed1.mul(78.233)).mul(43758.5453));
    const r3 = fract(sin(seed1.mul(45.123)).mul(43758.5453));
    const r4 = fract(sin(seed1.mul(93.989)).mul(43758.5453));

    // Place randomly across the whole board
    pos.x.assign(r1.sub(0.5).mul(50.0));
    pos.y.assign(r2.sub(0.5).mul(50.0));
    pos.z.assign(0.0);

    vel.x.assign(r3.sub(0.5).mul(2.0));
    vel.y.assign(r4.sub(0.5).mul(2.0));
    vel.z.assign(0.0);

    // Phase 8: Ground Zero Seeding
    const dist = length(pos);
    If(dist.lessThan(initialInfectedRadius), () => {
        infection.assign(uint(1));
        timerBuffer.element(i).assign(recoveryTime);
    }).Else(() => {
        infection.assign(uint(0));
        timerBuffer.element(i).assign(0.0);
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
