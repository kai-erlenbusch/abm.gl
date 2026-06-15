import { 
    Fn, float, vec3, abs, If, Loop, instanceIndex, uint, int,
    atomicAdd, atomicLoad, atomicStore, min, max,
    workgroupBarrier, workgroupId, workgroupArray, clamp, floor, ceil,
    fract, sin, length, texture, vec2, Break, invocationLocalIndex, mod, select, sqrt
} from 'three/tsl';

export const pcgHash = Fn(([seed]) => {
    const state = uint(seed).mul(uint(747796405)).add(uint(2891336453));
    const word = state.shiftRight(state.shiftRight(28).add(uint(4))).bitXor(state).mul(uint(277803737));
    return float(word.shiftRight(22).bitXor(word)).div(float(4294967295.0));
});

/**
 * 1. Flocking Behavior Primitive
 * Calculates separation, alignment, and cohesion entirely on the GPU.
 */
export const flockingBehavior = Fn(([positions, velocities, policyMapTexture, infectionBuffer, timerBuffer, deltaUniform]) => {
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

    const myInfection = infectionBuffer.element(i);
    
    // SIR Recovery Logic (Branchless)
    const myTimer = timerBuffer.element(i);
    const isInfectedCond = myInfection.equal(uint(1));
    
    myTimer.subAssign(deltaUniform.mul(select(isInfectedCond, float(1.0), float(0.0))));
    
    infectionBuffer.element(i).assign(
        select(
            isInfectedCond.and(myTimer.lessThanEqual(0.0)),
            uint(2),
            myInfection
        )
    );
});

export const telemetryAggregateNode = Fn(([positions, velocities, policyMapTexture, aggregateBuffer, infectionBuffer]) => {
    const i = instanceIndex;
    const pos = positions.element(i);
    const vel = velocities.element(i);
    
    const u = clamp(pos.x.add(25.0).div(50.0), 0.0, 1.0);
    const v = clamp(pos.y.add(25.0).div(50.0), 0.0, 1.0);
    const speedLocal = texture(policyMapTexture, vec2(u, v)).r;

    // Divide 50x50 world (-25 to 25) into 10x10 grid. Each cell is 5x5.
    const normX = pos.x.add(25.0);
    const normY = float(25.0).sub(pos.y);
    
    const col = floor(normX.div(5.0));
    const row = floor(normY.div(5.0));
    
    const safeCol = clamp(col, 0, 9);
    const safeRow = clamp(row, 0, 9);
    
    const gridIndex = safeRow.mul(10).add(safeCol);
    const speedIndex = gridIndex.mul(4);
    const countIndex = gridIndex.mul(4).add(1);
    const infectedIndex = gridIndex.mul(4).add(2);
    const recoveredIndex = gridIndex.mul(4).add(3);

    const speed = length(vel).mul(speedLocal);
    
    const myInfection = infectionBuffer.element(i);
    const isInfected = select(myInfection.equal(uint(1)), uint(1), uint(0));
    const isRecovered = select(myInfection.equal(uint(2)), uint(1), uint(0));

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

export const spatialCountNode = Fn(([positions, cellCountBuffer, agentCountLimit]: any) => {
    // Pad to multiple of 256
    const i = instanceIndex;
    const localId = invocationLocalIndex;
    
    const sharedArray = workgroupArray('uint', 256);
    
    // Bounds check for padding
    If(i.lessThan(agentCountLimit), () => {
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
        sharedArray.element(localId).assign(9999999);
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
// PASS 2: 3-Pass Parallel Prefix Sum (Blelloch Scan Variant)
// ------------------------------------------------------------------------------------------------

// Pass 2a: Local Chunk Scan
export const spatialPrefixSum_ChunkNode = Fn(([cellCountBuffer, cellOffsetBuffer, chunkSumsBuffer]: any) => {
    const globalId = instanceIndex;
    const localId = invocationLocalIndex;
    const groupId = workgroupId.x;
    
    const sharedArray = workgroupArray('uint', 256);
    
    // Load data into shared memory
    const count = select(globalId.lessThan(10240), uint(cellCountBuffer.element(globalId)), uint(0));
    sharedArray.element(localId).assign(count);
    workgroupBarrier();
    
    // Up-Sweep (Reduce)
    for (let offset = 1; offset < 256; offset *= 2) {
        const i = localId.mul(offset * 2).add(offset * 2 - 1);
        If(i.lessThan(256), () => {
            sharedArray.element(i).addAssign(sharedArray.element(i.sub(offset)));
        });
        workgroupBarrier();
    }
    
    // Clear the last element for Down-Sweep
    If(localId.equal(255), () => {
        // Save the total chunk sum
        If(groupId.lessThan(40), () => {
            chunkSumsBuffer.element(groupId).assign(sharedArray.element(uint(255)));
        });
        sharedArray.element(uint(255)).assign(uint(0));
    });
    workgroupBarrier();
    
    // Down-Sweep
    for (let offset = 128; offset > 0; offset /= 2) {
        const i = localId.mul(offset * 2).add(offset * 2 - 1);
        If(i.lessThan(256), () => {
            const temp = sharedArray.element(i.sub(offset)).toVar();
            sharedArray.element(i.sub(offset)).assign(sharedArray.element(i));
            sharedArray.element(i).addAssign(temp);
        });
        workgroupBarrier();
    }
    
    // Write out the exclusive prefix sum for this chunk
    If(globalId.lessThan(10240), () => {
        cellOffsetBuffer.element(globalId).assign(sharedArray.element(localId));
    });
});

// Pass 2b: Block Scan (Sequential scan of the chunk sums)
export const spatialPrefixSum_BlockNode = Fn(([chunkSumsBuffer]: any) => {
    const i = instanceIndex;
    If(i.equal(uint(0)), () => {
        const sum = uint(0).toVar();
        Loop(40, ({ i: j }) => {
            const jUint = uint(j);
            const count = uint(chunkSumsBuffer.element(jUint));
            chunkSumsBuffer.element(jUint).assign(sum);
            sum.addAssign(count);
        });
    });
});

// Pass 2c: Scatter/Add global block offsets to local chunks
export const spatialPrefixSum_ScatterNode = Fn(([cellOffsetBuffer, cellOffsetAtomicBuffer, chunkSumsBuffer]: any) => {
    const globalId = instanceIndex;
    const groupId = workgroupId.x;
    
    If(globalId.lessThan(10240), () => {
        const blockOffset = chunkSumsBuffer.element(groupId);
        const finalOffset = cellOffsetBuffer.element(globalId).add(blockOffset);
        cellOffsetBuffer.element(globalId).assign(finalOffset);
        atomicStore(cellOffsetAtomicBuffer.element(globalId), finalOffset);
    });
});

export const spatialScatterNode = Fn(([positions, velocities, cellOffsetAtomicBuffer, sortedAgentIndicesBuffer, sortedPositionsBuffer, sortedVelocitiesBuffer, agentCountLimit]: any) => {
    // 1M threads
    const i = instanceIndex;
    If(i.lessThan(agentCountLimit), () => {
        const pos = positions.element(i);
        const vel = velocities.element(i);
        const normX = pos.x.add(25.0);
        const normY = pos.y.add(25.0);
        
        // 100x100 grid, cell size 0.5
        const col = clamp(floor(normX.div(0.5)), 0, 99);
        const row = clamp(floor(normY.div(0.5)), 0, 99);
        const gridIndex = row.mul(100).add(col);
        
        const slot = atomicAdd(cellOffsetAtomicBuffer.element(gridIndex), uint(1));
        sortedAgentIndicesBuffer.element(slot).assign(i);
        sortedPositionsBuffer.element(slot).assign(pos);
        sortedVelocitiesBuffer.element(slot).assign(vel);
    });
});

export const spatialCollisionNode = Fn(([positions, velocities, cellCountBuffer, cellOffsetBuffer, sortedAgentIndicesBuffer, sortedPositionsBuffer, sortedVelocitiesBuffer, infectionBuffer, timerBuffer, infectionRadius, transmissionProb, recoveryTime, seedUniform, agentCountLimit]: any) => {
    // 1M threads
    const i = instanceIndex;
    If(i.lessThan(agentCountLimit), () => {
        const pos = positions.element(i);
        const vel = velocities.element(i);
        const myInfection = infectionBuffer.element(i);
        
        const normX = pos.x.add(25.0);
        const normY = pos.y.add(25.0);
    
    // 100x100 grid, cell size 0.5
    const col = uint(clamp(floor(normX.div(0.5)), 0, 99));
    const row = uint(clamp(floor(normY.div(0.5)), 0, 99));
    
    const separation = vec2(0, 0).toVar();
    const neighborsCount = uint(0).toVar();
    
    // Dynamic search bounds based on infection radius.
    // Cell size is 0.5 units.
    const searchRadius = int(ceil(infectionRadius.div(0.5)));
    // Guarantee at least enough span for 0.5 physical separation (ceil(0.5 / 0.5) = 1)
    const effectiveSearchRadius = max(1, searchRadius);
    const gridSpan = effectiveSearchRadius.mul(2).add(1);
    
    Loop(gridSpan, ({ i: rIndex }) => {
        Loop(gridSpan, ({ i: cIndex }) => {
            const rOffset = int(rIndex).sub(effectiveSearchRadius);
            const cOffset = int(cIndex).sub(effectiveSearchRadius);
            
            const neighborCol = uint(clamp(int(col).add(cOffset), 0, 99));
            const neighborRow = uint(clamp(int(row).add(rOffset), 0, 99));
            const neighborGridIndex = neighborRow.mul(uint(100)).add(neighborCol);
            
            const startIdx = cellOffsetBuffer.element(neighborGridIndex);
            const count = uint(cellCountBuffer.element(neighborGridIndex));
            
            // Uniform Strided Sampling: Cap ALU per cell (8 per cell max)
            const stride = max(uint(1), count.div(uint(8)));
            const loopCap = min(count, uint(8));
            
            Loop(loopCap, ({ i: j }) => {
                const jUint = uint(j);
                const sortedIndex = startIdx.add(jUint.mul(stride));
                const otherAgentId = sortedAgentIndicesBuffer.element(sortedIndex);
                
                If(otherAgentId.notEqual(i), () => {
                    const otherPos = sortedPositionsBuffer.element(sortedIndex);
                    
                    const delta = pos.sub(otherPos);
                    const distSq = delta.dot(delta);
                    
                    // Repulsion threshold (0.5 * 0.5 = 0.25)
                    If(distSq.lessThan(0.25).and(distSq.greaterThan(0.000001)), () => {
                        const dist = sqrt(distSq);
                        const pushDir = delta.normalize();
                        const pushStrength = float(0.5).sub(dist); 
                        separation.addAssign(pushDir.mul(pushStrength));
                        neighborsCount.addAssign(uint(1));
                    });

                    // Phase 7: Viral Transmission (Decoupled from physical repulsion via distSq and state check)
                    If(myInfection.notEqual(uint(2)), () => {
                        const infRadSq = infectionRadius.mul(infectionRadius);
                        If(distSq.lessThan(infRadSq).and(distSq.greaterThan(0.000001)), () => {
                            If(myInfection.equal(uint(0)), () => {
                                If(infectionBuffer.element(otherAgentId).equal(uint(1)), () => {
                                    // Generate pseudo-random value [0, 1] for this contact using PCG Hash
                                    const contactSeed = pos.x.mul(13.5).add(pos.y.mul(41.2)).add(seedUniform).add(float(otherAgentId));
                                    const rand = pcgHash(contactSeed);
                                    
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
        });
    });
    
    If(neighborsCount.greaterThan(uint(0)), () => {
        // Average the separation force
        const avgSeparation = separation.div(float(neighborsCount));
        
        // Push the agent and normalize its velocity
        vel.addAssign(avgSeparation.mul(0.2)); 
        vel.assign(vel.normalize());
    });
    });
});

export const setupEpidemicNode = Fn(([positions, velocities, infectionBuffer, timerBuffer, seed, initialInfectedRadius, recoveryTime]: any) => {
    const i = instanceIndex;
    const pos = positions.element(i);
    const vel = velocities.element(i);
    const infection = infectionBuffer.element(i);

    // Robust PRNG using PCG Hash
    // We can just use the global index and the seed, no float32 drift hacks needed!
    const seedBase = float(i).add(seed.mul(10000.0));
    
    const r1 = pcgHash(seedBase.mul(12.9898));
    const r2 = pcgHash(seedBase.mul(78.233));
    const r3 = pcgHash(seedBase.mul(45.123));
    const r4 = pcgHash(seedBase.mul(93.989));

    // Place randomly across the whole board
    pos.x.assign(r1.sub(0.5).mul(50.0));
    pos.y.assign(r2.sub(0.5).mul(50.0));

    vel.x.assign(r3.sub(0.5).mul(2.0));
    vel.y.assign(r4.sub(0.5).mul(2.0));

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
