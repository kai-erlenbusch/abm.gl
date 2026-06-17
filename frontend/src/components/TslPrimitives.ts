import { 
    Fn, float, vec3, abs, If, Loop, instanceIndex, uint, int,
    atomicAdd, atomicLoad, atomicStore, min, max,
    workgroupBarrier, workgroupId, workgroupArray, clamp, floor, ceil,
    fract, sin, length, texture, vec2, Break, invocationLocalIndex, mod, select, sqrt
} from 'three/tsl';

export const prngHash = Fn(([seed]) => {
    let x = uint(seed);
    x = x.shiftLeft(uint(13)).bitXor(x);
    x = x.mul(x.mul(x).mul(uint(15731)).add(uint(789221))).add(uint(1376312589));
    return float(x.bitAnd(uint(0x7fffffff))).div(float(0x7fffffff));
});

export const hashUInt = Fn(([seed]) => {
    let x = uint(seed);
    x = x.shiftLeft(uint(13)).bitXor(x);
    x = x.mul(x.mul(x).mul(uint(15731)).add(uint(789221))).add(uint(1376312589));
    return x;
});

/**
 * 1. Flocking Behavior Primitive
 * Calculates separation, alignment, and cohesion entirely on the GPU.
 */
export const flockingBehavior = Fn(([positions, positions_next, velocities_next, infection_next, timer_next, deltaUniform, agentCountLimit]: any) => {
    const i = instanceIndex;
    If(i.lessThan(agentCountLimit), () => {
        const pos = positions.element(i);
        const vel = velocities_next.element(i);
        
        const speedLocal = float(10.0);

        const newPos = pos.add(vel.mul(speedLocal).mul(deltaUniform));

        const wrappedX = newPos.x.add(75.0).mod(50.0).sub(25.0);
        const wrappedY = newPos.y.add(75.0).mod(50.0).sub(25.0);
        positions_next.element(i).assign(vec2(wrappedX, wrappedY));
        
        // Phase 7: SIR Recovery Logic
        const myInfection = infection_next.element(i);
        const myTimer = timer_next.element(i);
        const isInfectedCond = myInfection.equal(uint(1));
        
        myTimer.subAssign(deltaUniform.mul(select(isInfectedCond, float(1.0), float(0.0))));
        
        infection_next.element(i).assign(
            select(
                isInfectedCond.and(myTimer.lessThanEqual(0.0)),
                uint(2),
                myInfection
            )
        );
    });
});

export const telemetryAggregateNode = Fn(([positions, velocities, policyMapTexture, aggregateBuffer, infectionBuffer, agentCountLimit]: any) => {
    const i = instanceIndex;
    If(i.lessThan(agentCountLimit), () => {
        const pos = positions.element(i);
        const vel = velocities.element(i);
        
        const u = clamp(pos.x.add(25.0).div(50.0), 0.0, 1.0);
        const v = clamp(pos.y.add(25.0).div(50.0), 0.0, 1.0);
        const speedLocal = float(1.0);

        // Divide 50x50 world (-25 to 25) into 10x10 grid. Each cell is 5x5.
        const normX = pos.x.add(25.0);
        const normY = float(25.0).sub(pos.y);
        
        const col = floor(normX.div(5.0));
        const row = floor(normY.div(5.0));
        
        const safeCol = uint(clamp(col, 0, 9));
        const safeRow = uint(clamp(row, 0, 9));
        
        const gridIndex = safeRow.mul(uint(10)).add(safeCol);
        const speedIndex = gridIndex.mul(uint(4));
        const countIndex = gridIndex.mul(uint(4)).add(uint(1));
        const infectedIndex = gridIndex.mul(uint(4)).add(uint(2));
        const recoveredIndex = gridIndex.mul(uint(4)).add(uint(3));

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
});

export const resetAggregate = Fn(([aggregateBuffer]) => {
    // Run this with 300 threads to clear the entire spatial grid buffer
    const i = instanceIndex;
    atomicStore(aggregateBuffer.element(i), uint(0));
});



export const spatialCollisionNode = Fn(([velocities, velocities_next, infectionBuffer, infection_next, timerBuffer, timer_next, cellCountBuffer, cellOffsetBuffer, sortedAgentIndicesBuffer, sortedPositionsBuffer, infectionRadius, transmissionProb, recoveryTime, seedUniform, agentCountLimit, deltaUniform]: any) => {
    // 1M threads
    const sortedThreadId = instanceIndex;
    If(sortedThreadId.lessThan(agentCountLimit), () => {
        // Fix Warp Divergence: Threads in a warp process spatially adjacent agents
        const realAgentId = sortedAgentIndicesBuffer.element(sortedThreadId);
        const pos = sortedPositionsBuffer.element(sortedThreadId);
        
        // Read from Primary State A, convert to variables for local accumulation
        const vel = velocities.element(realAgentId).toVar();
        const myInfection = infectionBuffer.element(realAgentId).toVar();
        const myTimer = timerBuffer.element(realAgentId).toVar();
        
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
                
                // Volume Exclusion Cap: Physically, only a finite amount of matter can occupy a cell.
                // Reading linearly ensures perfectly coalesced memory access from our Prefix Sum sorted buffers.
                const loopCap = min(count, uint(4096));
                
                Loop(loopCap, ({ i: j }) => {
                    const jUint = uint(j);
                    // LINEAR READ: No stride. Fixes Memory Coalescing!
                    const sortedIndex = startIdx.add(jUint);
                    const otherAgentId = sortedAgentIndicesBuffer.element(sortedIndex);
                    
                    If(otherAgentId.notEqual(realAgentId), () => {
                        const otherPos = sortedPositionsBuffer.element(sortedIndex);
                        
                        const delta = pos.sub(otherPos);
                        const distSq = delta.dot(delta);
                        
                        // Repulsion threshold (0.5 * 0.5 = 0.25)
                        If(distSq.lessThan(0.25), () => {
                            If(distSq.greaterThan(0.000001), () => {
                                const dist = sqrt(distSq);
                                const pushDir = delta.normalize();
                                const pushStrength = float(0.5).sub(dist); 
                                separation.addAssign(pushDir.mul(pushStrength));
                                neighborsCount.addAssign(uint(1));
                            }).Else(() => {
                                // Micro-jitter for perfectly overlapping particles to push them apart deterministically
                                const dirX = select(realAgentId.greaterThan(otherAgentId), float(1.0), float(-1.0));
                                separation.addAssign(vec2(dirX, 0.0).mul(0.1));
                                neighborsCount.addAssign(uint(1));
                            });
                        });

                        // Phase 7: Viral Transmission (Decoupled from physical repulsion via distSq and state check)
                        If(myInfection.notEqual(uint(2)), () => {
                            const infRadSq = infectionRadius.mul(infectionRadius);
                            If(distSq.lessThan(infRadSq), () => {
                                If(myInfection.equal(uint(0)), () => {
                                    If(infectionBuffer.element(otherAgentId).equal(uint(1)), () => {
                                        // Generate pseudo-random value [0, 1] for this contact using PCG Hash
                                        const contactSeed = uint(pos.x.mul(13.5).add(pos.y.mul(41.2)).add(seedUniform).add(float(otherAgentId)).mul(1000.0));
                                        const rand = prngHash(contactSeed);
                                        
                                        If(rand.lessThan(transmissionProb), () => {
                                            myInfection.assign(uint(1));
                                            myTimer.assign(recoveryTime);
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
            const avgSeparation = separation.div(float(neighborsCount));
            // Scale repulsion force by delta time (normalized to 60 FPS = 0.016s) to fix hyperspeed at high refresh rates
            const timeScaledForce = avgSeparation.mul(0.2).mul(deltaUniform.mul(60.0));
            vel.addAssign(timeScaledForce); 
            
            // Prevent NaN from normalization if velocity becomes perfectly zero
            const lenSq = vel.dot(vel);
            If(lenSq.greaterThan(0.000001), () => {
                vel.assign(vel.normalize());
            }).Else(() => {
                vel.assign(vec2(1.0, 0.0));
            });
        });

        // Write Final Computed State to Ping-Pong Buffers (_next)
        velocities_next.element(realAgentId).assign(vel);
        infection_next.element(realAgentId).assign(myInfection);
        timer_next.element(realAgentId).assign(myTimer);
    });
});

export const setupEpidemicNode = Fn(([positions, velocities, infectionBuffer, timerBuffer, seed, initialInfectedRadius, recoveryTime, agentCountLimit]: any) => {
    const i = instanceIndex;
    If(i.lessThan(agentCountLimit), () => {
        const pos = positions.element(i);
        const vel = velocities.element(i);
        const infection = infectionBuffer.element(i);

        // Robust PRNG using integer hash to prevent banding hyperplanes
        const uSeed = uint(seed.mul(10000.0));
        const base = uint(i).add(uSeed);
        
        // Feed the output of the hash back into itself to break linear resonance
        const h1 = hashUInt(base);
        const h2 = hashUInt(h1);
        const h3 = hashUInt(h2);
        const h4 = hashUInt(h3);
        
        // Convert to float [0, 1) using 24 bits of entropy to avoid f32 precision loss
        const r1 = float(h1.bitAnd(uint(0xffffff))).div(float(16777216.0));
        const r2 = float(h2.bitAnd(uint(0xffffff))).div(float(16777216.0));
        const r3 = float(h3.bitAnd(uint(0xffffff))).div(float(16777216.0));
        const r4 = float(h4.bitAnd(uint(0xffffff))).div(float(16777216.0));

        // Place randomly across the whole board
        pos.assign(vec2(
            r1.sub(0.5).mul(50.0),
            r2.sub(0.5).mul(50.0)
        ));

        const initialVel = vec2(
            r3.sub(0.5).mul(2.0),
            r4.sub(0.5).mul(2.0)
        );
        vel.assign(initialVel.normalize());

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
});

/**
 * 3. GPU Data Aggregation Primitive
 * Reduces the 100,000+ agent states down to a 10-float array
 * before reading back to Javascript, avoiding VRAM bottlenecks.
 */
export const aggregateStats = Fn(([agentData, outputBuffer]) => {
    // Parallel reduction logic
});
