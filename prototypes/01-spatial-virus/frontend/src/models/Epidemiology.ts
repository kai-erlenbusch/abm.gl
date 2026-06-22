import { Fn, uint, int, float, instanceIndex, vec2, If, Loop, clamp, floor, ceil, max, min, sqrt } from 'three/tsl';
import { prngHash } from '../engine/math/PRNG';

export const setupEpidemicNode = Fn(([positions, velocities, infectionBuffer, timerBuffer, seed, initialInfectedRadius, recoveryTime, agentCountLimit, worldSize]: any) => {
    const i = instanceIndex;
    If(i.lessThan(agentCountLimit), () => {
        const pos = positions.element(i);
        const vel = velocities.element(i);
        const infection = infectionBuffer.element(i);

        const uSeed = uint(seed.mul(10000.0));
        const base = uint(i).add(uSeed);
        
        const h1 = uint(prngHash(base).mul(4294967295.0));
        const h2 = uint(prngHash(h1).mul(4294967295.0));
        const h3 = uint(prngHash(h2).mul(4294967295.0));
        const h4 = uint(prngHash(h3).mul(4294967295.0));
        
        const r1 = float(h1).div(4294967295.0);
        const r2 = float(h2).div(4294967295.0);
        const r3 = float(h3).div(4294967295.0);
        const r4 = float(h4).div(4294967295.0);

        pos.assign(vec2(
            r1.sub(0.5).mul(worldSize),
            r2.sub(0.5).mul(worldSize)
        ));

        const initialVel = vec2(
            r3.sub(0.5).mul(2.0),
            r4.sub(0.5).mul(2.0)
        );
        vel.assign(initialVel.normalize());

        const distSq = pos.x.mul(pos.x).add(pos.y.mul(pos.y));
        const initRadSq = initialInfectedRadius.mul(initialInfectedRadius);
        If(distSq.lessThan(initRadSq), () => {
            infection.assign(uint(1));
            timerBuffer.element(i).assign(recoveryTime);
        }).Else(() => {
            infection.assign(uint(0));
            timerBuffer.element(i).assign(0.0);
        });
    });
});

export const epidemicCollisionNode = Fn(([velocities, infectionBuffer, timerBuffer, cellCountBuffer, cellOffsetBuffer, sortedAgentIndicesBuffer, sortedPositionsBuffer, infectionRadius, transmissionProb, recoveryTime, seedUniform, agentCountLimit, deltaUniform, worldOffset, cellSize, gridDimX, gridDimY]: any) => {
    const sortedThreadId = instanceIndex;
    If(sortedThreadId.lessThan(agentCountLimit), () => {
        const realAgentId = sortedAgentIndicesBuffer.element(sortedThreadId);
        const pos = sortedPositionsBuffer.element(sortedThreadId);
        const vel = velocities.element(realAgentId);
        
        const myInfection = infectionBuffer.element(realAgentId).toVar();
        const myTimer = timerBuffer.element(realAgentId).toVar();
        
        const normX = pos.x.add(worldOffset);
        const normY = pos.y.add(worldOffset);
    
        const col = uint(clamp(floor(normX.div(cellSize)), float(0), float(gridDimX).sub(1.0)));
        const row = uint(clamp(floor(normY.div(cellSize)), float(0), float(gridDimY).sub(1.0)));
        
        const separation = vec2(0, 0).toVar();
        const neighborsCount = uint(0).toVar();
        
        const searchRadius = int(ceil(infectionRadius.div(cellSize)));
        const effectiveSearchRadius = max(1, searchRadius);
        const gridSpan = effectiveSearchRadius.mul(2).add(1);
        
        Loop(gridSpan, ({ i: rIndex }) => {
            Loop(gridSpan, ({ i: cIndex }) => {
                const rOffset = int(rIndex).sub(effectiveSearchRadius);
                const cOffset = int(cIndex).sub(effectiveSearchRadius);
                
                const neighborCol = uint(clamp(int(col).add(cOffset), int(0), int(gridDimX).sub(1)));
                const neighborRow = uint(clamp(int(row).add(rOffset), int(0), int(gridDimY).sub(1)));
                const neighborGridIndex = neighborRow.mul(gridDimX).add(neighborCol);
                
                const startIdx = cellOffsetBuffer.element(neighborGridIndex);
                const count = uint(cellCountBuffer.element(neighborGridIndex));
                
                // Volume Exclusion Cap
                const loopCap = min(count, uint(4096));
                
                Loop(loopCap, ({ i: j }) => {
                    const jUint = uint(j);
                    const sortedIndex = startIdx.add(jUint);
                    const otherAgentId = sortedAgentIndicesBuffer.element(sortedIndex);
                    
                    If(otherAgentId.notEqual(realAgentId), () => {
                        const otherPos = sortedPositionsBuffer.element(sortedIndex);
                        
                        const delta = pos.sub(otherPos);
                        const distSq = delta.dot(delta);
                        
                        // Repulsion threshold uses a physical separation metric proportional to cellSize
                        // Assuming 0.5 physical radius per agent for pushing apart
                        const pushDistSq = float(0.25);
                        If(distSq.lessThan(pushDistSq).and(distSq.greaterThan(0.000001)), () => {
                            const dist = sqrt(distSq);
                            const pushDir = delta.normalize();
                            const pushStrength = float(0.5).sub(dist); 
                            separation.addAssign(pushDir.mul(pushStrength));
                            neighborsCount.addAssign(uint(1));
                        });

                        If(myInfection.notEqual(uint(2)), () => {
                            const infRadSq = infectionRadius.mul(infectionRadius);
                            If(distSq.lessThan(infRadSq).and(distSq.greaterThan(0.000001)), () => {
                                If(myInfection.equal(uint(0)), () => {
                                    If(infectionBuffer.element(otherAgentId).equal(uint(1)), () => {
                                        const contactSeed = uint(pos.x.mul(13.5).add(pos.y.mul(41.2)).add(seedUniform).add(float(otherAgentId)).mul(1000.0));
                                        const rand = prngHash(contactSeed);
                                        
                                        If(rand.lessThan(transmissionProb), () => {
                                            infectionBuffer.element(realAgentId).assign(uint(1));
                                            timerBuffer.element(realAgentId).assign(recoveryTime);
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
            const timeScaledForce = avgSeparation.mul(0.2).mul(deltaUniform.mul(60.0));
            vel.addAssign(timeScaledForce); 
            
            const lenSq = vel.dot(vel);
            If(lenSq.greaterThan(0.000001), () => {
                vel.assign(vel.normalize());
            }).Else(() => {
                vel.assign(vec2(1.0, 0.0));
            });
        });
    });
});
