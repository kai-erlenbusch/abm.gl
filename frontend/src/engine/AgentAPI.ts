// @ts-nocheck
import { Fn, instanceIndex, uint, Loop, If, clamp, floor, min, max, sqrt } from 'three/tsl';

/**
 * NetLogo-inspired API for writing generic agent logic in WebGPU.
 */
export const AgentAPI = {
    
    /**
     * Given the SpatialGrid structures, iterates over all agents within a physical radius
     * and executes the callback with the neighbor's agent ID.
     */
    inRadius: (pos: any, radius: any, grid: any, sortedPositions: any, sortedAgentIndices: any, callback: (neighborId: any, distSq: any) => void) => {
        const normX = pos.x.add(25.0);
        const normY = pos.y.add(25.0);
        
        const col = uint(clamp(floor(normX.div(0.5)), 0, 99));
        const row = uint(clamp(floor(normY.div(0.5)), 0, 99));
        
        const searchRadius = int(ceil(radius.div(0.5)));
        const effectiveSearchRadius = max(1, searchRadius);
        const gridSpan = effectiveSearchRadius.mul(2).add(1);

        Loop(gridSpan, ({ i: rIndex }) => {
            Loop(gridSpan, ({ i: cIndex }) => {
                const rOffset = int(rIndex).sub(effectiveSearchRadius);
                const cOffset = int(cIndex).sub(effectiveSearchRadius);
                
                const neighborCol = uint(clamp(int(col).add(cOffset), 0, 99));
                const neighborRow = uint(clamp(int(row).add(rOffset), 0, 99));
                const neighborGridIndex = neighborRow.mul(uint(100)).add(neighborCol);
                
                const startIdx = grid.nodes.offset.element(neighborGridIndex);
                const count = uint(grid.nodes.count.element(neighborGridIndex));
                
                const loopCap = min(count, uint(1024)); // Cap per cell
                
                Loop(loopCap, ({ i: j }) => {
                    const sortedIndex = startIdx.add(uint(j));
                    const otherPos = sortedPositions.element(sortedIndex);
                    
                    const delta = pos.sub(otherPos);
                    const distSq = delta.dot(delta);
                    const radSq = radius.mul(radius);
                    
                    If(distSq.lessThan(radSq).and(distSq.greaterThan(0.000001)), () => {
                        const neighborAgentId = sortedAgentIndices.element(sortedIndex);
                        callback(neighborAgentId, distSq);
                    });
                });
            });
        });
    },

    /**
     * Ask agents. Similar to `ask turtles [...]`
     * Evaluates a condition, and if true, executes the logic.
     */
    ask: (condition: any, logic: () => void) => {
        If(condition, logic);
    },

    /**
     * Move forward in the direction of velocity.
     * Modifies position buffer directly.
     */
    fd: (pos: any, vel: any, speed: any) => {
        pos.addAssign(vel.mul(speed));
    },

    /**
     * Bounces agent off the bounds of a (-25, 25) coordinate system.
     */
    bounce: (pos: any, vel: any) => {
        If(pos.x.greaterThan(25).or(pos.x.lessThan(-25)), () => {
            vel.x.mulAssign(-1);
        });
        If(pos.y.greaterThan(25).or(pos.y.lessThan(-25)), () => {
            vel.y.mulAssign(-1);
        });
    }
};
