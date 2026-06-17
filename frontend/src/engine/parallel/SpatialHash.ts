import { Fn, uint, float, instanceIndex, atomicStore, atomicAdd, If, clamp, floor } from 'three/tsl';

export const spatialResetNode = Fn(([cellCountBuffer, cellOffsetAtomicBuffer, cellCountLimit]: any) => {
    const i = instanceIndex;
    If(i.lessThan(cellCountLimit), () => {
        atomicStore(cellCountBuffer.element(i), uint(0));
        atomicStore(cellOffsetAtomicBuffer.element(i), uint(0));
    });
});

export const spatialCountNode = Fn(([positions, cellCountBuffer, agentCountLimit, worldOffset, cellSize, gridDimX, gridDimY]: any) => {
    const i = instanceIndex;
    If(i.lessThan(agentCountLimit), () => {
        const pos = positions.element(i);
        const normX = pos.x.add(worldOffset);
        const normY = pos.y.add(worldOffset);
        
        const col = uint(clamp(floor(normX.div(cellSize)), float(0), float(gridDimX).sub(1.0)));
        const row = uint(clamp(floor(normY.div(cellSize)), float(0), float(gridDimY).sub(1.0)));
        const gridIndex = row.mul(gridDimX).add(col);
        
        atomicAdd(cellCountBuffer.element(gridIndex), uint(1));
    });
});

export const spatialScatterNode = Fn(([positions, velocities, cellOffsetAtomicBuffer, sortedAgentIndicesBuffer, sortedPositionsBuffer, sortedVelocitiesBuffer, agentCountLimit, worldOffset, cellSize, gridDimX, gridDimY]: any) => {
    const i = instanceIndex;
    If(i.lessThan(agentCountLimit), () => {
        const pos = positions.element(i);
        const vel = velocities.element(i);
        const normX = pos.x.add(worldOffset);
        const normY = pos.y.add(worldOffset);
        
        const col = uint(clamp(floor(normX.div(cellSize)), float(0), float(gridDimX).sub(1.0)));
        const row = uint(clamp(floor(normY.div(cellSize)), float(0), float(gridDimY).sub(1.0)));
        const gridIndex = row.mul(gridDimX).add(col);
        
        const slot = atomicAdd(cellOffsetAtomicBuffer.element(gridIndex), uint(1));
        sortedAgentIndicesBuffer.element(slot).assign(i);
        sortedPositionsBuffer.element(slot).assign(pos);
        sortedVelocitiesBuffer.element(slot).assign(vel);
    });
});
