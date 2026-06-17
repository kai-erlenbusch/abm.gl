import { Fn, float, uint, instanceIndex, atomicAdd, atomicStore, clamp, floor, If, select, length } from 'three/tsl';

export const telemetryAggregateNode = Fn(([positions, velocities, policyMapTexture, aggregateBuffer, infectionBuffer, agentCountLimit, worldSize, worldOffset]: any) => {
    const globalId = instanceIndex;

    If(globalId.lessThan(agentCountLimit), () => {
        const pos = positions.element(globalId);
        const vel = velocities.element(globalId);
        const myInfection = infectionBuffer.element(globalId);
        
        const normX = pos.x.add(worldOffset);
        // Telemetry uses worldOffset to flip Y axis (e.g. 25.0 - pos.y maps 25..-25 to 0..50)
        const normY = worldOffset.sub(pos.y);
        
        // Telemetry uses 10x10 grid, so cell size is worldSize / 10
        const telemetryCellSize = worldSize.div(10.0);
        const col = uint(clamp(floor(normX.div(telemetryCellSize)), float(0), float(9)));
        const row = uint(clamp(floor(normY.div(telemetryCellSize)), float(0), float(9)));
        
        const gridIndex = row.mul(uint(10)).add(col);
        const speedIndex = gridIndex.mul(uint(4));
        const countIndex = speedIndex.add(uint(1));
        const infectedIndex = speedIndex.add(uint(2));
        const recoveredIndex = speedIndex.add(uint(3));
        
        const speed = length(vel);
        
        const isInfected = select(myInfection.equal(uint(1)), uint(1), uint(0));
        const isRecovered = select(myInfection.equal(uint(2)), uint(1), uint(0));
        
        atomicAdd(aggregateBuffer.element(speedIndex), uint(speed.mul(100.0)));
        atomicAdd(aggregateBuffer.element(countIndex), uint(1));
        atomicAdd(aggregateBuffer.element(infectedIndex), isInfected);
        atomicAdd(aggregateBuffer.element(recoveredIndex), isRecovered);
    });
});

export const resetAggregate = Fn(([aggregateBuffer]) => {
    const i = instanceIndex;
    atomicStore(aggregateBuffer.element(i), uint(0));
});
