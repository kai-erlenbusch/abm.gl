import { Fn, uint, float, instanceIndex, If, vec2, select } from 'three/tsl';

export const flockingBehavior = Fn(([positions, velocities, infectionBuffer, timerBuffer, deltaUniform, agentCountLimit, worldSize, worldOffset]: any) => {
    const i = instanceIndex;
    If(i.lessThan(agentCountLimit), () => {
        const pos = positions.element(i);
        const vel = velocities.element(i);
        
        const speedLocal = float(10.0);

        const newPos = pos.add(vel.mul(speedLocal).mul(deltaUniform));

        const wrappedX = newPos.x.add(worldSize.mul(1.5)).mod(worldSize).sub(worldOffset);
        const wrappedY = newPos.y.add(worldSize.mul(1.5)).mod(worldSize).sub(worldOffset);
        positions.element(i).assign(vec2(wrappedX, wrappedY));
        
        // SIR Recovery Logic (Temporarily still here due to buffer coupling)
        const myInfection = infectionBuffer.element(i);
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
});
