import { Fn, uint, instanceIndex, invocationLocalIndex, workgroupArray, select, workgroupBarrier, workgroupId, Loop, If, atomicStore } from 'three/tsl';// Pass 2a: Local Chunk Scan
export const spatialPrefixSum_ChunkNode = Fn(([cellCountBuffer, cellOffsetBuffer, chunkSumsBuffer, cellCountLimit]: any) => {
    const globalId = instanceIndex;
    const localId = invocationLocalIndex;
    const groupId = workgroupId.x;
    
    const sharedArray = workgroupArray('uint', 256);
    
    // Load data into shared memory
    const count = select(globalId.lessThan(cellCountLimit), uint(cellCountBuffer.element(globalId)), uint(0));
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
        // Number of chunks is roughly ceil(cellCountLimit / 256)
        const maxChunks = uint(cellCountLimit).add(255).div(256);
        If(groupId.lessThan(maxChunks), () => {
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
    If(globalId.lessThan(cellCountLimit), () => {
        cellOffsetBuffer.element(globalId).assign(sharedArray.element(localId));
    });
});

// Pass 2b: Block Scan (Sequential scan of the chunk sums)
export const spatialPrefixSum_BlockNode = Fn(([chunkSumsBuffer, chunkCountLimit]: any) => {
    const i = instanceIndex;
    If(i.equal(uint(0)), () => {
        const sum = uint(0).toVar();
        Loop(chunkCountLimit, ({ i: j }) => {
            const jUint = uint(j);
            const count = uint(chunkSumsBuffer.element(jUint));
            chunkSumsBuffer.element(jUint).assign(sum);
            sum.addAssign(count);
        });
    });
});

// Pass 2c: Scatter/Add global block offsets to local chunks
export const spatialPrefixSum_ScatterNode = Fn(([cellOffsetBuffer, cellOffsetAtomicBuffer, chunkSumsBuffer, cellCountLimit]: any) => {
    const globalId = instanceIndex;
    const groupId = workgroupId.x;
    
    If(globalId.lessThan(cellCountLimit), () => {
        const blockOffset = chunkSumsBuffer.element(groupId);
        const finalOffset = cellOffsetBuffer.element(globalId).add(blockOffset);
        cellOffsetBuffer.element(globalId).assign(finalOffset);
        atomicStore(cellOffsetAtomicBuffer.element(globalId), finalOffset);
    });
});
