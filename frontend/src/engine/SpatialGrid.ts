// @ts-nocheck
import { StorageBufferAttribute } from 'three/webgpu';
import { storage, Fn, instanceIndex, uint, atomicStore, atomicAdd, atomicLoad, If, clamp, floor, workgroupArray, workgroupBarrier, workgroupId, invocationLocalIndex, Loop, int, max, min, vec2, sqrt, select } from 'three/tsl';

export class SpatialGrid {
  cellCount: number;
  agentCount: number;
  
  attributes: Record<string, StorageBufferAttribute>;
  nodes: Record<string, any>;

  constructor(gridSizeX: number, gridSizeY: number, agentCount: number) {
    this.cellCount = gridSizeX * gridSizeY;
    this.agentCount = agentCount;
    this.attributes = {};
    this.nodes = {};

    this.initBuffers();
  }

  initBuffers() {
    // Spatial Grid Buffers
    const cellCountArray = new Uint32Array(this.cellCount);
    const cellOffsetArray = new Uint32Array(this.cellCount);
    const cellOffsetAtomicArray = new Uint32Array(this.cellCount);
    const chunkCount = Math.ceil(this.cellCount / 256);
    const chunkSumsArray = new Uint32Array(chunkCount);
    
    // Sorted Agent Output
    const sortedAgentArray = new Uint32Array(this.agentCount);
    const sortedPosArray = new Float32Array(this.agentCount * 2);
    const sortedVelArray = new Float32Array(this.agentCount * 2);

    this.attributes.count = new StorageBufferAttribute(cellCountArray, 1);
    this.attributes.offset = new StorageBufferAttribute(cellOffsetArray, 1);
    this.attributes.offsetAtomic = new StorageBufferAttribute(cellOffsetAtomicArray, 1);
    this.attributes.chunkSums = new StorageBufferAttribute(chunkSumsArray, 1);
    this.attributes.sortedIndices = new StorageBufferAttribute(sortedAgentArray, 1);
    this.attributes.sortedPositions = new StorageBufferAttribute(sortedPosArray, 2);
    this.attributes.sortedVelocities = new StorageBufferAttribute(sortedVelArray, 2);

    this.nodes.countAtomic = storage(this.attributes.count, 'uint', this.cellCount).toAtomic();
    this.nodes.count = storage(this.attributes.count, 'uint', this.cellCount);
    
    this.nodes.offset = storage(this.attributes.offset, 'uint', this.cellCount);
    this.nodes.offsetAtomic = storage(this.attributes.offsetAtomic, 'uint', this.cellCount).toAtomic();
    
    this.nodes.chunkSums = storage(this.attributes.chunkSums, 'uint', chunkCount);
    this.nodes.sortedIndices = storage(this.attributes.sortedIndices, 'uint', this.agentCount);
    this.nodes.sortedPositions = storage(this.attributes.sortedPositions, 'vec2', this.agentCount);
  }

  dispose() {
    for (const key in this.attributes) {
      this.attributes[key].dispose();
    }
  }

  // --- TSL Primitive Nodes ---

  getResetNode() {
      const { countAtomic, offset, offsetAtomic } = this.nodes;
      return Fn(() => {
          const i = instanceIndex;
          If(i.lessThan(this.cellCount), () => {
              atomicStore(countAtomic.element(i), uint(0));
              atomicStore(offsetAtomic.element(i), uint(0));
          });
      })().compute(this.cellCount);
  }

  getCountNode(positionsNode: any) {
    const { countAtomic } = this.nodes;
    const limit = this.agentCount;
    const compute = Fn(() => {
        const i = instanceIndex;
        If(i.lessThan(limit), () => {
            const pos = positionsNode.element(i);
            const normX = pos.x.add(25.0);
            const normY = pos.y.add(25.0);
            const col = uint(clamp(floor(normX.div(0.5)), 0, 99));
            const row = uint(clamp(floor(normY.div(0.5)), 0, 99));
            const gridIndex = row.mul(uint(100)).add(col);
            atomicAdd(countAtomic.element(gridIndex), uint(1));
        });
    })();
    const pass = compute.compute(limit);
    pass.workgroupSize = [256, 1, 1];
    return pass;
  }

  getPrefixSumChunkNode() {
      const { chunkSums, count, offset } = this.nodes;
      const limit = this.cellCount;
    const chunkLim = Math.ceil(this.cellCount / 256);
    const compute = Fn(() => {
        const globalId = instanceIndex;
        const localId = invocationLocalIndex;
        const groupId = globalId.div(256);
        const sharedArray = workgroupArray('uint', 256);
        
        const c = select(globalId.lessThan(limit), uint(count.element(globalId)), uint(0));
        sharedArray.element(localId).assign(c);
        workgroupBarrier();
        
        for (let off = 1; off < 256; off *= 2) {
            const i = localId.mul(uint(off * 2)).add(uint(off * 2 - 1));
            If(i.lessThan(uint(256)), () => {
                sharedArray.element(i).addAssign(sharedArray.element(i.sub(uint(off))));
            });
            workgroupBarrier();
        }
        
        If(localId.equal(255), () => {
            If(groupId.lessThan(chunkLim), () => {
                chunkSums.element(groupId).assign(sharedArray.element(uint(255)));
            });
            sharedArray.element(uint(255)).assign(uint(0));
        });
        workgroupBarrier();
        
        for (let off = 128; off > 0; off /= 2) {
            const i = localId.mul(uint(off * 2)).add(uint(off * 2 - 1));
            If(i.lessThan(uint(256)), () => {
                const temp = sharedArray.element(i.sub(uint(off))).toVar();
                sharedArray.element(i.sub(uint(off))).assign(sharedArray.element(i));
                sharedArray.element(i).addAssign(temp);
            });
            workgroupBarrier();
        }
        
        If(globalId.lessThan(limit), () => {
            offset.element(globalId).assign(sharedArray.element(localId));
        });
    })();
    const pass = compute.compute(limit);
    pass.workgroupSize = [256, 1, 1];
    return pass;
  }

  getPrefixSumBlockNode() {
      const { chunkSums } = this.nodes;
      const chunkLim = Math.ceil(this.cellCount / 256);
      return Fn(() => {
          const i = instanceIndex;
          If(i.equal(uint(0)), () => {
              const sum = uint(0).toVar();
              Loop(chunkLim, ({ i: j }) => {
                  const jUint = uint(j);
                  const c = uint(chunkSums.element(jUint));
                  chunkSums.element(jUint).assign(sum);
                  sum.addAssign(c);
              });
          });
      })().compute(1);
  }

  getPrefixSumScatterNode() {
      const { offset, offsetAtomic, chunkSums } = this.nodes;
      const limit = this.cellCount;
      const compute = Fn(() => {
          const globalId = instanceIndex;
          const groupId = workgroupId.x;
          If(globalId.lessThan(limit), () => {
              const blockOffset = chunkSums.element(groupId);
              const finalOffset = offset.element(globalId).add(blockOffset);
              offset.element(globalId).assign(finalOffset);
              atomicStore(offsetAtomic.element(globalId), finalOffset);
          });
      })();
      const pass = compute.compute(limit);
      pass.workgroupSize = [256, 1, 1];
      return pass;
  }

  getAgentScatterNode(positionsNode: any, velocitiesNode: any) {
      const { offsetAtomic, sortedIndices, sortedPositions } = this.nodes;
      const limit = this.agentCount;
      return Fn(() => {
          const i = instanceIndex;
          If(i.lessThan(limit), () => {
              const pos = positionsNode.element(i);
              const normX = pos.x.add(25.0);
              const normY = pos.y.add(25.0);
              const col = uint(clamp(floor(normX.div(0.5)), 0, 99));
              const row = uint(clamp(floor(normY.div(0.5)), 0, 99));
              const gridIndex = row.mul(uint(100)).add(col);
              
              const slot = atomicAdd(offsetAtomic.element(gridIndex), uint(1));
              sortedIndices.element(slot).assign(i);
              sortedPositions.element(slot).assign(pos);
          });
      })().compute(limit);
  }
}
