// @ts-nocheck
import { StorageBufferAttribute } from 'three/webgpu';
import { storage } from 'three/tsl';
import { spatialResetNode, spatialCountNode, spatialScatterNode } from './parallel/SpatialHash';
import { spatialPrefixSum_ChunkNode, spatialPrefixSum_BlockNode, spatialPrefixSum_ScatterNode } from './parallel/PrefixSum';

export class SpatialGrid {
  cellCount: number;
  agentCount: number;
  gridSizeX: number;
  gridSizeY: number;
  cellSize: number;
  
  attributes: Record<string, StorageBufferAttribute>;
  nodes: Record<string, any>;

  constructor(gridSizeX: number, gridSizeY: number, agentCount: number, cellSize: number = 0.5) {
    this.gridSizeX = gridSizeX;
    this.gridSizeY = gridSizeY;
    this.cellSize = cellSize;
    
    // Directive B: Prefix Sum 256-Padding Math
    const rawCellCount = gridSizeX * gridSizeY;
    this.cellCount = Math.ceil(rawCellCount / 256) * 256;
    
    this.agentCount = agentCount;
    this.attributes = {};
    this.nodes = {};

    this.initBuffers();
  }

  initBuffers() {
    const cellCountArray = new Uint32Array(this.cellCount);
    const cellOffsetArray = new Uint32Array(this.cellCount);
    const cellOffsetAtomicArray = new Uint32Array(this.cellCount);
    const chunkCount = Math.ceil(this.cellCount / 256);
    const chunkSumsArray = new Uint32Array(chunkCount);
    
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
    this.nodes.sortedVelocities = storage(this.attributes.sortedVelocities, 'vec2', this.agentCount);
  }

  dispose() {
    for (const key in this.attributes) {
      this.attributes[key].dispose();
    }
  }

  // --- Wrapper passes ---
  getResetNode() {
      const { countAtomic, offsetAtomic } = this.nodes;
      const pass = spatialResetNode(countAtomic, offsetAtomic, this.cellCount).compute(this.cellCount);
      return pass;
  }

  getCountNode(positionsNode: any, worldOffsetNode: any, gridDimXNode: any, gridDimYNode: any, cellSizeNode: any) {
    const { countAtomic } = this.nodes;
    const pass = spatialCountNode(positionsNode, countAtomic, this.agentCount, worldOffsetNode, cellSizeNode, gridDimXNode, gridDimYNode).compute(this.agentCount);
    pass.workgroupSize = [256, 1, 1];
    return pass;
  }

  getPrefixSumChunkNode() {
    const { chunkSums, count, offset } = this.nodes;
    const pass = spatialPrefixSum_ChunkNode(count, offset, chunkSums, this.cellCount).compute(this.cellCount);
    pass.workgroupSize = [256, 1, 1];
    return pass;
  }

  getPrefixSumBlockNode() {
      const { chunkSums } = this.nodes;
      const chunkLim = Math.ceil(this.cellCount / 256);
      return spatialPrefixSum_BlockNode(chunkSums, chunkLim).compute(1);
  }

  getPrefixSumScatterNode() {
      const { offset, offsetAtomic, chunkSums } = this.nodes;
      const pass = spatialPrefixSum_ScatterNode(offset, offsetAtomic, chunkSums, this.cellCount).compute(this.cellCount);
      pass.workgroupSize = [256, 1, 1];
      return pass;
  }

  getAgentScatterNode(positionsNode: any, velocitiesNode: any, worldOffsetNode: any, gridDimXNode: any, gridDimYNode: any, cellSizeNode: any) {
      const { offsetAtomic, sortedIndices, sortedPositions, sortedVelocities } = this.nodes;
      const pass = spatialScatterNode(positionsNode, velocitiesNode, offsetAtomic, sortedIndices, sortedPositions, sortedVelocities, this.agentCount, worldOffsetNode, cellSizeNode, gridDimXNode, gridDimYNode).compute(this.agentCount);
      pass.workgroupSize = [256, 1, 1];
      return pass;
  }
}
