// @ts-nocheck
import { StorageBufferAttribute } from 'three/webgpu';
import { storage, Fn, instanceIndex, If } from 'three/tsl';

/**
 * AgentDataStore manages the Structure of Arrays (SoA) for our agents.
 * It implements Ping-Pong buffering (Read from A, Write to B) to prevent
 * race conditions during massively parallel GPU execution.
 */
export class AgentDataStore {
  agentCount: number;
  attributes: Record<string, StorageBufferAttribute>;
  nodes: Record<string, any>;

  constructor(agentCount: number) {
    this.agentCount = agentCount;
    this.attributes = {};
    this.nodes = {};
  }

  /**
   * Registers a new agent property (e.g. position, velocity, state).
   */
  addProperty(name: string, itemSize: number, dataType: 'vec2' | 'vec3' | 'float' | 'uint' | 'int') {
    const arrayType = dataType === 'uint' ? Uint32Array : (dataType === 'int' ? Int32Array : Float32Array);
    
    // State A (Read/Write)
    const arrA = new arrayType(this.agentCount * itemSize);
    const attrA = new StorageBufferAttribute(arrA as any, itemSize);
    this.attributes[name] = attrA;
    this.nodes[name] = storage(attrA, dataType, this.agentCount);
    
    // State B (Ping-Pong / Next State)
    const arrB = new arrayType(this.agentCount * itemSize);
    const attrB = new StorageBufferAttribute(arrB as any, itemSize);
    this.attributes[`${name}_next`] = attrB;
    this.nodes[`${name}_next`] = storage(attrB, dataType, this.agentCount);
  }

  getArray(name: string) {
    return this.attributes[name].array;
  }

  getNode(name: string) {
    return this.nodes[name];
  }

  getNextNode(name: string) {
    return this.nodes[`${name}_next`];
  }

  // Swaps State A and State B (conceptually, by dispatching a copy or flipping pointers in a real engine)
  // In WebGPU TSL, we can just write a compute pass to copy next -> current.
  getCopyPassNode(agentCountLimit: any) {
     const p_pos = this.nodes['position'];
     const n_pos = this.nodes['position_next'];
     const p_vel = this.nodes['velocity'];
     const n_vel = this.nodes['velocity_next'];
     const p_inf = this.nodes['infection'];
     const n_inf = this.nodes['infection_next'];
     const p_tim = this.nodes['timer'];
     const n_tim = this.nodes['timer_next'];
     
     // @ts-ignore
     const copyFn = Fn(() => {
         // @ts-ignore
         const i = instanceIndex;
         If(i.lessThan(agentCountLimit), () => {
             p_pos.element(i).assign(n_pos.element(i));
             p_vel.element(i).assign(n_vel.element(i));
             p_inf.element(i).assign(n_inf.element(i));
             p_tim.element(i).assign(n_tim.element(i));
         });
     });
     
     return copyFn();
  }

  dispose() {
    for (const key in this.attributes) {
      this.attributes[key].dispose();
    }
  }
}
