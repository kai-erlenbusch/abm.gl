import { wgslFn, storage, uint, instanceIndex, invocationLocalIndex } from 'three/tsl';
import { StorageBufferAttribute } from 'three/webgpu';

const posBuf = storage(new StorageBufferAttribute(new Float32Array(800), 2), 'vec2', 400);
const velBuf = storage(new StorageBufferAttribute(new Float32Array(800), 2), 'vec2', 400);
const aggBuf = storage(new StorageBufferAttribute(new Uint32Array(400), 1), 'uint', 400).toAtomic();
const infBuf = storage(new StorageBufferAttribute(new Uint32Array(400), 1), 'uint', 400);

const fn = wgslFn(`
    var<workgroup> sharedData: array<atomic<u32>, 400>;
    fn telemetryAggregate(
        positions: ptr<storage, array<vec2<f32>>, read>,
        velocities: ptr<storage, array<vec2<f32>>, read>,
        aggregateBuffer: ptr<storage, array<atomic<u32>>, read_write>,
        infectionBuffer: ptr<storage, array<u32>, read>,
        agentCountLimit: u32,
        globalId: u32,
        localId: u32
    ) {}
`);

try {
  fn({
        positions: posBuf,
        velocities: velBuf,
        aggregateBuffer: aggBuf,
        infectionBuffer: infBuf,
        agentCountLimit: uint(400),
        globalId: instanceIndex,
        localId: invocationLocalIndex
  });
  console.log('Passed wgslFn successfully');
} catch (e) {
  console.error(e);
}
