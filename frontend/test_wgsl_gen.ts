import { wgslFn, storage, uint, instanceIndex, invocationLocalIndex, Fn, StorageBufferAttribute } from 'three/tsl';
import WebGPURenderer from 'three/src/renderers/webgpu/WebGPURenderer.js';

const posBuf = storage(new StorageBufferAttribute(new Float32Array(800), 2), 'vec2', 400);
const velBuf = storage(new StorageBufferAttribute(new Float32Array(800), 2), 'vec2', 400);
const aggBuf = storage(new StorageBufferAttribute(new Uint32Array(400), 1), 'uint', 400).toAtomic();
const infBuf = storage(new StorageBufferAttribute(new Uint32Array(400), 1), 'uint', 400);

const telemetryAggregateWGSL = wgslFn(`
    var<workgroup> sharedData: array<atomic<u32>, 400>;

    fn telemetryAggregate(
        positions: ptr<storage, array<vec2<f32>>, read>,
        velocities: ptr<storage, array<vec2<f32>>, read>,
        aggregateBuffer: ptr<storage, array<atomic<u32>>, read_write>,
        infectionBuffer: ptr<storage, array<u32>, read>,
        agentCountLimit: u32,
        globalId: u32,
        localId: u32
    ) {
        for (var i: u32 = localId; i < 400u; i += 256u) {
            atomicStore(&sharedData[i], 0u);
        }
        workgroupBarrier();
        if (globalId < agentCountLimit) {
            atomicAdd(&sharedData[0], 1u);
        }
        workgroupBarrier();
        for (var j: u32 = localId; j < 400u; j += 256u) {
            let val = atomicLoad(&sharedData[j]);
            if (val > 0u) {
                atomicAdd(&aggregateBuffer[j], val);
            }
        }
    }
`);

const telemetryAggregateNode = Fn(() => {
    telemetryAggregateWGSL({
        positions: posBuf,
        velocities: velBuf,
        aggregateBuffer: aggBuf,
        infectionBuffer: infBuf,
        agentCountLimit: uint(400),
        globalId: instanceIndex,
        localId: invocationLocalIndex
    });
});

async function test() {
    const renderer = new WebGPURenderer();
    await renderer.init();
    const computeNode = telemetryAggregateNode().compute(400);
    // Force compilation by asking renderer to compile it
    const pipeline = await renderer.computePipelines.get(computeNode);
    // Wait, getting pipeline requires internal API. Let's just mock a minimal pass.
}
