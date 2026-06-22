import { Fn, uint, Loop, min } from 'three/tsl';
import { NodeShaderStage } from 'three/src/nodes/core/constants.js';
import { WebGPUNodeBuilder } from 'three/src/renderers/webgpu/nodes/WebGPUNodeBuilder.js';

const testNode = Fn(([count]) => {
    const cappedCount = min(count, uint(256));
    Loop(cappedCount, ({ i }) => {
        // do nothing
    });
});
console.log('Test successful');
