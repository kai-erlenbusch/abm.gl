import * as tsl from 'three/tsl';
console.log("Exports containing 'workgroup':");
for (const key of Object.keys(tsl)) {
    if (key.toLowerCase().includes('local') || key.toLowerCase().includes('invocation') || key.toLowerCase().includes('thread')) {
        console.log(key);
    }
}
