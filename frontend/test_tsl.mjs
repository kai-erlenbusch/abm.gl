import * as tsl from 'three/tsl';
console.log("Exports containing 'workgroup':");
for (const key of Object.keys(tsl)) {
    if (key.toLowerCase().includes('workgroup') || key.toLowerCase().includes('barrier') || key.toLowerCase().includes('shared') || key.toLowerCase().includes('scan')) {
        console.log(key);
    }
}
