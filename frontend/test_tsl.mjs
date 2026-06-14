import * as tsl from 'three/tsl';
console.log("Exports containing 'workgroup':");
for (const key of Object.keys(tsl)) {
    if (key.toLowerCase() === 'break') {
        console.log(key);
    }
}
