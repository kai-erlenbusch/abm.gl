import { Fn, uint } from 'three/tsl';
const node = Fn(() => {})();
const comp = node.compute(10240);
if (typeof comp.workgroupSize === 'function') {
  comp.workgroupSize(256);
  console.log('function called');
} else {
  comp.workgroupSize = [256, 1, 1];
  console.log('property set');
}
