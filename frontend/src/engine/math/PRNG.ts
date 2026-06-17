import { Fn, uint, float } from 'three/tsl';

export const prngHash = Fn(([seed]) => {
    let x = uint(seed);
    x = x.shiftLeft(uint(13)).bitXor(x);
    x = x.mul(x.mul(x).mul(uint(15731)).add(uint(789221))).add(uint(1376312589));
    return float(x.bitAnd(uint(0x7fffffff))).div(float(0x7fffffff));
});
