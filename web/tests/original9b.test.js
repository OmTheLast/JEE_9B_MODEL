import test from 'node:test';
import assert from 'node:assert/strict';
import { unpackEmbedding, makePositions, repeatingTail } from '../original9b-math.js';
test('affine embedding preserves unsigned nibbles and noninteger offsets', () => {
 const w = new Uint32Array(512).fill(0xfedcba98), s = new Float32Array(64).fill(.25), b = new Float32Array(64).fill(-1.5);
 assert.deepEqual([...unpackEmbedding(w,s,b).slice(0,8)], [.5,.75,1,1.25,1.5,1.75,2,2.25]);
});
test('image positions use a 12 by 12 grid and resume after its extent',()=>{
 const p=makePositions([10,11,...Array(144).fill(99),12],99);
 assert.deepEqual(p.positions.map(x=>x.at(-1)),[14,14,14]);
 assert.deepEqual(p.positions.map(x=>x[145]),[2,13,13]);
 assert.equal(p.delta,-132); assert.throws(()=>makePositions([10,99,11],99));
});
test('loop guard detects three repeated spans but permits two and distinct working',()=>{
 const a=[1,2,3,4,5,6,7,8]; assert.equal(repeatingTail([...a,...a]),false); assert.equal(repeatingTail([...a,...a,...a]),true);
 assert.equal(repeatingTail([...a,...a,...a,9]),false);
});
