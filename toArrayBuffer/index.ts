import * as sm from '@shumai/shumai'
import { startSamplingProfiler } from 'bun:jsc';

startSamplingProfiler('toArrayBuffer/profiles')
const array1 = new Float32Array(128)
const array2 = new Float32Array(128)
for (let i = 0; i < 128; ++i) {
  array1[i] = Math.random()
  array2[i] = Math.random()
}
const a = new sm.Tensor(array1)
const b = new sm.Tensor(array2)

let res: Float32Array
for (let i = 0; i < 100000; ++i) {
  const c = a.add(b)
  res = c.toFloat32Array()
}
