const { Tensor, bytesUsed } = require('./napi_impl/js/index.cjs');

const fillArray = (arr) => {
  const len = arr.length;
  for (let i = 0; i < len; i++) {
    arr[i] = Math.random();
  }
  return arr;
}

const test = () => {
  const t0 = performance.now() / 1e3;
  let a
  for (let i = 0; i < 10000; i++) {
    const backingArray = fillArray(new Float64Array(1000));
    a = new Tensor(backingArray).toFloat64Array();
  }
  const t1 = performance.now() / 1e3;
  const time = t1 - t0;
  const bytes = bytesUsed()
  return { time, bytes };
}

const runTest = (runs) => {
  const times = []
  const endBytes = []
  for (let i = 0; i < runs; ++i) {
    const { time, bytes } = test()
    times[i] = time;
    endBytes[i] = bytes;
  }
  console.log(`avg time (${runs} runs) to init 10k Tensors of 1k elements and return as TypedArray (dtype = Float64): ${times.reduce((a, b) => a + b, 0) / runs} seconds`)
  console.log(`avg bytesUsed (${runs} runs) to init 10k Tensors of 1k elements and return as TypedArray (dtype = Float64): ${Number(endBytes.reduce((a, b) => a + b, BigInt(0))) / runs} bytes`)
}

runTest(1)