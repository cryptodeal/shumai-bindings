const { dlopen, suffix, ptr, FFIType, toArrayBuffer } = require('bun:ffi')
const { cwd } = require('process')

const file = `${cwd()}/libflashlight_binding.${suffix}`

const { symbols: fl } = dlopen(file, {
  init: {},
  bytesUsed: {
    returns: FFIType.u64
  },
  tensorFromFloat32Buffer: {
    args: [FFIType.i64, FFIType.ptr],
    returns: FFIType.ptr
  },
  tensorFromFloat64Buffer: {
    args: [FFIType.i64, FFIType.ptr],
    returns: FFIType.ptr
  },
  tensorFromInt8Buffer: {
    args: [FFIType.i64, FFIType.ptr],
    returns: FFIType.ptr
  },
  tensorFromInt16Buffer: {
    args: [FFIType.i64, FFIType.ptr],
    returns: FFIType.ptr
  },
  tensorFromInt32Buffer: {
    args: [FFIType.i64, FFIType.ptr],
    returns: FFIType.ptr
  },
  tensorFromInt64Buffer: {
    args: [FFIType.i64, FFIType.ptr],
    returns: FFIType.ptr
  },
  tensorFromUint8Buffer: {
    args: [FFIType.i64, FFIType.ptr],
    returns: FFIType.ptr
  },
  tensorFromUint16Buffer: {
    args: [FFIType.i64, FFIType.ptr],
    returns: FFIType.ptr
  },
  tensorFromUint32Buffer: {
    args: [FFIType.i64, FFIType.ptr],
    returns: FFIType.ptr
  },
  tensorFromUint64Buffer: {
    args: [FFIType.i64, FFIType.ptr],
    returns: FFIType.ptr
  },
  _bytes: {
    args: [FFIType.ptr],
    returns: FFIType.u64
  },
  destroyTensor: {
    args: [FFIType.ptr, FFIType.ptr]
  },
  genTensorDestroyer: {
    returns: FFIType.ptr
  },
  _float32Buffer: {
    args: [FFIType.ptr],
    returns: FFIType.ptr
  },
  _float64Buffer: {
    args: [FFIType.ptr],
    returns: FFIType.ptr
  },
  _int16Buffer: {
    args: [FFIType.ptr],
    returns: FFIType.ptr
  },
  _int32Buffer: {
    args: [FFIType.ptr],
    returns: FFIType.ptr
  },
  _int64Buffer: {
    args: [FFIType.ptr],
    returns: FFIType.ptr
  },
  _uint8Buffer: {
    args: [FFIType.ptr],
    returns: FFIType.ptr
  },
  _uint16Buffer: {
    args: [FFIType.ptr],
    returns: FFIType.ptr
  },
  _uint32Buffer: {
    args: [FFIType.ptr],
    returns: FFIType.ptr
  },
  _uint64Buffer: {
    args: [FFIType.ptr],
    returns: FFIType.ptr
  },
  _float32Scalar: {
    args: [FFIType.ptr],
    returns: FFIType.f32
  },
  _float64Scalar: {
    args: [FFIType.ptr],
    returns: FFIType.f64
  },
  _boolInt8Scalar: {
    args: [FFIType.ptr],
    returns: FFIType.i8
  },
  _int16Scalar: {
    args: [FFIType.ptr],
    returns: FFIType.i16
  },
  _int32Scalar: {
    args: [FFIType.ptr],
    returns: FFIType.i32
  },
  _int64Scalar: {
    args: [FFIType.ptr],
    returns: FFIType.i64
  },
  _uint8Scalar: {
    args: [FFIType.ptr],
    returns: FFIType.u8
  },
  _uint16Scalar: {
    args: [FFIType.ptr],
    returns: FFIType.u16
  },
  _uint32Scalar: {
    args: [FFIType.ptr],
    returns: FFIType.u32
  },
  _uint64Scalar: {
    args: [FFIType.ptr],
    returns: FFIType.u64
  },
  _asContiguousTensor: {
    args: [FFIType.ptr],
    returns: FFIType.ptr
  },
  _elements: {
    args: [FFIType.ptr],
    returns: FFIType.u64
  }
})

fl.init();

function wrapFLTensor(closure, ptr) {
  const _ptr = closure(ptr);
  const t = new Tensor({ _ptr })
  return t;
}

class Tensor {

  get elements() {
    return Number(fl._elements.native(this._ptr))
  }

  _injest_ptr(_ptr) {
    this._ptr = _ptr
    const byteLength = Number(fl._bytes.native(_ptr))
    this._underlying = toArrayBuffer(
      _ptr,
      0,
      byteLength,
      fl.genTensorDestroyer.native()
    )
  }

  asContiguousTensor() {
    return wrapFLTensor(fl._asContiguousTensor.native, this._ptr)
  }
  
  constructor(t) {
    if (t.hasOwnProperty('_ptr')) {
      this._injest_ptr(t._ptr)
      return;
    }

    if (t instanceof Float32Array || t.constructor === Float32Array) {
      const len_ = t.length
      const len = len_.constructor === BigInt ? len_ : BigInt(len_ || 0)
      this._injest_ptr(fl.tensorFromFloat32Buffer.native(len, ptr(t)))
      return
    } else if (t instanceof Float64Array || t.constructor === Float64Array) {
      const len_ = t.length
      const len = len_.constructor === BigInt ? len_ : BigInt(len_ || 0)
      this._injest_ptr(fl.tensorFromFloat64Buffer.native(len, ptr(t)))
      return
    } else if (t instanceof Int8Array || t.constructor === Int8Array) {
      const len_ = t.length
      const len = len_.constructor === BigInt ? len_ : BigInt(len_ || 0)
      this._injest_ptr(fl.tensorFromInt8Buffer.native(len, ptr(t)))
      return
    } else if (t instanceof Int16Array || t.constructor === Int16Array) {
      const len_ = t.length
      const len = len_.constructor === BigInt ? len_ : BigInt(len_ || 0)
      this._injest_ptr(fl.tensorFromInt16Buffer.native(len, ptr(t)))
      return
    } else if (t instanceof Int32Array || t.constructor === Int32Array) {
       const len_ = t.length
      const len = len_.constructor === BigInt ? len_ : BigInt(len_ || 0)
      this._injest_ptr(fl.tensorFromInt32Buffer.native(len, ptr(t)))
      return
    } else if (t instanceof BigInt64Array || t.constructor === BigInt64Array) {
      const len_ = t.length
      const len = len_.constructor === BigInt ? len_ : BigInt(len_ || 0)
      this._injest_ptr(fl.tensorFromInt64Buffer.native(len, ptr(t)))
      return
    } else if (t instanceof Uint8Array || t.constructor === Uint8Array) {
      const len_ = t.length
      const len = len_.constructor === BigInt ? len_ : BigInt(len_ || 0)
      this._injest_ptr(fl.tensorFromUint8Buffer.native(len, ptr(t)))
      return
    } else if (t instanceof Uint16Array || t.constructor === Uint16Array) {
      const len_ = t.length
      const len = len_.constructor === BigInt ? len_ : BigInt(len_ || 0)
      this._injest_ptr(fl.tensorFromUint16Buffer.native(len, ptr(t)))
      return
    } else if (t instanceof Uint32Array || t.constructor === Uint32Array) {
      const len_ = t.length
      const len = len_.constructor === BigInt ? len_ : BigInt(len_ || 0)
      this._injest_ptr(fl.tensorFromUint32Buffer.native(len, ptr(t)))
      return
    } else if (t instanceof BigUint64Array || t.constructor === BigUint64Array) {
      const len_ = t.length
      const len = len_.constructor === BigInt ? len_ : BigInt(len_ || 0)
      this._injest_ptr(fl.tensorFromUint64Buffer.native(len, ptr(obj)))
      return
    } else {
      this._ptr = t._ptr
    }
  }

  get ptr() {
    return this._ptr
  }

  toFloat32Array() {
    const contig = this.asContiguousTensor()
    const elems = contig.elements
    return new Float32Array(toArrayBuffer(fl._float32Buffer.native(contig.ptr), 0, elems * 4))
  }

  toFloat64Array() {
    const contig = this.asContiguousTensor()
    const elems = contig.elements
    return new Float64Array(toArrayBuffer(fl._float64Buffer.native(contig.ptr), 0, elems * 8))
  }

  toBoolInt8Array() {
    const contig = this.asContiguousTensor()
    const elems = contig.elements
    return new Int8Array(toArrayBuffer(fl._boolInt8Buffer.native(contig.ptr), 0, elems))
  }

  toInt16Array() {
    const contig = this.asContiguousTensor()
    const elems = contig.elements
    return new Int16Array(toArrayBuffer(fl._int16Buffer.native(contig.ptr), 0, elems * 2))
  }

  toInt32Array() {
    const contig = this.asContiguousTensor()
    const elems = contig.elements
    return new Int32Array(toArrayBuffer(fl._int32Buffer.native(contig.ptr), 0, elems * 4))
  }

  toBigInt64Array() {
    const contig = this.asContiguousTensor()
    const elems = contig.elements
    return new BigInt64Array(toArrayBuffer(fl._int64Buffer.native(contig.ptr), 0, elems * 8))
  }

  toUint8Array() {
    const contig = this.asContiguousTensor()
    const elems = contig.elements
    return new Uint8Array(toArrayBuffer(fl._uint8Buffer.native(contig.ptr), 0, elems))
  }

  toUint16Array() {
    const contig = this.asContiguousTensor()
    const elems = contig.elements
    return new Uint16Array(toArrayBuffer(fl._uint16Buffer.native(contig.ptr), 0, elems * 2))
  }

  toUint32Array() {
    const contig = this.asContiguousTensor()
    const elems = contig.elements
    return new Uint32Array(toArrayBuffer(fl._uint32Buffer.native(contig.ptr), 0, elems * 4))
  }

  toBigUint64Array() {
    const contig = this.asContiguousTensor()
    const elems = contig.elements
    return new BigUint64Array(toArrayBuffer(fl._uint64Buffer.native(contig.ptr), 0, elems * 8))
  }

  toFloat16() {
    return fl._float16Scalar.native(this.ptr)
  }

  toFloat32() {
    return fl._float32Scalar.native(this.ptr)
  }

  toFloat64() {
    return fl._float64Scalar.native(this.ptr)
  }

  toBoolInt8() {
    return fl._boolInt8Scalar.native(this.ptr)
  }

  toInt16() {
    return fl._int16Scalar.native(this.ptr)
  }

  toInt32() {
    return fl._int32Scalar.native(this.ptr)
  }

  toBigInt64() {
    return fl._int64Scalar.native(this.ptr)
  }

  toUint8() {
    return fl._uint8Scalar.native(this.ptr)
  }

  toUint16() {
    return fl._uint16Scalar.native(this.ptr)
  }

  toUint32() {
    return fl._uint32Scalar.native(this.ptr)
  }

  toBigUint64() {
    return fl._uint64Scalar.native(this.ptr)
  }
}

const bytesUsed = () => {
	return fl.bytesUsed.native();
}

module.exports = { bytesUsed, Tensor };