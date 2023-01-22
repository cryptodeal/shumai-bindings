const { _add, _asContiguousTensor, _tensorFromFloat32Array, _tensorFromBoolInt8Array, _tensorFromInt16Array, _tensorFromInt32Array, _tensorFromInt64Array, _tensorFromUint8Array, _tensorFromUint16Array, _tensorFromUint32Array, _tensorFromUint64Array, _tensorFromFloat64Array, _init, _bytesUsed, _toFloat32Scalar, _toFloat64Scalar, _toBoolInt8Scalar, _toInt16Scalar, _toInt32Scalar, _toInt64Scalar, _toUint8Scalar, _toUint16Scalar, _toUint32Scalar, _toUint64Scalar, _toFloat32Array, _toFloat64Array, _toBoolInt8Array, _toInt16Array, _toInt32Array, _toInt64Array, _toUint8Array, _toUint16Array, _toUint32Array, _toUint64Array } = require('../../build/Release/shumai_bindings.node');

_init();

class Tensor {
  #native_self
  constructor(t) {
    if (t instanceof Tensor) {
      this.#native_self = t.#native_self;
      return;
    } else if (t instanceof Float32Array || t.constructor === Float32Array) {
      this.#native_self = _tensorFromFloat32Array(t);
      return;
    } else if (t instanceof Float64Array || t.constructor === Float64Array) {
      this.#native_self = _tensorFromFloat64Array(t);
      return;
    } else if (t instanceof Int8Array || t.constructor === Int8Array) {
      this.#native_self = _tensorFromBoolInt8Array(t);
      return;
    } else if (t instanceof Int16Array || t.constructor === Int16Array) {
      this.#native_self = _tensorFromInt16Array(t);
      return;
    } else if (t instanceof Int32Array || t.constructor === Int32Array) {
      this.#native_self = _tensorFromInt32Array(t);
      return;
    } else if (t instanceof BigInt64Array || t.constructor === BigInt64Array) {
      this.#native_self = _tensorFromInt64Array(t);
      return;
    } else if (t instanceof Uint8Array || t.constructor === Uint8Array) {
      this.#native_self = _tensorFromUint8Array(t);
      return;
    } else if (t instanceof Uint16Array || t.constructor === Uint16Array) {
      this.#native_self = _tensorFromUint16Array(t);
      return;
    } else if (t instanceof Uint32Array || t.constructor === Uint32Array) {
      this.#native_self = _tensorFromUint32Array(t);
      return;
    } else if (t instanceof BigUint64Array || t.constructor === BigUint64Array) {
      this.#native_self = _tensorFromUint64Array(t);
      return;
    } else {
      this.#native_self = t
      return;
    }
  }

  asContiguousTensor() {
    return _asContiguousTensor(this.#native_self);
  }

  toFloat32Scalar() {
    return _toFloat32Scalar(this.#native_self)
  }

  toFloat64Scalar() {
    return _toFloat64Scalar(this.#native_self)
  }

  toBoolInt8Scalar() {
    return _toBoolInt8Scalar(this.#native_self)
  }

  toInt16Scalar() {    
    return _toInt16Scalar(this.#native_self)
  }

  toInt32Scalar() {
    return _toInt32Scalar(this.#native_self)
  }

  toBigInt64Scalar() {
    return _toInt64Scalar(this.#native_self)
  }

  toUint8Scalar() {
    return _toUint8Scalar(this.#native_self)
  }

  toUint16Scalar() {
    return _toUint16Scalar(this.#native_self)
  }

  toUint32Scalar() {
    return _toUint32Scalar(this.#native_self)
  }

  toBigUint64Scalar() {
    return _toUint64Scalar(this.#native_self)
  }

  toFloat32Array() {
    return _toFloat32Array(this.#native_self)
  }

  toFloat64Array() {
    return _toFloat64Array(this.#native_self)
  }

  toBoolInt8Array() {
    const contig = this.asContiguousTensor()
    return _toBoolInt8Array(contig.#native_self)
  }

  toInt16Array() {
    const contig = this.asContiguousTensor()
    return _toInt16Array(contig.#native_self)
  }

  toInt32Array() {
    const contig = this.asContiguousTensor()
    return _toInt32Array(contig.#native_self)
  }

  toBigInt64Array() {
    const contig = this.asContiguousTensor()
    return _toInt64Array(contig.#native_self)
  } 

  toUint8Array() { 
    const contig = this.asContiguousTensor()
    return _toUint8Array(contig.#native_self)
  }

  toUint16Array() {
    const contig = this.asContiguousTensor()
    return _toUint16Array(contig.#native_self)
  }

  toUint32Array() {
    const contig = this.asContiguousTensor()
    return _toUint32Array(contig.#native_self)
  }

  toBigUint64Array() {
    const contig = this.asContiguousTensor()
    return _toUint64Array(contig.#native_self)
  }

  add(other) {
    return new Tensor(_add(this.#native_self, other.#native_self))
  }
}

const bytesUsed = () => {
	return _bytesUsed();
}

module.exports = { Tensor, bytesUsed };