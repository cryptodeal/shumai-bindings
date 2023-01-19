#include <napi.h>
#include <atomic>
#include <cstdio>
#include <iostream>
#include <string>
#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/tensor/AutogradExtension.h"
#include "flashlight/fl/autograd/tensor/AutogradOps.h"
#include "flashlight/fl/common/DynamicBenchmark.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/runtime/Device.h"
#include "flashlight/fl/runtime/Stream.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/TensorAdapter.h"

// globally scoped variables
static std::atomic<size_t> g_bytes_used = 0;

// helpers
static inline void DeleteTensor(Napi::Env env, void* ptr) {
  auto* val = static_cast<fl::Tensor*>(ptr);
  if (val->hasAdapter()) {
    auto byte_count = static_cast<int64_t>(val->bytes());
    g_bytes_used -= byte_count;
    Napi::MemoryManagement::AdjustExternalMemory(env, -byte_count);
  }
  delete val;
}

template <typename T>
static inline void DeleteArrayBuffer(Napi::Env env,
                                     void* /*data*/,
                                     std::vector<T>* hint) {
  size_t bytes = hint->size() * sizeof(T);
  std::unique_ptr<std::vector<T>> vectorPtrToDelete(hint);
  Napi::MemoryManagement::AdjustExternalMemory(env, -bytes);
}

static inline Napi::External<fl::Tensor> ExternalizeTensor(Napi::Env env,
                                                           fl::Tensor* ptr) {
  return Napi::External<fl::Tensor>::New(env, ptr, DeleteTensor);
}

template <typename T>
static inline T* UnExternalize(Napi::Value val) {
  return val.As<Napi::External<T>>().Data();
}

// methods to init tensor from underlying buffer via JS TypedArray
static Napi::Value _tensorFromFloat32Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsTypedArray()) {
    Napi::Error::New(env,
                     "`tensorFromFloat32Array` epects args[0] to be "
                     "instanceof `Float32Array`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::TypedArray _tmp_typed_array = info[0].As<Napi::TypedArray>();
  int64_t length = static_cast<int64_t>(_tmp_typed_array.ElementLength());
  if (_tmp_typed_array.TypedArrayType() != napi_float32_array) {
    Napi::Error::New(env,
                     "`tensorFromFloat32Array` epects args[0] to be "
                     "instanceof `Float32Array`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  float* ptr = _tmp_typed_array.As<Napi::TypedArrayOf<float>>().Data();
  auto* t = new fl::Tensor(
      fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
  auto _out_bytes_used = static_cast<int64_t>(t->bytes());
  g_bytes_used += _out_bytes_used;
  Napi::MemoryManagement::AdjustExternalMemory(env, _out_bytes_used);
  Napi::External<fl::Tensor> wrapped = ExternalizeTensor(env, t);
  return wrapped;
}

static Napi::Value _tensorFromFloat64Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::TypedArray _tmp_typed_array = info[0].As<Napi::TypedArray>();
  int64_t length = static_cast<int64_t>(_tmp_typed_array.ElementLength());
  double* ptr = _tmp_typed_array.As<Napi::TypedArrayOf<double>>().Data();
  auto* t = new fl::Tensor(
      fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
  auto _out_bytes_used = static_cast<int64_t>(t->bytes());
  g_bytes_used += _out_bytes_used;
  Napi::MemoryManagement::AdjustExternalMemory(env, _out_bytes_used);
  Napi::External<fl::Tensor> wrapped = ExternalizeTensor(env, t);
  return wrapped;
}

static Napi::Value _tensorFromBoolInt8Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsTypedArray()) {
    Napi::Error::New(env,
                     "`tensorFromBoolInt8Array` epects args[0] to be "
                     "instanceof `Int8Array`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::TypedArray _tmp_typed_array = info[0].As<Napi::TypedArray>();
  int64_t length = static_cast<int64_t>(_tmp_typed_array.ElementLength());
  if (_tmp_typed_array.TypedArrayType() != napi_int8_array) {
    Napi::Error::New(env,
                     "`tensorFromBoolInt8Array` epects args[0] to be "
                     "instanceof `Int8Array`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  char* ptr = reinterpret_cast<char*>(
      _tmp_typed_array.As<Napi::TypedArrayOf<int8_t>>().Data());
  auto* t = new fl::Tensor(
      fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
  auto _out_bytes_used = static_cast<int64_t>(t->bytes());
  g_bytes_used += _out_bytes_used;
  Napi::MemoryManagement::AdjustExternalMemory(env, _out_bytes_used);
  Napi::External<fl::Tensor> wrapped = ExternalizeTensor(env, t);
  return wrapped;
}

static Napi::Value _tensorFromInt16Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsTypedArray()) {
    Napi::Error::New(env,
                     "`tensorFromInt16Array` epects args[0] to be "
                     "instanceof `Int16Array`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::TypedArray _tmp_typed_array = info[0].As<Napi::TypedArray>();
  int64_t length = static_cast<int64_t>(_tmp_typed_array.ElementLength());
  if (_tmp_typed_array.TypedArrayType() != napi_int16_array) {
    Napi::Error::New(env,
                     "`tensorFromInt16Array` epects args[0] to be "
                     "instanceof `Int16Array`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  int16_t* ptr = _tmp_typed_array.As<Napi::TypedArrayOf<int16_t>>().Data();
  auto* t = new fl::Tensor(
      fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
  auto _out_bytes_used = static_cast<int64_t>(t->bytes());
  g_bytes_used += _out_bytes_used;
  Napi::MemoryManagement::AdjustExternalMemory(env, _out_bytes_used);
  Napi::External<fl::Tensor> wrapped = ExternalizeTensor(env, t);
  return wrapped;
}

static Napi::Value _tensorFromInt32Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsTypedArray()) {
    Napi::Error::New(env,
                     "`tensorFromInt32Array` epects args[0] to be "
                     "instanceof `Int32Array`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::TypedArray _tmp_typed_array = info[0].As<Napi::TypedArray>();
  int64_t length = static_cast<int64_t>(_tmp_typed_array.ElementLength());
  if (_tmp_typed_array.TypedArrayType() != napi_int32_array) {
    Napi::Error::New(env,
                     "`tensorFromInt32Array` epects args[0] to be "
                     "instanceof `Int32Array`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  int32_t* ptr = _tmp_typed_array.As<Napi::TypedArrayOf<int32_t>>().Data();
  auto* t = new fl::Tensor(
      fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
  auto _out_bytes_used = static_cast<int64_t>(t->bytes());
  g_bytes_used += _out_bytes_used;
  Napi::MemoryManagement::AdjustExternalMemory(env, _out_bytes_used);
  Napi::External<fl::Tensor> wrapped = ExternalizeTensor(env, t);
  return wrapped;
}

static Napi::Value _tensorFromInt64Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsTypedArray()) {
    Napi::Error::New(env,
                     "`tensorFromInt64Array` epects args[0] to be "
                     "instanceof `BigInt64Array`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::TypedArray _tmp_typed_array = info[0].As<Napi::TypedArray>();
  int64_t length = static_cast<int64_t>(_tmp_typed_array.ElementLength());
  if (_tmp_typed_array.TypedArrayType() != napi_bigint64_array) {
    Napi::Error::New(env,
                     "`tensorFromInt64Array` epects args[0] to be "
                     "instanceof `BigInt64Array`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  int64_t* ptr = _tmp_typed_array.As<Napi::TypedArrayOf<int64_t>>().Data();
  auto* t = new fl::Tensor(
      fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
  auto _out_bytes_used = static_cast<int64_t>(t->bytes());
  g_bytes_used += _out_bytes_used;
  Napi::MemoryManagement::AdjustExternalMemory(env, _out_bytes_used);
  Napi::External<fl::Tensor> wrapped = ExternalizeTensor(env, t);
  return wrapped;
}

static Napi::Value _tensorFromUint8Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsTypedArray()) {
    Napi::Error::New(env,
                     "`tensorFromUint8Array` epects args[0] to be "
                     "instanceof `Uint8Array`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::TypedArray _tmp_typed_array = info[0].As<Napi::TypedArray>();
  int64_t length = static_cast<int64_t>(_tmp_typed_array.ElementLength());
  if (_tmp_typed_array.TypedArrayType() != napi_uint8_array) {
    Napi::Error::New(env,
                     "`tensorFromUint8Array` epects args[0] to be "
                     "instanceof `Uint8Array`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  uint8_t* ptr = _tmp_typed_array.As<Napi::TypedArrayOf<uint8_t>>().Data();
  auto* t = new fl::Tensor(
      fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
  auto _out_bytes_used = static_cast<int64_t>(t->bytes());
  g_bytes_used += _out_bytes_used;
  Napi::MemoryManagement::AdjustExternalMemory(env, _out_bytes_used);
  Napi::External<fl::Tensor> wrapped = ExternalizeTensor(env, t);
  return wrapped;
}

static Napi::Value _tensorFromUint16Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsTypedArray()) {
    Napi::Error::New(env,
                     "`tensorFromUint16Array` epects args[0] to be "
                     "instanceof `Uint16Array`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::TypedArray _tmp_typed_array = info[0].As<Napi::TypedArray>();
  int64_t length = static_cast<int64_t>(_tmp_typed_array.ElementLength());
  if (_tmp_typed_array.TypedArrayType() != napi_uint16_array) {
    Napi::Error::New(env,
                     "`tensorFromUint16Array` epects args[0] to be "
                     "instanceof `Uint16Array`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  uint16_t* ptr = _tmp_typed_array.As<Napi::TypedArrayOf<uint16_t>>().Data();
  auto* t = new fl::Tensor(
      fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
  auto _out_bytes_used = static_cast<int64_t>(t->bytes());
  g_bytes_used += _out_bytes_used;
  Napi::MemoryManagement::AdjustExternalMemory(env, _out_bytes_used);
  Napi::External<fl::Tensor> wrapped = ExternalizeTensor(env, t);
  return wrapped;
}

static Napi::Value _tensorFromUint32Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsTypedArray()) {
    Napi::Error::New(env,
                     "`tensorFromUint32Array` epects args[0] to be "
                     "instanceof `Uint32Array`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::TypedArray _tmp_typed_array = info[0].As<Napi::TypedArray>();
  int64_t length = static_cast<int64_t>(_tmp_typed_array.ElementLength());
  if (_tmp_typed_array.TypedArrayType() != napi_uint32_array) {
    Napi::Error::New(env,
                     "`tensorFromUint32Array` epects args[0] to be "
                     "instanceof `Uint32Array`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  uint32_t* ptr = _tmp_typed_array.As<Napi::TypedArrayOf<uint32_t>>().Data();
  auto* t = new fl::Tensor(
      fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
  auto _out_bytes_used = static_cast<int64_t>(t->bytes());
  g_bytes_used += _out_bytes_used;
  Napi::MemoryManagement::AdjustExternalMemory(env, _out_bytes_used);
  Napi::External<fl::Tensor> wrapped = ExternalizeTensor(env, t);
  return wrapped;
}

static Napi::Value _tensorFromUint64Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsTypedArray()) {
    Napi::Error::New(env,
                     "`tensorFromUint64Array` epects args[0] to be "
                     "instanceof `BigUint64Array`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::TypedArray _tmp_typed_array = info[0].As<Napi::TypedArray>();
  int64_t length = static_cast<int64_t>(_tmp_typed_array.ElementLength());
  if (_tmp_typed_array.TypedArrayType() != napi_biguint64_array) {
    Napi::Error::New(env,
                     "`tensorFromUint64Array` epects args[0] to be "
                     "instanceof `BigUint64Array`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  uint64_t* ptr = _tmp_typed_array.As<Napi::TypedArrayOf<uint64_t>>().Data();
  auto* t = new fl::Tensor(
      fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
  auto _out_bytes_used = static_cast<int64_t>(t->bytes());
  g_bytes_used += _out_bytes_used;
  Napi::MemoryManagement::AdjustExternalMemory(env, _out_bytes_used);
  Napi::External<fl::Tensor> wrapped = ExternalizeTensor(env, t);
  return wrapped;
}

static Napi::Value _tensorFromFloat32Buffer(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsArrayBuffer()) {
    Napi::Error::New(env,
                     "`tensorFromFloat32Buffer` epects args[0] to be "
                     "instanceof `ArrayBuffer`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::ArrayBuffer buf = info[0].As<Napi::ArrayBuffer>();
  int64_t length = static_cast<int64_t>(buf.ByteLength() / sizeof(float));
  float* ptr = (float*)buf.Data();
  auto* t = new fl::Tensor(
      fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
  auto _out_bytes_used = static_cast<int64_t>(t->bytes());
  g_bytes_used += _out_bytes_used;
  Napi::MemoryManagement::AdjustExternalMemory(env, _out_bytes_used);
  Napi::External<fl::Tensor> wrapped = ExternalizeTensor(env, t);
  return wrapped;
}

static Napi::Value _tensorFromFloat64Buffer(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::ArrayBuffer buf = info[0].As<Napi::ArrayBuffer>();
  int64_t length = static_cast<int64_t>(buf.ByteLength() / sizeof(double));
  double* ptr = (double*)buf.Data();
  auto* t = new fl::Tensor(
      fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
  auto _out_bytes_used = static_cast<int64_t>(t->bytes());
  g_bytes_used += _out_bytes_used;
  Napi::MemoryManagement::AdjustExternalMemory(env, _out_bytes_used);
  Napi::External<fl::Tensor> wrapped = ExternalizeTensor(env, t);
  return wrapped;
}

static Napi::Value _tensorFromBoolInt8Buffer(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsArrayBuffer()) {
    Napi::Error::New(env,
                     "`tensorFromBoolInt8Buffer` epects args[0] to be "
                     "instanceof `ArrayBuffer`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::ArrayBuffer buf = info[0].As<Napi::ArrayBuffer>();
  int64_t length = static_cast<int64_t>(buf.ByteLength() / sizeof(int8_t));
  char* ptr = reinterpret_cast<char*>(buf.Data());
  auto* t = new fl::Tensor(
      fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
  auto _out_bytes_used = static_cast<int64_t>(t->bytes());
  g_bytes_used += _out_bytes_used;
  Napi::MemoryManagement::AdjustExternalMemory(env, _out_bytes_used);
  Napi::External<fl::Tensor> wrapped = ExternalizeTensor(env, t);
  return wrapped;
}

static Napi::Value _tensorFromInt16Buffer(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsArrayBuffer()) {
    Napi::Error::New(
        env,
        "`tensorFromInt16Buffer` epects args[0] to be instanceof `ArrayBuffer`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::ArrayBuffer buf = info[0].As<Napi::ArrayBuffer>();
  int64_t length = static_cast<int64_t>(buf.ByteLength() / sizeof(int16_t));
  int16_t* ptr = (int16_t*)buf.Data();
  auto* t = new fl::Tensor(
      fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
  auto _out_bytes_used = static_cast<int64_t>(t->bytes());
  g_bytes_used += _out_bytes_used;
  Napi::MemoryManagement::AdjustExternalMemory(env, _out_bytes_used);
  Napi::External<fl::Tensor> wrapped = ExternalizeTensor(env, t);
  return wrapped;
}

static Napi::Value _tensorFromInt32Buffer(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsArrayBuffer()) {
    Napi::Error::New(
        env,
        "`tensorFromInt32Buffer` epects args[0] to be instanceof `ArrayBuffer`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::ArrayBuffer buf = info[0].As<Napi::ArrayBuffer>();
  int64_t length = static_cast<int64_t>(buf.ByteLength() / sizeof(int32_t));
  int32_t* ptr = (int32_t*)buf.Data();
  auto* t = new fl::Tensor(
      fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
  auto _out_bytes_used = static_cast<int64_t>(t->bytes());
  g_bytes_used += _out_bytes_used;
  Napi::MemoryManagement::AdjustExternalMemory(env, _out_bytes_used);
  Napi::External<fl::Tensor> wrapped = ExternalizeTensor(env, t);
  return wrapped;
}

static Napi::Value _tensorFromInt64Buffer(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsArrayBuffer()) {
    Napi::Error::New(env,
                     "`tensorFtensorFromInt64BufferromFloat32Buffer` epects "
                     "args[0] to be instanceof `ArrayBuffer`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::ArrayBuffer buf = info[0].As<Napi::ArrayBuffer>();
  int64_t length = static_cast<int64_t>(buf.ByteLength() / sizeof(int64_t));
  int64_t* ptr = (int64_t*)buf.Data();
  auto* t = new fl::Tensor(
      fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
  auto _out_bytes_used = static_cast<int64_t>(t->bytes());
  g_bytes_used += _out_bytes_used;
  Napi::MemoryManagement::AdjustExternalMemory(env, _out_bytes_used);
  Napi::External<fl::Tensor> wrapped = ExternalizeTensor(env, t);
  return wrapped;
}

static Napi::Value _tensorFromUint8Buffer(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsArrayBuffer()) {
    Napi::Error::New(
        env,
        "`tensorFromUint8Buffer` epects args[0] to be instanceof `ArrayBuffer`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::ArrayBuffer buf = info[0].As<Napi::ArrayBuffer>();
  int64_t length = static_cast<int64_t>(buf.ByteLength() / sizeof(uint8_t));
  uint8_t* ptr = (uint8_t*)buf.Data();
  auto* t = new fl::Tensor(
      fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
  auto _out_bytes_used = static_cast<int64_t>(t->bytes());
  g_bytes_used += _out_bytes_used;
  Napi::MemoryManagement::AdjustExternalMemory(env, _out_bytes_used);
  Napi::External<fl::Tensor> wrapped = ExternalizeTensor(env, t);
  return wrapped;
}

static Napi::Value _tensorFromUint16Buffer(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsArrayBuffer()) {
    Napi::Error::New(env,
                     "`tensorFromUint16Buffer` epects args[0] to be instanceof "
                     "`ArrayBuffer`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::ArrayBuffer buf = info[0].As<Napi::ArrayBuffer>();
  int64_t length = static_cast<int64_t>(buf.ByteLength() / sizeof(uint16_t));
  uint16_t* ptr = (uint16_t*)buf.Data();
  auto* t = new fl::Tensor(
      fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
  auto _out_bytes_used = static_cast<int64_t>(t->bytes());
  g_bytes_used += _out_bytes_used;
  Napi::MemoryManagement::AdjustExternalMemory(env, _out_bytes_used);
  Napi::External<fl::Tensor> wrapped = ExternalizeTensor(env, t);
  return wrapped;
}

static Napi::Value _tensorFromUint32Buffer(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsArrayBuffer()) {
    Napi::Error::New(env,
                     "`tensorFromUint32Buffer` epects args[0] to be instanceof "
                     "`ArrayBuffer`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::ArrayBuffer buf = info[0].As<Napi::ArrayBuffer>();
  int64_t length = static_cast<int64_t>(buf.ByteLength() / sizeof(uint32_t));
  uint32_t* ptr = (uint32_t*)buf.Data();
  auto* t = new fl::Tensor(
      fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
  auto _out_bytes_used = static_cast<int64_t>(t->bytes());
  g_bytes_used += _out_bytes_used;
  Napi::MemoryManagement::AdjustExternalMemory(env, _out_bytes_used);
  Napi::External<fl::Tensor> wrapped = ExternalizeTensor(env, t);
  return wrapped;
}

static Napi::Value _tensorFromUint64Buffer(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsArrayBuffer()) {
    Napi::Error::New(env,
                     "`tensorFromUint64Buffer` epects args[0] to be instanceof "
                     "`ArrayBuffer`")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  Napi::ArrayBuffer buf = info[0].As<Napi::ArrayBuffer>();
  int64_t length = static_cast<int64_t>(buf.ByteLength() / sizeof(uint64_t));
  uint64_t* ptr = (uint64_t*)buf.Data();
  auto* t = new fl::Tensor(
      fl::Tensor::fromBuffer({length}, ptr, fl::MemoryLocation::Host));
  auto _out_bytes_used = static_cast<int64_t>(t->bytes());
  g_bytes_used += _out_bytes_used;
  Napi::MemoryManagement::AdjustExternalMemory(env, _out_bytes_used);
  Napi::External<fl::Tensor> wrapped = ExternalizeTensor(env, t);
  return wrapped;
}

static Napi::Value _toFloat32Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "`toFloat32Array` expects args[0] to be native "
                         "`Tensor` (typeof `Napi::External<fl::Tensor>`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  fl::Tensor* t = UnExternalize<fl::Tensor>(info[0]);
  auto* contig_tensor = new fl::Tensor(t->asContiguousTensor());
  size_t elemLen = static_cast<size_t>(t->elements());
  size_t byteLen = elemLen * sizeof(float);
  void* ptr = contig_tensor->astype(fl::dtype::f32).host<float>();
  Napi::ArrayBuffer buff = Napi::ArrayBuffer::New(env, ptr, byteLen);
  Napi::TypedArrayOf<float> out =
      Napi::TypedArrayOf<float>::New(env, elemLen, buff, 0, napi_float32_array);
  return out;
}

static Napi::Value _toFloat64Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor* t = UnExternalize<fl::Tensor>(info[0]);
  fl::Tensor* contig_tensor = new fl::Tensor(t->asContiguousTensor());
  size_t elemLen = static_cast<size_t>(contig_tensor->elements());
  size_t byteLen = elemLen * sizeof(double);
  double* ptr = contig_tensor->astype(fl::dtype::f64).host<double>();
  std::unique_ptr<std::vector<double>> nativeArray =
      std::make_unique<std::vector<double>>(ptr, ptr + elemLen);
  delete contig_tensor;
  Napi::ArrayBuffer buff =
      Napi::ArrayBuffer::New(env, nativeArray->data(), byteLen,
                             DeleteArrayBuffer<double>, nativeArray.get());
  Napi::MemoryManagement::AdjustExternalMemory(env, byteLen);
  nativeArray.release();
  Napi::TypedArrayOf<double> out = Napi::TypedArrayOf<double>::New(
      env, elemLen, buff, 0, napi_float64_array);
  return out;
}

static Napi::Value _toBoolInt8Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "`toBoolInt8Array` expects args[0] to be native "
                         "`Tensor` (typeof `Napi::External<fl::Tensor>`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  fl::Tensor* t = UnExternalize<fl::Tensor>(info[0]);
  size_t elemLen = static_cast<size_t>(t->elements());
  size_t byteLen = elemLen * sizeof(int8_t);
  void* ptr = t->astype(fl::dtype::b8).host<int>();
  Napi::ArrayBuffer buff = Napi::ArrayBuffer::New(env, ptr, byteLen);
  Napi::TypedArrayOf<int8_t> out =
      Napi::TypedArrayOf<int8_t>::New(env, elemLen, buff, 0, napi_int8_array);
  return out;
}

static Napi::Value _toInt16Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "`toInt16Array` expects args[0] to be native `Tensor` "
                         "(typeof `Napi::External<fl::Tensor>`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  fl::Tensor* t = UnExternalize<fl::Tensor>(info[0]);
  size_t elemLen = static_cast<size_t>(t->elements());
  size_t byteLen = elemLen * sizeof(int16_t);
  void* ptr = t->astype(fl::dtype::s16).host<int>();
  Napi::ArrayBuffer buff = Napi::ArrayBuffer::New(env, ptr, byteLen);
  Napi::TypedArrayOf<int16_t> out =
      Napi::TypedArrayOf<int16_t>::New(env, elemLen, buff, 0, napi_int16_array);
  return out;
}

static Napi::Value _toInt32Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "`toInt32Array` expects args[0] to be native `Tensor` "
                         "(typeof `Napi::External<fl::Tensor>`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  fl::Tensor* t = UnExternalize<fl::Tensor>(info[0]);
  size_t elemLen = static_cast<size_t>(t->elements());
  size_t byteLen = elemLen * sizeof(int32_t);
  void* ptr = t->astype(fl::dtype::s32).host<int>();
  Napi::ArrayBuffer buff = Napi::ArrayBuffer::New(env, ptr, byteLen);
  Napi::TypedArrayOf<int32_t> out =
      Napi::TypedArrayOf<int32_t>::New(env, elemLen, buff, 0, napi_int32_array);
  return out;
}

static Napi::Value _toInt64Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "`toInt64Array` expects args[0] to be native `Tensor` "
                         "(typeof `Napi::External<fl::Tensor>`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  fl::Tensor* t = UnExternalize<fl::Tensor>(info[0]);
  size_t elemLen = static_cast<size_t>(t->elements());
  size_t byteLen = elemLen * sizeof(int64_t);
  void* ptr = t->astype(fl::dtype::s64).host<int>();
  Napi::ArrayBuffer buff = Napi::ArrayBuffer::New(env, ptr, byteLen);
  Napi::TypedArrayOf<int64_t> out = Napi::TypedArrayOf<int64_t>::New(
      env, elemLen, buff, 0, napi_bigint64_array);
  return out;
}

static Napi::Value _toUint8Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "`toUint8Array` expects args[0] to be native `Tensor` "
                         "(typeof `Napi::External<fl::Tensor>`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  fl::Tensor* t = UnExternalize<fl::Tensor>(info[0]);
  size_t elemLen = static_cast<size_t>(t->elements());
  size_t byteLen = elemLen * sizeof(uint8_t);
  void* ptr = t->astype(fl::dtype::u8).host<unsigned>();
  Napi::ArrayBuffer buff = Napi::ArrayBuffer::New(env, ptr, byteLen);
  Napi::TypedArrayOf<uint8_t> out =
      Napi::TypedArrayOf<uint8_t>::New(env, elemLen, buff, 0, napi_uint8_array);
  return out;
}

static Napi::Value _toUint16Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "`toUint16Array` expects args[0] to be native "
                         "`Tensor` (typeof `Napi::External<fl::Tensor>`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  fl::Tensor* t = UnExternalize<fl::Tensor>(info[0]);
  size_t elemLen = static_cast<size_t>(t->elements());
  size_t byteLen = elemLen * sizeof(uint16_t);
  void* ptr = t->astype(fl::dtype::u16).host<unsigned>();
  Napi::ArrayBuffer buff = Napi::ArrayBuffer::New(env, ptr, byteLen);
  Napi::TypedArrayOf<uint16_t> out = Napi::TypedArrayOf<uint16_t>::New(
      env, elemLen, buff, 0, napi_uint16_array);
  return out;
}

static Napi::Value _toUint32Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "`toUint32Array` expects args[0] to be native "
                         "`Tensor` (typeof `Napi::External<fl::Tensor>`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  fl::Tensor* t = UnExternalize<fl::Tensor>(info[0]);
  size_t elemLen = static_cast<size_t>(t->elements());
  size_t byteLen = elemLen * sizeof(uint32_t);
  void* ptr = t->astype(fl::dtype::u32).host<unsigned>();
  Napi::ArrayBuffer buff = Napi::ArrayBuffer::New(env, ptr, byteLen);
  Napi::TypedArrayOf<uint32_t> out = Napi::TypedArrayOf<uint32_t>::New(
      env, elemLen, buff, 0, napi_uint32_array);
  return out;
}

static Napi::Value _toUint64Array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "`toUint64Array` expects args[0] to be native "
                         "`Tensor` (typeof `Napi::External<fl::Tensor>`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  fl::Tensor* t = UnExternalize<fl::Tensor>(info[0]);
  size_t elemLen = static_cast<size_t>(t->elements());
  size_t byteLen = elemLen * sizeof(uint64_t);
  void* ptr = t->astype(fl::dtype::u64).host<unsigned>();
  Napi::ArrayBuffer buff = Napi::ArrayBuffer::New(env, ptr, byteLen);
  Napi::TypedArrayOf<uint64_t> out = Napi::TypedArrayOf<uint64_t>::New(
      env, elemLen, buff, 0, napi_biguint64_array);
  return out;
}

static Napi::Value _toFloat32Scalar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "`toFloat32Scalar` expects args[0] to be native "
                         "`Tensor` (typeof `Napi::External<fl::Tensor>`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  fl::Tensor* t = UnExternalize<fl::Tensor>(info[0]);
  return Napi::Number::New(env, t->asScalar<float>());
}

static Napi::Value _toFloat64Scalar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "`toFloat64Scalar` expects args[0] to be native "
                         "`Tensor` (typeof `Napi::External<fl::Tensor>`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  fl::Tensor* t = UnExternalize<fl::Tensor>(info[0]);
  return Napi::Number::New(env, t->asScalar<double>());
}

static Napi::Value _toBoolInt8Scalar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "`toBoolInt8Scalar` expects args[0] to be native "
                         "`Tensor` (typeof `Napi::External<fl::Tensor>`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  fl::Tensor* t = UnExternalize<fl::Tensor>(info[0]);
  return Napi::Number::New(env, t->asScalar<char>());
}

static Napi::Value _toInt16Scalar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "`toInt16Scalar` expects args[0] to be native "
                         "`Tensor` (typeof `Napi::External<fl::Tensor>`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  fl::Tensor* t = UnExternalize<fl::Tensor>(info[0]);
  return Napi::Number::New(env, t->asScalar<int16_t>());
}

static Napi::Value _toInt32Scalar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "`toInt32Scalar` expects args[0] to be native "
                         "`Tensor` (typeof `Napi::External<fl::Tensor>`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  fl::Tensor* t = UnExternalize<fl::Tensor>(info[0]);
  return Napi::Number::New(env, t->asScalar<int32_t>());
}

static Napi::Value _toInt64Scalar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "`toInt64Scalar` expects args[0] to be native "
                         "`Tensor` (typeof `Napi::External<fl::Tensor>`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  fl::Tensor* t = UnExternalize<fl::Tensor>(info[0]);
  return Napi::BigInt::New(env, t->asScalar<int64_t>());
}

static Napi::Value _toUint8Scalar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "`toUint8Scalar` expects args[0] to be native "
                         "`Tensor` (typeof `Napi::External<fl::Tensor>`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  fl::Tensor* t = UnExternalize<fl::Tensor>(info[0]);
  return Napi::Number::New(env, t->asScalar<uint8_t>());
}

static Napi::Value _toUint16Scalar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "`toUint16Scalar` expects args[0] to be native "
                         "`Tensor` (typeof `Napi::External<fl::Tensor>`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  fl::Tensor* t = UnExternalize<fl::Tensor>(info[0]);
  return Napi::Number::New(env, t->asScalar<uint16_t>());
}

static Napi::Value _toUint32Scalar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "`toUint32Scalar` expects args[0] to be native "
                         "`Tensor` (typeof `Napi::External<fl::Tensor>`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  fl::Tensor* t = UnExternalize<fl::Tensor>(info[0]);
  return Napi::Number::New(env, t->asScalar<uint32_t>());
}

static Napi::Value _toUint64Scalar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "`toUint64Scalar` expects args[0] to be native "
                         "`Tensor` (typeof `Napi::External<fl::Tensor>`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  fl::Tensor* t = UnExternalize<fl::Tensor>(info[0]);
  return Napi::BigInt::New(env, t->asScalar<uint64_t>());
}

static Napi::Value _add(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    Napi::TypeError::New(env, "`add` expects exactly 2 args")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env,
                         "`add` expects args[0] to be native `Tensor` (typeof "
                         "`Napi::External<fl::Tensor>`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  fl::Tensor* lhs = UnExternalize<fl::Tensor>(info[0]);
  if (!info[1].IsExternal()) {
    Napi::TypeError::New(env,
                         "`add` expects args[1] to be native `Tensor` (typeof "
                         "`Napi::External<fl::Tensor>`)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  fl::Tensor* rhs = UnExternalize<fl::Tensor>(info[1]);
  fl::Tensor _res;
  _res = fl::add(*(lhs), *(rhs));
  auto _out_byte_count = static_cast<int64_t>(_res.bytes());
  g_bytes_used += _out_byte_count;
  Napi::MemoryManagement::AdjustExternalMemory(env, _out_byte_count);
  auto* out = new fl::Tensor(_res);
  Napi::External<fl::Tensor> _external_out = ExternalizeTensor(env, out);
  return _external_out;
}

static Napi::Value _asContiguousTensor(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  fl::Tensor* _tmp_external = UnExternalize<fl::Tensor>(info[0]);
  auto _res = _tmp_external->asContiguousTensor();
  auto _out_byte_count = static_cast<int64_t>(_res.bytes());
  g_bytes_used += _out_byte_count;
  Napi::MemoryManagement::AdjustExternalMemory(env, _out_byte_count);
  auto* out = new fl::Tensor(_res);
  Napi::External<fl::Tensor> _external_out = ExternalizeTensor(env, out);
  return _external_out;
}

// global methods
static void _init(const Napi::CallbackInfo& /*info*/) {
  fl::init();
}

static Napi::Value _bytesUsed(const Napi::CallbackInfo& info) {
  return Napi::BigInt::New(info.Env(), static_cast<int64_t>(g_bytes_used));
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "_tensorFromFloat32Buffer"),
              Napi::Function::New(env, _tensorFromFloat32Buffer));
  exports.Set(Napi::String::New(env, "_tensorFromFloat64Buffer"),
              Napi::Function::New(env, _tensorFromFloat64Buffer));
  exports.Set(Napi::String::New(env, "_tensorFromBoolInt8Buffer"),
              Napi::Function::New(env, _tensorFromBoolInt8Buffer));
  exports.Set(Napi::String::New(env, "_tensorFromInt16Buffer"),
              Napi::Function::New(env, _tensorFromInt16Buffer));
  exports.Set(Napi::String::New(env, "_tensorFromInt32Buffer"),
              Napi::Function::New(env, _tensorFromInt32Buffer));
  exports.Set(Napi::String::New(env, "_tensorFromInt64Buffer"),
              Napi::Function::New(env, _tensorFromInt64Buffer));
  exports.Set(Napi::String::New(env, "_tensorFromUint8Buffer"),
              Napi::Function::New(env, _tensorFromUint8Buffer));
  exports.Set(Napi::String::New(env, "_tensorFromUint16Buffer"),
              Napi::Function::New(env, _tensorFromUint16Buffer));
  exports.Set(Napi::String::New(env, "_tensorFromUint32Buffer"),
              Napi::Function::New(env, _tensorFromUint32Buffer));
  exports.Set(Napi::String::New(env, "_tensorFromUint64Buffer"),
              Napi::Function::New(env, _tensorFromUint64Buffer));
  exports.Set(Napi::String::New(env, "_init"), Napi::Function::New(env, _init));
  exports.Set(Napi::String::New(env, "_bytesUsed"),
              Napi::Function::New(env, _bytesUsed));
  exports.Set(Napi::String::New(env, "_toFloat32Array"),
              Napi::Function::New(env, _toFloat32Array));
  exports.Set(Napi::String::New(env, "_toFloat64Array"),
              Napi::Function::New(env, _toFloat64Array));
  exports.Set(Napi::String::New(env, "_toBoolInt8Array"),
              Napi::Function::New(env, _toBoolInt8Array));
  exports.Set(Napi::String::New(env, "_toInt16Array"),
              Napi::Function::New(env, _toInt16Array));
  exports.Set(Napi::String::New(env, "_toInt32Array"),
              Napi::Function::New(env, _toInt32Array));
  exports.Set(Napi::String::New(env, "_toInt64Array"),
              Napi::Function::New(env, _toInt64Array));
  exports.Set(Napi::String::New(env, "_toUint8Array"),
              Napi::Function::New(env, _toUint8Array));
  exports.Set(Napi::String::New(env, "_toUint16Array"),
              Napi::Function::New(env, _toUint16Array));
  exports.Set(Napi::String::New(env, "_toUint32Array"),
              Napi::Function::New(env, _toUint32Array));
  exports.Set(Napi::String::New(env, "_toUint64Array"),
              Napi::Function::New(env, _toUint64Array));
  exports.Set(Napi::String::New(env, "_toFloat32Scalar"),
              Napi::Function::New(env, _toFloat32Scalar));
  exports.Set(Napi::String::New(env, "_toFloat64Scalar"),
              Napi::Function::New(env, _toFloat64Scalar));
  exports.Set(Napi::String::New(env, "_toBoolInt8Scalar"),
              Napi::Function::New(env, _toBoolInt8Scalar));
  exports.Set(Napi::String::New(env, "_toInt16Scalar"),
              Napi::Function::New(env, _toInt16Scalar));
  exports.Set(Napi::String::New(env, "_toInt32Scalar"),
              Napi::Function::New(env, _toInt32Scalar));
  exports.Set(Napi::String::New(env, "_toInt64Scalar"),
              Napi::Function::New(env, _toInt64Scalar));
  exports.Set(Napi::String::New(env, "_toUint8Scalar"),
              Napi::Function::New(env, _toUint8Scalar));
  exports.Set(Napi::String::New(env, "_toUint16Scalar"),
              Napi::Function::New(env, _toUint16Scalar));
  exports.Set(Napi::String::New(env, "_toUint32Scalar"),
              Napi::Function::New(env, _toUint32Scalar));
  exports.Set(Napi::String::New(env, "_toUint64Scalar"),
              Napi::Function::New(env, _toUint64Scalar));
  exports.Set(Napi::String::New(env, "_add"), Napi::Function::New(env, _add));
  exports.Set(Napi::String::New(env, "_tensorFromFloat32Array"),
              Napi::Function::New(env, _tensorFromFloat32Array));
  exports.Set(Napi::String::New(env, "_tensorFromFloat64Array"),
              Napi::Function::New(env, _tensorFromFloat64Array));
  exports.Set(Napi::String::New(env, "_tensorFromBoolInt8Array"),
              Napi::Function::New(env, _tensorFromBoolInt8Array));
  exports.Set(Napi::String::New(env, "_tensorFromInt16Array"),
              Napi::Function::New(env, _tensorFromInt16Array));
  exports.Set(Napi::String::New(env, "_tensorFromInt32Array"),
              Napi::Function::New(env, _tensorFromInt32Array));
  exports.Set(Napi::String::New(env, "_tensorFromInt64Array"),
              Napi::Function::New(env, _tensorFromInt64Array));
  exports.Set(Napi::String::New(env, "_tensorFromUint8Array"),
              Napi::Function::New(env, _tensorFromUint8Array));
  exports.Set(Napi::String::New(env, "_tensorFromUint16Array"),
              Napi::Function::New(env, _tensorFromUint16Array));
  exports.Set(Napi::String::New(env, "_tensorFromUint32Array"),
              Napi::Function::New(env, _tensorFromUint32Array));
  exports.Set(Napi::String::New(env, "_tensorFromUint64Array"),
              Napi::Function::New(env, _tensorFromUint64Array));
  exports.Set(Napi::String::New(env, "_asContiguousTensor"),
              Napi::Function::New(env, _asContiguousTensor));
  return exports;
}

NODE_API_MODULE(addon, Init)
