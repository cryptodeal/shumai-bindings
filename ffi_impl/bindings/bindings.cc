#include <atomic>
#include <cstdlib>
#include <iostream>
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

#define FMT_RESET "\033[0m"
#define FMT_RED "\033[31m"
#define FMT_GRAY "\033[90m"
#define FMT_YELLOW "\033[33m"
#define FMT_CYAN "\033[36m"
#define FMT_BOLD_WHITE "\033[1m\033[97m"
#define FMT_BOLD_ITALIC_WHITE "\033[1m\033[3m\033[97m"

#define HANDLE_EXCEPTION(what)                                         \
  {                                                                    \
    std::cerr << FMT_RED << "native code error" << FMT_GRAY << ": "    \
              << FMT_BOLD_WHITE << what << FMT_RESET << FMT_GRAY       \
              << "\n                  at " << FMT_BOLD_ITALIC_WHITE    \
              << __func__ << FMT_RESET << FMT_GRAY << " (" << FMT_CYAN \
              << __FILE__ << FMT_GRAY << ":" << FMT_YELLOW << __LINE__ \
              << FMT_GRAY << ")" << FMT_RESET << std::endl;            \
    return nullptr;                                                    \
  }

#if 0
#include <mutex>
static std::mutex g_op_mutex;
#define LOCK_GUARD std::lock_guard<std::mutex> guard(g_op_mutex);
#else
#define LOCK_GUARD
#endif

static std::atomic<size_t> g_bytes_used = 0;

extern "C"
{
  void init()
  {
    fl::init();
  }

  size_t bytesUsed()
  {
    return g_bytes_used;
  }

  size_t _bytes(void *t)
  {
    LOCK_GUARD
    auto *tensor = reinterpret_cast<fl::Tensor *>(t);
    return tensor->bytes();
  }

  void destroyTensor(void *t, void * /*ignore*/)
  {
    LOCK_GUARD
    auto *tensor = reinterpret_cast<fl::Tensor *>(t);
    if (tensor->hasAdapter())
    {
      g_bytes_used -= tensor->bytes();
    }
    delete tensor;
  }

  typedef void (*JSTypedArrayBytesDeallocator)(void *bytes,
                                               void *deallocatorContext);

  JSTypedArrayBytesDeallocator genTensorDestroyer()
  {
    return destroyTensor;
  }

  float *_float32Buffer(void *t)
  {
    try
    {
      LOCK_GUARD
      auto *tensor = reinterpret_cast<fl::Tensor *>(t);
      return tensor->astype(fl::dtype::f32).host<float>();
    }
    catch (std::exception const &e)
    {
      HANDLE_EXCEPTION(e.what());
    }
    catch (...)
    {
      HANDLE_EXCEPTION("[unknown]");
    }
  }

  float *_float64Buffer(void *t)
  {
    try
    {
      LOCK_GUARD
      auto *tensor = reinterpret_cast<fl::Tensor *>(t);
      return tensor->astype(fl::dtype::f64).host<float>();
    }
    catch (std::exception const &e)
    {
      HANDLE_EXCEPTION(e.what());
    }
    catch (...)
    {
      HANDLE_EXCEPTION("[unknown]");
    }
  }

  int *_boolInt8Buffer(void *t)
  {
    try
    {
      LOCK_GUARD
      auto *tensor = reinterpret_cast<fl::Tensor *>(t);
      return tensor->astype(fl::dtype::b8).host<int>();
    }
    catch (std::exception const &e)
    {
      HANDLE_EXCEPTION(e.what());
    }
    catch (...)
    {
      HANDLE_EXCEPTION("[unknown]");
    }
  }

  int *_int16Buffer(void *t)
  {
    try
    {
      LOCK_GUARD
      auto *tensor = reinterpret_cast<fl::Tensor *>(t);
      return tensor->astype(fl::dtype::s16).host<int>();
    }
    catch (std::exception const &e)
    {
      HANDLE_EXCEPTION(e.what());
    }
    catch (...)
    {
      HANDLE_EXCEPTION("[unknown]");
    }
  }

  int *_int32Buffer(void *t)
  {
    try
    {
      LOCK_GUARD
      auto *tensor = reinterpret_cast<fl::Tensor *>(t);
      return tensor->astype(fl::dtype::s32).host<int>();
    }
    catch (std::exception const &e)
    {
      HANDLE_EXCEPTION(e.what());
    }
    catch (...)
    {
      HANDLE_EXCEPTION("[unknown]");
    }
  }

  int *_int64Buffer(void *t)
  {
    try
    {
      LOCK_GUARD
      auto *tensor = reinterpret_cast<fl::Tensor *>(t);
      return tensor->astype(fl::dtype::s64).host<int>();
    }
    catch (std::exception const &e)
    {
      HANDLE_EXCEPTION(e.what());
    }
    catch (...)
    {
      HANDLE_EXCEPTION("[unknown]");
    }
  }

  unsigned *_uint8Buffer(void *t)
  {
    try
    {
      LOCK_GUARD
      auto *tensor = reinterpret_cast<fl::Tensor *>(t);
      return tensor->astype(fl::dtype::u8).host<unsigned>();
    }
    catch (std::exception const &e)
    {
      HANDLE_EXCEPTION(e.what());
    }
    catch (...)
    {
      HANDLE_EXCEPTION("[unknown]");
    }
  }

  unsigned *_uint16Buffer(void *t)
  {
    try
    {
      LOCK_GUARD
      auto *tensor = reinterpret_cast<fl::Tensor *>(t);
      return tensor->astype(fl::dtype::u16).host<unsigned>();
    }
    catch (std::exception const &e)
    {
      HANDLE_EXCEPTION(e.what());
    }
    catch (...)
    {
      HANDLE_EXCEPTION("[unknown]");
    }
  }

  unsigned *_uint32Buffer(void *t)
  {
    try
    {
      LOCK_GUARD
      auto *tensor = reinterpret_cast<fl::Tensor *>(t);
      return tensor->astype(fl::dtype::u32).host<unsigned>();
    }
    catch (std::exception const &e)
    {
      HANDLE_EXCEPTION(e.what());
    }
    catch (...)
    {
      HANDLE_EXCEPTION("[unknown]");
    }
  }

  unsigned *_uint64Buffer(void *t)
  {
    try
    {
      LOCK_GUARD
      auto *tensor = reinterpret_cast<fl::Tensor *>(t);
      return tensor->astype(fl::dtype::u64).host<unsigned>();
    }
    catch (std::exception const &e)
    {
      HANDLE_EXCEPTION(e.what());
    }
    catch (...)
    {
      HANDLE_EXCEPTION("[unknown]");
    }
  }

  float _float16Scalar(void *t)
  {
    LOCK_GUARD
    auto *tensor = reinterpret_cast<fl::Tensor *>(t);
    return tensor->asScalar<float>();
  }

  float _float32Scalar(void *t)
  {
    LOCK_GUARD
    auto *tensor = reinterpret_cast<fl::Tensor *>(t);
    return tensor->asScalar<float>();
  }

  float _float64Scalar(void *t)
  {
    LOCK_GUARD
    auto *tensor = reinterpret_cast<fl::Tensor *>(t);
    return tensor->asScalar<float>();
  }

  char _boolInt8Scalar(void *t)
  {
    LOCK_GUARD
    auto *tensor = reinterpret_cast<fl::Tensor *>(t);
    return tensor->asScalar<char>();
  }

  int16_t _int16Scalar(void *t)
  {
    LOCK_GUARD
    auto *tensor = reinterpret_cast<fl::Tensor *>(t);
    return tensor->asScalar<int16_t>();
  }

  int32_t _int32Scalar(void *t)
  {
    LOCK_GUARD
    auto *tensor = reinterpret_cast<fl::Tensor *>(t);
    return tensor->asScalar<int32_t>();
  }

  int64_t _int64Scalar(void *t)
  {
    LOCK_GUARD
    auto *tensor = reinterpret_cast<fl::Tensor *>(t);
    return tensor->asScalar<int64_t>();
  }

  uint8_t _uint8Scalar(void *t)
  {
    LOCK_GUARD
    auto *tensor = reinterpret_cast<fl::Tensor *>(t);
    return tensor->asScalar<uint8_t>();
  }

  uint16_t _uint16Scalar(void *t)
  {
    LOCK_GUARD
    auto *tensor = reinterpret_cast<fl::Tensor *>(t);
    return tensor->asScalar<uint16_t>();
  }

  uint32_t _uint32Scalar(void *t)
  {
    LOCK_GUARD
    auto *tensor = reinterpret_cast<fl::Tensor *>(t);
    return tensor->asScalar<uint32_t>();
  }

  uint64_t _uint64Scalar(void *t)
  {
    LOCK_GUARD
    auto *tensor = reinterpret_cast<fl::Tensor *>(t);
    return tensor->asScalar<uint64_t>();
  }

  void *tensorFromFloat32Buffer(int64_t numel, void *ptr)
  {
    try
    {
      LOCK_GUARD
      auto *t = new fl::Tensor(
          fl::Tensor::fromBuffer({numel}, (float *)ptr, fl::MemoryLocation::Host));
      g_bytes_used += t->bytes();
      return t;
    }
    catch (std::exception const &e)
    {
      HANDLE_EXCEPTION(e.what());
    }
    catch (...)
    {
      HANDLE_EXCEPTION("[unknown]");
    }
  }

  void *tensorFromFloat64Buffer(int64_t numel, void *ptr)
  {
    try
    {
      LOCK_GUARD
      auto *t = new fl::Tensor(fl::Tensor::fromBuffer({numel}, (double *)ptr,
                                                      fl::MemoryLocation::Host));
      g_bytes_used += t->bytes();
      return t;
    }
    catch (std::exception const &e)
    {
      HANDLE_EXCEPTION(e.what());
    }
    catch (...)
    {
      HANDLE_EXCEPTION("[unknown]");
    }
  }

  void *tensorFromInt8Buffer(int64_t numel, void *ptr)
  {
    try
    {
      LOCK_GUARD
      auto *t = new fl::Tensor(
          fl::Tensor::fromBuffer({numel}, (char *)ptr, fl::MemoryLocation::Host));
      g_bytes_used += t->bytes();
      return t;
    }
    catch (std::exception const &e)
    {
      HANDLE_EXCEPTION(e.what());
    }
    catch (...)
    {
      HANDLE_EXCEPTION("[unknown]");
    }
  }

  void *tensorFromInt16Buffer(int64_t numel, void *ptr)
  {
    try
    {
      LOCK_GUARD
      auto *t = new fl::Tensor(fl::Tensor::fromBuffer({numel}, (int16_t *)ptr,
                                                      fl::MemoryLocation::Host));
      g_bytes_used += t->bytes();
      return t;
    }
    catch (std::exception const &e)
    {
      HANDLE_EXCEPTION(e.what());
    }
    catch (...)
    {
      HANDLE_EXCEPTION("[unknown]");
    }
  }

  void *tensorFromInt32Buffer(int64_t numel, void *ptr)
  {
    try
    {
      LOCK_GUARD
      auto *t = new fl::Tensor(fl::Tensor::fromBuffer({numel}, (int32_t *)ptr,
                                                      fl::MemoryLocation::Host));
      g_bytes_used += t->bytes();
      return t;
    }
    catch (std::exception const &e)
    {
      HANDLE_EXCEPTION(e.what());
    }
    catch (...)
    {
      HANDLE_EXCEPTION("[unknown]");
    }
  }

  void *tensorFromInt64Buffer(int64_t numel, void *ptr)
  {
    try
    {
      LOCK_GUARD
      auto *t = new fl::Tensor(fl::Tensor::fromBuffer({numel}, (int64_t *)ptr,
                                                      fl::MemoryLocation::Host));
      g_bytes_used += t->bytes();
      return t;
    }
    catch (std::exception const &e)
    {
      HANDLE_EXCEPTION(e.what());
    }
    catch (...)
    {
      HANDLE_EXCEPTION("[unknown]");
    }
  }

  size_t _elements(void *t)
  {
    LOCK_GUARD
    auto *tensor = reinterpret_cast<fl::Tensor *>(t);
    return tensor->elements();
  }

  void *tensorFromUint8Buffer(int64_t numel, void *ptr)
  {
    try
    {
      LOCK_GUARD
      auto *t = new fl::Tensor(fl::Tensor::fromBuffer({numel}, (uint8_t *)ptr,
                                                      fl::MemoryLocation::Host));
      g_bytes_used += t->bytes();
      return t;
    }
    catch (std::exception const &e)
    {
      HANDLE_EXCEPTION(e.what());
    }
    catch (...)
    {
      HANDLE_EXCEPTION("[unknown]");
    }
  }

  void *tensorFromUint16Buffer(int64_t numel, void *ptr)
  {
    try
    {
      LOCK_GUARD
      auto *t = new fl::Tensor(fl::Tensor::fromBuffer({numel}, (uint16_t *)ptr,
                                                      fl::MemoryLocation::Host));
      g_bytes_used += t->bytes();
      return t;
    }
    catch (std::exception const &e)
    {
      HANDLE_EXCEPTION(e.what());
    }
    catch (...)
    {
      HANDLE_EXCEPTION("[unknown]");
    }
  }

  void *tensorFromUint32Buffer(int64_t numel, void *ptr)
  {
    try
    {
      LOCK_GUARD
      auto *t = new fl::Tensor(fl::Tensor::fromBuffer({numel}, (uint32_t *)ptr,
                                                      fl::MemoryLocation::Host));
      g_bytes_used += t->bytes();
      return t;
    }
    catch (std::exception const &e)
    {
      HANDLE_EXCEPTION(e.what());
    }
    catch (...)
    {
      HANDLE_EXCEPTION("[unknown]");
    }
  }

  void *tensorFromUint64Buffer(int64_t numel, void *ptr)
  {
    try
    {
      LOCK_GUARD
      auto *t = new fl::Tensor(fl::Tensor::fromBuffer({numel}, (uint64_t *)ptr,
                                                      fl::MemoryLocation::Host));
      g_bytes_used += t->bytes();
      return t;
    }
    catch (std::exception const &e)
    {
      HANDLE_EXCEPTION(e.what());
    }
    catch (...)
    {
      HANDLE_EXCEPTION("[unknown]");
    }
  }

  void *_asContiguousTensor(void *t)
  {
    try
    {
      LOCK_GUARD
      auto *tensor = reinterpret_cast<fl::Tensor *>(t);
      auto *new_tensor = new fl::Tensor(tensor->asContiguousTensor());
      g_bytes_used += new_tensor->bytes();
      return new_tensor;
    }
    catch (std::exception const &e)
    {
      HANDLE_EXCEPTION(e.what());
    }
    catch (...)
    {
      HANDLE_EXCEPTION("[unknown]");
    }
  }
}