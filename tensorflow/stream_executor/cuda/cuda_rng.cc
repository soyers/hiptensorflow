/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "cuda/include/hiprng/hiprng.h"
#include "tensorflow/stream_executor/cuda/cuda_rng.h"

#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_helpers.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/dso_loader.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/rng.h"

// Formats hiprngStatus_t to output prettified values into a log stream.
std::ostream &operator<<(std::ostream &in, const hiprngStatus_t &status) {
#define OSTREAM_HIPRNG_STATUS(__name) \
  case HIPRNG_STATUS_##__name:        \
    in << "HIPRNG_STATUS_" #__name;   \
    return in;

  switch (status) {
    OSTREAM_HIPRNG_STATUS(SUCCESS)
    OSTREAM_HIPRNG_STATUS(VERSION_MISMATCH)
    OSTREAM_HIPRNG_STATUS(NOT_INITIALIZED)
    OSTREAM_HIPRNG_STATUS(ALLOCATION_FAILED)
    OSTREAM_HIPRNG_STATUS(TYPE_ERROR)
    OSTREAM_HIPRNG_STATUS(OUT_OF_RANGE)
    OSTREAM_HIPRNG_STATUS(LENGTH_NOT_MULTIPLE)
    OSTREAM_HIPRNG_STATUS(LAUNCH_FAILURE)
    OSTREAM_HIPRNG_STATUS(PREEXISTING_FAILURE)
    OSTREAM_HIPRNG_STATUS(INITIALIZATION_FAILED)
    OSTREAM_HIPRNG_STATUS(ARCH_MISMATCH)
    OSTREAM_HIPRNG_STATUS(INTERNAL_ERROR)
    default:
      in << "hiprngStatus_t(" << static_cast<int>(status) << ")";
      return in;
  }
}

namespace perftools {
namespace gputools {
namespace cuda {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kCuRandPlugin);

namespace dynload {

#define PERFTOOLS_GPUTOOLS_HIPRNG_WRAP(__name)                              \
  struct DynLoadShim__##__name {                                            \
    static const char *kName;                                               \
    using FuncPointerT = std::add_pointer<decltype(::__name)>::type;        \
    static void *GetDsoHandle() {                                           \
      static auto status = internal::CachedDsoLoader::GetCurandDsoHandle(); \
      return status.ValueOrDie();                                           \
    }                                                                       \
    static FuncPointerT LoadOrDie() {                                       \
      void *f;                                                              \
      port::Status s = port::Env::Default()->GetSymbolFromLibrary(          \
          GetDsoHandle(), kName, &f);                                       \
      CHECK(s.ok()) << "could not find " << kName                           \
                    << " in hiprng DSO; dlerror: " << s.error_message();    \
      return reinterpret_cast<FuncPointerT>(f);                             \
    }                                                                       \
    static FuncPointerT DynLoad() {                                         \
      static FuncPointerT f = LoadOrDie();                                  \
      return f;                                                             \
    }                                                                       \
    template <typename... Args>                                             \
    hiprngStatus_t operator()(CUDAExecutor *parent, Args... args) {         \
      cuda::ScopedActivateExecutorContext sac{parent};                      \
      return DynLoad()(args...);                                            \
    }                                                                       \
  } __name;                                                                 \
  const char *DynLoadShim__##__name::kName = #__name;

PERFTOOLS_GPUTOOLS_HIPRNG_WRAP(hiprngCreateGenerator);
PERFTOOLS_GPUTOOLS_HIPRNG_WRAP(hiprngDestroyGenerator);
PERFTOOLS_GPUTOOLS_HIPRNG_WRAP(hiprngSetStream);
PERFTOOLS_GPUTOOLS_HIPRNG_WRAP(hiprngGenerateUniform);
PERFTOOLS_GPUTOOLS_HIPRNG_WRAP(hiprngGenerateUniformDouble);
PERFTOOLS_GPUTOOLS_HIPRNG_WRAP(hiprngSetPseudoRandomGeneratorSeed);
PERFTOOLS_GPUTOOLS_HIPRNG_WRAP(hiprngSetGeneratorOffset);
PERFTOOLS_GPUTOOLS_HIPRNG_WRAP(hiprngGenerateNormal);
PERFTOOLS_GPUTOOLS_HIPRNG_WRAP(hiprngGenerateNormalDouble);

}  // namespace dynload

template <typename T>
string TypeString();

template <>
string TypeString<float>() {
  return "float";
}

template <>
string TypeString<double>() {
  return "double";
}

template <>
string TypeString<std::complex<float>>() {
  return "std::complex<float>";
}

template <>
string TypeString<std::complex<double>>() {
  return "std::complex<double>";
}

CUDARng::CUDARng(CUDAExecutor *parent) : parent_(parent), rng_(nullptr) {}

CUDARng::~CUDARng() {
  if (rng_ != nullptr) {
    dynload::hiprngDestroyGenerator(parent_, rng_);
  }
}

bool CUDARng::Init() {
  mutex_lock lock{mu_};
  CHECK(rng_ == nullptr);

  hiprngStatus_t ret =
      dynload::hiprngCreateGenerator(parent_, &rng_, HIPRNG_RNG_PSEUDO_DEFAULT);
  if (ret != HIPRNG_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to create random number generator: " << ret;
    return false;
  }

  CHECK(rng_ != nullptr);
  return true;
}

bool CUDARng::SetStream(Stream *stream) {
  hiprngStatus_t ret =
      dynload::hiprngSetStream(parent_, rng_, AsCUDAStreamValue(stream));
  if (ret != HIPRNG_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set stream for random generation: " << ret;
    return false;
  }

  return true;
}

// Returns true if std::complex stores its contents as two consecutive
// elements. Tests int, float and double, as the last two are independent
// specializations.
constexpr bool ComplexIsConsecutiveFloats() {
  return sizeof(std::complex<int>) == 8 && sizeof(std::complex<float>) == 8 &&
      sizeof(std::complex<double>) == 16;
}

template <typename T>
bool CUDARng::DoPopulateRandUniformInternal(Stream *stream,
                                            DeviceMemory<T> *v) {
  mutex_lock lock{mu_};
  static_assert(ComplexIsConsecutiveFloats(),
                "std::complex values are not stored as consecutive values");

  if (!SetStream(stream)) {
    return false;
  }

  // std::complex<T> is currently implemented as two consecutive T variables.
  uint64 element_count = v->ElementCount();
  if (std::is_same<T, std::complex<float>>::value ||
      std::is_same<T, std::complex<double>>::value) {
    element_count *= 2;
  }

  hiprngStatus_t ret;
  if (std::is_same<T, float>::value ||
      std::is_same<T, std::complex<float>>::value) {
    ret = dynload::hiprngGenerateUniform(
        parent_, rng_, reinterpret_cast<float *>(CUDAMemoryMutable(v)),
        element_count);
  } else {
    ret = dynload::hiprngGenerateUniformDouble(
        parent_, rng_, reinterpret_cast<double *>(CUDAMemoryMutable(v)),
        element_count);
  }
  if (ret != HIPRNG_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to do uniform generation of " << v->ElementCount()
               << " " << TypeString<T>() << "s at " << v->opaque() << ": "
               << ret;
    return false;
  }

  return true;
}

bool CUDARng::DoPopulateRandUniform(Stream *stream, DeviceMemory<float> *v) {
  return DoPopulateRandUniformInternal(stream, v);
}

bool CUDARng::DoPopulateRandUniform(Stream *stream, DeviceMemory<double> *v) {
  return DoPopulateRandUniformInternal(stream, v);
}

bool CUDARng::DoPopulateRandUniform(Stream *stream,
                                    DeviceMemory<std::complex<float>> *v) {
  return DoPopulateRandUniformInternal(stream, v);
}

bool CUDARng::DoPopulateRandUniform(Stream *stream,
                                    DeviceMemory<std::complex<double>> *v) {
  return DoPopulateRandUniformInternal(stream, v);
}

template <typename ElemT, typename FuncT>
bool CUDARng::DoPopulateRandGaussianInternal(Stream *stream, ElemT mean,
                                             ElemT stddev,
                                             DeviceMemory<ElemT> *v,
                                             FuncT func) {
  mutex_lock lock{mu_};

  if (!SetStream(stream)) {
    return false;
  }

  uint64 element_count = v->ElementCount();
  hiprngStatus_t ret =
      func(parent_, rng_, CUDAMemoryMutable(v), element_count, mean, stddev);

  if (ret != HIPRNG_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to do gaussian generation of " << v->ElementCount()
               << " floats at " << v->opaque() << ": " << ret;
    return false;
  }

  return true;
}

bool CUDARng::DoPopulateRandGaussian(Stream *stream, float mean, float stddev,
                                     DeviceMemory<float> *v) {
  return DoPopulateRandGaussianInternal(stream, mean, stddev, v,
                                        dynload::hiprngGenerateNormal);
}

bool CUDARng::DoPopulateRandGaussian(Stream *stream, double mean, double stddev,
                                     DeviceMemory<double> *v) {
  return DoPopulateRandGaussianInternal(stream, mean, stddev, v,
                                        dynload::hiprngGenerateNormalDouble);
}

bool CUDARng::SetSeed(Stream *stream, const uint8 *seed, uint64 seed_bytes) {
  mutex_lock lock{mu_};
  CHECK(rng_ != nullptr);

  if (!CheckSeed(seed, seed_bytes)) {
    return false;
  }

  if (!SetStream(stream)) {
    return false;
  }

  // Requires 8 bytes of seed data; checked in RngSupport::CheckSeed (above)
  // (which itself requires 16 for API consistency with host RNG fallbacks).
  hiprngStatus_t ret = dynload::hiprngSetPseudoRandomGeneratorSeed(
      parent_, rng_, *(reinterpret_cast<const uint64 *>(seed)));
  if (ret != HIPRNG_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set rng seed: " << ret;
    return false;
  }

  ret = dynload::hiprngSetGeneratorOffset(parent_, rng_, 0);
  if (ret != HIPRNG_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to reset rng position: " << ret;
    return false;
  }
  return true;
}

}  // namespace cuda
}  // namespace gputools
}  // namespace perftools

namespace gpu = ::perftools::gputools;

REGISTER_MODULE_INITIALIZER(register_hiprng, {
  gpu::port::Status status =
      gpu::PluginRegistry::Instance()
          ->RegisterFactory<gpu::PluginRegistry::RngFactory>(
              gpu::cuda::kCudaPlatformId, gpu::cuda::kCuRandPlugin, "cuRAND",
              [](gpu::internal::StreamExecutorInterface
                     *parent) -> gpu::rng::RngSupport * {
                gpu::cuda::CUDAExecutor *cuda_executor =
                    dynamic_cast<gpu::cuda::CUDAExecutor *>(parent);
                if (cuda_executor == nullptr) {
                  LOG(ERROR)
                      << "Attempting to initialize an instance of the cuRAND "
                      << "support library with a non-CUDA StreamExecutor";
                  return nullptr;
                }

                gpu::cuda::CUDARng *rng = new gpu::cuda::CUDARng(cuda_executor);
                if (!rng->Init()) {
                  // Note: Init() will log a more specific error.
                  delete rng;
                  return nullptr;
                }
                return rng;
              });

  if (!status.ok()) {
    LOG(ERROR) << "Unable to register cuRAND factory: "
               << status.error_message();
  }

  // Prime the cuRAND DSO. The loader will log more information.
  auto statusor = gpu::internal::CachedDsoLoader::GetCurandDsoHandle();
  if (!statusor.ok()) {
    LOG(INFO) << "Unable to load cuRAND DSO.";
  }

  gpu::PluginRegistry::Instance()->SetDefaultFactory(gpu::cuda::kCudaPlatformId,
                                                     gpu::PluginKind::kRng,
                                                     gpu::cuda::kCuRandPlugin);
});
