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

// Include HIPBLAS headers early, and then set EIGEN_HAS_CUDA_FP16
// if we have new enough CUDA (which we will only know after including
// cuda.h). This ensures that Eigen's Half.h does not attempt to make its own
// __half typedef if CUDA has already defined one (and conversely, that we do
// not include <cuda_fp16.h> after Half.h has made its typedef).
#include "cuda/include/hipblas/hipblas.h"

#if CUDA_VERSION >= 7050
#define EIGEN_HAS_CUDA_FP16
#endif

#if CUDA_VERSION >= 8000
#define SE_CUDA_DATA_HALF CUDA_R_16F
#else
#define SE_CUDA_DATA_HALF HIPBLAS_DATA_HALF
#endif

#include "tensorflow/stream_executor/cuda/cuda_blas.h"

#include <complex>

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
#include "tensorflow/stream_executor/lib/status_macros.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/scratch_allocator.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace perftools {
namespace gputools {
namespace cuda {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kCuBlasPlugin);

namespace dynload {

#define PERFTOOLS_GPUTOOLS_HIPBLAS_WRAP(__name)                              \
  struct DynLoadShim__##__name {                                            \
    static const char *kName;                                               \
    using FuncPointerT = std::add_pointer<decltype(::__name)>::type;        \
    static void *GetDsoHandle() {                                           \
      static auto status = internal::CachedDsoLoader::GetCublasDsoHandle(); \
      return status.ValueOrDie();                                           \
    }                                                                       \
    static FuncPointerT LoadOrDie() {                                       \
      void *f;                                                              \
      port::Status s = port::Env::Default()->GetSymbolFromLibrary(          \
          GetDsoHandle(), kName, &f);                                       \
      CHECK(s.ok()) << "could not find " << kName                           \
                    << " in HIPBLAS DSO; dlerror: " << s.error_message();    \
      return reinterpret_cast<FuncPointerT>(f);                             \
    }                                                                       \
    static FuncPointerT DynLoad() {                                         \
      static FuncPointerT f = LoadOrDie();                                  \
      return f;                                                             \
    }                                                                       \
    template <typename... Args>                                             \
    hipblasStatus_t operator()(CUDAExecutor *parent, Args... args) {         \
      cuda::ScopedActivateExecutorContext sac{parent};                      \
      return DynLoad()(args...);                                            \
    }                                                                       \
  } __name;                                                                 \
  const char *DynLoadShim__##__name::kName = #__name;

#define PERFTOOLS_GPUTOOLS_HIPBLAS_V2_WRAP(__name) \
  PERFTOOLS_GPUTOOLS_HIPBLAS_WRAP(__name)

#define HIPBLAS_BLAS_ROUTINE_EACH(__macro) \
/*  __macro(hipblasSnrm2)                    \
  __macro(hipblasDnrm2)                    \
  __macro(hipblasScnrm2)                   \
  __macro(hipblasDznrm2)                 */  \
  __macro(hipblasSdot)                     \
  __macro(hipblasDdot)                     \
/*  __macro(hipblasCdotu)                    \
  __macro(hipblasCdotc)                    \
  __macro(hipblasZdotu)                    \
  __macro(hipblasZdotc)                   */ \
  __macro(hipblasSscal)                    \
  __macro(hipblasDscal)                    \
  __macro(hipblasCscal)                    \
  __macro(hipblasCsscal)                   \
  __macro(hipblasZscal)                    \
  __macro(hipblasZdscal)                   \
  __macro(hipblasSaxpy)                    \
  __macro(hipblasDaxpy)                    \
/*__macro(hipblasCaxpy)                    \
  __macro(hipblasZaxpy)                  */ \
  __macro(hipblasScopy)                    \
  __macro(hipblasDcopy)                    \
/*__macro(hipblasCcopy)                    \
  __macro(hipblasZcopy)                    \
  __macro(hipblasSswap)                    \
  __macro(hipblasDswap)                    \
  __macro(hipblasCswap)                    \
  __macro(hipblasZswap)                    \
  __macro(hipblasIsamax)                   \
  __macro(hipblasIdamax)                   \
  __macro(hipblasIcamax)                   \
  __macro(hipblasIzamax)                   \
  __macro(hipblasIsamin)                   \
  __macro(hipblasIdamin)                   \
  __macro(hipblasIcamin)                   \
  __macro(hipblasIzamin)                 */  \
  __macro(hipblasSasum)                    \
  __macro(hipblasDasum)                    \
/*  __macro(hipblasScasum)                   \
  __macro(hipblasDzasum)                   \
  __macro(hipblasSrot)                     \
  __macro(hipblasDrot)                     \
  __macro(hipblasCrot)                     \
  __macro(hipblasCsrot)                    \
  __macro(hipblasZrot)                     \
  __macro(hipblasZdrot)                    \
  __macro(hipblasSrotg)                    \
  __macro(hipblasDrotg)                    \
  __macro(hipblasCrotg)                    \
  __macro(hipblasZrotg)                    \
  __macro(hipblasSrotm)                    \
  __macro(hipblasDrotm)                    \
  __macro(hipblasSrotmg)                   \
  __macro(hipblasDrotmg)                 */  \
  __macro(hipblasSgemv)                    \
  __macro(hipblasDgemv)                    \
/*  __macro(hipblasCgemv)                    \
  __macro(hipblasZgemv)                    \
  __macro(hipblasSgbmv)                    \
  __macro(hipblasDgbmv)                    \
  __macro(hipblasCgbmv)                    \
  __macro(hipblasZgbmv)                    \
  __macro(hipblasStrmv)                    \
  __macro(hipblasDtrmv)                    \
  __macro(hipblasCtrmv)                    \
  __macro(hipblasZtrmv)                    \
  __macro(hipblasStbmv)                    \
  __macro(hipblasDtbmv)                    \
  __macro(hipblasCtbmv)                    \
  __macro(hipblasZtbmv)                    \
  __macro(hipblasStpmv)                    \
  __macro(hipblasDtpmv)                    \
  __macro(hipblasCtpmv)                    \
  __macro(hipblasZtpmv)                    \
  __macro(hipblasStrsv)                    \
  __macro(hipblasDtrsv)                    \
  __macro(hipblasCtrsv)                    \
  __macro(hipblasZtrsv)                    \
  __macro(hipblasStpsv)                    \
  __macro(hipblasDtpsv)                    \
  __macro(hipblasCtpsv)                    \
  __macro(hipblasZtpsv)                    \
  __macro(hipblasStbsv)                    \
  __macro(hipblasDtbsv)                    \
  __macro(hipblasCtbsv)                    \
  __macro(hipblasZtbsv)                    \
  __macro(hipblasSsymv)                    \
  __macro(hipblasDsymv)                    \
  __macro(hipblasCsymv)                    \
  __macro(hipblasZsymv)                    \
  __macro(hipblasChemv)                    \
  __macro(hipblasZhemv)                    \
  __macro(hipblasSsbmv)                    \
  __macro(hipblasDsbmv)                    \
  __macro(hipblasChbmv)                    \
  __macro(hipblasZhbmv)                    \
  __macro(hipblasSspmv)                    \
  __macro(hipblasDspmv)                    \
  __macro(hipblasChpmv)                    \
  __macro(hipblasZhpmv)                  */  \
  __macro(hipblasSger)                     \
/*  __macro(hipblasDger)                     \
  __macro(hipblasCgeru)                    \
  __macro(hipblasCgerc)                    \
  __macro(hipblasZgeru)                    \
  __macro(hipblasZgerc)                    \
  __macro(hipblasSsyr)                     \
  __macro(hipblasDsyr)                     \
  __macro(hipblasCsyr)                     \
  __macro(hipblasZsyr)                     \
  __macro(hipblasCher)                     \
  __macro(hipblasZher)                     \
  __macro(hipblasSspr)                     \
  __macro(hipblasDspr)                     \
  __macro(hipblasChpr)                     \
  __macro(hipblasZhpr)                     \
  __macro(hipblasSsyr2)                    \
  __macro(hipblasDsyr2)                    \
  __macro(hipblasCsyr2)                    \
  __macro(hipblasZsyr2)                    \
  __macro(hipblasCher2)                    \
  __macro(hipblasZher2)                    \
  __macro(hipblasSspr2)                    \
  __macro(hipblasDspr2)                    \
  __macro(hipblasChpr2)                    \
  __macro(hipblasZhpr2)                  */  \
  __macro(hipblasSgemm)                    \
  __macro(hipblasDgemm)                    \
  __macro(hipblasCgemm)                    \
  __macro(hipblasZgemm)                    \
/*  __macro(hipblasSsyrk)                    \
  __macro(hipblasDsyrk)                    \
  __macro(hipblasCsyrk)                    \
  __macro(hipblasZsyrk)                    \
  __macro(hipblasCherk)                    \
  __macro(hipblasZherk)                    \
  __macro(hipblasSsyr2k)                   \
  __macro(hipblasDsyr2k)                   \
  __macro(hipblasCsyr2k)                   \
  __macro(hipblasZsyr2k)                   \
  __macro(hipblasCher2k)                   \
  __macro(hipblasZher2k)                   \
  __macro(hipblasSsyrkx)                   \
  __macro(hipblasDsyrkx)                   \
  __macro(hipblasCsyrkx)                   \
  __macro(hipblasZsyrkx)                   \
  __macro(hipblasCherkx)                   \
  __macro(hipblasZherkx)                   \
  __macro(hipblasSsymm)                    \
  __macro(hipblasDsymm)                    \
  __macro(hipblasCsymm)                    \
  __macro(hipblasZsymm)                    \
  __macro(hipblasChemm)                    \
  __macro(hipblasZhemm)                    \
  __macro(hipblasStrsm)                    \
  __macro(hipblasDtrsm)                    \
  __macro(hipblasCtrsm)                    \
  __macro(hipblasZtrsm)                    \
  __macro(hipblasStrmm)                    \
  __macro(hipblasDtrmm)                    \
  __macro(hipblasCtrmm)                    \
  __macro(hipblasZtrmm)                    \
  __macro(hipblasSgeam)                    \
  __macro(hipblasDgeam)                    \
  __macro(hipblasCgeam)                    \
  __macro(hipblasZgeam)                    \
  __macro(hipblasSdgmm)                    \
  __macro(hipblasDdgmm)                    \
  __macro(hipblasCdgmm)                    \
  __macro(hipblasZdgmm)*/

PERFTOOLS_GPUTOOLS_HIPBLAS_V2_WRAP(hipblasCreate)
PERFTOOLS_GPUTOOLS_HIPBLAS_V2_WRAP(hipblasDestroy)
PERFTOOLS_GPUTOOLS_HIPBLAS_V2_WRAP(hipblasSetStream)
//PERFTOOLS_GPUTOOLS_HIPBLAS_V2_WRAP(hipblasSetPointerMode)
//PERFTOOLS_GPUTOOLS_HIPBLAS_V2_WRAP(hipblasGetPointerMode)
PERFTOOLS_GPUTOOLS_HIPBLAS_WRAP(hipblasSgemmBatched)
PERFTOOLS_GPUTOOLS_HIPBLAS_WRAP(hipblasDgemmBatched)
PERFTOOLS_GPUTOOLS_HIPBLAS_WRAP(hipblasCgemmBatched)
PERFTOOLS_GPUTOOLS_HIPBLAS_WRAP(hipblasZgemmBatched)
HIPBLAS_BLAS_ROUTINE_EACH(PERFTOOLS_GPUTOOLS_HIPBLAS_V2_WRAP)

#if CUDA_VERSION >= 7050
//PERFTOOLS_GPUTOOLS_HIPBLAS_WRAP(hipblasSgemmEx)
#endif

}  // namespace dynload

static string ToString(hipblasStatus_t status) {
  switch (status) {
    case HIPBLAS_STATUS_SUCCESS:
      return "HIPBLAS_STATUS_SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED:
      return "HIPBLAS_STATUS_NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED:
      return "HIPBLAS_STATUS_ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE:
      return "HIPBLAS_STATUS_INVALID_VALUE";
    //case HIPBLAS_STATUS_ARCH_MISMATCH:
      //return "HIPBLAS_STATUS_ARCH_MISMATCH";
    case HIPBLAS_STATUS_MAPPING_ERROR:
      return "HIPBLAS_STATUS_MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED:
      return "HIPBLAS_STATUS_EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR:
      return "HIPBLAS_STATUS_INTERNAL_ERROR";
    default:
      return port::StrCat("<invalid hipblas status: ", status, ">");
  }
}

// HIPBLAS has interfaces that permit pointers to be passed from either the host
// memory space or the device memory space; however, you must instruct it as to
// which address space those pointers are in with hipblasSetPointerMode.
//
// This helper sets the HIPBLAS pointer mode to a desired value for a HIPBLAS call
// you are about to perform in a given scope.
//
// The prior HIPBLAS pointer mode is retained and restored when this object goes
// out of scope.
/*class ScopedCublasPointerMode {
 public:
  // Note that, because the setting of the hipblas pointer mode is fallible,
  // construction of this scoped datatype must be paired with a call to
  // Init().
  //
  // Parameters:
  //  handle: The hipblas library handle to act upon in setting the pointer mode.
  explicit ScopedCublasPointerMode(CUDAExecutor *parent, hipblasHandle_t handle)
      : parent_(parent), handle_(handle), ok_(false) {}

  // Attempts the switch to the requested scoped pointer mode, new_mode.
  //
  // Note that when false is returned, an appropriate error has already been
  // logged.
  bool Init(hipblasPointerMode_t new_mode) {
    hipblasStatus_t ret =
        dynload::hipblasGetPointerMode_v2(parent_, handle_, &old_mode_);
    if (ret != HIPBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "failed to get old hipblas pointer mode: " << ToString(ret);
      return ok_ = false;
    }

    ret = dynload::hipblasSetPointerMode_v2(parent_, handle_, new_mode);
    if (ret != HIPBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "failed to set new hipblas pointer mode: " << ToString(ret);
      return ok_ = false;
    }

    return ok_ = true;
  }

  // Switches back to the prior pointer mode, if the switch operation was
  // successful in the first place.
  ~ScopedCublasPointerMode() {
    if (ok_) {
      hipblasStatus_t ret =
          dynload::hipblasSetPointerMode_v2(parent_, handle_, old_mode_);
      if (ret != HIPBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set former hipblas pointer mode: "
                   << ToString(ret);
      }
    }
  }

 private:
  CUDAExecutor *parent_;   // Executor establishing this pointer mode for.
  hipblasHandle_t handle_;  // Handle to the HIPBLAS instance of interest.
  hipblasPointerMode_t old_mode_;  // Prior HIPBLAS pointer mode, to be restored.
  bool ok_;                       // Whether the change was successful.
};*/

bool CUDABlas::Init() {
  hipblasStatus_t ret = dynload::hipblasCreate(parent_, &blas_);
  if (ret != HIPBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to create hipblas handle: " << ToString(ret);
    return false;
  }

  return true;
}

CUDABlas::CUDABlas(cuda::CUDAExecutor *parent)
    : parent_(CHECK_NOTNULL(parent)), blas_(nullptr) {}

CUDABlas::~CUDABlas() {
  if (blas_ != nullptr) {
    dynload::hipblasDestroy(parent_, blas_);
  }
}

bool CUDABlas::SetStream(Stream *stream) {
  CHECK(stream != nullptr);
  CHECK(AsCUDAStreamValue(stream) != nullptr);
  CHECK(blas_ != nullptr);
  hipblasStatus_t ret =
      dynload::hipblasSetStream(parent_, blas_, AsCUDAStreamValue(stream));
  if (ret != HIPBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set stream for HIPBLAS calls: " << ToString(ret);
    return false;
  }

  return true;
}

namespace {

// Helper functions transforming blas arguments into HIPBLAS arguments.

hipblasOperation_t CUDABlasTranspose(blas::Transpose trans) {
  switch (trans) {
    case blas::Transpose::kNoTranspose:
      return HIPBLAS_OP_N;
    case blas::Transpose::kTranspose:
      return HIPBLAS_OP_T;
    case blas::Transpose::kConjugateTranspose:
      return HIPBLAS_OP_C;
    default:
      LOG(FATAL) << "Invalid value of blas::Transpose.";
  }
}

/*hipblasFillMode_t CUDABlasUpperLower(blas::UpperLower uplo) {
  switch (uplo) {
    case blas::UpperLower::kUpper:
      return HIPBLAS_FILL_MODE_UPPER;
    case blas::UpperLower::kLower:
      return HIPBLAS_FILL_MODE_LOWER;
    default:
      LOG(FATAL) << "Invalid value of blas::UpperLower.";
  }
}*/

/*hipblasDiagType_t CUDABlasDiagonal(blas::Diagonal diag) {
  switch (diag) {
    case blas::Diagonal::kUnit:
      return HIPBLAS_DIAG_UNIT;
    case blas::Diagonal::kNonUnit:
      return HIPBLAS_DIAG_NON_UNIT;
    default:
      LOG(FATAL) << "Invalid value of blas::Diagonal.";
  }
}*/

/*hipblasSideMode_t CUDABlasSide(blas::Side side) {
  switch (side) {
    case blas::Side::kLeft:
      return HIPBLAS_SIDE_LEFT;
    case blas::Side::kRight:
      return HIPBLAS_SIDE_RIGHT;
    default:
      LOG(FATAL) << "Invalid value of blas::Side.";
  }
}*/

}  // namespace

template <typename FuncT, typename... Args>
bool CUDABlas::DoBlasInternal(FuncT hipblas_func, Stream *stream,
                              bool pointer_mode_host, Args... args) {
  mutex_lock lock{mu_};

  CHECK(blas_ != nullptr);
  if (!SetStream(stream)) {
    return false;
  }

  /*ScopedCublasPointerMode pointer_mode{parent_, blas_};
  if (!pointer_mode.Init(pointer_mode_host ? HIPBLAS_POINTER_MODE_HOST
                                           : HIPBLAS_POINTER_MODE_DEVICE)) {
    return false;
  }*/

  hipblasStatus_t ret = hipblas_func(parent_, blas_, args...);
  if (ret != HIPBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to run HIPBLAS routine " << hipblas_func.kName << ": "
               << ToString(ret);
    return false;
  }

  return true;
}

bool CUDABlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *result) {
  return DoBlasInternal(dynload::hipblasSasum, stream,
                        false /* = pointer_mode_host */, elem_count,
                        const_cast<float*>(CUDAMemory(x)), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) {
  return DoBlasInternal(dynload::hipblasDasum, stream,
                        false /* = pointer_mode_host */, elem_count,
                        const_cast<double*>(CUDAMemory(x)), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<float> *result) {
  return false;//DoBlasInternal(
      //dynload::hipblasScasum, stream, false /* = pointer_mode_host */,
      //elem_count, CUDAComplex(CUDAMemory(x)), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<double> *result) {
  return false;//DoBlasInternal(
      //dynload::hipblasDzasum, stream, false /* = pointer_mode_host */,
      //elem_count, CUDAComplex(CUDAMemory(x)), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasAxpy(Stream *stream, uint64 elem_count, float alpha,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(dynload::hipblasSaxpy, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        CUDAMemory(x), incx, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasAxpy(Stream *stream, uint64 elem_count, double alpha,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(dynload::hipblasDaxpy, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        CUDAMemory(x), incx, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasAxpy(Stream *stream, uint64 elem_count,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return false;//DoBlasInternal(dynload::hipblasCaxpy, stream,
                        //true /* = pointer_mode_host */, elem_count,
                        //CUDAComplex(&alpha), CUDAComplex(CUDAMemory(x)), incx,
                        //CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasAxpy(Stream *stream, uint64 elem_count,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return false;//DoBlasInternal(dynload::hipblasZaxpy, stream,
                        //true /* = pointer_mode_host */, elem_count,
                        //CUDAComplex(&alpha), CUDAComplex(CUDAMemory(x)), incx,
                        //CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(dynload::hipblasScopy, stream,
                        true /* = pointer_mode_host */, elem_count,
                        CUDAMemory(x), incx, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(dynload::hipblasDcopy, stream,
                        true /* = pointer_mode_host */, elem_count,
                        CUDAMemory(x), incx, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return false;//DoBlasInternal(dynload::hipblasCcopy, stream,
                        //true /* = pointer_mode_host */, elem_count,
                        //CUDAComplex(CUDAMemory(x)), incx,
                        //CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return false;//DoBlasInternal(dynload::hipblasZcopy, stream,
                        //true /* = pointer_mode_host */, elem_count,
                        //CUDAComplex(CUDAMemory(x)), incx,
                        //CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasDot(Stream *stream, uint64 elem_count,
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *result) {
  return DoBlasInternal(
      dynload::hipblasSdot, stream, false /* = pointer_mode_host */, elem_count,
      CUDAMemory(x), incx, CUDAMemory(y), incy, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasDot(Stream *stream, uint64 elem_count,
                         const DeviceMemory<double> &x, int incx,
                         const DeviceMemory<double> &y, int incy,
                         DeviceMemory<double> *result) {
  return DoBlasInternal(
      dynload::hipblasDdot, stream, false /* = pointer_mode_host */, elem_count,
      CUDAMemory(x), incx, CUDAMemory(y), incy, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasDotc(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *result) {
  return false;//DoBlasInternal(
      //dynload::hipblasCdotc, stream, false /* = pointer_mode_host */, elem_count,
      //CUDAComplex(CUDAMemory(x)), incx, CUDAComplex(CUDAMemory(y)), incy,
      //CUDAComplex(CUDAMemoryMutable(result)));
}

bool CUDABlas::DoBlasDotc(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *result) {
  return false;//DoBlasInternal(
      //dynload::hipblasZdotc, stream, false /* = pointer_mode_host */, elem_count,
      //CUDAComplex(CUDAMemory(x)), incx, CUDAComplex(CUDAMemory(y)), incy,
      //CUDAComplex(CUDAMemoryMutable(result)));
}

bool CUDABlas::DoBlasDotu(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *result) {
  return false;//DoBlasInternal(
      //dynload::hipblasCdotu, stream, false /* = pointer_mode_host */, elem_count,
      //CUDAComplex(CUDAMemory(x)), incx, CUDAComplex(CUDAMemory(y)), incy,
      //CUDAComplex(CUDAMemoryMutable(result)));
}

bool CUDABlas::DoBlasDotu(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *result) {
  return false;//DoBlasInternal(
      //dynload::hipblasZdotu, stream, false /* = pointer_mode_host */, elem_count,
      //CUDAComplex(CUDAMemory(x)), incx, CUDAComplex(CUDAMemory(y)), incy,
      //CUDAComplex(CUDAMemoryMutable(result)));
}

bool CUDABlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *result) {
  return false;//DoBlasInternal(dynload::hipblasSnrm2, stream,
                        //false /* = pointer_mode_host */, elem_count,
                        //CUDAMemory(x), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) {
  return false;//DoBlasInternal(dynload::hipblasDnrm2, stream,
                        //false /* = pointer_mode_host */, elem_count,
                        //CUDAMemory(x), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<float> *result) {
  return false;//DoBlasInternal(
      //dynload::hipblasScnrm2, stream, false /* = pointer_mode_host */,
      //elem_count, CUDAComplex(CUDAMemory(x)), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<double> *result) {
  return false;//DoBlasInternal(
      //dynload::hipblasDznrm2, stream, false /* = pointer_mode_host */,
      //elem_count, CUDAComplex(CUDAMemory(x)), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<float> *x, int incx,
                         DeviceMemory<float> *y, int incy, float c, float s) {
  return false;//DoBlasInternal(
      //dynload::hipblasSrot, stream, true /* = pointer_mode_host */, elem_count,
      //CUDAMemoryMutable(x), incx, CUDAMemoryMutable(y), incy, &c, &s);
}

bool CUDABlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<double> *x, int incx,
                         DeviceMemory<double> *y, int incy, double c,
                         double s) {
  return false;//DoBlasInternal(
      //dynload::hipblasDrot, stream, true /* = pointer_mode_host */, elem_count,
      //CUDAMemoryMutable(x), incx, CUDAMemoryMutable(y), incy, &c, &s);
}

bool CUDABlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<std::complex<float>> *x, int incx,
                         DeviceMemory<std::complex<float>> *y, int incy,
                         float c, float s) {
  return false;//DoBlasInternal(dynload::hipblasCsrot, stream,
                        //true /* = pointer_mode_host */, elem_count,
                        //CUDAComplex(CUDAMemoryMutable(x)), incx,
                        //CUDAComplex(CUDAMemoryMutable(y)), incy, &c, &s);
}

bool CUDABlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<std::complex<double>> *x, int incx,
                         DeviceMemory<std::complex<double>> *y, int incy,
                         double c, double s) {
  return false;//DoBlasInternal(dynload::hipblasZdrot, stream,
                        //true /* = pointer_mode_host */, elem_count,
                        //CUDAComplex(CUDAMemoryMutable(x)), incx,
                        //CUDAComplex(CUDAMemoryMutable(y)), incy, &c, &s);
}

bool CUDABlas::DoBlasRotg(Stream *stream, DeviceMemory<float> *a,
                          DeviceMemory<float> *b, DeviceMemory<float> *c,
                          DeviceMemory<float> *s) {
  return false;//DoBlasInternal(dynload::hipblasSrotg, stream,
                        //false /* = pointer_mode_host */, CUDAMemoryMutable(a),
                        //CUDAMemoryMutable(b), CUDAMemoryMutable(c),
                        //CUDAMemoryMutable(s));
}

bool CUDABlas::DoBlasRotg(Stream *stream, DeviceMemory<double> *a,
                          DeviceMemory<double> *b, DeviceMemory<double> *c,
                          DeviceMemory<double> *s) {
  return false;//DoBlasInternal(dynload::hipblasDrotg, stream,
                        //false /* = pointer_mode_host */,
                        //CUDAComplex(CUDAMemoryMutable(a)), CUDAMemoryMutable(b),
                        //CUDAMemoryMutable(c), CUDAMemoryMutable(s));
}

bool CUDABlas::DoBlasRotg(Stream *stream, DeviceMemory<std::complex<float>> *a,
                          DeviceMemory<std::complex<float>> *b,
                          DeviceMemory<float> *c,
                          DeviceMemory<std::complex<float>> *s) {
  return false;//DoBlasInternal(
      //dynload::hipblasCrotg, stream, false /* = pointer_mode_host */,
      //CUDAComplex(CUDAMemoryMutable(a)), CUDAComplex(CUDAMemoryMutable(b)),
      //CUDAComplex(CUDAMemoryMutable(c)), CUDAComplex(CUDAMemoryMutable(s)));
}

bool CUDABlas::DoBlasRotg(Stream *stream, DeviceMemory<std::complex<double>> *a,
                          DeviceMemory<std::complex<double>> *b,
                          DeviceMemory<double> *c,
                          DeviceMemory<std::complex<double>> *s) {
  return false;//DoBlasInternal(
      //dynload::hipblasZrotg, stream, false /* = pointer_mode_host */,
      //CUDAComplex(CUDAMemoryMutable(a)), CUDAComplex(CUDAMemoryMutable(b)),
      //CUDAComplex(CUDAMemoryMutable(c)), CUDAComplex(CUDAMemoryMutable(s)));
}

bool CUDABlas::DoBlasRotm(Stream *stream, uint64 elem_count,
                          DeviceMemory<float> *x, int incx,
                          DeviceMemory<float> *y, int incy,
                          const DeviceMemory<float> &param) {
  return false;//DoBlasInternal(dynload::hipblasSrotm, stream,
                        //false /* = pointer_mode_host */, elem_count,
                        //CUDAMemoryMutable(x), incx, CUDAMemoryMutable(y), incy,
                        //CUDAMemory(param));
}

bool CUDABlas::DoBlasRotm(Stream *stream, uint64 elem_count,
                          DeviceMemory<double> *x, int incx,
                          DeviceMemory<double> *y, int incy,
                          const DeviceMemory<double> &param) {
  return false;//DoBlasInternal(dynload::hipblasDrotm, stream,
                        //false /* = pointer_mode_host */, elem_count,
                        //CUDAMemoryMutable(x), incx, CUDAMemoryMutable(y), incy,
                        //CUDAMemory(param));
}

bool CUDABlas::DoBlasRotmg(Stream *stream, DeviceMemory<float> *d1,
                           DeviceMemory<float> *d2, DeviceMemory<float> *x1,
                           const DeviceMemory<float> &y1,
                           DeviceMemory<float> *param) {
  return false;//DoBlasInternal(dynload::hipblasSrotmg, stream,
                        //false /* = pointer_mode_host */, CUDAMemoryMutable(d1),
                        //CUDAMemoryMutable(d2), CUDAMemoryMutable(x1),
                        //CUDAMemory(y1), CUDAMemoryMutable(param));
}

bool CUDABlas::DoBlasRotmg(Stream *stream, DeviceMemory<double> *d1,
                           DeviceMemory<double> *d2, DeviceMemory<double> *x1,
                           const DeviceMemory<double> &y1,
                           DeviceMemory<double> *param) {
  return false;//DoBlasInternal(dynload::hipblasDrotmg, stream,
                        //false /* = pointer_mode_host */, CUDAMemoryMutable(d1),
                        //CUDAMemoryMutable(d2), CUDAMemoryMutable(x1),
                        //CUDAMemory(y1), CUDAMemoryMutable(param));
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64 elem_count, float alpha,
                          DeviceMemory<float> *x, int incx) {
  return DoBlasInternal(dynload::hipblasSscal, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64 elem_count, double alpha,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(dynload::hipblasDscal, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64 elem_count, float alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(
      dynload::hipblasCsscal, stream, true /* = pointer_mode_host */, elem_count,
      CUDAComplex(&alpha), CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64 elem_count, double alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(
      dynload::hipblasZdscal, stream, true /* = pointer_mode_host */, elem_count,
      CUDAComplex(&alpha), CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64 elem_count,
                          std::complex<float> alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(
      dynload::hipblasCscal, stream, true /* = pointer_mode_host */, elem_count,
      CUDAComplex(&alpha), CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64 elem_count,
                          std::complex<double> alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(
      dynload::hipblasZscal, stream, true /* = pointer_mode_host */, elem_count,
      CUDAComplex(&alpha), CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<float> *x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return false;//DoBlasInternal(dynload::hipblasSswap, stream,
                        //true /* = pointer_mode_host */, elem_count,
                        //CUDAMemoryMutable(x), incx, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<double> *x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return false;//DoBlasInternal(dynload::hipblasDswap, stream,
                        //true /* = pointer_mode_host */, elem_count,
                        //CUDAMemoryMutable(x), incx, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<std::complex<float>> *x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return false;//DoBlasInternal(dynload::hipblasCswap, stream,
                        //true /* = pointer_mode_host */, elem_count,
                        //CUDAComplex(CUDAMemoryMutable(x)), incx,
                        //CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<std::complex<double>> *x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return false;//DoBlasInternal(dynload::hipblasZswap, stream,
                        //true /* = pointer_mode_host */, elem_count,
                        //CUDAComplex(CUDAMemoryMutable(x)), incx,
                        //CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<float> &x, int incx,
                           DeviceMemory<int> *result) {
  return false;//DoBlasInternal(dynload::hipblasIsamax, stream,
                        //false /* = pointer_mode_host */, elem_count,
                        //CUDAMemory(x), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<double> &x, int incx,
                           DeviceMemory<int> *result) {
  return false;//DoBlasInternal(dynload::hipblasIdamax, stream,
                        //false /* = pointer_mode_host */, elem_count,
                        //CUDAMemory(x), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<float>> &x, int incx,
                           DeviceMemory<int> *result) {
  return false;//DoBlasInternal(
      //dynload::hipblasIcamax, stream, false /* = pointer_mode_host */,
      //elem_count, CUDAComplex(CUDAMemory(x)), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<double>> &x,
                           int incx, DeviceMemory<int> *result) {
  return false;//DoBlasInternal(
      //dynload::hipblasIzamax, stream, false /* = pointer_mode_host */,
      //elem_count, CUDAComplex(CUDAMemory(x)), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<float> &x, int incx,
                           DeviceMemory<int> *result) {
  return false;//DoBlasInternal(
      //dynload::hipblasIsamin, stream, false /* = pointer_mode_host */,
      //elem_count, CUDAComplex(CUDAMemory(x)), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<double> &x, int incx,
                           DeviceMemory<int> *result) {
  return false;//DoBlasInternal(
      //dynload::hipblasIdamin, stream, false /* = pointer_mode_host */,
      //elem_count, CUDAComplex(CUDAMemory(x)), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<float>> &x, int incx,
                           DeviceMemory<int> *result) {
  return false;//DoBlasInternal(
      //dynload::hipblasIcamin, stream, false /* = pointer_mode_host */,
      //elem_count, CUDAComplex(CUDAMemory(x)), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<double>> &x,
                           int incx, DeviceMemory<int> *result) {
  return false;//DoBlasInternal(
      //dynload::hipblasIzamin, stream, false /* = pointer_mode_host */,
      //elem_count, CUDAComplex(CUDAMemory(x)), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  return false;//DoBlasInternal(
      //dynload::hipblasSgbmv, stream, true /* = pointer_mode_host */,
      //CUDABlasTranspose(trans), m, n, kl, ku, &alpha, CUDAMemory(a), lda,
      //CUDAMemory(x), incx, &beta, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return false;//DoBlasInternal(
      //dynload::hipblasDgbmv, stream, true /* = pointer_mode_host */,
      //CUDABlasTranspose(trans), m, n, kl, ku, &alpha, CUDAMemory(a), lda,
      //CUDAMemory(x), incx, &beta, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return false;//DoBlasInternal(
      //dynload::hipblasCgbmv, stream, true /* = pointer_mode_host */,
      //CUDABlasTranspose(trans), m, n, kl, ku, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(x)), incx,
      //CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return false;//DoBlasInternal(
      //dynload::hipblasZgbmv, stream, true /* = pointer_mode_host */,
      //CUDABlasTranspose(trans), m, n, kl, ku, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(x)), incx,
      //CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(
      dynload::hipblasSgemv, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(trans), m, n, &alpha, const_cast<float*>(CUDAMemory(a)), lda, const_cast<float*>(CUDAMemory(x)),
      incx, &beta, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, double alpha, const DeviceMemory<double> &a,
                          int lda, const DeviceMemory<double> &x, int incx,
                          double beta, DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(
      dynload::hipblasDgemv, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(trans), m, n, &alpha, CUDAMemory(a), lda, CUDAMemory(x),
      incx, &beta, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return false;//DoBlasInternal(
      //dynload::hipblasCgemv, stream, true /* = pointer_mode_host */,
      //CUDABlasTranspose(trans), m, n, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(x)), incx,
      //CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return false;//DoBlasInternal(
      //dynload::hipblasZgemv, stream, true /* = pointer_mode_host */,
      //CUDABlasTranspose(trans), m, n, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(x)), incx,
      //CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasGer(Stream *stream, uint64 m, uint64 n, float alpha,
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *a, int lda) {
  return DoBlasInternal(
      dynload::hipblasSger, stream, true /* = pointer_mode_host */, m, n, &alpha,
      CUDAMemory(x), incx, CUDAMemory(y), incy, CUDAMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasGer(Stream *stream, uint64 m, uint64 n, double alpha,
                         const DeviceMemory<double> &x, int incx,
                         const DeviceMemory<double> &y, int incy,
                         DeviceMemory<double> *a, int lda) {
  return false;//DoBlasInternal(
      //dynload::hipblasDger, stream, true /* = pointer_mode_host */, m, n, &alpha,
      //CUDAMemory(x), incx, CUDAMemory(y), incy, CUDAMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasGerc(Stream *stream, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  return false;//DoBlasInternal(
      //dynload::hipblasCgerc, stream, true /* = pointer_mode_host */, m, n,
      //CUDAComplex(&alpha), CUDAComplex(CUDAMemory(x)), incx,
      //CUDAComplex(CUDAMemory(y)), incy, CUDAComplex(CUDAMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasGerc(Stream *stream, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  return false;//DoBlasInternal(
      //dynload::hipblasZgerc, stream, true /* = pointer_mode_host */, m, n,
      //CUDAComplex(&alpha), CUDAComplex(CUDAMemory(x)), incx,
      //CUDAComplex(CUDAMemory(y)), incy, CUDAComplex(CUDAMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasGeru(Stream *stream, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  return false;//DoBlasInternal(
      //dynload::hipblasCgeru, stream, true /* = pointer_mode_host */, m, n,
      //CUDAComplex(&alpha), CUDAComplex(CUDAMemory(x)), incx,
      //CUDAComplex(CUDAMemory(y)), incy, CUDAComplex(CUDAMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasGeru(Stream *stream, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  return false;//DoBlasInternal(
      //dynload::hipblasZgeru, stream, true /* = pointer_mode_host */, m, n,
      //CUDAComplex(&alpha), CUDAComplex(CUDAMemory(x)), incx,
      //CUDAComplex(CUDAMemory(y)), incy, CUDAComplex(CUDAMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return false;//DoBlasInternal(
      //dynload::hipblasChbmv, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), n, k, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(x)), incx,
      //CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return false;//DoBlasInternal(
      //dynload::hipblasZhbmv, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), n, k, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(x)), incx,
      //CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return false;//DoBlasInternal(
      //dynload::hipblasChemv, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), n, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(x)), incx,
      //CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return false;//DoBlasInternal(
      //dynload::hipblasZhemv, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), n, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(x)), incx,
      //CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha,
                         const DeviceMemory<std::complex<float>> &x, int incx,
                         DeviceMemory<std::complex<float>> *a, int lda) {
  return false;//DoBlasInternal(
      //dynload::hipblasCher, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), n, &alpha, CUDAComplex(CUDAMemory(x)), incx,
      //CUDAComplex(CUDAMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha,
                         const DeviceMemory<std::complex<double>> &x, int incx,
                         DeviceMemory<std::complex<double>> *a, int lda) {
  return false;//DoBlasInternal(
      //dynload::hipblasZher, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), n, &alpha, CUDAComplex(CUDAMemory(x)), incx,
      //CUDAComplex(CUDAMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  return false;//DoBlasInternal(
      //dynload::hipblasCher2, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), n, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(x)), incx, CUDAComplex(CUDAMemory(y)), incy,
      //CUDAComplex(CUDAMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  return false;//DoBlasInternal(
      //dynload::hipblasZher2, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), n, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(x)), incx, CUDAComplex(CUDAMemory(y)), incy,
      //CUDAComplex(CUDAMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &ap,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return false;//DoBlasInternal(
      //dynload::hipblasChpmv, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), n, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(ap)), CUDAComplex(CUDAMemory(x)), incx,
      //CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &ap,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return false;//DoBlasInternal(
      //dynload::hipblasZhpmv, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), n, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(ap)), CUDAComplex(CUDAMemory(x)), incx,
      //CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha,
                         const DeviceMemory<std::complex<float>> &x, int incx,
                         DeviceMemory<std::complex<float>> *ap) {
  return false;//DoBlasInternal(
      //dynload::hipblasChpr, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), n, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(x)), incx, CUDAComplex(CUDAMemoryMutable(ap)));
}

bool CUDABlas::DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha,
                         const DeviceMemory<std::complex<double>> &x, int incx,
                         DeviceMemory<std::complex<double>> *ap) {
  return false;//DoBlasInternal(
      //dynload::hipblasZhpr, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), n, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(x)), incx, CUDAComplex(CUDAMemoryMutable(ap)));
}

bool CUDABlas::DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *ap) {
  return false;//DoBlasInternal(
      //dynload::hipblasChpr2, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), n, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(x)), incx, CUDAComplex(CUDAMemory(y)), incy,
      //CUDAComplex(CUDAMemoryMutable(ap)));
}

bool CUDABlas::DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *ap) {
  return false;//DoBlasInternal(
      //dynload::hipblasZhpr2, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), n, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(x)), incx, CUDAComplex(CUDAMemory(y)), incy,
      //CUDAComplex(CUDAMemoryMutable(ap)));
}

bool CUDABlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  return false;//DoBlasInternal(
      //dynload::hipblasSsbmv, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), n, k, &alpha, CUDAMemory(a), lda, CUDAMemory(x),
      //incx, &beta, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, double alpha, const DeviceMemory<double> &a,
                          int lda, const DeviceMemory<double> &x, int incx,
                          double beta, DeviceMemory<double> *y, int incy) {
  return false;//DoBlasInternal(
      //dynload::hipblasDsbmv, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), n, k, &alpha, CUDAMemory(a), lda, CUDAMemory(x),
      //incx, &beta, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &ap,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  return false;//DoBlasInternal(dynload::hipblasSspmv, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(ap),
                        //CUDAMemory(x), incx, &beta, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &ap,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return false;//DoBlasInternal(dynload::hipblasDspmv, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(ap),
                        //CUDAMemory(x), incx, &beta, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha, const DeviceMemory<float> &x, int incx,
                         DeviceMemory<float> *ap) {
  return false;//DoBlasInternal(dynload::hipblasSspr, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(x),
                        //incx, CUDAMemoryMutable(ap));
}

bool CUDABlas::DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha, const DeviceMemory<double> &x, int incx,
                         DeviceMemory<double> *ap) {
  return false;//DoBlasInternal(dynload::hipblasDspr, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(x),
                        //incx, CUDAMemoryMutable(ap));
}

bool CUDABlas::DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &x, int incx,
                          const DeviceMemory<float> &y, int incy,
                          DeviceMemory<float> *ap) {
  return false;//DoBlasInternal(dynload::hipblasSspr2, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(x),
                        //incx, CUDAMemory(y), incy, CUDAMemoryMutable(ap));
}

bool CUDABlas::DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &x, int incx,
                          const DeviceMemory<double> &y, int incy,
                          DeviceMemory<double> *ap) {
  return false;//DoBlasInternal(dynload::hipblasDspr2, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(x),
                        //incx, CUDAMemory(y), incy, CUDAMemoryMutable(ap));
}

bool CUDABlas::DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  return false;//DoBlasInternal(dynload::hipblasSsymv, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(a), lda,
                        //CUDAMemory(x), incx, &beta, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return false;//DoBlasInternal(dynload::hipblasDsymv, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(a), lda,
                        //CUDAMemory(x), incx, &beta, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha, const DeviceMemory<float> &x, int incx,
                         DeviceMemory<float> *a, int lda) {
  return false;//DoBlasInternal(dynload::hipblasSsyr, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(x),
                        //incx, CUDAMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha, const DeviceMemory<double> &x, int incx,
                         DeviceMemory<double> *a, int lda) {
  return false;//DoBlasInternal(dynload::hipblasDsyr, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(x),
                        //incx, CUDAMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &x, int incx,
                          const DeviceMemory<float> &y, int incy,
                          DeviceMemory<float> *a, int lda) {
  return false;//DoBlasInternal(dynload::hipblasSsyr2, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(x),
                        //incx, CUDAMemory(y), incy, CUDAMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &x, int incx,
                          const DeviceMemory<double> &y, int incy,
                          DeviceMemory<double> *a, int lda) {
  return false;//DoBlasInternal(dynload::hipblasDsyr2, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(x),
                        //incx, CUDAMemory(y), incy, CUDAMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return false;//DoBlasInternal(dynload::hipblasStbmv, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        //CUDABlasDiagonal(diag), n, k, CUDAMemory(a), lda,
                        //CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return false;//DoBlasInternal(dynload::hipblasDtbmv, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        //CUDABlasDiagonal(diag), n, k, CUDAMemory(a), lda,
                        //CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<float>> &a,
                          int lda, DeviceMemory<std::complex<float>> *x,
                          int incx) {
  return false;//DoBlasInternal(
      //dynload::hipblasCtbmv, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
      //CUDABlasDiagonal(diag), n, k, CUDAComplex(CUDAMemory(a)), lda,
      //CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<double>> &a,
                          int lda, DeviceMemory<std::complex<double>> *x,
                          int incx) {
  return false;//DoBlasInternal(
      //dynload::hipblasZtbmv, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
      //CUDABlasDiagonal(diag), n, k, CUDAComplex(CUDAMemory(a)), lda,
      //CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return false;//DoBlasInternal(dynload::hipblasStbsv, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        //CUDABlasDiagonal(diag), n, k, CUDAMemory(a), lda,
                        //CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return false;//DoBlasInternal(dynload::hipblasDtbsv, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        //CUDABlasDiagonal(diag), n, k, CUDAMemory(a), lda,
                        //CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<float>> &a,
                          int lda, DeviceMemory<std::complex<float>> *x,
                          int incx) {
  return false;//DoBlasInternal(
      //dynload::hipblasCtbsv, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
      //CUDABlasDiagonal(diag), n, k, CUDAComplex(CUDAMemory(a)), lda,
      //CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<double>> &a,
                          int lda, DeviceMemory<std::complex<double>> *x,
                          int incx) {
  return false;//DoBlasInternal(
      //dynload::hipblasZtbsv, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
      //CUDABlasDiagonal(diag), n, k, CUDAComplex(CUDAMemory(a)), lda,
      //CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &ap, DeviceMemory<float> *x,
                          int incx) {
  return false;//DoBlasInternal(
      //dynload::hipblasStpmv, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
      //CUDABlasDiagonal(diag), n, CUDAMemory(ap), CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &ap,
                          DeviceMemory<double> *x, int incx) {
  return false;//DoBlasInternal(
      //dynload::hipblasDtpmv, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
      //CUDABlasDiagonal(diag), n, CUDAMemory(ap), CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &ap,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return false;//DoBlasInternal(dynload::hipblasCtpmv, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        //CUDABlasDiagonal(diag), n, CUDAComplex(CUDAMemory(ap)),
                        //CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &ap,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return false;//DoBlasInternal(dynload::hipblasZtpmv, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        //CUDABlasDiagonal(diag), n, CUDAComplex(CUDAMemory(ap)),
                        //CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &ap, DeviceMemory<float> *x,
                          int incx) {
  return false;//DoBlasInternal(
      //dynload::hipblasStpsv, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
      //CUDABlasDiagonal(diag), n, CUDAMemory(ap), CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &ap,
                          DeviceMemory<double> *x, int incx) {
  return false;//DoBlasInternal(
      //dynload::hipblasDtpsv, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
      //CUDABlasDiagonal(diag), n, CUDAMemory(ap), CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &ap,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return false;//DoBlasInternal(dynload::hipblasCtpsv, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        //CUDABlasDiagonal(diag), n, CUDAComplex(CUDAMemory(ap)),
                        //CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &ap,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return false;//DoBlasInternal(dynload::hipblasZtpsv, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        //CUDABlasDiagonal(diag), n, CUDAComplex(CUDAMemory(ap)),
                        //CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return false;//DoBlasInternal(dynload::hipblasStrmv, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        //CUDABlasDiagonal(diag), n, CUDAMemory(a), lda,
                        //CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return false;//DoBlasInternal(dynload::hipblasDtrmv, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        //CUDABlasDiagonal(diag), n, CUDAMemory(a), lda,
                        //CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return false;//DoBlasInternal(dynload::hipblasCtrmv, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        //CUDABlasDiagonal(diag), n, CUDAComplex(CUDAMemory(a)),
                        //lda, CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return false;//DoBlasInternal(dynload::hipblasZtrmv, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        //CUDABlasDiagonal(diag), n, CUDAComplex(CUDAMemory(a)),
                        //lda, CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return false;//DoBlasInternal(dynload::hipblasStrsv, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        //CUDABlasDiagonal(diag), n, CUDAMemory(a), lda,
                        //CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return false;//DoBlasInternal(dynload::hipblasDtrsv, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        //CUDABlasDiagonal(diag), n, CUDAMemory(a), lda,
                        //CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return false;//DoBlasInternal(dynload::hipblasCtrsv, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        //CUDABlasDiagonal(diag), n, CUDAComplex(CUDAMemory(a)),
                        //lda, CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return false;//DoBlasInternal(dynload::hipblasZtrsv, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        //CUDABlasDiagonal(diag), n, CUDAComplex(CUDAMemory(a)),
                        //lda, CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasGemm(
    Stream *stream, blas::Transpose transa,
    blas::Transpose transb, uint64 m, uint64 n, uint64 k,
    float alpha, const DeviceMemory<Eigen::half> &a, int lda,
    const DeviceMemory<Eigen::half> &b, int ldb, float beta,
    DeviceMemory<Eigen::half> *c, int ldc) {
#if CUDA_VERSION >= 7050
  VLOG(1) << port::Printf(
      "doing HIPBLAS SGEMM: at=%d bt=%d m=%llu n=%llu "
      "k=%llu alpha=%f a=%p lda=%d b=%p ldb=%d beta=%f "
      "c=%p ldc=%d",
      static_cast<int>(transa), static_cast<int>(transb), m, n, k, alpha,
      a.opaque(), lda, b.opaque(), ldb, beta, c->opaque(), ldc);
  if (transa == blas::Transpose::kNoTranspose) {
    if (lda < static_cast<int64>(m)) {
      LOG(WARNING) << "GEMM lda was smaller than m (no transpose case); "
                      "precondition violation";
    }
  } else {
    if (lda < static_cast<int64>(k)) {
      LOG(WARNING) << "GEMM lda (" << lda << ") was smaller than k (" << k
                   << ") (transpose case); precondition violation";
    }
  }
  if (transb == blas::Transpose::kNoTranspose) {
    if (ldb < static_cast<int64>(k)) {
      LOG(WARNING) << "GEMM ldb (" << ldb << ") was smaller than k (" << k
                   << ") (no transpose case); precondition violation";
    }
  } else {
    if (ldb < static_cast<int64>(n)) {
      LOG(WARNING) << "GEMM ldb was smaller than n (transpose case); "
                      "precondition violation";
    }
  }
  // TODO(sesse): Consider supporting the Hgemm interface, which uses half
  // calculations internally (faster on newer devices, such as Pascal and TX1,
  // but less precise).
  return false;//DoBlasInternal(
      //dynload::hipblasSgemmEx, stream, true /* = pointer_mode_host */,
      //CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k, &alpha,
      //CUDAMemory(a), SE_CUDA_DATA_HALF, lda,
      //CUDAMemory(b), SE_CUDA_DATA_HALF, ldb,
      //&beta,
      //CUDAMemoryMutable(c), SE_CUDA_DATA_HALF, ldc);
#else
  LOG(ERROR) << "fp16 sgemm is not implemented in this HIPBLAS version "
             << "(need at least CUDA 7.5)";
  return false;
#endif
}

bool CUDABlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &b, int ldb, float beta,
                          DeviceMemory<float> *c, int ldc) {
  VLOG(1) << port::Printf(
      "doing HIPBLAS SGEMM: at=%d bt=%d m=%llu n=%llu "
      "k=%llu alpha=%f a=%p lda=%d b=%p ldb=%d beta=%f "
      "c=%p ldc=%d",
      static_cast<int>(transa), static_cast<int>(transb), m, n, k, alpha,
      a.opaque(), lda, b.opaque(), ldb, beta, c->opaque(), ldc);
  if (transa == blas::Transpose::kNoTranspose) {
    if (lda < static_cast<int64>(m)) {
      LOG(WARNING) << "GEMM lda was smaller than m (no transpose case); "
                      "precondition violation";
    }
  } else {
    if (lda < static_cast<int64>(k)) {
      LOG(WARNING) << "GEMM lda (" << lda << ") was smaller than k (" << k
                   << ") (transpose case); precondition violation";
    }
  }
  if (transb == blas::Transpose::kNoTranspose) {
    if (ldb < static_cast<int64>(k)) {
      LOG(WARNING) << "GEMM ldb (" << ldb << ") was smaller than k (" << k
                   << ") (no transpose case); precondition violation";
    }
  } else {
    if (ldb < static_cast<int64>(n)) {
      LOG(WARNING) << "GEMM ldb was smaller than n (transpose case); "
                      "precondition violation";
    }
  }
  return DoBlasInternal(
      dynload::hipblasSgemm, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k, &alpha,
      const_cast<float*>(CUDAMemory(a)), lda, const_cast<float*>(CUDAMemory(b)), ldb, &beta, CUDAMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &b, int ldb, double beta,
                          DeviceMemory<double> *c, int ldc) {
  return DoBlasInternal(
      dynload::hipblasDgemm, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k, &alpha,
      CUDAMemory(a), lda, CUDAMemory(b), ldb, &beta, CUDAMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  return DoBlasInternal(
      dynload::hipblasCgemm, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k,
      CUDAComplex(&alpha), const_cast<const hipFloatComplex*>(CUDAComplex(CUDAMemory(a))), lda,
      const_cast<const hipFloatComplex*>(CUDAComplex(CUDAMemory(b))), ldb, CUDAComplex(&beta),
      CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  return DoBlasInternal(
      dynload::hipblasZgemm, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k,
      CUDAComplex(&alpha), CUDAComplex(CUDAMemory(a)), lda,
      CUDAComplex(CUDAMemory(b)), ldb, CUDAComplex(&beta),
      CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

template <typename T, typename FuncT>
port::Status CUDABlas::DoBlasGemmBatchedInternal(
    FuncT hipblas_func, Stream *stream, blas::Transpose transa,
    blas::Transpose transb, uint64 m, uint64 n, uint64 k, T alpha,
    const port::ArraySlice<DeviceMemory<T> *> &a_ptrs_to_wrappers, int lda,
    const port::ArraySlice<DeviceMemory<T> *> &b_ptrs_to_wrappers, int ldb,
    T beta, const port::ArraySlice<DeviceMemory<T> *> &c_ptrs_to_wrappers,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  std::vector<T *> a_raw_ptrs, b_raw_ptrs, c_raw_ptrs;
  for (int i = 0; i < batch_count; ++i) {
    a_raw_ptrs.push_back(static_cast<T *>(a_ptrs_to_wrappers[i]->opaque()));
    b_raw_ptrs.push_back(static_cast<T *>(b_ptrs_to_wrappers[i]->opaque()));
    c_raw_ptrs.push_back(static_cast<T *>(c_ptrs_to_wrappers[i]->opaque()));
  }

  typedef typename CUDAComplexT<T>::type CUDA_T;

  const size_t size = batch_count * sizeof(CUDA_T *);

  // Device-side copy of pointers to matrices.
  DeviceMemory<CUDA_T *> a;
  DeviceMemory<CUDA_T *> b;
  DeviceMemory<CUDA_T *> c;

  // If temporary space is allocated for device-side copies of pointers to
  // matrices, that temporary space should not be freed until this function
  // returns. Although the values for these unique_ptrs are not set here, they
  // are declared at this scope so they will be destroyed when the function
  // returns.
  //
  // If a scratch allocator is provided, these pointers will not be used at all.
  std::unique_ptr<TemporaryDeviceMemory<CUDA_T *>> a_temporary;
  std::unique_ptr<TemporaryDeviceMemory<CUDA_T *>> b_temporary;
  std::unique_ptr<TemporaryDeviceMemory<CUDA_T *>> c_temporary;

  // Decide how to allocate device-side copy of pointers to matrices based on
  // whether a scratch allocator was passed.
  if (scratch_allocator != nullptr) {
    SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> a_bytes,
                        scratch_allocator->AllocateBytes(stream, size));
    SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> b_bytes,
                        scratch_allocator->AllocateBytes(stream, size));
    SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> c_bytes,
                        scratch_allocator->AllocateBytes(stream, size));
    a = DeviceMemory<CUDA_T *>(a_bytes);
    b = DeviceMemory<CUDA_T *>(b_bytes);
    c = DeviceMemory<CUDA_T *>(c_bytes);
  } else {
    SE_ASSIGN_OR_RETURN(a_temporary,
                        stream->AllocateTemporaryArray<CUDA_T *>(batch_count));
    SE_ASSIGN_OR_RETURN(b_temporary,
                        stream->AllocateTemporaryArray<CUDA_T *>(batch_count));
    SE_ASSIGN_OR_RETURN(c_temporary,
                        stream->AllocateTemporaryArray<CUDA_T *>(batch_count));
    a = DeviceMemory<CUDA_T *>(*a_temporary->mutable_device_memory());
    b = DeviceMemory<CUDA_T *>(*b_temporary->mutable_device_memory());
    c = DeviceMemory<CUDA_T *>(*c_temporary->mutable_device_memory());
  }

  if (!stream->ThenMemcpy(&a, a_raw_ptrs.data(), size).ok() ||
      !stream->ThenMemcpy(&b, b_raw_ptrs.data(), size).ok() ||
      !stream->ThenMemcpy(&c, c_raw_ptrs.data(), size).ok()) {
    return port::Status(port::error::INTERNAL,
                        "failed to copy memory from host to device in "
                        "CUDABlas::DoBlasGemmBatched");
  }

  bool ok = DoBlasInternal(
      hipblas_func, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k,
      CUDAComplex(&alpha), const_cast<const CUDA_T **>(CUDAMemory(a)), lda,
      const_cast<const CUDA_T **>(CUDAMemory(b)), ldb, CUDAComplex(&beta),
      const_cast<CUDA_T **>(CUDAMemory(c)), ldc, batch_count);

  if (ok) {
    return port::Status::OK();
  }
  return port::Status(port::error::INTERNAL,
                      "failed BLAS call, see log for details");
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, float alpha,
    const port::ArraySlice<DeviceMemory<float> *> &a_array, int lda,
    const port::ArraySlice<DeviceMemory<float> *> &b_array, int ldb, float beta,
    const port::ArraySlice<DeviceMemory<float> *> &c_array, int ldc,
    int batch_count, ScratchAllocator *scratch_allocator) {
  SE_RETURN_STATUS_AS_BOOL(DoBlasGemmBatchedInternal(
      dynload::hipblasSgemmBatched, stream, transa, transb, m, n, k, alpha,
      a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count,
      scratch_allocator));
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, double alpha,
    const port::ArraySlice<DeviceMemory<double> *> &a_array, int lda,
    const port::ArraySlice<DeviceMemory<double> *> &b_array, int ldb,
    double beta, const port::ArraySlice<DeviceMemory<double> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  SE_RETURN_STATUS_AS_BOOL(DoBlasGemmBatchedInternal(
      dynload::hipblasDgemmBatched, stream, transa, transb, m, n, k, alpha,
      a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count,
      scratch_allocator));
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, std::complex<float> alpha,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &a_array,
    int lda,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &b_array,
    int ldb, std::complex<float> beta,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  SE_RETURN_STATUS_AS_BOOL(DoBlasGemmBatchedInternal(
      dynload::hipblasCgemmBatched, stream, transa, transb, m, n, k, alpha,
      a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count,
      scratch_allocator));
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, std::complex<double> alpha,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &a_array,
    int lda,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &b_array,
    int ldb, std::complex<double> beta,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  SE_RETURN_STATUS_AS_BOOL(DoBlasGemmBatchedInternal(
      dynload::hipblasZgemmBatched, stream, transa, transb, m, n, k, alpha,
      a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count,
      scratch_allocator));
}

bool CUDABlas::DoBlasHemm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  return false;//DoBlasInternal(
      //dynload::hipblasChemm, stream, true /* = pointer_mode_host */,
      //CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(b)), ldb,
      //CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasHemm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  return false;//DoBlasInternal(
      //dynload::hipblasZhemm, stream, true /* = pointer_mode_host */,
      //CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(b)), ldb,
      //CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasHerk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          float alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          float beta, DeviceMemory<std::complex<float>> *c,
                          int ldc) {
  return false;//DoBlasInternal(dynload::hipblasCherk, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        //k, CUDAComplex(&alpha), CUDAComplex(CUDAMemory(a)), lda,
                        //&beta, CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasHerk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          double alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          double beta, DeviceMemory<std::complex<double>> *c,
                          int ldc) {
  return false;//DoBlasInternal(dynload::hipblasZherk, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                       // k, CUDAComplex(&alpha), CUDAComplex(CUDAMemory(a)), lda,
                       // &beta, CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasHer2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<float> alpha,
                           const DeviceMemory<std::complex<float>> &a, int lda,
                           const DeviceMemory<std::complex<float>> &b, int ldb,
                           float beta, DeviceMemory<std::complex<float>> *c,
                           int ldc) {
  return false;//DoBlasInternal(dynload::hipblasCher2k, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        //k, CUDAComplex(&alpha), CUDAComplex(CUDAMemory(a)), lda,
                        //CUDAComplex(CUDAMemory(b)), ldb, &beta,
                        //CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasHer2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<double> alpha,
                           const DeviceMemory<std::complex<double>> &a, int lda,
                           const DeviceMemory<std::complex<double>> &b, int ldb,
                           double beta, DeviceMemory<std::complex<double>> *c,
                           int ldc) {
  return false;//DoBlasInternal(dynload::hipblasZher2k, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        //k, CUDAComplex(&alpha), CUDAComplex(CUDAMemory(a)), lda,
                        //CUDAComplex(CUDAMemory(b)), ldb, &beta,
                        //CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &b, int ldb, float beta,
                          DeviceMemory<float> *c, int ldc) {
  return false;//DoBlasInternal(
      //dynload::hipblasSsymm, stream, true /* = pointer_mode_host */,
      //CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n, &alpha, CUDAMemory(a),
      //lda, CUDAMemory(b), ldb, &beta, CUDAMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &b, int ldb, double beta,
                          DeviceMemory<double> *c, int ldc) {
  return false;//DoBlasInternal(
      //dynload::hipblasDsymm, stream, true /* = pointer_mode_host */,
      //CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n, &alpha, CUDAMemory(a),
      //lda, CUDAMemory(b), ldb, &beta, CUDAMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  return false;//DoBlasInternal(
      //dynload::hipblasCsymm, stream, true /* = pointer_mode_host */,
      //CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(b)), ldb,
      //CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  return false;//DoBlasInternal(
      //dynload::hipblasZsymm, stream, true /* = pointer_mode_host */,
      //CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(b)), ldb,
      //CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          float beta, DeviceMemory<float> *c, int ldc) {
  return false;//DoBlasInternal(
      //dynload::hipblasSsyrk, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n, k, &alpha,
      //CUDAMemory(a), lda, &beta, CUDAMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          double beta, DeviceMemory<double> *c, int ldc) {
  return false;//DoBlasInternal(
      //dynload::hipblasDsyrk, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n, k, &alpha,
      //CUDAMemory(a), lda, &beta, CUDAMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  return false;//DoBlasInternal(
      //dynload::hipblasCsyrk, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n, k,
      //CUDAComplex(&alpha), CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(&beta),
      //CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  return false;//DoBlasInternal(
      //dynload::hipblasZsyrk, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n, k,
      //CUDAComplex(&alpha), CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(&beta),
      //CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           float alpha, const DeviceMemory<float> &a, int lda,
                           const DeviceMemory<float> &b, int ldb, float beta,
                           DeviceMemory<float> *c, int ldc) {
  return false;//DoBlasInternal(
      //dynload::hipblasSsyr2k, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n, k, &alpha,
      //CUDAMemory(a), lda, CUDAMemory(b), ldb, &beta, CUDAMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           double alpha, const DeviceMemory<double> &a, int lda,
                           const DeviceMemory<double> &b, int ldb, double beta,
                           DeviceMemory<double> *c, int ldc) {
  return false;//DoBlasInternal(
      //dynload::hipblasDsyr2k, stream, true /* = pointer_mode_host */,
      //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n, k, &alpha,
      //CUDAMemory(a), lda, CUDAMemory(b), ldb, &beta, CUDAMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<float> alpha,
                           const DeviceMemory<std::complex<float>> &a, int lda,
                           const DeviceMemory<std::complex<float>> &b, int ldb,
                           std::complex<float> beta,
                           DeviceMemory<std::complex<float>> *c, int ldc) {
  return false;//DoBlasInternal(dynload::hipblasCsyr2k, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        //k, CUDAComplex(&alpha), CUDAComplex(CUDAMemory(a)), lda,
                        //CUDAComplex(CUDAMemory(b)), ldb, CUDAComplex(&beta),
                        //CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<double> alpha,
                           const DeviceMemory<std::complex<double>> &a, int lda,
                           const DeviceMemory<std::complex<double>> &b, int ldb,
                           std::complex<double> beta,
                           DeviceMemory<std::complex<double>> *c, int ldc) {
  return false;//DoBlasInternal(dynload::hipblasZsyr2k, stream,
                        //true /* = pointer_mode_host */,
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        //k, CUDAComplex(&alpha), CUDAComplex(CUDAMemory(a)), lda,
                        //CUDAComplex(CUDAMemory(b)), ldb, CUDAComplex(&beta),
                        //CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
  return false;//DoBlasInternal(
      //dynload::hipblasStrmm, stream, true /* = pointer_mode_host */,
      //CUDABlasSide(side), CUDABlasUpperLower(uplo), CUDABlasTranspose(transa),
      //CUDABlasDiagonal(diag), m, n, &alpha, CUDAMemory(a), lda,
      //CUDAMemoryMutable(b), ldb, CUDAMemoryMutable(b), ldb);
}

bool CUDABlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
  return false;//DoBlasInternal(
      //dynload::hipblasDtrmm, stream, true /* = pointer_mode_host */,
      //CUDABlasSide(side), CUDABlasUpperLower(uplo), CUDABlasTranspose(transa),
      //CUDABlasDiagonal(diag), m, n, &alpha, CUDAMemory(a), lda,
      //CUDAMemoryMutable(b), ldb, CUDAMemoryMutable(b), ldb);
}

bool CUDABlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
  return false;//DoBlasInternal(
      //dynload::hipblasCtrmm, stream, true /* = pointer_mode_host */,
      //CUDABlasSide(side), CUDABlasUpperLower(uplo), CUDABlasTranspose(transa),
      //CUDABlasDiagonal(diag), m, n, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemoryMutable(b)), ldb,
      //CUDAComplex(CUDAMemoryMutable(b)), ldb);
}

bool CUDABlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
  return false;//DoBlasInternal(
      //dynload::hipblasZtrmm, stream, true /* = pointer_mode_host */,
      //CUDABlasSide(side), CUDABlasUpperLower(uplo), CUDABlasTranspose(transa),
      //CUDABlasDiagonal(diag), m, n, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemoryMutable(b)), ldb,
      //CUDAComplex(CUDAMemoryMutable(b)), ldb);
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
  return false;//DoBlasInternal(dynload::hipblasStrsm, stream,
                        //true /* = pointer_mode_host */, CUDABlasSide(side),
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(transa),
                        //CUDABlasDiagonal(diag), m, n, &alpha, CUDAMemory(a),
                        //lda, CUDAMemoryMutable(b), ldb);
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
  return false;//DoBlasInternal(dynload::hipblasDtrsm, stream,
                        //true /* = pointer_mode_host */, CUDABlasSide(side),
                        //CUDABlasUpperLower(uplo), CUDABlasTranspose(transa),
                        //CUDABlasDiagonal(diag), m, n, &alpha, CUDAMemory(a),
                        //lda, CUDAMemoryMutable(b), ldb);
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
  return false;//DoBlasInternal(
      //dynload::hipblasCtrsm, stream, true /* = pointer_mode_host */,
      //CUDABlasSide(side), CUDABlasUpperLower(uplo), CUDABlasTranspose(transa),
      //CUDABlasDiagonal(diag), m, n, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemoryMutable(b)), ldb);
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
  return false;//DoBlasInternal(
      //dynload::hipblasZtrsm, stream, true /* = pointer_mode_host */,
      //CUDABlasSide(side), CUDABlasUpperLower(uplo), CUDABlasTranspose(transa),
      //CUDABlasDiagonal(diag), m, n, CUDAComplex(&alpha),
      //CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemoryMutable(b)), ldb);
}

}  // namespace cuda

namespace gpu = ::perftools::gputools;

void initialize_hipblas() {
  gpu::port::Status status =
      gpu::PluginRegistry::Instance()
          ->RegisterFactory<gpu::PluginRegistry::BlasFactory>(
              gpu::cuda::kCudaPlatformId, gpu::cuda::kCuBlasPlugin, "HIPBLAS",
              [](gpu::internal::StreamExecutorInterface
                     *parent) -> gpu::blas::BlasSupport * {
                gpu::cuda::CUDAExecutor *cuda_executor =
                    dynamic_cast<gpu::cuda::CUDAExecutor *>(parent);
                if (cuda_executor == nullptr) {
                  LOG(ERROR)
                      << "Attempting to initialize an instance of the HIPBLAS "
                      << "support library with a non-CUDA StreamExecutor";
                  return nullptr;
                }

                gpu::cuda::CUDABlas *blas =
                    new gpu::cuda::CUDABlas(cuda_executor);
                if (!blas->Init()) {
                  // Note: Init() will log a more specific error.
                  delete blas;
                  return nullptr;
                }
                return blas;
              });

  if (!status.ok()) {
    LOG(ERROR) << "Unable to register HIPBLAS factory: "
               << status.error_message();
  }

  // Prime the HIPBLAS DSO. The loader will log more information.
  auto statusor = gpu::internal::CachedDsoLoader::GetCublasDsoHandle();
  if (!statusor.ok()) {
    LOG(INFO) << "Unable to load HIPBLAS DSO.";
  }

  gpu::PluginRegistry::Instance()->SetDefaultFactory(gpu::cuda::kCudaPlatformId,
                                                     gpu::PluginKind::kBlas,
                                                     gpu::cuda::kCuBlasPlugin);
}

}  // namespace gputools
}  // namespace perftools

REGISTER_MODULE_INITIALIZER(register_hipblas,
                            { perftools::gputools::initialize_hipblas(); });
