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

// Include cuBLAS headers early, and then set EIGEN_HAS_CUDA_FP16
// if we have new enough CUDA (which we will only know after including
// cuda.h). This ensures that Eigen's Half.h does not attempt to make its own
// __half typedef if CUDA has already defined one (and conversely, that we do
// not include <cuda_fp16.h> after Half.h has made its typedef).
//#include "cuda/include/hip_runtime.h"
#include "cuda/include/hipblas.h"

#if CUDA_VERSION >= 7050
#define EIGEN_HAS_CUDA_FP16
#endif

#if CUDA_VERSION >= 8000
#define SE_CUDA_DATA_HALF CUDA_R_16F
#else
#define SE_CUDA_DATA_HALF CUBLAS_DATA_HALF
#endif

#include "tensorflow/stream_executor/cuda/cuda_blas.h"
//#undef __CUDA_RUNTIME_H__

#include <dlfcn.h>

#include <complex>

#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_helpers.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/dso_loader.h"
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
//#include "tensorflow/stream_executor/cuda/cuda_blas.h"

namespace perftools 
{
namespace gputools 
{
namespace cuda 
{

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kCuBlasPlugin);

namespace dynload {

#define PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(__name)                              \
  struct DynLoadShim__##__name {                                            \
    static const char *kName;                                               \
    using FuncPointerT = std::add_pointer<decltype(::__name)>::type;        \
    static void *GetDsoHandle() {                                           \
      static auto status = internal::CachedDsoLoader::GetCublasDsoHandle(); \
      return status.ValueOrDie();                                           \
    }                                                                       \
    static FuncPointerT DynLoad() {                                         \
      static void *f = dlsym(GetDsoHandle(), kName);                        \
      CHECK(f != nullptr) << "could not find " << kName                     \
                          << " in cuBLAS DSO; dlerror: " << dlerror();      \
      return reinterpret_cast<FuncPointerT>(f);                             \
    }                                                                       \
    template <typename... Args>                                             \
    hipblasStatus_t operator()(CUDAExecutor * parent, Args... args) {       \
      cuda::ScopedActivateExecutorContext sac{parent};                      \
      return DynLoad()(args...);                                            \
    }                                                                       \
  } __name;                                                                 \
  const char *DynLoadShim__##__name::kName = #__name;

#define PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP(__name) \
  PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(__name)

#define CUBLAS_BLAS_ROUTINE_EACH(__macro) \
  __macro(hipblasSdot)                     \
  __macro(hipblasDdot)                     \
  __macro(hipblasSscal)                    \
  __macro(hipblasDscal)                    \
  __macro(hipblasZscal)                   \
  __macro(hipblasCscal)                   \
  __macro(hipblasZdscal)                   \
  __macro(hipblasCsscal)                   \
  __macro(hipblasSaxpy)                    \
  __macro(hipblasScopy)                    \
  __macro(hipblasDcopy)                    \
  __macro(hipblasSasum)                    \
  __macro(hipblasDasum)                    \
  __macro(hipblasSgemv)                    \
  __macro(hipblasSger)                     \
  __macro(hipblasSgemm)                    \
  __macro(hipblasDgemm)                    \
  __macro(hipblasCgemm)			   \
  __macro(hipblasZgemm)			   

PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP(hipblasCreate) 
PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP(hipblasDestroy) 
PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(hipblasSgemmBatched)
PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(hipblasDgemmBatched) 
PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(hipblasCgemmBatched)
PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(hipblasZgemmBatched) 

CUBLAS_BLAS_ROUTINE_EACH(PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP)

#if CUDA_VERSION >= 7050
PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(hipblasSgemmEx)
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
    case HIPBLAS_STATUS_ARCH_MISMATCH:
      return "HIPBLAS_STATUS_ARCH_MISMATCH";
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



bool CUDABlas::Init() {
  //cupblasStatus_t ret = dynload::cublasCreate_v2(parent_, &blas_);
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
//    dynload::cublasDestroy_v2(parent_, blas_);
    dynload::hipblasDestroy(parent_, blas_);
  }
}



namespace {

// Helper functions transforming blas arguments into cuBLAS arguments.

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

}  // namespace


template <typename FuncT, typename... Args>
bool CUDABlas::DoBlasInternal(FuncT cublas_func, Stream *stream,
                              bool pointer_mode_host, Args... args) {
  mutex_lock lock{mu_};

  CHECK(blas_ != nullptr);

  hipblasStatus_t ret = cublas_func(parent_, blas_, args...);
  if (ret != HIPBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to run hipBLAS routine " << cublas_func.kName << ": "
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
                        CUDAMemory(x), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasAsum(Stream *stream, uint64 elem_count, 
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) {
  return DoBlasInternal(dynload::hipblasDasum, stream,
                        false /* = pointer_mode_host */, elem_count,
                        CUDAMemory(x), incx, CUDAMemoryMutable(result));
}


bool CUDABlas::DoBlasAxpy(Stream *stream, uint64 elem_count, float alpha, 
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(dynload::hipblasSaxpy, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        CUDAMemory(x), incx, CUDAMemoryMutable(y), incy);
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

bool CUDABlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m, 
                          uint64 n, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(
      dynload::hipblasSgemv, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(trans), m, n, &alpha, CUDAMemory(a), lda, CUDAMemory(x),
      incx, &beta, CUDAMemoryMutable(y), incy);
}


bool CUDABlas::DoBlasGer(Stream *stream, uint64 m, uint64 n, float alpha, 
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *a, int lda) {
  return DoBlasInternal(
      dynload::hipblasSger, stream, true /* = pointer_mode_host */, m, n, &alpha,
      CUDAMemory(x), incx, CUDAMemory(y), incy, CUDAMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasGemm(
    Stream *stream, blas::Transpose transa,
    blas::Transpose transb, uint64 m, uint64 n, uint64 k,
    float alpha, const DeviceMemory<Eigen::half> &a, int lda,
    const DeviceMemory<Eigen::half> &b, int ldb, float beta,
    DeviceMemory<Eigen::half> *c, int ldc) {
#if CUDA_VERSION >= 7050
  VLOG(1) << port::Printf(
      "doing cuBLAS SGEMM: at=%d bt=%d m=%llu n=%llu "
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
  return DoBlasInternal(
      dynload::hipblasSgemmEx, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k, &alpha,
      CUDAMemory(a), SE_CUDA_DATA_HALF, lda,
      CUDAMemory(b), SE_CUDA_DATA_HALF, ldb,
      &beta,
      CUDAMemoryMutable(c), SE_CUDA_DATA_HALF, ldc);
#else
  LOG(ERROR) << "fp16 sgemm is not implemented in this cuBLAS version "
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
      "doing hipBLAS SGEMM: at=%d bt=%d m=%llu n=%llu "
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
      CUDAMemory(a), lda, CUDAMemory(b), ldb, &beta, CUDAMemoryMutable(c), ldc);
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
      CUDAComplex(&alpha), CUDAComplex(CUDAMemory(a)), lda,
      CUDAComplex(CUDAMemory(b)), ldb, CUDAComplex(&beta),
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
    FuncT cublas_func, Stream *stream, blas::Transpose transa,
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
      cublas_func, stream, true /* = pointer_mode_host */,
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

}  // namespace cuda


namespace gpu = ::perftools::gputools;

void initialize_cublas() {
  gpu::port::Status status =
      gpu::PluginRegistry::Instance()
          ->RegisterFactory<gpu::PluginRegistry::BlasFactory>(
              gpu::cuda::kCudaPlatformId, gpu::cuda::kCuBlasPlugin, "cuBLAS",
              [](gpu::internal::StreamExecutorInterface
                     *parent) -> gpu::blas::BlasSupport * {
                gpu::cuda::CUDAExecutor *cuda_executor =
                    dynamic_cast<gpu::cuda::CUDAExecutor *>(parent);
                if (cuda_executor == nullptr) {
                  LOG(ERROR)
                      << "Attempting to initialize an instance of the hipBLAS "
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
    LOG(ERROR) << "Unable to register cuBLAS factory: "
               << status.error_message();
  }

  // Prime the hipBLAS DSO. The loader will log more information.
  auto statusor = gpu::internal::CachedDsoLoader::GetCublasDsoHandle();
  if (!statusor.ok()) {
    LOG(INFO) << "Unable to load hipBLAS DSO.";
  }

  gpu::PluginRegistry::Instance()->SetDefaultFactory(gpu::cuda::kCudaPlatformId,
                                                     gpu::PluginKind::kBlas,
                                                     gpu::cuda::kCuBlasPlugin);
}

}  // namespace gputools
}  // namespace perftools

REGISTER_MODULE_INITIALIZER(register_cublas,
                            { perftools::gputools::initialize_cublas(); });
