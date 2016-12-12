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

#ifdef __HIP_PLATFORM_HCC__

#include "tensorflow/stream_executor/cuda/cuda_dnn.h"

#include <dlfcn.h>
#include <functional>
#include <memory>

#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/stream_executor/cuda/cuda_diagnostics.h"
#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/cuda/cuda_timer.h"
#include "tensorflow/stream_executor/dnn.h"
#include "tensorflow/stream_executor/dso_loader.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/stream_executor/lib/threadpool.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/scratch_allocator.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
// clang-format off
#include "cuda/include/mlopen.h"
// clang-format on

namespace {

// Converts (via narrowing) a type T value to a type U, and checks that the
// value has no value change due to the conversion.
template <typename WideT, typename NarrowT>
NarrowT CheckedNarrowing(const WideT& wide) {
  NarrowT narrow = wide;
  CHECK_EQ(narrow, wide)
      << "checked narrowing failed; values not equal post-conversion";
  return narrow;
}

// Returns the "Compatibility" version number from the CuDNN version number.
// This is the number that tries to indicate ABI compatibility.
//
// For example, if mlopen_version is 5107, the compatibility version
// number will be 5100.
/*size_t mlopenCompatibilityVersion(size_t mlopen_version) {
  return (mlopen_version / 100) * 100;
}
*/
}  // namespace

namespace perftools {
namespace gputools {

using dnn::BatchDescriptor;
using dnn::FilterDescriptor;
using dnn::ConvolutionDescriptor;
using dnn::PoolingDescriptor;
using dnn::NormalizeDescriptor;

namespace cuda {

//PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kCuDnnPlugin);

string ToString(mlopenStatus_t status) {
  switch (status) {
    case mlopenStatusSuccess:
      return "MLOPEN_STATUS_SUCCESS";
    case mlopenStatusNotInitialized:
      return "MLOPEN_STATUS_NOT_INITIALIZED";
    case mlopenStatusAllocFailed:
      return "MLOPEN_STATUS_ALLOC_FAILED";
    case mlopenStatusBadParm:
      return "MLOPEN_STATUS_BAD_PARAM";
    case mlopenStatusInternalError:
      return "MLOPEN_STATUS_INTERNAL_ERROR";
    case mlopenStatusInvalidValue:
      return "MLOPEN_STATUS_INVALID_VALUE";
    case mlopenStatusNotImplemented:
      return "MLOPEN_STATUS_NOT_SUPPORTED";
    case mlopenStatusUnknownError:
      return "MLOPEN_STATUS_UNKNOWN";
    default:
      return port::StrCat("<unknown mlopen status: ", static_cast<int>(status),
                          ">");
  }
}

namespace dynload {

static port::ThreadPool* InitCudnnThreadpool() {
  port::ThreadPool* mlopen_threadpool_;
  port::ThreadOptions options;
  // TBD(keveman): Conservatively setting the stack size and guard size to 2MB,
  // until we can get some guarantees from NVIDIA on the minimum stack space
  // they will work with.
  options.stack_size = 2 * 1024 * 1024;
  options.guard_size = 2 * 1024 * 1024;
  mlopen_threadpool_ = new port::ThreadPool(port::Env::Default(), options,
                                           "mlopen_threadpool", 1);
  CHECK(mlopen_threadpool_);
  return mlopen_threadpool_;
}

static mutex mlopen_threadpool_mu(LINKER_INITIALIZED);
static port::ThreadPool* GetCudaThreadpool() {
  mutex_lock lock(mlopen_threadpool_mu);
  static port::ThreadPool* mlopen_threadpool = InitCudnnThreadpool();
  return mlopen_threadpool;
}

// Retrieves the CUDNN DSO, dies on failure.
void* GetDsoHandle() {
  static auto result = internal::CachedDsoLoader::GetCudnnDsoHandle();
  return result.ValueOrDie();
}

// Calls mlopenGetVersion in the loaded DSO.
/*size_t mlopenGetVersion() {
  static void* f = dlsym(GetDsoHandle(), "mlopenGetVersion");
  if (f == nullptr) {
    LOG(FATAL) << "could not find mlopenGetVersion in mlopen DSO; dlerror: "
               << dlerror();
  }
  auto callable = reinterpret_cast<size_t (*)(void)>(f);
  return callable();
}*/

#define PERFTOOLS_GPUTOOLS_MLOPEN_WRAP(__name)                        \
  struct DynLoadShim__##__name {                                     \
    static const char* kName;                                        \
    typedef std::add_pointer<decltype(::__name)>::type FuncPointerT; \
    static FuncPointerT DynLoad() {                                  \
      static void* f = dlsym(GetDsoHandle(), kName);                 \
      if (f == nullptr) {                                            \
        LOG(FATAL) << "could not find " << kName                     \
                   << " in mlopen DSO; dlerror: " << dlerror();       \
      }                                                              \
      return reinterpret_cast<FuncPointerT>(f);                      \
    }                                                                \
    template <typename... Args>                                      \
    mlopenStatus_t operator()(CUDAExecutor* parent, Args... args) {   \
      cuda::ScopedActivateExecutorContext sac{parent};               \
      mlopenStatus_t retval = DynLoad()(args...);                     \
      return retval;                                                 \
    }                                                                \
  } __name;                                                          \
  const char* DynLoadShim__##__name::kName = #__name;

// clang-format off
#define MLOPEN_DNN_ROUTINE_EACH(__macro)                   \
  __macro(mlopenGetConvolutionForwardOutputDim)          \
  __macro(mlopenFindConvolutionForwardAlgorithm)            \
  __macro(mlopenCreateTensorDescriptor)                    \
  __macro(mlopenDestroyTensorDescriptor)                   \
  __macro(mlopenSetNdPoolingDescriptor)                    \
  __macro(mlopenSetLRNDescriptor)                          \
  __macro(mlopenCreateConvolutionDescriptor)               \
  __macro(mlopenCreatePoolingDescriptor)                   \
  __macro(mlopenDestroyPoolingDescriptor)                  \
  __macro(mlopenCreateLRNDescriptor)                       \
  __macro(mlopenDestroyLRNDescriptor)                      \
  __macro(mlopenDestroyConvolutionDescriptor)              \
  __macro(mlopenCreate)                                    \
  __macro(mlopenDestroy)                                   \
  /*__macro(mlopenSetStream)*/                                 \
  __macro(mlopenActivationForward)                         \
  __macro(mlopenConvolutionForward)                        \
  /*__macro(mlopenConvolutionBackwardBias)*/                   \
  /*__macro(mlopenGetConvolutionForwardWorkspaceSize)*/        \
  __macro(mlopenTransformTensor)                           \
  __macro(mlopenInitConvolutionDescriptor)                \
  __macro(mlopenSetTensorDescriptor)                     \
  __macro(mlopenPoolingForward)                            \
  __macro(mlopenPoolingBackward)                           \
  __macro(mlopenLRNForward)                    \
  __macro(mlopenLRNBackward)                   \
  __macro(mlopenConvolutionBackwardData)                   \
  __macro(mlopenConvolutionBackwardWeights)    \
  __macro(mlopenGetKernelTime)                 \
  __macro(mlopenEnableProfiling)

MLOPEN_DNN_ROUTINE_EACH(PERFTOOLS_GPUTOOLS_MLOPEN_WRAP)

#define MLOPEN_DNN_ROUTINE_EACH_AFTER_R3(__macro)              \
  __macro(mlopenConvolutionBackwardWeightsGetWorkSpaceSize)     \
  __macro(mlopenFindConvolutionBackwardDataAlgorithm)           \
  __macro(mlopenFindConvolutionBackwardWeightsAlgorithm)         \
  /*__macro(mlopenGetConvolutionBackwardDataWorkspaceSize)*/
MLOPEN_DNN_ROUTINE_EACH_AFTER_R3(PERFTOOLS_GPUTOOLS_MLOPEN_WRAP)
#undef MLOPEN_DNN_ROUTINE_EACH_AFTER_R3

// APIs in R5
#define MLOPEN_DNN_ROUTINE_EACH_R5(__macro)                    \
  __macro(mlopenCreateActivationDescriptor)                    \
  __macro(mlopenSetActivationDescriptor)                       \
  __macro(mlopenGetActivationDescriptor)                       \
  __macro(mlopenDestroyActivationDescriptor)                   \

MLOPEN_DNN_ROUTINE_EACH_R5(PERFTOOLS_GPUTOOLS_MLOPEN_WRAP)
#undef MLOPEN_DNN_ROUTINE_EACH_R5

#undef MLOPEN_DNN_ROUTINE_EACH

}  // namespace dynload

namespace {

mlopenHandle_t ToHandle(void* opaque_handle) {
  return static_cast<mlopenHandle_t>(opaque_handle);
}

mlopenConvFwdAlgorithm_t ToConvForwardAlgo(dnn::AlgorithmType algorithm) {
  mlopenConvFwdAlgorithm_t algo = mlopenConvFwdAlgorithm_t(algorithm);
  switch (algo) {
    case mlopenConvolutionFwdAlgoGEMM:
    case mlopenConvolutionFwdAlgoDirect:
    case mlopenConvolutionFwdAlgoFFT:
    case mlopenConvolutionFwdAlgoWinograd:
      return algo;
    default:
      LOG(FATAL) << "Unsupported MLOpen convolution forward algorithm: "
                 << algorithm;
  }
}

mlopenConvBwdDataAlgorithm_t ToConvBackwardDataAlgo(
    dnn::AlgorithmType algorithm) {
  mlopenConvBwdDataAlgorithm_t algo = mlopenConvBwdDataAlgorithm_t(algorithm);
  switch (algo) {
    case mlopenConvolutionBwdDataAlgo_0:
      return algo;
    default:
      LOG(FATAL)
          << "Unsupported MLOpen convolution backward algorithm for data: "
          << algorithm;
  }
}

mlopenConvBwdWeightsAlgorithm_t ToConvBackwardFilterAlgo(
    dnn::AlgorithmType algorithm) {
  mlopenConvBwdWeightsAlgorithm_t algo =
      mlopenConvBwdWeightsAlgorithm_t(algorithm);
  switch (algo) {
    case mlopenConvolutionBwdWeightsAlgoGEMM:
    case mlopenConvolutionBwdWeightsAlgoDirect:
      return algo;
    default:
      LOG(FATAL)
          << "Unsupported MLOpen convolution backward algorithm for filter: "
          << algorithm;
  }
}

}  // namespace

CudnnSupport::CudnnSupport(CUDAExecutor* parent)
    : parent_(parent), dnn_handle_(nullptr) {}

CudnnSupport::~CudnnSupport() {
  auto status = dynload::mlopenDestroy(parent_, ToHandle(dnn_handle_));
  if (status != mlopenStatusSuccess) {
    LOG(ERROR) << "could not destroy mlopen handle: " << ToString(status);
  }
}

port::Status CudnnSupport::Init() {
  auto status = dynload::mlopenCreate(
      parent_, reinterpret_cast<mlopenHandle_t*>(&dnn_handle_), 0, (hipStream_t*)NULL);
  if (status == mlopenStatusSuccess) {
    return port::Status::OK();
  }

  LOG(ERROR) << "could not create mlopen handle: " << ToString(status);
/*  if (status == CUDNN_STATUS_NOT_INITIALIZED) {
    // This is the error code that the driver returns when we're not running a
    // sufficient CUDA driver -- mlopen requires 6.5+ compatibility, which
    // starts with the 340.XX driver series.
    auto result = cuda::Diagnostician::FindKernelDriverVersion();
    if (!result.ok()) {
      LOG(ERROR) << "error retrieving driver version: "
                 << DriverVersionStatusToString(result);
    } else {
      const auto& version = result.ValueOrDie();
      LOG(INFO) << "running driver version: " << DriverVersionToString(version);
      // OS X kernel driver does not report version accurately
#if !defined(__APPLE__)
      if (std::get<0>(version) < 340) {
        LOG(ERROR)
            << "mlopen library is only supported on 340.XX+ driver versions";
      }
#endif
    }
  }*/

  return port::Status{port::error::INTERNAL,
                      port::StrCat("mlopen library could not create a handle: ",
                                   ToString(status))};
}

// Turns a BatchDescriptor structure into a mlopen tensor handle within a scope.
class ScopedTensorDescriptor {
 public:
  ScopedTensorDescriptor(CUDAExecutor* parent,
                         const BatchDescriptor& batch_descriptor,
                         mlopenDataType_t elem_type)
      : parent_(parent), handle_(nullptr) {
    mlopenStatus_t status =
        dynload::mlopenCreateTensorDescriptor(parent_, &handle_);
    if (status != mlopenStatusSuccess) {
      LOG(FATAL) << "could not create mlopen tensor descriptor: "
                 << ToString(status);
    }

    switch (batch_descriptor.layout()) {
      case dnn::DataLayout::kBatchYXDepth:
      case dnn::DataLayout::kBatchDepthYX:
        break;
      default:
        LOG(FATAL) << "Unsupported tensor format "
                   << DataLayoutString(batch_descriptor.layout());
        break;
    }

    const int nd = batch_descriptor.ndims() + 2;
    // cuDNN requires the strides and dims to be ordered as BDYX.
    std::vector<int64> strides64 =
        batch_descriptor.full_strides(dnn::DataLayout::kBatchDepthYX);
    std::vector<int64> dims64 =
        batch_descriptor.full_dims(dnn::DataLayout::kBatchDepthYX);

    // cuDNN requires arrays of ints.
    std::vector<int> strides(nd);
    std::vector<int> dims(nd);
    std::transform(strides64.cbegin(), strides64.cend(), strides.begin(),
                   &CheckedNarrowing<int64, int>);
    std::transform(dims64.cbegin(), dims64.cend(), dims.begin(),
                   &CheckedNarrowing<int64, int>);
    status = dynload::mlopenSetTensorDescriptor(
        parent_, handle_, elem_type, nd, dims.data(), strides.data());

    if (status != mlopenStatusSuccess) {
      LOG(FATAL) << "could not set mlopen tensor descriptor: "
                 << ToString(status);
    }
  }

  ~ScopedTensorDescriptor() {
    mlopenStatus_t status =
        dynload::mlopenDestroyTensorDescriptor(parent_, handle_);
    if (status != mlopenStatusSuccess) {
      LOG(ERROR) << "could not destroy mlopen tensor descriptor: "
                 << ToString(status);
    }
  }

  mlopenTensorDescriptor_t handle() const { return handle_; }

 private:
  CUDAExecutor* parent_;            // Parent executor. Not owned.
  mlopenTensorDescriptor_t handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedTensorDescriptor);
};

// Turns a FilterDescriptor structure into a mlopen filter handle within a scope.
class ScopedFilterDescriptor {
 public:
  ScopedFilterDescriptor(CUDAExecutor* parent,
                         const FilterDescriptor& filter_descriptor,
                         const BatchDescriptor& batch_descriptor,
                         mlopenDataType_t elem_type)
      : parent_(parent), handle_(nullptr) {
    mlopenStatus_t status =
        dynload::mlopenCreateTensorDescriptor(parent_, &handle_);
    if (status != mlopenStatusSuccess) {
      LOG(FATAL) << "could not create mlopen filter descriptor: "
                 << ToString(status);
    }

    std::vector<int> dims(2 + filter_descriptor.ndims());
    dims[0] = filter_descriptor.output_feature_map_count();
    dims[1] = filter_descriptor.input_feature_map_count();
    const auto& spatial_dims = filter_descriptor.input_filter_dims();
    std::copy(spatial_dims.begin(), spatial_dims.end(), dims.begin() + 2);

    std::vector<int> strides(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size()-2; i >= 0; i--) {
      strides[i] = strides[i + 1] * dims[i + 1];
    }

    status = dynload::mlopenSetTensorDescriptor(parent_, handle_, elem_type,
                                                 dims.size(), dims.data(), strides.data());
    if (status != mlopenStatusSuccess) {
      LOG(FATAL) << "could not set mlopen filter descriptor: "
                 << ToString(status);
    }
  }

  ~ScopedFilterDescriptor() {
    mlopenStatus_t status =
        dynload::mlopenDestroyTensorDescriptor(parent_, handle_);
    if (status != mlopenStatusSuccess) {
      LOG(ERROR) << "could not destroy mlopen filter descriptor: "
                 << ToString(status);
    }
  }

  mlopenTensorDescriptor_t handle() const { return handle_; }

 private:
  // Parent executor object. Not owned.
  CUDAExecutor* parent_;

  // mlopen filter descriptor this object creates. Owned.
  mlopenTensorDescriptor_t handle_;

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedFilterDescriptor);
};

// Turns a ConvolutionDescriptor structure into a mlopen convolution handle
// within a scope.
class ScopedConvolutionDescriptor {
 public:
  ScopedConvolutionDescriptor(
      CUDAExecutor* parent, const ConvolutionDescriptor& convolution_descriptor,
      mlopenDataType_t data_type)
      : parent_(parent), handle_(nullptr) {
    mlopenStatus_t status =
        dynload::mlopenCreateConvolutionDescriptor(parent_, &handle_);
    if (status != mlopenStatusSuccess) {
      LOG(FATAL) << "could not create mlopen convolution descriptor: "
                 << ToString(status);
    }
    const auto& strides64 = convolution_descriptor.strides();
    const auto& padding64 = convolution_descriptor.padding();

    // cuDNN requires arrays of ints.
    std::vector<int> strides(convolution_descriptor.ndims());
    std::vector<int> padding(convolution_descriptor.ndims());
    std::transform(strides64.cbegin(), strides64.cend(), strides.begin(),
                   &CheckedNarrowing<int64, int>);
    std::transform(padding64.cbegin(), padding64.cend(), padding.begin(),
                   &CheckedNarrowing<int64, int>);
    std::vector<int> upscale(convolution_descriptor.ndims(), 1);

    status = dynload::mlopenInitConvolutionDescriptor(
        parent_, handle_, mlopenCrossCorrelation, padding[0], padding[1],
        strides[0], strides[1], upscale[0], upscale[1]);

    if (status != mlopenStatusSuccess) {
      LOG(FATAL) << "could not set mlopen convolution descriptor: "
                 << ToString(status);
    }
  }

  ~ScopedConvolutionDescriptor() {
    mlopenStatus_t status =
        dynload::mlopenDestroyConvolutionDescriptor(parent_, handle_);
    if (status != mlopenStatusSuccess) {
      LOG(ERROR) << "could not destroy mlopen convolution descriptor: "
                 << ToString(status);
    }
  }

  mlopenConvolutionDescriptor_t handle() const { return handle_; }

 private:
  CUDAExecutor* parent_;                 // Parent executor. Not owned.
  mlopenConvolutionDescriptor_t handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedConvolutionDescriptor);
};

// Turns a PoolingDescriptor structure into a mlopen pooling descriptor handle
// within a scope.
class ScopedPoolingDescriptor {
 public:
  ScopedPoolingDescriptor(CUDAExecutor* parent,
                          const PoolingDescriptor& pooling_descriptor)
      : parent_(parent), handle_(nullptr) {
    mlopenStatus_t status =
        dynload::mlopenCreatePoolingDescriptor(parent_, &handle_);
    if (status != mlopenStatusSuccess) {
      LOG(FATAL) << "could not create mlopen pooling descriptor: "
                 << ToString(status);
    }

    const std::vector<int64> strides64 = pooling_descriptor.strides();
    const std::vector<int64> padding64 = pooling_descriptor.padding();
    const std::vector<int64> shape64 = pooling_descriptor.window();

    const int nd = pooling_descriptor.ndims();
    std::vector<int> shape(nd);
    std::vector<int> padding(nd);
    std::vector<int> strides(nd);
    std::transform(strides64.cbegin(), strides64.cend(), strides.begin(),
                   &CheckedNarrowing<int64, int>);
    std::transform(padding64.cbegin(), padding64.cend(), padding.begin(),
                   &CheckedNarrowing<int64, int>);
    std::transform(shape64.cbegin(), shape64.cend(), shape.begin(),
                   &CheckedNarrowing<int64, int>);
    status = dynload::mlopenSetNdPoolingDescriptor(
        parent_, handle_,
        (pooling_descriptor.mode() == dnn::PoolingMode::kMaximum
             ? mlopenPoolingMax
             : mlopenPoolingAverage),
        nd, shape.data(), padding.data(), strides.data());
    if (status != mlopenStatusSuccess) {
      LOG(FATAL) << "could not set mlopen pooling descriptor: "
                 << ToString(status);
    }
  }
  ~ScopedPoolingDescriptor() {
    mlopenStatus_t status =
        dynload::mlopenDestroyPoolingDescriptor(parent_, handle_);
    if (status != mlopenStatusSuccess) {
      LOG(ERROR) << "could not destroy mlopen pooling descriptor: "
                 << ToString(status);
    }
  }

  mlopenPoolingDescriptor_t handle() const { return handle_; }

 private:
  CUDAExecutor* parent_;             // Parent executor. Not owned.
  mlopenPoolingDescriptor_t handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedPoolingDescriptor);
};

// Turns a NormalizeDescriptor structure into a mlopen LRN descriptor handle.
class ScopedNormalizeDescriptor {
 public:
  ScopedNormalizeDescriptor(CUDAExecutor* parent,
                            const NormalizeDescriptor& normalize_descriptor)
      : parent_(parent), handle_(nullptr) {
    mlopenStatus_t status = dynload::mlopenCreateLRNDescriptor(parent_, &handle_);
    if (status != mlopenStatusSuccess) {
      LOG(FATAL) << "could not create mlopen LRN descriptor: "
                 << ToString(status);
    }

    // The range specifies that the indices in the closed range
    // [i - range, i + range] should be included in the normalization for index
    // i. The lrnN value is the total number of elements in the range, so
    // lrnN = 2*range + 1.
    unsigned lrnN = 2 * normalize_descriptor.range() + 1;

    // Note that SE defines the normalization operation as
    //
    //  U_i = V_i / ((bias +  alpha      * (sum_j V_j^2)) ^ beta)
    //
    // but cuDNN defines it as
    //
    //  U_i = V_i / ((bias + (alpha / n) * (sum_j V_j^2)) ^ beta)
    //
    // i.e. there is a factor of n difference between the meaning of the alphas
    // in the two contexts. The cuDNN alpha is n times the SE alpha.
    double lrnAlpha = lrnN * normalize_descriptor.alpha();

    double lrnBeta = normalize_descriptor.beta();
    double lrnK = normalize_descriptor.bias();
    status = dynload::mlopenSetLRNDescriptor(parent_, handle_, mlopenLRNCrossChannel, lrnN, lrnAlpha,
                                            lrnBeta, lrnK);
    if (status != mlopenStatusSuccess) {
      LOG(FATAL) << "could not set mlopen LRN descriptor: " << ToString(status);
    }
  }

  ~ScopedNormalizeDescriptor() {
    mlopenStatus_t status = dynload::mlopenDestroyLRNDescriptor(parent_, handle_);
    if (status != mlopenStatusSuccess) {
      LOG(ERROR) << "could not destroy mlopen LRN descriptor: "
                 << ToString(status);
    }
  }

  mlopenLRNDescriptor_t handle() const { return handle_; }

 private:
  CUDAExecutor* parent_;         // Parent executor. Not owned.
  mlopenLRNDescriptor_t handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedNormalizeDescriptor);
};

#if 0//CUDNN_VERSION >= 5000
// Turns a ActivationDescriptor structure into a mlopen activation
// descriptor handle within a scope.
class ScopedActivationDescriptor {
 public:
  ScopedActivationDescriptor(CUDAExecutor* parent,
                             dnn::ActivationMode activation_mode,
                             double value_max)
      : parent_(parent), handle_(nullptr) {
    mlopenStatus_t status =
        dynload::mlopenCreateActivationDescriptor(parent_, &handle_);
    if (status != mlopenStatusSuccess) {
      LOG(FATAL) << "could not create mlopen activation descriptor: "
                 << ToString(status);
    }

    double relu_ceiling = 0.0;
    mlopenActivationMode_t mode;
    switch (activation_mode) {
      case dnn::ActivationMode::kRelu6:
        relu_ceiling = 6.0;
        mode = CUDNN_ACTIVATION_CLIPPED_RELU;
        break;
      case dnn::ActivationMode::kReluX:
        relu_ceiling = value_max;
        mode = CUDNN_ACTIVATION_CLIPPED_RELU;
        break;
      case dnn::ActivationMode::kRelu:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case dnn::ActivationMode::kSigmoid:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      case dnn::ActivationMode::kTanh:
        mode = CUDNN_ACTIVATION_TANH;
        break;
      default:
        LOG(FATAL) << "unrecognized activation mode: "
                   << static_cast<int>(activation_mode);
    }

    // Always propagate nans.
    mlopenNanPropagation_t nan_propagation = CUDNN_PROPAGATE_NAN;
    status = dynload::mlopenSetActivationDescriptor(
        parent_, handle_,
        mode, nan_propagation, relu_ceiling);
    if (status != mlopenStatusSuccess) {
      LOG(FATAL) << "could not set mlopen activation descriptor: "
                 << ToString(status);
    }
  }

  ~ScopedActivationDescriptor() {
    mlopenStatus_t status =
        dynload::mlopenDestroyActivationDescriptor(parent_, handle_);
    if (status != mlopenStatusSuccess) {
      LOG(ERROR) << "could not destroy mlopen activation descriptor: "
                 << ToString(status);
    }
  }

  mlopenActivationDescriptor_t handle() const { return handle_; }

 private:
  CUDAExecutor* parent_;                // Parent executor. Not owned.
  mlopenActivationDescriptor_t handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedActivationDescriptor);
};
#endif

namespace {

#if 0//CUDNN_VERSION >= 5000

mlopenRNNInputMode_t ToCudnnRnnInputMode(dnn::RnnInputMode input_mode) {
  switch (input_mode) {
    case dnn::RnnInputMode::kRnnLinearSkip:
    case dnn::RnnInputMode::kRnnSkipInput:
      return static_cast<mlopenRNNInputMode_t>(input_mode);
    default:
      LOG(FATAL) << "Invalid RNN input mode: " << static_cast<int>(input_mode);
  }
}

mlopenDirectionMode_t ToCudnnRnnDirectionMode(
    dnn::RnnDirectionMode direction_mode) {
  switch (direction_mode) {
    case dnn::RnnDirectionMode::kRnnUnidirectional:
    case dnn::RnnDirectionMode::kRnnBidirectional:
      return static_cast<mlopenDirectionMode_t>(direction_mode);
    default:
      LOG(FATAL) << "Invalid RNN direction mode: "
                 << static_cast<int>(direction_mode);
  }
}

mlopenRNNMode_t ToCudnnRnnMode(dnn::RnnMode rnn_mode) {
  switch (rnn_mode) {
    case dnn::RnnMode::kRnnRelu:
    case dnn::RnnMode::kRnnTanh:
    case dnn::RnnMode::kRnnLstm:
    case dnn::RnnMode::kRnnGru:
      return static_cast<mlopenRNNMode_t>(rnn_mode);
    default:
      LOG(FATAL) << "Invalid RNN Mode: " << static_cast<int>(rnn_mode);
  }
}

mlopenDataType_t ToCudnnDataType(dnn::DataType data_type) {
  switch (data_type) {
    case dnn::DataType::kFloat:
    case dnn::DataType::kDouble:
    case dnn::DataType::kHalf:
      return static_cast<mlopenDataType_t>(data_type);
    default:
      LOG(FATAL) << "Invalid DNN data type: " << static_cast<int>(data_type);
  }
}

int CudnnDataTypeToByteSize(mlopenDataType_t data_type) {
  switch (data_type) {
    case CUDNN_DATA_FLOAT:
      return sizeof(float);
    case CUDNN_DATA_DOUBLE:
      return sizeof(double);
    case CUDNN_DATA_HALF:
      return sizeof(Eigen::half);
    default:
      LOG(FATAL) << "Invalid DNN data type: " << static_cast<int>(data_type);
  }
}

#endif  // CUDNN_VERSION

template <typename Base>
class MixinBase : public Base {};
template <>
class MixinBase<void> {};

}  // namespace

#if 0//CUDNN_VERSION >= 5000

#define CUDNN_RETURN_IF_FAIL(STATUS, ...)                                \
  if (!SE_PREDICT_TRUE((STATUS) == mlopenStatusSuccess)) {              \
    string error_msg = port::StrCat(ToString(STATUS), " ", __VA_ARGS__); \
    SetFailure(port::Status(port::error::UNKNOWN, error_msg));           \
    LOG(ERROR) << error_msg;                                             \
    return;                                                              \
  }

template <typename Base>
class CudnnDescriptorCommon : public MixinBase<Base> {
 public:
  bool ok() const { return status_.ok(); }
  port::Status Status() const { return status_; }

 protected:
  void SetFailure(const port::Status& status) { status_.Update(status); }
  port::Status status_;
};

class CudnnDropoutDescriptor : public CudnnDescriptorCommon<void> {
 public:
  CudnnDropoutDescriptor(CUDAExecutor* parent, mlopenHandle_t mlopen_handle,
                         float dropout, uint64 seed,
                         ScratchAllocator* state_allocator)
      : parent_(parent), handle_(nullptr) {
    mlopenStatus_t status;
    status = dynload::mlopenCreateDropoutDescriptor(parent_, &handle_);
    CUDNN_RETURN_IF_FAIL(status, "Failed to create dropout descriptor");

    if (dropout == 0.f) {
      return;
    }

    DeviceMemory<uint8> state_memory;
    if (state_allocator) {
      size_t state_sizes_in_bytes = 0;
      status = dynload::mlopenDropoutGetStatesSize(parent_, mlopen_handle,
                                                  &state_sizes_in_bytes);
      CUDNN_RETURN_IF_FAIL(status, "Failed to query dropout state sizes");

      auto allocated =
          state_allocator->AllocateBytes(nullptr, state_sizes_in_bytes);
      if (!allocated.ok() ||
          (state_memory = allocated.ValueOrDie()) == nullptr) {
        string error_msg =
            port::StrCat("Fail to allocate Cudnn dropout state memory");
        status_ = port::Status(port::error::UNKNOWN, error_msg);
        LOG(ERROR) << error_msg;
        return;
      }
    }
    status = dynload::mlopenSetDropoutDescriptor(parent_, handle_, mlopen_handle,
                                                dropout, state_memory.opaque(),
                                                state_memory.size(), seed);
    CUDNN_RETURN_IF_FAIL(status, "Failed to set dropout descriptor");
  }

  ~CudnnDropoutDescriptor() {
    if (handle_) {
      mlopenStatus_t status =
          dynload::mlopenDestroyDropoutDescriptor(parent_, handle_);
      CUDNN_RETURN_IF_FAIL(status, "Failed to destroy Cudnn dropout handle: ");
    }
  }

  mlopenDropoutDescriptor_t handle() const {
    if (!ok()) return nullptr;
    return handle_;
  }

 private:
  CUDAExecutor* parent_;
  mlopenDropoutDescriptor_t handle_;
  float dropout_;
  uint64 seed_;
  port::Status status_;
  SE_DISALLOW_COPY_AND_ASSIGN(CudnnDropoutDescriptor);
};

class CudnnRnnParamsDescriptor : public CudnnDescriptorCommon<void> {
 public:
  typedef dnn::RnnDescriptor::ParamsRegion ParamsRegion;
  typedef dnn::RnnDescriptor::ParamsRegions ParamsRegions;
  CudnnRnnParamsDescriptor(CUDAExecutor* parent, mlopenHandle_t mlopen_handle,
                           const CudnnRnnDescriptor& rnn_desc);
  ~CudnnRnnParamsDescriptor() {
    mlopenStatus_t status =
        dynload::mlopenDestroyFilterDescriptor(parent_, handle_);
    CUDNN_RETURN_IF_FAIL(status, "Failed to destroy RNN filter desciptor");
  }
  mlopenFilterDescriptor_t handle() const {
    if (!ok()) return nullptr;
    return handle_;
  }
  int64 params_size_in_bytes() const { return params_size_in_bytes_; }
  ParamsRegions params_weights() const {
    if (!ok()) return ParamsRegions();
    return weights_;
  }
  ParamsRegions params_biases() const {
    if (!ok()) return ParamsRegions();
    return biases_;
  }

 private:
  int GetRegionCountPerLayer() const;
  CUDAExecutor* parent_;
  mlopenFilterDescriptor_t handle_;
  const CudnnRnnDescriptor* rnn_desc_;
  int64 params_size_in_bytes_;
  ParamsRegions weights_;
  ParamsRegions biases_;
  port::Status status_;
  SE_DISALLOW_COPY_AND_ASSIGN(CudnnRnnParamsDescriptor);
};

class CudnnRnnDescriptor : public CudnnDescriptorCommon<dnn::RnnDescriptor> {
 public:
  CudnnRnnDescriptor(CUDAExecutor* parent, mlopenHandle_t mlopen_handle,
                     int num_layers, int hidden_size, int input_size,
                     mlopenRNNInputMode_t input_mode,
                     mlopenDirectionMode_t direction_mode,
                     mlopenRNNMode_t rnn_mode, mlopenDataType_t data_type,
                     float dropout, uint64 seed,
                     ScratchAllocator* state_allocator)
      : parent_(parent),
        rnn_desc_(nullptr),
        num_layers_(num_layers),
        hidden_size_(hidden_size),
        input_size_(input_size),
        input_mode_(input_mode),
        direction_mode_(direction_mode),
        rnn_mode_(rnn_mode),
        data_type_(data_type) {
    // Create the dropout handle.
    mlopen_dropout_desc_.reset(new CudnnDropoutDescriptor(
        parent, mlopen_handle, dropout, seed, state_allocator));
    if (!mlopen_dropout_desc_->ok()) {
      SetFailure(mlopen_dropout_desc_->Status());
      return;
    }

    // Create the RNN handle
    mlopenStatus_t status =
        dynload::mlopenCreateRNNDescriptor(parent_, &rnn_desc_);
    CUDNN_RETURN_IF_FAIL(status, "Unable to create RNN descriptor");
    status = dynload::mlopenSetRNNDescriptor(
        parent, rnn_desc_ /*rnnDesc*/, hidden_size /*hiddenSize*/,
        num_layers /*numLayers*/, dropout_handle() /*dropoutDesc*/,
        input_mode /*inputMode*/, direction_mode /*direction*/,
        rnn_mode /*mode*/, data_type /*dataType*/);
    CUDNN_RETURN_IF_FAIL(status, "Unable to update RNN descriptor");

    // Create the params handle.
    mlopen_params_desc_.reset(
        new CudnnRnnParamsDescriptor(parent, mlopen_handle, *this));
    if (!mlopen_params_desc_->ok()) {
      SetFailure(mlopen_params_desc_->Status());
      return;
    }
  }
  ~CudnnRnnDescriptor() override {
    if (rnn_desc_) {
      mlopenStatus_t status =
          dynload::mlopenDestroyRNNDescriptor(parent_, rnn_desc_);
      CUDNN_RETURN_IF_FAIL(status, "Unable to destroy RNN descriptor");
    }
  }
  mlopenRNNDescriptor_t handle() const {
    if (!ok()) return nullptr;
    return rnn_desc_;
  }
  int num_layers() const { return num_layers_; }
  int hidden_size() const { return hidden_size_; }
  int input_size() const { return input_size_; }
  mlopenRNNInputMode_t input_mode() const { return input_mode_; }
  mlopenDirectionMode_t direction_mode() const { return direction_mode_; }
  mlopenRNNMode_t rnn_mode() const { return rnn_mode_; }
  mlopenDataType_t data_type() const { return data_type_; }
  int64 ParamsSizeInBytes() const override {
    return mlopen_params_desc_->params_size_in_bytes();
  }
  mlopenDropoutDescriptor_t dropout_handle() const {
    if (!mlopen_dropout_desc_) return nullptr;
    return mlopen_dropout_desc_->handle();
  }
  mlopenFilterDescriptor_t params_handle() const {
    if (!mlopen_params_desc_) return nullptr;
    return mlopen_params_desc_->handle();
  }
  ParamsRegions ParamsWeightRegions() const override {
    if (!ok()) return ParamsRegions();
    return mlopen_params_desc_->params_weights();
  }
  ParamsRegions ParamsBiasRegions() const override {
    if (!ok()) return ParamsRegions();
    return mlopen_params_desc_->params_biases();
  }

 private:
  CUDAExecutor* parent_;
  mlopenRNNDescriptor_t rnn_desc_;
  int num_layers_;
  int hidden_size_;
  int input_size_;
  mlopenRNNInputMode_t input_mode_;
  mlopenDirectionMode_t direction_mode_;
  mlopenRNNMode_t rnn_mode_;
  mlopenDataType_t data_type_;
  port::Status status_;
  std::unique_ptr<CudnnDropoutDescriptor> mlopen_dropout_desc_;
  std::unique_ptr<CudnnRnnParamsDescriptor> mlopen_params_desc_;
  SE_DISALLOW_COPY_AND_ASSIGN(CudnnRnnDescriptor);
};

CudnnRnnParamsDescriptor::CudnnRnnParamsDescriptor(
    CUDAExecutor* parent, mlopenHandle_t mlopen_handle,
    const CudnnRnnDescriptor& rnn_desc)
    : parent_(parent),
      handle_(nullptr),
      rnn_desc_(&rnn_desc),
      params_size_in_bytes_(0) {
  mlopenTensorDescriptor_t input_desc = nullptr;
  {
    // Query the params size.
    auto status = dynload::mlopenCreateTensorDescriptor(parent, &input_desc);
    CUDNN_RETURN_IF_FAIL(status, "Cudnn fails to create tensor descriptor");
    int dims[] = {1, rnn_desc.input_size(), 1};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    status = dynload::mlopenSetTensorNdDescriptor(
        parent, input_desc /*tensorDesc*/, rnn_desc.data_type() /*dataType*/,
        sizeof(dims) / sizeof(dims[0]) /*nbDims*/, dims /*dimA*/,
        strides /*strideA*/);
    CUDNN_RETURN_IF_FAIL(status, "Cudnn fails to set tensor descriptor");

    size_t params_size = 0;
    status = dynload::mlopenGetRNNParamsSize(
        parent, mlopen_handle /*handle*/, rnn_desc.handle() /*rnnDesc*/,
        input_desc /*xDesc*/, &params_size /*sizeInBytes*/,
        rnn_desc.data_type() /*dataType*/);
    CUDNN_RETURN_IF_FAIL(status, "Cudnn fails to get RNN parameter size");
    params_size_in_bytes_ = static_cast<int64>(params_size);
  }

  {
    // Create the params descriptor.
    auto status = dynload::mlopenCreateFilterDescriptor(parent, &handle_);
    CUDNN_RETURN_IF_FAIL(status, "Cudnn fails to create RNN filter descriptor");
    int dims[] = {static_cast<int>(params_size_in_bytes_), 1, 1};
    status = dynload::mlopenSetFilterNdDescriptor(
        parent, handle_ /*filterDesc*/, rnn_desc.data_type() /*dataType*/,
        CUDNN_TENSOR_NCHW /*format*/, sizeof(dims) / sizeof(dims[0]) /*nbDims*/,
        dims /*filterDimA*/);
    CUDNN_RETURN_IF_FAIL(status, "Cudnn fails to update RNN filter descriptor");
  }

  {
    // Create the weights and biases into the params buffer
    int region_count_per_layer = GetRegionCountPerLayer();
    mlopenFilterDescriptor_t region_desc_handle = nullptr;
    auto status =
        dynload::mlopenCreateFilterDescriptor(parent, &region_desc_handle);
    CUDNN_RETURN_IF_FAIL(status, "Cudnn fails to create filter descriptor");
    for (int layer = 0; layer < rnn_desc.num_layers(); layer++) {
      for (int region = 0; region < region_count_per_layer; region++) {
        for (int type = 0; type < 2; type++) {
          void* offset = nullptr;
          if (type == 0) {
            status = dynload::mlopenGetRNNLinLayerMatrixParams(
                parent, mlopen_handle /*handle*/, rnn_desc.handle() /*rnnDesc*/,
                layer /*layer*/, input_desc /*xDesc*/, handle_ /*wDesc*/,
                nullptr /*w*/, region /*linLayerID*/,
                region_desc_handle /*linLayerMatDesc*/,
                &offset /*linLayerMat*/);
            CUDNN_RETURN_IF_FAIL(
                status, "Cudnn fails to call mlopenGetRNNLinLayerMatrixParams");
          } else {
            status = dynload::mlopenGetRNNLinLayerBiasParams(
                parent, mlopen_handle /*rnnDesc*/, rnn_desc.handle() /*rnnDesc*/,
                layer /*layer*/, input_desc /*xDesc*/, handle_ /*wDesc*/,
                nullptr /*w*/, region /*linLayerID*/,
                region_desc_handle /*linLayerBiasDesc*/,
                &offset /*linLayerBias*/);
            CUDNN_RETURN_IF_FAIL(
                status, "Cudnn fails to call mlopenGetRNNLinLayerBiasParams");
          }
          int dims[] = {1, 1, 1};
          mlopenDataType_t data_type;
          mlopenTensorFormat_t tensor_format;
          int n_dims;
          status = dynload::mlopenGetFilterNdDescriptor(
              parent, region_desc_handle /*filterDesc*/,
              sizeof(dims) / sizeof(dims[0]) /*nbDimsRequested*/,
              &data_type /*dataType*/, &tensor_format /*format*/,
              &n_dims /*nbDims*/, dims /*filterDimA*/);
          CUDNN_RETURN_IF_FAIL(status, "Cudnn fails to get filter description");
          int64 size = dims[0] * dims[1] * dims[2] *
                       CudnnDataTypeToByteSize(rnn_desc.data_type());
          auto region = ParamsRegion{reinterpret_cast<int64>(offset), size};
          if (type == 0) {
            weights_.push_back(region);
          } else {
            biases_.push_back(region);
          }
        }
      }
    }
    status = dynload::mlopenDestroyFilterDescriptor(parent, region_desc_handle);
    CUDNN_RETURN_IF_FAIL(status, "Cudnn fails to destroy filter descriptor");
  }

  {
    // Release the dummy input tensor descriptor.
    auto status = dynload::mlopenDestroyTensorDescriptor(parent, input_desc);
    CUDNN_RETURN_IF_FAIL(status, "Cudnn fails to destroy tensor descriptor");
  }
}

int CudnnRnnParamsDescriptor::GetRegionCountPerLayer() const {
  auto rnn_mode = rnn_desc_->rnn_mode();
  switch (rnn_mode) {
    case CUDNN_RNN_RELU:
    case CUDNN_RNN_TANH:
      return 2;
    case CUDNN_LSTM:
      return 8;
    case CUDNN_GRU:
      return 6;
    default:
      LOG(FATAL) << "Invalid RNN Mode: " << static_cast<int>(rnn_mode);
  }
}

class CudnnRnnSequenceTensorDescriptor
    : public CudnnDescriptorCommon<dnn::RnnSequenceTensorDescriptor> {
 public:
  CudnnRnnSequenceTensorDescriptor(CUDAExecutor* parent, int seq_length,
                                   int batch_size, int data_size,
                                   mlopenDataType_t data_type)
      : parent_(parent),
        seq_length_(seq_length),
        batch_size_(batch_size),
        data_size_(data_size),
        data_type_(data_type) {
    mlopenTensorDescriptor_t handle = nullptr;
    if (seq_length <= 0) {
      string error_msg =
          port::StrCat("sequence length must be positive: ", seq_length);
      LOG(ERROR) << error_msg;
      SetFailure(port::Status(port::error::UNKNOWN, error_msg));
      return;
    }
    mlopenStatus_t status =
        dynload::mlopenCreateTensorDescriptor(parent, &handle);
    CUDNN_RETURN_IF_FAIL(status, "Failed to create tensor descriptor");
    int dims[] = {batch_size, data_size, 1};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    status = dynload::mlopenSetTensorNdDescriptor(
        parent, handle /*tensorDesc*/, data_type /*dataType*/,
        sizeof(dims) / sizeof(dims[0]) /*nbDims*/, dims /*dimA*/,
        strides /*strideA*/);
    CUDNN_RETURN_IF_FAIL(status, "Failed to update tensor descriptor");
    // Replicate handle across the number of steps.
    handles_.assign(seq_length, handle);
  }

  ~CudnnRnnSequenceTensorDescriptor() override {
    // Only the first one needs to be destroyed. All others are the same.
    mlopenStatus_t status =
        dynload::mlopenDestroyTensorDescriptor(parent_, handles_[0]);
    CUDNN_RETURN_IF_FAIL(status, "Failed to destroy sequence tensor desciptor");
  }

  const mlopenTensorDescriptor_t* handles() const {
    if (!ok()) return nullptr;
    CHECK(!handles_.empty()) << "handles cannot be empty";
    return handles_.data();
  }

  int seq_length() const { return seq_length_; }
  int batch_size() const { return batch_size_; }
  int data_size() const { return data_size_; }

 private:
  CUDAExecutor* parent_;
  int seq_length_;
  int batch_size_;
  int data_size_;
  mlopenDataType_t data_type_;
  std::vector<mlopenTensorDescriptor_t> handles_;
  port::Status status_;
  SE_DISALLOW_COPY_AND_ASSIGN(CudnnRnnSequenceTensorDescriptor);
};

class CudnnRnnStateTensorDescriptor
    : public CudnnDescriptorCommon<dnn::RnnStateTensorDescriptor> {
 public:
  CudnnRnnStateTensorDescriptor(CUDAExecutor* parent, int num_layers,
                                int batch_size, int data_size,
                                mlopenDataType_t data_type)
      : parent_(parent),
        handle_(nullptr),
        num_layers_(num_layers),
        batch_size_(batch_size),
        data_size_(data_size),
        data_type_(data_type) {
    mlopenStatus_t status =
        dynload::mlopenCreateTensorDescriptor(parent, &handle_);
    CUDNN_RETURN_IF_FAIL(status, "Failed to create tensor descriptor");
    int dims[] = {num_layers, batch_size, data_size};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    status = dynload::mlopenSetTensorNdDescriptor(
        parent, handle_ /*tensorDesc*/, data_type /*dataType*/,
        sizeof(dims) / sizeof(dims[0]) /*nbDims*/, dims /*dimA*/,
        strides /*strideA*/);
    CUDNN_RETURN_IF_FAIL(status, "Failed to update tensor descriptor");
  }

  ~CudnnRnnStateTensorDescriptor() override {
    if (!handle_) {
      mlopenStatus_t status =
          dynload::mlopenDestroyTensorDescriptor(parent_, handle_);
      CUDNN_RETURN_IF_FAIL(status, "Unable to destroy RNN state tensor");
    }
  }

  mlopenTensorDescriptor_t handle() const {
    if (!ok()) return nullptr;
    return handle_;
  }
  int num_layers() const { return num_layers_; }
  int batch_size() const { return batch_size_; }
  int data_size() const { return data_size_; }

 private:
  CUDAExecutor* parent_;
  mlopenTensorDescriptor_t handle_;
  int num_layers_;
  int batch_size_;
  int data_size_;
  port::Status status_;
  mlopenDataType_t data_type_;
  SE_DISALLOW_COPY_AND_ASSIGN(CudnnRnnStateTensorDescriptor);
};

namespace {

struct RnnModelDims {
  int num_layers = 0;
  int batch_size = 0;
  int seq_length = 0;
  int hidden_size = 0;
  int input_size = 0;
  int dir_count = 0;
};

template <class T>
bool ExtractAndCheckRnnForward(
    const CudnnRnnDescriptor& rnn_desc,
    const CudnnRnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<T>& input_data,
    const CudnnRnnStateTensorDescriptor& input_h_desc,
    const DeviceMemory<T>& input_h_data,
    const CudnnRnnStateTensorDescriptor& input_c_desc,
    const DeviceMemory<T>& input_c_data, const DeviceMemory<T>& params,
    const CudnnRnnSequenceTensorDescriptor& output_desc,
    const DeviceMemory<T>& output_data,
    const CudnnRnnStateTensorDescriptor& output_h_desc,
    const DeviceMemory<T>& output_h_data,
    const CudnnRnnStateTensorDescriptor& output_c_desc,
    const DeviceMemory<T>& output_c_data, RnnModelDims* model_dims) {
  // extract model parameters
  model_dims->num_layers = rnn_desc.num_layers();
  model_dims->batch_size = input_desc.batch_size();
  model_dims->seq_length = input_desc.seq_length();
  model_dims->hidden_size = rnn_desc.hidden_size();
  model_dims->input_size = input_desc.data_size();
  model_dims->dir_count =
      (rnn_desc.direction_mode() == CUDNN_BIDIRECTIONAL) ? 2 : 1;

  // check parameters
  if (!(input_h_desc.num_layers() ==
            model_dims->num_layers * model_dims->dir_count &&
        input_h_desc.batch_size() == model_dims->batch_size &&
        input_h_desc.data_size() == model_dims->hidden_size)) {
    LOG(ERROR) << "Invalid input_h shape";
    return false;
  }
  if (!(input_h_desc.num_layers() == input_c_desc.num_layers() &&
        input_h_desc.batch_size() == input_c_desc.batch_size() &&
        input_h_desc.data_size() == input_c_desc.data_size())) {
    LOG(ERROR) << "Invalid input_c shape";
    return false;
  }
  if (!(output_desc.seq_length() == model_dims->seq_length &&
        output_desc.batch_size() == model_dims->batch_size &&
        output_desc.data_size() ==
            model_dims->hidden_size * model_dims->dir_count)) {
    LOG(ERROR) << "Invalid output shape";
    return false;
  }
  if (!(input_h_desc.num_layers() == output_h_desc.num_layers() &&
        input_h_desc.batch_size() == output_h_desc.batch_size() &&
        input_h_desc.data_size() == output_h_desc.data_size())) {
    LOG(ERROR) << "Invalid output_h shape";
    return false;
  }
  if (!(input_h_desc.num_layers() == output_c_desc.num_layers() &&
        input_h_desc.batch_size() == output_c_desc.batch_size() &&
        input_h_desc.data_size() == output_c_desc.data_size())) {
    LOG(ERROR) << "Invalid output_h shape";
    return false;
  }

  return true;
}

bool CheckRNNParameterSize(CUDAExecutor* parent, mlopenHandle_t mlopen_handle,
                           const CudnnRnnDescriptor& rnn_desc,
                           const CudnnRnnSequenceTensorDescriptor& input_desc) {
  size_t params_size_in_bytes = 0;
  mlopenStatus_t status = dynload::mlopenGetRNNParamsSize(
      parent, mlopen_handle /*handle*/, rnn_desc.handle() /*rnnDesc*/,
      input_desc.handles()[0] /*xDesc*/, &params_size_in_bytes /*sizeInBytes*/,
      rnn_desc.data_type() /*dataType*/);
  if (status != mlopenStatusSuccess) {
    LOG(ERROR) << "Unable to check RNN param size: " << ToString(status);
    return false;
  }
  return static_cast<int64>(params_size_in_bytes) ==
         rnn_desc.ParamsSizeInBytes();
}

bool CreateRnnWorkspace(Stream* stream, CUDAExecutor* parent,
                        mlopenHandle_t mlopen_handle,
                        const CudnnRnnDescriptor& rnn_desc,
                        const CudnnRnnSequenceTensorDescriptor& input_desc,
                        ScratchAllocator* workspace_allocator,
                        DeviceMemory<uint8>* workspace) {
  // Query the workspace size.
  size_t workspace_size_in_bytes = 0;
  mlopenStatus_t status = dynload::mlopenGetRNNWorkspaceSize(
      parent, mlopen_handle /*handle*/, rnn_desc.handle() /*rnnDesc*/,
      input_desc.seq_length() /*seqLength*/, input_desc.handles() /*xDesc*/,
      &workspace_size_in_bytes /*sizeInBytes*/);
  if (status != mlopenStatusSuccess) {
    LOG(ERROR) << "Unable to query workspace size: " << ToString(status);
    return false;
  }
  // Allocate the workspace.
  if (workspace_size_in_bytes > 0) {
    auto allocated =
        workspace_allocator->AllocateBytes(stream, workspace_size_in_bytes);
    if (!allocated.ok() || (*workspace = allocated.ValueOrDie()) == nullptr) {
      LOG(ERROR) << "Failed to allocate RNN workspace";
      return false;
    }
  } else {
    *workspace = DeviceMemory<uint8>();
  }
  return true;
}

}  // namespace

template <class T>
bool CudnnSupport::DoRnnForwardImpl(
    Stream* stream, const CudnnRnnDescriptor& rnn_desc,
    const CudnnRnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<T>& input_data,
    const CudnnRnnStateTensorDescriptor& input_h_desc,
    const DeviceMemory<T>& input_h_data,
    const CudnnRnnStateTensorDescriptor& input_c_desc,
    const DeviceMemory<T>& input_c_data, const DeviceMemory<T>& params,
    const CudnnRnnSequenceTensorDescriptor& output_desc,
    DeviceMemory<T>* output_data,
    const CudnnRnnStateTensorDescriptor& output_h_desc,
    DeviceMemory<T>* output_h_data,
    const CudnnRnnStateTensorDescriptor& output_c_desc,
    DeviceMemory<T>* output_c_data, bool is_training,
    ScratchAllocator* reserve_space_allocator,
    ScratchAllocator* workspace_allocator) {
  // extract model parameters
  RnnModelDims model_dims;
  bool res = ExtractAndCheckRnnForward(
      rnn_desc, input_desc, input_data, input_h_desc, input_h_data,
      input_c_desc, input_c_data, params, output_desc, *output_data,
      output_h_desc, *output_h_data, output_c_desc, *output_c_data,
      &model_dims);
  if (!res) {
    LOG(ERROR) << "Invalid parameters for RNN Model";
    return false;
  }

  // check params size
  mutex_lock lock{dnn_handle_mutex_};

  if (!CheckRNNParameterSize(parent_, ToHandle(dnn_handle_), rnn_desc,
                             input_desc)) {
    LOG(ERROR) << "Invalid parameters";
    return false;
  }

  // create the workspace
  DeviceMemory<uint8> workspace;
  if (!CreateRnnWorkspace(stream, parent_, ToHandle(dnn_handle_), rnn_desc,
                          input_desc, workspace_allocator, &workspace)) {
    LOG(ERROR) << "Unable to create rnn workspace";
    return false;
  }

  // query the reserve space size
  // allocate the reserve space
  DeviceMemory<uint8> reserve_space;
  if (is_training) {
    size_t reserve_space_size_in_bytes = 0;
    mlopenStatus_t status = dynload::mlopenGetRNNTrainingReserveSize(
        parent_, ToHandle(dnn_handle_) /*handle*/,
        rnn_desc.handle() /*rnnDesc*/, model_dims.seq_length /*seqLength*/,
        input_desc.handles() /*xDesc*/,
        &reserve_space_size_in_bytes /*sizeInBytes*/);
    if (status != mlopenStatusSuccess) {
      LOG(ERROR) << "Unable to query reserve space size: " << ToString(status);
      return false;
    }

    if (reserve_space_size_in_bytes > 0) {
      auto allocated = reserve_space_allocator->AllocateBytes(
          stream, reserve_space_size_in_bytes);
      if (!allocated.ok() ||
          (reserve_space = allocated.ValueOrDie()) == nullptr) {
        LOG(ERROR) << "Fail to allocate RNN reserve space";
        return false;
      }
    }
  }

  // make the forward call
  if (!is_training) {
    mlopenStatus_t status = dynload::mlopenRNNForwardInference(
        parent_, ToHandle(dnn_handle_) /*handle*/,
        rnn_desc.handle() /*rnnDesc*/, model_dims.seq_length /*seqLength*/,
        input_desc.handles() /*xDesc*/, input_data.opaque() /*x*/,
        input_h_desc.handle() /*hxDesc*/, input_h_data.opaque() /*hx*/,
        input_c_desc.handle() /*cxDesc*/, input_c_data.opaque() /*cx*/,
        rnn_desc.params_handle() /*wDesc*/, params.opaque() /*w*/,
        output_desc.handles() /*yDesc*/, output_data->opaque() /*y*/,
        output_h_desc.handle() /*hyDesc*/, output_h_data->opaque() /*hy*/,
        output_c_desc.handle() /*cyDesc*/, output_c_data->opaque() /*cy*/,
        workspace.opaque() /*workspace*/,
        workspace.size() /*workSpaceSizeInBytes*/);
    if (status != mlopenStatusSuccess) {
      LOG(ERROR) << "Failed to call mlopenRNNForwardInference: "
                 << ToString(status);
      return false;
    }
  } else {
    mlopenStatus_t status = dynload::mlopenRNNForwardTraining(
        parent_, ToHandle(dnn_handle_) /*handle*/,
        rnn_desc.handle() /*rnnDesc*/, model_dims.seq_length /*seqLength*/,
        input_desc.handles() /*xDesc*/, input_data.opaque() /*x*/,
        input_h_desc.handle() /*hxDesc*/, input_h_data.opaque() /*hx*/,
        input_c_desc.handle() /*cxDesc*/, input_c_data.opaque() /*cx*/,
        rnn_desc.params_handle() /*wDesc*/, params.opaque() /*w*/,
        output_desc.handles() /*yDesc*/, output_data->opaque() /*y*/,
        output_h_desc.handle() /*hyDesc*/, output_h_data->opaque() /*hy*/,
        output_c_desc.handle() /*cyDesc*/, output_c_data->opaque() /*cy*/,
        workspace.opaque() /*workspace*/,
        workspace.size() /*workSpaceSizeInBytes*/,
        reserve_space.opaque() /*reserveSpace*/,
        reserve_space.size() /*reserveSpaceSizeInBytes*/);
    if (status != mlopenStatusSuccess) {
      LOG(ERROR) << "Failed to call mlopenRNNForwardTraining"
                 << ToString(status);
      return false;
    }
  }

  return true;
}

template <class T>
bool CudnnSupport::DoRnnBackwardImpl(
    Stream* stream, const CudnnRnnDescriptor& rnn_desc,
    const CudnnRnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<T>& input_data,
    const CudnnRnnStateTensorDescriptor& input_h_desc,
    const DeviceMemory<T>& input_h_data,
    const CudnnRnnStateTensorDescriptor& input_c_desc,
    const DeviceMemory<T>& input_c_data, const DeviceMemory<T>& params,
    const CudnnRnnSequenceTensorDescriptor& output_desc,
    const DeviceMemory<T>& output_data,
    const CudnnRnnStateTensorDescriptor& output_h_desc,
    const DeviceMemory<T>& output_h_data,
    const CudnnRnnStateTensorDescriptor& output_c_desc,
    const DeviceMemory<T>& output_c_data,
    const DeviceMemory<float>& output_backprop_data,
    const DeviceMemory<float>& output_h_backprop_data,
    const DeviceMemory<float>& output_c_backprop_data,
    DeviceMemory<float>* input_backprop_data,
    DeviceMemory<float>* input_h_backprop_data,
    DeviceMemory<float>* input_c_backprop_data,
    DeviceMemory<float>* params_backprop_data,
    DeviceMemory<uint8>* reserve_space_data,
    ScratchAllocator* workspace_allocator) {
  // extract model parameters
  RnnModelDims model_dims;
  bool res = ExtractAndCheckRnnForward(
      rnn_desc, input_desc, input_data, input_h_desc, input_h_data,
      input_c_desc, input_c_data, params, output_desc, output_data,
      output_h_desc, output_h_data, output_c_desc, output_c_data, &model_dims);
  if (!res) {
    LOG(ERROR) << "Invalid parameters for RNN Model";
    return false;
  }

  // check params size
  mutex_lock lock{dnn_handle_mutex_};

  if (!CheckRNNParameterSize(parent_, ToHandle(dnn_handle_), rnn_desc,
                             input_desc)) {
    LOG(ERROR) << "Invalid parameters";
    return false;
  }

  // create the workspace
  DeviceMemory<uint8> workspace;
  if (!CreateRnnWorkspace(stream, parent_, ToHandle(dnn_handle_), rnn_desc,
                          input_desc, workspace_allocator, &workspace)) {
    LOG(ERROR) << "Unable to create rnn workspace";
    return false;
  }

  // make the backward data call
  mlopenStatus_t status = dynload::mlopenRNNBackwardData(
      parent_, ToHandle(dnn_handle_) /*handle*/, rnn_desc.handle() /*rnnDesc*/,
      model_dims.seq_length /*seqLength*/, output_desc.handles() /*yDesc*/,
      output_data.opaque() /*y*/, output_desc.handles() /*dyDesc*/,
      output_backprop_data.opaque() /*dy*/, output_h_desc.handle() /*dhyDesc*/,
      output_h_backprop_data.opaque() /*dhy*/,
      output_c_desc.handle() /*dcyDesc*/,
      output_c_backprop_data.opaque() /*dcy*/,
      rnn_desc.params_handle() /*wDesc*/, params.opaque() /*w*/,
      input_h_desc.handle() /*hxDesc*/, input_h_data.opaque() /*hx*/,
      input_c_desc.handle() /*cxDesc*/, input_c_data.opaque() /*cx*/,
      input_desc.handles() /*dxDesc*/, input_backprop_data->opaque() /*dx*/,
      input_h_desc.handle() /*dhxDesc*/,
      input_h_backprop_data->opaque() /*dhx*/,
      input_c_desc.handle() /*dcxDesc*/,
      input_c_backprop_data->opaque() /*dcx*/, workspace.opaque() /*workspace*/,
      workspace.size() /*workSpaceSizeInBytes*/,
      reserve_space_data->opaque() /*reserveSpace*/,
      reserve_space_data->size() /*reserveSpaceSizeInBytes*/);
  if (status != mlopenStatusSuccess) {
    LOG(ERROR) << "Failed to call mlopenRNNBackwardData: " << ToString(status);
    return false;
  }

  if (params_backprop_data != nullptr) {
    // Clear the dw to zeros.
    stream->ThenMemZero(params_backprop_data, params_backprop_data->size());
    // make the backward weight call
    status = dynload::mlopenRNNBackwardWeights(
        parent_, ToHandle(dnn_handle_) /*handle*/,
        rnn_desc.handle() /*rnnDesc*/, model_dims.seq_length /*seqLength*/,
        input_desc.handles() /*xDesc*/, input_data.opaque() /*x*/,
        input_h_desc.handle() /*hxDesc*/, input_h_data.opaque() /*hx*/,
        output_desc.handles() /*yDesc*/, output_data.opaque() /*y*/,
        workspace.opaque() /*workspace*/,
        workspace.size() /*workSpaceSizeInBytes*/,
        rnn_desc.params_handle() /*dwDesc*/,
        params_backprop_data->opaque() /*dw*/,
        reserve_space_data->opaque() /*reserveSpace*/,
        reserve_space_data->size() /*reserveSpaceSizeInBytes*/);
    if (status != mlopenStatusSuccess) {
      LOG(ERROR) << "Failed to call mlopenRNNBackwardWeights: "
                 << ToString(status);
      return false;
    }
  }

  return true;
}

#endif  // CUDNN_VERSION

port::StatusOr<std::unique_ptr<dnn::RnnDescriptor>>
CudnnSupport::createRnnDescriptor(int num_layers, int hidden_size,
                                  int input_size, dnn::RnnInputMode input_mode,
                                  dnn::RnnDirectionMode direction_mode,
                                  dnn::RnnMode rnn_mode,
                                  dnn::DataType data_type, float dropout,
                                  uint64 seed,
                                  ScratchAllocator* state_allocator) {
#if 0//CUDNN_VERSION >= 5000
  mutex_lock lock{dnn_handle_mutex_};
  std::unique_ptr<CudnnRnnDescriptor> rnn_desc(new CudnnRnnDescriptor(
      parent_, ToHandle(dnn_handle_), num_layers, hidden_size, input_size,
      ToCudnnRnnInputMode(input_mode), ToCudnnRnnDirectionMode(direction_mode),
      ToCudnnRnnMode(rnn_mode), ToCudnnDataType(data_type), dropout, seed,
      state_allocator));
  if (!rnn_desc->ok()) {
    return rnn_desc->Status();
  }
  return port::StatusOr<std::unique_ptr<dnn::RnnDescriptor>>(
      std::move(rnn_desc));
#else
  string error_msg =
      port::StrCat("createRnnDescriptor not implemented in MLOpen");

  LOG(ERROR) << error_msg;
  return port::Status{port::error::UNIMPLEMENTED, error_msg};
#endif  // CUDNN_VERSION
}

port::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
CudnnSupport::createRnnSequenceTensorDescriptor(int seq_length, int batch_size,
                                                int data_size,
                                                dnn::DataType data_type) {
#if 0//CUDNN_VERSION >= 5000
  std::unique_ptr<CudnnRnnSequenceTensorDescriptor> seq_desc(
      new CudnnRnnSequenceTensorDescriptor(parent_, seq_length, batch_size,
                                           data_size,
                                           ToCudnnDataType(data_type)));
  if (!seq_desc->ok()) {
    return seq_desc->Status();
  }
  return port::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>(
      std::move(seq_desc));
#else
  string error_msg = port::StrCat(
      "createRnnSequenceTensorDescriptor not implemented in MLOpen");

  LOG(ERROR) << error_msg;
  return port::Status{port::error::UNIMPLEMENTED, error_msg};
#endif  // CUDNN_VERSION
}

port::StatusOr<std::unique_ptr<dnn::RnnStateTensorDescriptor>>
CudnnSupport::createRnnStateTensorDescriptor(int num_layer, int batch_size,
                                             int data_size,
                                             dnn::DataType data_type) {
#if 0//CUDNN_VERSION >= 5000
  std::unique_ptr<CudnnRnnStateTensorDescriptor> state_desc(
      new CudnnRnnStateTensorDescriptor(parent_, num_layer, batch_size,
                                        data_size, ToCudnnDataType(data_type)));
  if (!state_desc->ok()) {
    return state_desc->Status();
  }
  return port::StatusOr<std::unique_ptr<dnn::RnnStateTensorDescriptor>>(
      std::move(state_desc));
#else
  string error_msg = port::StrCat(
      "createRnnStateTensorDescriptor not implemented in MLOpen");

  LOG(ERROR) << error_msg;
  return port::Status{port::error::UNIMPLEMENTED, error_msg};
#endif  // CUDNN_VERSION
}

bool CudnnSupport::DoRnnForward(
    Stream* stream, const dnn::RnnDescriptor& rnn_desc,
    const dnn::RnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<float>& input_data,
    const dnn::RnnStateTensorDescriptor& input_h_desc,
    const DeviceMemory<float>& input_h_data,
    const dnn::RnnStateTensorDescriptor& input_c_desc,
    const DeviceMemory<float>& input_c_data, const DeviceMemory<float>& params,
    const dnn::RnnSequenceTensorDescriptor& output_desc,
    DeviceMemory<float>* output_data,
    const dnn::RnnStateTensorDescriptor& output_h_desc,
    DeviceMemory<float>* output_h_data,
    const dnn::RnnStateTensorDescriptor& output_c_desc,
    DeviceMemory<float>* output_c_data, bool is_training,
    ScratchAllocator* reserve_space_allocator,
    ScratchAllocator* workspace_allocator) {
#if 0//CUDNN_VERSION >= 5000
  const CudnnRnnDescriptor& mlopen_rnn_desc =
      static_cast<const CudnnRnnDescriptor&>(rnn_desc);
  const CudnnRnnSequenceTensorDescriptor& mlopen_input_desc =
      static_cast<const CudnnRnnSequenceTensorDescriptor&>(input_desc);
  const CudnnRnnStateTensorDescriptor& mlopen_input_h_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(input_h_desc);
  const CudnnRnnStateTensorDescriptor& mlopen_input_c_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(input_c_desc);
  const CudnnRnnSequenceTensorDescriptor& mlopen_output_desc =
      static_cast<const CudnnRnnSequenceTensorDescriptor&>(output_desc);
  const CudnnRnnStateTensorDescriptor& mlopen_output_h_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(output_h_desc);
  const CudnnRnnStateTensorDescriptor& mlopen_output_c_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(output_c_desc);

  return DoRnnForwardImpl<float>(
      stream, mlopen_rnn_desc, mlopen_input_desc, input_data, mlopen_input_h_desc,
      input_h_data, mlopen_input_c_desc, input_c_data, params, mlopen_output_desc,
      output_data, mlopen_output_h_desc, output_h_data, mlopen_output_c_desc,
      output_c_data, is_training, reserve_space_allocator, workspace_allocator);
#else
  return false;
#endif  // CUDNN_VERSION
}

bool CudnnSupport::DoRnnBackward(
    Stream* stream, const dnn::RnnDescriptor& rnn_desc,
    const dnn::RnnSequenceTensorDescriptor& input_desc,
    const DeviceMemory<float>& input_data,
    const dnn::RnnStateTensorDescriptor& input_h_desc,
    const DeviceMemory<float>& input_h_data,
    const dnn::RnnStateTensorDescriptor& input_c_desc,
    const DeviceMemory<float>& input_c_data, const DeviceMemory<float>& params,
    const dnn::RnnSequenceTensorDescriptor& output_desc,
    const DeviceMemory<float>& output_data,
    const dnn::RnnStateTensorDescriptor& output_h_desc,
    const DeviceMemory<float>& output_h_data,
    const dnn::RnnStateTensorDescriptor& output_c_desc,
    const DeviceMemory<float>& output_c_data,
    const DeviceMemory<float>& output_backprop_data,
    const DeviceMemory<float>& output_h_backprop_data,
    const DeviceMemory<float>& output_c_backprop_data,
    DeviceMemory<float>* input_backprop_data,
    DeviceMemory<float>* input_h_backprop_data,
    DeviceMemory<float>* input_c_backprop_data,
    DeviceMemory<float>* params_backprop_data,
    DeviceMemory<uint8>* reserve_space_data,
    ScratchAllocator* workspace_allocator) {
#if 0//CUDNN_VERSION >= 5000
  const CudnnRnnDescriptor& mlopen_rnn_desc =
      static_cast<const CudnnRnnDescriptor&>(rnn_desc);
  const CudnnRnnSequenceTensorDescriptor& mlopen_input_desc =
      static_cast<const CudnnRnnSequenceTensorDescriptor&>(input_desc);
  const CudnnRnnStateTensorDescriptor& mlopen_input_h_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(input_h_desc);
  const CudnnRnnStateTensorDescriptor& mlopen_input_c_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(input_c_desc);
  const CudnnRnnSequenceTensorDescriptor& mlopen_output_desc =
      static_cast<const CudnnRnnSequenceTensorDescriptor&>(output_desc);
  const CudnnRnnStateTensorDescriptor& mlopen_output_h_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(output_h_desc);
  const CudnnRnnStateTensorDescriptor& mlopen_output_c_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(output_c_desc);

  return DoRnnBackwardImpl<float>(
      stream, mlopen_rnn_desc, mlopen_input_desc, input_data, mlopen_input_h_desc,
      input_h_data, mlopen_input_c_desc, input_c_data, params, mlopen_output_desc,
      output_data, mlopen_output_h_desc, output_h_data, mlopen_output_c_desc,
      output_c_data, output_backprop_data, output_h_backprop_data,
      output_c_backprop_data, input_backprop_data, input_h_backprop_data,
      input_c_backprop_data, params_backprop_data, reserve_space_data,
      workspace_allocator);
#else
  return false;
#endif  // CUDNN_VERSION
}

template <class T>
bool CudnnSupport::DoConvolveImpl(
    Stream* stream, int mlopen_type,  // Actually mlopenDataType_t.
    const BatchDescriptor& batch_descriptor, const DeviceMemory<T>& input_data,
    const FilterDescriptor& filter_descriptor,
    const DeviceMemory<T>& filter_data,
    const ConvolutionDescriptor& convolution_descriptor,
    const BatchDescriptor& output_descriptor, DeviceMemory<T>* output_data,
    ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  ScopedTensorDescriptor input_nd{parent_, batch_descriptor,
      static_cast<mlopenDataType_t>(mlopen_type)};
  ScopedTensorDescriptor output_nd{parent_, output_descriptor,
      static_cast<mlopenDataType_t>(mlopen_type)};
  ScopedFilterDescriptor filter{parent_, filter_descriptor, batch_descriptor,
      static_cast<mlopenDataType_t>(mlopen_type)};
  // TODO(sesse): Figure out under what circumstances cuDNN would
  // accept CUDNN_DATA_HALF here; probably related to compute capability
  // and cuDNN version; at least cuDNN 4 on TITAN X only supports
  // CUDNN_DATA_FLOAT even for half input.
  ScopedConvolutionDescriptor conv{parent_, convolution_descriptor,
      mlopenFloat};

  mutex_lock lock{dnn_handle_mutex_};
/*  auto status = dynload::mlopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != mlopenStatusSuccess) {
    LOG(FATAL) << "failed to set stream for mlopen handle: " << ToString(status);
  }*/
  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  const bool is_profiling = output_profile_result != nullptr;
  mlopenConvFwdAlgorithm_t algo;
  DeviceMemory<uint8> scratch;

  if (algorithm_config.algorithm() == dnn::kDefaultAlgorithm) {
    // With the default algorithm, use Cudnn's heuristics.
    auto get_algorithm = [&](bool specify_limit)
        SHARED_LOCKS_REQUIRED(dnn_handle_mutex_) {
          mlopenConvPreference_t preference =
              specify_limit ? mlopenConvolutionWorkSpaceLimit
                            : mlopenConvolutionNoWorkspace;

          auto memory_limit_bytes =
              scratch_allocator == nullptr
                  ? 0
                  : scratch_allocator->GetMemoryLimitInBytes(stream);
          if (memory_limit_bytes < 0) {
            memory_limit_bytes = 0;
          }

          mlopenConvFwdAlgorithm_t algo_to_use;
          algo_to_use = mlopenConvolutionFwdAlgoDirect;
#if 0
          status = dynload::mlopenFindConvolutionForwardAlgorithm(
              parent_, ToHandle(dnn_handle_), input_nd.handle(),
              filter.handle(), conv.handle(), output_nd.handle(),
              /*preference=*/preference,
              /*memoryLimitInBytes=*/memory_limit_bytes,
              /*algo=*/&algo_to_use);
          CHECK_EQ(status, mlopenStatusSuccess)
              << "Unable to find a suitable "
                 "algorithm for doing forward "
                 "convolution";
#endif
          return algo_to_use;
        };

    algo = get_algorithm(/*specify_limit=*/scratch_allocator != nullptr);

#if 0
    if (scratch_allocator != nullptr) {
      size_t size_in_bytes;
      status = dynload::mlopenGetConvolutionForwardWorkspaceSize(
          parent_, ToHandle(dnn_handle_), /*srcDesc=*/input_nd.handle(),
          /*filterDesc=*/filter.handle(), /*convDesc=*/conv.handle(),
          /*destDesc=*/output_nd.handle(), /*algo=*/algo,
          /*sizeInBytes=*/&size_in_bytes);
      if (status == mlopenStatusSuccess && size_in_bytes != 0) {
        auto allocated =
            scratch_allocator->AllocateBytes(stream, size_in_bytes);
        if (allocated.ok()) {
          scratch = allocated.ValueOrDie();
        }
      }
    }
#endif

    // If we didn't allocate any scratch space (perhaps because of failed
    // allocation), we force a switch back to the "no workspace" algorithm.
    if (scratch == nullptr) {
      algo = get_algorithm(/*specify_limit=*/false);
    }
  } else {
    // An algorithm has been specified.
    algo = ToConvForwardAlgo(algorithm_config.algorithm());

    size_t size_in_bytes = 0;
#if 0
    status = dynload::mlopenGetConvolutionForwardWorkspaceSize(
        parent_, ToHandle(dnn_handle_), /*srcDesc=*/input_nd.handle(),
        /*filterDesc=*/filter.handle(), /*convDesc=*/conv.handle(),
        /*destDesc=*/output_nd.handle(), /*algo=*/algo,
        /*sizeInBytes=*/&size_in_bytes);
    if (status != mlopenStatusSuccess) {
      if (is_profiling) {
        // Silently return when we are profiling.
        return false;
      }
      LOG(FATAL) << "Cannot query the size of workspace needed for the given "
                    "algorithm: "
                 << algorithm_config.algorithm();
    }
    if (size_in_bytes != 0) {
      if (scratch_allocator == nullptr) {
        LOG(FATAL) << "An allocator must be specified when scratch memory is "
                      "needed";
      }
      auto allocated = scratch_allocator->AllocateBytes(stream, size_in_bytes);
      if (is_profiling && !allocated.ok()) {
        // Silently return when we are profiling.
        return false;
      }
      if (allocated.ok()) {
        scratch = allocated.ValueOrDie();
      }
      if (scratch == nullptr) {
        CHECK(algorithm_config.algorithm_no_scratch() != dnn::kDefaultAlgorithm)
            << "The primary convolution algorithm failed memory allocation, "
               "while a secondary algorithm is not provided.";
        algo = ToConvForwardAlgo(algorithm_config.algorithm_no_scratch());
      }
    }
#endif
  }

//  std::unique_ptr<CUDATimer> timer;
  if (is_profiling) {
#if 0
    timer.reset(new CUDATimer(parent_));
    timer->Init();
    // The start and stop of the timer should be as close to the Cudnn call as
    // possible. It is still possible for other threads to issue workload on
    // to this stream. So it could take multiple profiling measurements.
    timer->Start(AsCUDAStream(stream));
#endif
    dynload::mlopenEnableProfiling(parent_, ToHandle(dnn_handle_), true);
  }

  void* t = NULL;

  auto status = dynload::mlopenConvolutionForward(
      parent_, ToHandle(dnn_handle_),
      /*alpha=*/&alpha, /*srcDesc=*/input_nd.handle(),
      /*srcData=*/input_data.opaque(), /*filterDesc=*/filter.handle(),
      /*filterData=*/filter_data.opaque(), /*convDesc=*/conv.handle(),
      /*algo=*/algo, /*beta=*/&beta, /*destDesc=*/output_nd.handle(), 
      /*destData=*/output_data->opaque(), /*workSpace=*//*scratch.opaque()*/t,
      /*workSpaceSizeInBytes=*//*scratch.size()*/0);
  if (is_profiling) {
    float ElapsedTime;
    dynload::mlopenGetKernelTime(parent_, ToHandle(dnn_handle_), &ElapsedTime);
//  timer->Stop(AsCUDAStream(stream));
    output_profile_result->set_is_valid(true);
    output_profile_result->set_algorithm(algo);
    output_profile_result->set_elapsed_time_in_ms(ElapsedTime);
//  timer->Destroy();
  }

  if (status != mlopenStatusSuccess) {
    // Silently return when we are profiling.
    if (!is_profiling) {
      LOG(FATAL) << "failed to enqueue convolution on stream: "
                 << ToString(status);
    }
    return false;
  }

  return true;
}

bool CudnnSupport::GetConvolveAlgorithms(
    std::vector<dnn::AlgorithmType>* out_algorithms) {
  out_algorithms->assign({
      // clang-format off
      mlopenConvolutionFwdAlgoGEMM,
      mlopenConvolutionFwdAlgoDirect,
      mlopenConvolutionFwdAlgoFFT,
      mlopenConvolutionFwdAlgoWinograd,
      // clang-format on
  });
  return true;
}

bool CudnnSupport::GetConvolveBackwardDataAlgorithms(
    std::vector<dnn::AlgorithmType>* out_algorithms) {
  out_algorithms->assign({
      // clang-format off
      mlopenConvolutionBwdDataAlgo_0,
      // clang-format on
  });
  return true;
}

bool CudnnSupport::GetConvolveBackwardFilterAlgorithms(
    std::vector<dnn::AlgorithmType>* out_algorithms) {
  out_algorithms->assign({
      // clang-format off
      mlopenConvolutionBwdWeightsAlgoGEMM,
      mlopenConvolutionBwdWeightsAlgoDirect,
      // clang-format on
  });
  return true;
}

bool CudnnSupport::DoConvolve(
    Stream* stream, const BatchDescriptor& batch_descriptor,
    const DeviceMemory<float>& input_data,
    const FilterDescriptor& filter_descriptor,
    const DeviceMemory<float>& filter_data,
    const ConvolutionDescriptor& convolution_descriptor,
    const BatchDescriptor& output_descriptor, DeviceMemory<float>* output_data,
    ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return DoConvolveImpl<float>(
      stream, mlopenFloat, batch_descriptor, input_data, filter_descriptor,
      filter_data, convolution_descriptor, output_descriptor, output_data,
      scratch_allocator, algorithm_config, output_profile_result);
}

bool CudnnSupport::DoConvolve(
    Stream* stream, const BatchDescriptor& batch_descriptor,
    const DeviceMemory<double>& input_data,
    const FilterDescriptor& filter_descriptor,
    const DeviceMemory<double>& filter_data,
    const ConvolutionDescriptor& convolution_descriptor,
    const BatchDescriptor& output_descriptor,
    DeviceMemory<double>* output_data) {
  LOG(ERROR) << "double-based DNN not yet implemented";
  return false;
}

bool CudnnSupport::DoConvolve(
    Stream* stream, const BatchDescriptor& batch_descriptor,
    const DeviceMemory<Eigen::half>& input_data,
    const FilterDescriptor& filter_descriptor,
    const DeviceMemory<Eigen::half>& filter_data,
    const ConvolutionDescriptor& convolution_descriptor,
    const BatchDescriptor& output_descriptor,
    DeviceMemory<Eigen::half>* output_data, ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return DoConvolveImpl<Eigen::half>(
      stream, mlopenHalf, batch_descriptor, input_data, filter_descriptor,
      filter_data, convolution_descriptor, output_descriptor, output_data,
      scratch_allocator, algorithm_config, output_profile_result);
}

template<class T>
DeviceMemory<T> CudnnSupport::MaybeTransformLayout(
    Stream* stream,
    int mlopen_type,  // Actually mlopenDataType_t.
    BatchDescriptor* output_descriptor,
    DeviceMemory<T> backward_output_data,
    std::unique_ptr<TemporaryDeviceMemory<T>>* transform_scratch) {
  if (output_descriptor->layout() == dnn::DataLayout::kBatchDepthYX) {
    return backward_output_data;
  }
  CHECK(output_descriptor->layout() == dnn::DataLayout::kBatchYXDepth);
  *transform_scratch =
      stream->AllocateTemporaryArray<T>(backward_output_data.ElementCount())
          .ConsumeValueOrDie();
  BatchDescriptor transformed_output_descriptor;
  transformed_output_descriptor.CloneFrom(*output_descriptor);
  transformed_output_descriptor.set_layout(dnn::DataLayout::kBatchDepthYX);
  ScopedTensorDescriptor orig_out_back_nd{
      parent_, *output_descriptor, static_cast<mlopenDataType_t>(mlopen_type)};
  ScopedTensorDescriptor transformed_out_back_nd{
      parent_, transformed_output_descriptor,
      static_cast<mlopenDataType_t>(mlopen_type)};

  float alpha = 1.0f;
  float beta = 0.0f;
  auto status = dynload::mlopenTransformTensor(
      parent_, ToHandle(dnn_handle_), &alpha, orig_out_back_nd.handle(),
      backward_output_data.opaque(), &beta, transformed_out_back_nd.handle(),
      (*transform_scratch)->mutable_device_memory()->opaque());

  if (status != mlopenStatusSuccess) {
    LOG(FATAL) << "Failed to transform the data layout.";
  }
  output_descriptor->set_layout(dnn::DataLayout::kBatchDepthYX);
  return (*transform_scratch)->device_memory();
}

template <class T>
bool CudnnSupport::DoConvolveBackwardDataImpl(
    Stream* stream,
    int mlopen_type,  // Actually mlopenDataType_t.
    const FilterDescriptor& filter_descriptor,
    const DeviceMemory<T>& filter_data,
    const BatchDescriptor& output_descriptor_in,
    DeviceMemory<T> backward_output_data,
    const ConvolutionDescriptor& convolution_descriptor,
    const BatchDescriptor& input_descriptor,
    DeviceMemory<T>* backward_input_data, ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  mutex_lock lock{dnn_handle_mutex_};
/*  auto status = dynload::mlopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != mlopenStatusSuccess) {
    LOG(FATAL) << "failed to set stream for mlopen handle: " << ToString(status);
  }*/

  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  // TBD(keveman): remove once cuDNN supports kBatchYXDepth for backward pass.
  BatchDescriptor output_descriptor;
  output_descriptor.CloneFrom(output_descriptor_in);
  std::unique_ptr<TemporaryDeviceMemory<T>> transform_scratch;
  backward_output_data = MaybeTransformLayout(
      stream, mlopen_type, &output_descriptor, backward_output_data,
      &transform_scratch);

  ScopedTensorDescriptor out_back_nd{parent_, output_descriptor,
                                     static_cast<mlopenDataType_t>(mlopen_type)};
  ScopedTensorDescriptor in_back_nd{parent_, input_descriptor,
                                    static_cast<mlopenDataType_t>(mlopen_type)};
  ScopedFilterDescriptor filter{parent_, filter_descriptor, input_descriptor,
                                static_cast<mlopenDataType_t>(mlopen_type)};
  // TODO(sesse): Figure out under what circumstances cuDNN would
  // accept CUDNN_DATA_HALF here; probably related to compute capability
  // and cuDNN version; at least cuDNN 4 on TITAN X only supports
  // CUDNN_DATA_FLOAT even for half input.
  ScopedConvolutionDescriptor conv{parent_, convolution_descriptor,
                                   mlopenFloat};

  const bool is_profiling = output_profile_result != nullptr;
  mlopenConvBwdDataAlgorithm_t algo;
  DeviceMemory<uint8> scratch;

  if (algorithm_config.algorithm() == dnn::kDefaultAlgorithm) {
    // With the default algorithm, use Cudnn's heuristics.
    auto get_algorithm = [&](bool specify_limit) SHARED_LOCKS_REQUIRED(
        dnn_handle_mutex_) -> mlopenConvBwdDataAlgorithm_t {
      mlopenConvPreference_t preference =
          specify_limit ? mlopenConvolutionWorkSpaceLimit
                        : mlopenConvolutionNoWorkspace;

      auto memory_limit_bytes =
          scratch_allocator == nullptr
              ? 0
              : scratch_allocator->GetMemoryLimitInBytes(stream);
      if (memory_limit_bytes < 0) {
        memory_limit_bytes = 0;
      }

      mlopenConvBwdDataAlgorithm_t algo_to_use;
      algo_to_use = mlopenConvolutionBwdDataAlgo_0;
#if 0
      mlopenStatus_t status = dynload::mlopenGetConvolutionBackwardDataAlgorithm(
          parent_, ToHandle(dnn_handle_),
          /*filterDesc=*/filter.handle(),
          /*diffDesc=*/out_back_nd.handle(),
          /*convDesc=*/conv.handle(),
          /*gradDesc=*/in_back_nd.handle(),
          /*preference=*/preference,
          /*memoryLimitInBytes=*/memory_limit_bytes,
          /*algo=*/&algo_to_use);
      CHECK_EQ(status, mlopenStatusSuccess) << "Unable to find a suitable "
                                                "algorithm for doing backward "
                                                "filter convolution";
#endif
      return algo_to_use;
    };

    algo = get_algorithm(/*specify_limit=*/scratch_allocator != nullptr);
#if 0
    if (scratch_allocator != nullptr) {
      size_t size_in_bytes;
      status = dynload::mlopenGetConvolutionBackwardDataWorkspaceSize(
          parent_, ToHandle(dnn_handle_),
          /*filterDesc=*/filter.handle(),
          /*diffDesc=*/out_back_nd.handle(),
          /*convDesc=*/conv.handle(),
          /*gradDesc=*/in_back_nd.handle(),
          /*algo=*/algo,
          /*sizeInBytes=*/&size_in_bytes);
      if (status == mlopenStatusSuccess && size_in_bytes != 0) {
        auto allocated =
            scratch_allocator->AllocateBytes(stream, size_in_bytes);
        if (allocated.ok()) {
          scratch = allocated.ValueOrDie();
        }
      }
    }
#endif

    // If we didn't allocate any scratch space (perhaps because of failed
    // allocation), we force a switch back to the "no workspace" algorithm.
    if (scratch == nullptr) {
      algo = get_algorithm(/*specify_limit=*/false);
    }
  } else {
    // An algorithm has been specified.
    algo = ToConvBackwardDataAlgo(algorithm_config.algorithm());
    size_t size_in_bytes = 0;
#if 0
    status = dynload::mlopenGetConvolutionBackwardDataWorkspaceSize(
        parent_, ToHandle(dnn_handle_),
        /*filterDesc=*/filter.handle(),
        /*diffDesc=*/out_back_nd.handle(),
        /*convDesc=*/conv.handle(),
        /*gradDesc=*/in_back_nd.handle(),
        /*algo=*/algo,
        /*sizeInBytes=*/&size_in_bytes);
    if (status != mlopenStatusSuccess) {
      if (is_profiling) {
        // Silently return when we are profiling.
        return false;
      }
      LOG(FATAL) << "Cannot query the size of workspace needed for the given "
                    "algorithm: "
                 << algorithm_config.algorithm();
    }
    if (size_in_bytes != 0) {
      if (scratch_allocator == nullptr) {
        LOG(FATAL) << "An allocator must be specified when scratch memory is "
                      "needed";
      }
      auto allocated = scratch_allocator->AllocateBytes(stream, size_in_bytes);
      if (is_profiling && !allocated.ok()) {
        // Silently return when we are profiling.
        return false;
      }
      if (allocated.ok()) {
        scratch = allocated.ValueOrDie();
      }
      if (scratch == nullptr) {
        CHECK(algorithm_config.algorithm_no_scratch() != dnn::kDefaultAlgorithm)
            << "The primary convolution algorithm failed memory allocation, "
               "while a secondary algorithm is not provided.";
        algo = ToConvBackwardDataAlgo(algorithm_config.algorithm_no_scratch());
      }
    }
#endif
  }

//  std::unique_ptr<CUDATimer> timer;
  if (is_profiling) {
#if 0
    timer.reset(new CUDATimer(parent_));
    timer->Init();
    // The start and stop of the timer should be as close to the Cudnn call as
    // possible. It is still possible for other threads to issue workload on
    // to this stream. So it could take multiple profiling measurements.
    timer->Start(AsCUDAStream(stream));
#endif
    dynload::mlopenEnableProfiling(parent_, ToHandle(dnn_handle_), true);
  }

  void* t = NULL;

  auto status = dynload::mlopenConvolutionBackwardData(
      parent_, ToHandle(dnn_handle_),
      /*alpha=*/&alpha,
      /*diffDesc=*/out_back_nd.handle(),
      /*diffData=*/backward_output_data.opaque(),
      /*filterDesc=*/filter.handle(),
      /*filterData=*/filter_data.opaque(),
      /*convDesc=*/conv.handle(),
      /*algo=*/algo,
      /*beta=*/&beta,
      /*gradDesc=*/in_back_nd.handle(),
      /*gradData=*/backward_input_data->opaque(),
      /*workSpace=*//*scratch.opaque()*/t,
      /*workSpaceSizeInBytes=*//*scratch.size()*/0);
  if (is_profiling) {
    float ElapsedTime;
    dynload::mlopenGetKernelTime(parent_, ToHandle(dnn_handle_), &ElapsedTime);
//    timer->Stop(AsCUDAStream(stream));
    output_profile_result->set_is_valid(true);
    output_profile_result->set_algorithm(algo);
    output_profile_result->set_elapsed_time_in_ms(ElapsedTime);
//    timer->Destroy();
  }
  if (status != mlopenStatusSuccess) {
    // Silently return when we are profiling.
    if (!is_profiling) {
      LOG(FATAL) << "failed to enqueue convolution on stream: "
                 << ToString(status);
    }
    return false;
  }
  return true;
}

bool CudnnSupport::DoConvolveBackwardData(
    Stream* stream, const FilterDescriptor& filter_descriptor,
    const DeviceMemory<float>& filter_data,
    const BatchDescriptor& output_descriptor_in,
    DeviceMemory<float> backward_output_data,
    const ConvolutionDescriptor& convolution_descriptor,
    const BatchDescriptor& input_descriptor,
    DeviceMemory<float>* backward_input_data,
    ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return DoConvolveBackwardDataImpl(
      stream, mlopenFloat, filter_descriptor, filter_data,
      output_descriptor_in, backward_output_data, convolution_descriptor,
      input_descriptor, backward_input_data, scratch_allocator,
      algorithm_config, output_profile_result);
}

bool CudnnSupport::DoConvolveBackwardData(
    Stream* stream, const FilterDescriptor& filter_descriptor,
    const DeviceMemory<Eigen::half>& filter_data,
    const BatchDescriptor& output_descriptor_in,
    DeviceMemory<Eigen::half> backward_output_data,
    const ConvolutionDescriptor& convolution_descriptor,
    const BatchDescriptor& input_descriptor,
    DeviceMemory<Eigen::half>* backward_input_data,
    ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return DoConvolveBackwardDataImpl(
      stream, mlopenHalf, filter_descriptor, filter_data,
      output_descriptor_in, backward_output_data, convolution_descriptor,
      input_descriptor, backward_input_data, scratch_allocator,
      algorithm_config, output_profile_result);
}

template <class T>
bool CudnnSupport::DoConvolveBackwardFilterImpl(
    Stream* stream, int mlopen_type,  // Actually mlopenDataType_t.
    const dnn::BatchDescriptor& input_descriptor,
    const DeviceMemory<T>& input_data,
    const dnn::BatchDescriptor& output_descriptor_in,
    DeviceMemory<T> backward_output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemory<T>* backward_filter_data, ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  mutex_lock lock{dnn_handle_mutex_};
/*  auto status = dynload::mlopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != mlopenStatusSuccess) {
    LOG(FATAL) << "failed to set stream for mlopen handle: " << ToString(status);
  }*/

  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  // TBD(keveman): remove once cuDNN supports kBatchYXDepth for backward pass.
  BatchDescriptor output_descriptor;
  output_descriptor.CloneFrom(output_descriptor_in);
  std::unique_ptr<TemporaryDeviceMemory<T>> transform_scratch;
  backward_output_data = MaybeTransformLayout(
      stream, static_cast<mlopenDataType_t>(mlopen_type),
      &output_descriptor, backward_output_data,
      &transform_scratch);

  ScopedTensorDescriptor out_back_nd{parent_, output_descriptor,
        static_cast<mlopenDataType_t>(mlopen_type)};
  ScopedTensorDescriptor input_nd{parent_, input_descriptor,
          static_cast<mlopenDataType_t>(mlopen_type)};
  ScopedFilterDescriptor filter{parent_, filter_descriptor, input_descriptor,
        static_cast<mlopenDataType_t>(mlopen_type)};
  // TODO(sesse): Figure out under what circumstances cuDNN would
  // accept CUDNN_DATA_HALF here; probably related to compute capability
  // and cuDNN version; at least cuDNN 4 on TITAN X only supports
  // CUDNN_DATA_FLOAT even for half input.
  ScopedConvolutionDescriptor conv{parent_, convolution_descriptor,
      mlopenFloat};

  const bool is_profiling = output_profile_result != nullptr;
  mlopenConvBwdWeightsAlgorithm_t algo;
  DeviceMemory<uint8> scratch;

  if (algorithm_config.algorithm() == dnn::kDefaultAlgorithm) {
    // With the default algorithm, use Cudnn's heuristics.

    // Lambda that retrieves the algorithm.
    // specify_limit will occur when we have a scratch allocator and it succeeds
    // in allocating; otherwise, we'll fall back to the "no workspace" version.
    auto get_algorithm = [&](bool specify_limit) SHARED_LOCKS_REQUIRED(
        dnn_handle_mutex_) {
      mlopenConvPreference_t preference =
          specify_limit ? mlopenConvolutionWorkSpaceLimit
                        : mlopenConvolutionNoWorkspace;

      auto memory_limit_bytes =
          scratch_allocator == nullptr
              ? 0
              : scratch_allocator->GetMemoryLimitInBytes(stream);
      if (memory_limit_bytes < 0) {
        memory_limit_bytes = 0;
      }

      mlopenConvBwdWeightsAlgorithm_t algo_to_use;
      algo_to_use = mlopenConvolutionBwdWeightsAlgoDirect;
#if 0
      mlopenStatus_t status =
          dynload::mlopenGetConvolutionBackwardFilterAlgorithm(
              parent_, ToHandle(dnn_handle_),
              /*srcDesc=*/input_nd.handle(),
              /*diffDesc=*/out_back_nd.handle(),
              /*convDesc=*/conv.handle(),
              /*gradDesc=*/filter.handle(),
              /*preference=*/preference,
              /*memoryLimitInBytes=*/memory_limit_bytes,
              /*algo=*/&algo_to_use);
      CHECK_EQ(status, mlopenStatusSuccess) << "Unable to find a suitable "
                                                "algorithm for doing backward "
                                                "filter convolution";
#endif
      return algo_to_use;
    };

    algo = get_algorithm(/*specify_limit=*/scratch_allocator != nullptr);

    if (scratch_allocator != nullptr) {
      size_t size_in_bytes;
      auto status = dynload::mlopenConvolutionBackwardWeightsGetWorkSpaceSize(
          parent_, /*diffDesc=*/out_back_nd.handle(),
          /*srcDesc=*/input_nd.handle() ,/*convDesc=*/conv.handle(),
          /*gradDesc=*/filter.handle(), /*sizeInBytes=*/&size_in_bytes);

      if (status == mlopenStatusSuccess && size_in_bytes != 0) {
        auto allocated =
            scratch_allocator->AllocateBytes(stream, size_in_bytes);
        if (allocated.ok()) {
          scratch = allocated.ValueOrDie();
        }
      }
    }

    // If we didn't allocate any scratch space (perhaps because of failed
    // allocation), we force a switch back to the "no workspace" algorithm.
    if (scratch == nullptr) {
      algo = get_algorithm(/*specify_limit=*/false);
    }
  } else {
    // An algorithm has been specified.
    algo = ToConvBackwardFilterAlgo(algorithm_config.algorithm());

    size_t size_in_bytes;
    auto status = dynload::mlopenConvolutionBackwardWeightsGetWorkSpaceSize(
        parent_, /*diffDesc=*/out_back_nd.handle(),
        /*srcDesc=*/input_nd.handle() ,/*convDesc=*/conv.handle(),
        /*gradDesc=*/filter.handle(), /*sizeInBytes=*/&size_in_bytes);

    if (status != mlopenStatusSuccess) {
      if (is_profiling) {
        // Silently return when we are profiling.
        return false;
      }
      LOG(FATAL) << "Cannot query the size of workspace needed for the given "
                    "algorithm: "
                 << algorithm_config.algorithm();
    }
    if (size_in_bytes != 0) {
      if (scratch_allocator == nullptr) {
        LOG(FATAL) << "An allocator must be specified when scratch memory is "
                      "needed";
      }
      auto allocated = scratch_allocator->AllocateBytes(stream, size_in_bytes);
      if (is_profiling && !allocated.ok()) {
        // Silently return when we are profiling.
        return false;
      }
      if (allocated.ok()) {
        scratch = allocated.ValueOrDie();
      }
      if (scratch == nullptr) {
        CHECK(algorithm_config.algorithm_no_scratch() != dnn::kDefaultAlgorithm)
            << "The primary convolution algorithm failed memory allocation, "
               "while a secondary algorithm is not provided.";
        algo =
            ToConvBackwardFilterAlgo(algorithm_config.algorithm_no_scratch());
      }
    }
  }

//  std::unique_ptr<CUDATimer> timer;
  if (is_profiling) {
#if 0
    timer.reset(new CUDATimer(parent_));
    timer->Init();
    // The start and stop of the timer should be as close to the Cudnn call as
    // possible. It is still possible for other threads to issue workload on
    // to this stream. So it could take multiple profiling measurements.
    timer->Start(AsCUDAStream(stream));
#endif
    dynload::mlopenEnableProfiling(parent_, ToHandle(dnn_handle_), true);
  }

  auto status = dynload::mlopenConvolutionBackwardWeights(
      parent_, ToHandle(dnn_handle_), /*alpha=*/&alpha,
      /*diffDesc=*/out_back_nd.handle(),
      /*diffData=*/backward_output_data.opaque(),
      /*srcDesc=*/input_nd.handle(),
      /*srcData=*/input_data.opaque(),
      /*convDesc=*/conv.handle(),
      /*algo=*/algo,
      /*beta=*/&beta,
      /*gradDesc=*/filter.handle(),
      /*gradData=*/backward_filter_data->opaque(),
      /*workSpace=*/scratch.opaque(),
      /*workSpaceSizeInBytes=*/scratch.size());
  if (is_profiling) {
    float ElapsedTime;
    dynload::mlopenGetKernelTime(parent_, ToHandle(dnn_handle_), &ElapsedTime);
//    timer->Stop(AsCUDAStream(stream));
    output_profile_result->set_is_valid(true);
    output_profile_result->set_algorithm(algo);
    output_profile_result->set_elapsed_time_in_ms(ElapsedTime);
//    timer->Destroy();
  }
  if (status != mlopenStatusSuccess) {
    // Silently return when we are profiling.
    if (!is_profiling) {
      LOG(FATAL) << "failed to enqueue convolution on stream: "
                 << ToString(status);
    }
    return false;
  }
  return true;
}

bool CudnnSupport::DoConvolveBackwardFilter(
    Stream* stream, const dnn::BatchDescriptor& input_descriptor,
    const DeviceMemory<float>& input_data,
    const dnn::BatchDescriptor& output_descriptor_in,
    DeviceMemory<float> backward_output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemory<float>* backward_filter_data,
    ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return DoConvolveBackwardFilterImpl(
      stream, mlopenFloat, input_descriptor, input_data,
      output_descriptor_in, backward_output_data, convolution_descriptor,
      filter_descriptor, backward_filter_data, scratch_allocator,
      algorithm_config, output_profile_result);
}

bool CudnnSupport::DoConvolveBackwardFilter(
    Stream* stream, const dnn::BatchDescriptor& input_descriptor,
    const DeviceMemory<Eigen::half>& input_data,
    const dnn::BatchDescriptor& output_descriptor_in,
    DeviceMemory<Eigen::half> backward_output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceMemory<Eigen::half>* backward_filter_data,
    ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  return DoConvolveBackwardFilterImpl(
      stream, mlopenHalf, input_descriptor, input_data,
      output_descriptor_in, backward_output_data, convolution_descriptor,
      filter_descriptor, backward_filter_data, scratch_allocator,
      algorithm_config, output_profile_result);
}

template <class T>
bool CudnnSupport::DoConvolveBackwardBiasImpl(
    Stream* stream, int mlopen_type,  // Actually mlopenDataType_t.
    const dnn::BatchDescriptor& input_descriptor,
    const DeviceMemory<T>& input_data,
    const dnn::BatchDescriptor& bias_descriptor,
    DeviceMemory<T>* backward_bias_data) {
#if 0
  mutex_lock lock{dnn_handle_mutex_};
/*  auto status = dynload::mlopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != mlopenStatusSuccess) {
    LOG(FATAL) << "failed to set stream for mlopen handle: " << ToString(status);
  }*/

  ScopedTensorDescriptor input_nd{parent_, input_descriptor,
                                  static_cast<mlopenDataType_t>(mlopen_type)};
  ScopedTensorDescriptor bias_nd{parent_, bias_descriptor,
                                 static_cast<mlopenDataType_t>(mlopen_type)};

  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  status = dynload::mlopenConvolutionBackwardBias(
      parent_, ToHandle(dnn_handle_), &alpha, input_nd.handle(),
      input_data.opaque(), &beta, bias_nd.handle(),
      backward_bias_data->opaque());
  if (status != mlopenStatusSuccess) {
    LOG(FATAL) << "failed to enqueue backward convolution on stream: "
               << ToString(status);
    return false;
  }
  return true;
#else
  LOG(FATAL) << "BackwardBias not yet implemented";
  return false;
#endif
}

bool CudnnSupport::DoConvolveBackwardBias(
    Stream* stream, const BatchDescriptor& input_descriptor,
    const DeviceMemory<double>& input_data,
    const BatchDescriptor& bias_descriptor,
    DeviceMemory<double>* backward_bias_data) {
  return DoConvolveBackwardBiasImpl(stream, mlopenDouble, input_descriptor,
                                    input_data, bias_descriptor,
                                    backward_bias_data);
}

bool CudnnSupport::DoConvolveBackwardBias(
    Stream* stream, const BatchDescriptor& input_descriptor,
    const DeviceMemory<float>& input_data,
    const BatchDescriptor& bias_descriptor,
    DeviceMemory<float>* backward_bias_data) {
  return DoConvolveBackwardBiasImpl(stream, mlopenFloat, input_descriptor,
                                    input_data, bias_descriptor,
                                    backward_bias_data);
}

bool CudnnSupport::DoConvolveBackwardBias(
    Stream* stream, const BatchDescriptor& input_descriptor,
    const DeviceMemory<Eigen::half>& input_data,
    const BatchDescriptor& bias_descriptor,
    DeviceMemory<Eigen::half>* backward_bias_data) {
  return DoConvolveBackwardBiasImpl(stream, mlopenHalf, input_descriptor,
                                    input_data, bias_descriptor,
                                    backward_bias_data);
}

bool CudnnSupport::DoMatMul(Stream* stream,
                            const DeviceMemory<float>& input_data,
                            const DeviceMemory<float>& weights,
                            const dnn::BatchDescriptor& input_dimensions,
                            const dnn::BatchDescriptor& output_dimensions,
                            DeviceMemory<float>* output_data) {
  if (input_dimensions.count() != output_dimensions.count()) {
    LOG(ERROR) << "MatMul input and output dimensions are not compatible.";
    return false;
  }

  // We do not permute the input or output, instead we just
  // reinterpret the layout. We are working with row-major matrices
  // and the rows of the input and output correspond to batch, so
  // batch has to be outermost in both the input and output.
  //
  // By adding transposes to the BLAS gemm call we could perhaps make
  // the kYXDepthBatch layout work as well, but there has been no need
  // for that so far.
  if (input_dimensions.layout() != dnn::DataLayout::kBatchYXDepth &&
      input_dimensions.layout() != dnn::DataLayout::kBatchDepthYX) {
    LOG(ERROR) << "Unsupported MatMul input layout.";
    return false;
  }
  if (output_dimensions.layout() != dnn::DataLayout::kBatchYXDepth &&
      output_dimensions.layout() != dnn::DataLayout::kBatchDepthYX) {
    LOG(ERROR) << "Unsupported MatMul output layout.";
    return false;
  }

  if (output_dimensions.width() == 1 && output_dimensions.height() == 1) {
    // This is a fast path that also supports the kBatchYXDepth layout.

    // The matrices here are in row-major format while BLAS expects
    // column-major, i.e. our matrices are transposed as far as BLAS
    // is concerned. So we need to compute output^T =
    // input^T*weights^T. There is no parameter for transposing the
    // output in BLAS gemm, but instead we can transpose both sides of
    // the equality to see that this is equivalent to
    // output=weights*input. So we only need to swap the order of
    // weights and input in the matrix product to correct for the
    // row-major versus column-major difference.
    const float alpha = 1.0f;  // Take the matrix product without scaling it.
    const float beta = 0.0f;   // Ignore the original values in output_data.
    const int64 m = output_dimensions.NodesAcrossFeatureMaps();
    const int64 n = input_dimensions.count();
    const int64 k = input_dimensions.NodesAcrossFeatureMaps();
    stream->ThenBlasGemm(blas::Transpose::kNoTranspose,
                         blas::Transpose::kNoTranspose, m, n, k, alpha, weights,
                         m, input_data, k, beta, output_data, m);
  } else {
    // This is a slower and more complex path that supports output
    // width() * height() > 1, though it only supports the
    // kBatchYXDepth layout. Does support kBatchDepthYX if output
    // feature_map_count() == 1, as then there is no difference
    // between the two layouts.
    //
    // The operation here is the same as above, except that we have to
    // do the matrix multiplication for each (y,x) output coordinate
    // separately. We then interpret weights as containing K = width()
    // * height() different matrices, which we all multiply onto the
    // matrix from input_data, yielding K matrix products. We then
    // combine these together into one matrix by concatenating all the
    // first rows of these matrices, then all the seconds rows and so
    // on. We can do this with a batched matrix multiplication, where
    // the result is written to a different submatrix of the output
    // for each matrix multiplication.
    //
    // The reason that we only support the kBatchYXDepth output layout
    // is that we have to do something in the depth for each (y,x)
    // coordinate. The kBatchYXDepth layout has the depth information
    // for each point (y,x) in contiguous memory while the
    // kBatchDepthYX layout does not.
    //
    // TODO(broune): Consider a special case for when output depth ==
    // 1, as then possibly this could all be done as one matrix
    // multiplication instead of a batched one, which should be
    // faster. Another possibility would be to add a weights layout
    // parameter and then support kBatchDepthYX for a different
    // weights layout.
    if (output_dimensions.layout() != dnn::DataLayout::kBatchYXDepth &&
        !(output_dimensions.layout() == dnn::DataLayout::kBatchDepthYX &&
          output_dimensions.feature_map_count() == 1)) {
      LOG(ERROR) << "Unsupported MatMul output layout.";
      return false;
    }

    const float alpha = 1.0f;  // Take the matrix product without scaling it.
    const float beta = 0.0f;   // Ignore the original values in output_data.
    const uint64 m = output_dimensions.feature_map_count();
    const uint64 n = input_dimensions.count();
    const uint64 k = input_dimensions.NodesAcrossFeatureMaps();
    const int lda = m;
    const int ldb = k;
    const int ldc = output_dimensions.NodesAcrossFeatureMaps();
    const int batch_count = output_dimensions.NodesPerFeatureMap();

    std::vector<DeviceMemory<float>> a(batch_count);
    std::vector<DeviceMemory<float>> b(batch_count);
    std::vector<DeviceMemory<float>> c(batch_count);
    for (int i = 0; i < batch_count; ++i) {
      const int weights_offset = i * input_dimensions.NodesAcrossFeatureMaps() *
                                 output_dimensions.feature_map_count();
      a[i] = DeviceMemory<float>::MakeFromByteSize(
          const_cast<float*>(reinterpret_cast<const float*>(weights.opaque())) +
              weights_offset,
          weights.ElementCount() - weights_offset);

      b[i] = input_data;

      const int output_offset = i * output_dimensions.feature_map_count();
      c[i] = DeviceMemory<float>::MakeFromByteSize(
          const_cast<float*>(
              reinterpret_cast<const float*>(output_data->opaque())) +
              output_offset,
          output_data->ElementCount() - output_offset);
    }
    const auto toPtrs = [](std::vector<DeviceMemory<float>>& v) {
      std::vector<DeviceMemory<float>*> ptrs;
      for (auto& mem : v) {
        ptrs.push_back(&mem);
      }
      return ptrs;
    };

    stream->ThenBlasGemmBatched(blas::Transpose::kNoTranspose,
                                blas::Transpose::kNoTranspose, m, n, k, alpha,
                                toPtrs(a), lda, toPtrs(b), ldb, beta, toPtrs(c),
                                ldc, batch_count);
  }

  return stream->ok();
}

bool CudnnSupport::DoBiasAdd(Stream* stream,
                             const DeviceMemory<float>& input_data,
                             const DeviceMemory<float>& biases,
                             const dnn::BatchDescriptor& dimensions,
                             DeviceMemory<float>* output_data) {
  ScopedTensorDescriptor input_descriptor{parent_, dimensions,
                                          mlopenFloat};

  BatchDescriptor bias_dimensions;
  bias_dimensions.set_count(1)
      .set_feature_map_count(dimensions.feature_map_count())
      .set_height(1)
      .set_width(1)
      .set_layout(dnn::DataLayout::kBatchYXDepth);
  ScopedTensorDescriptor bias_descriptor{parent_, bias_dimensions,
                                         mlopenFloat};

  // mlopenAddTensor after R3 is in-place, so we need to copy input_data to
  // output_data before doing the addition, unless the input and
  // output are at the same address.
  if (input_data.opaque() != output_data->opaque()) {
    stream->ThenMemcpy(output_data, input_data,
                       dimensions.ElementCount() * sizeof(float));
    if (!stream->ok()) {
      LOG(ERROR)
          << "stream " << stream
          << " could not enqueue a tensor copy as part of bias addition.";
      return false;
    }
  }

  mutex_lock lock{dnn_handle_mutex_};
/*  auto status = dynload::mlopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != mlopenStatusSuccess) {
    LOG(ERROR) << "failed to set stream for mlopen handle: " << ToString(status);
    return false;
  }*/

  const float alpha = 1.0f;
  const float beta = 1.0f;

  auto status = dynload::mlopenTransformTensor(
      parent_, ToHandle(dnn_handle_), &alpha, bias_descriptor.handle(),
      biases.opaque(), &beta, input_descriptor.handle(),
      output_data->opaque());

  if (status != mlopenStatusSuccess) {
    LOG(ERROR) << "stream " << stream << " could not enqueue bias addition.";
    return false;
  }

  return true;
}

bool CudnnSupport::DoActivate(Stream* stream,
                              dnn::ActivationMode activation_mode,
                              const dnn::BatchDescriptor& dimensions,
                              const DeviceMemory<float>& input_data,
                              DeviceMemory<float>* output_data) {
#if 0
  mutex_lock lock{dnn_handle_mutex_};
/*  auto status = dynload::mlopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != mlopenStatusSuccess) {
    LOG(ERROR) << "failed to set stream for mlopen handle: " << ToString(status);
    return false;
  }*/

#if CUDNN_VERSION >= 5000
  ScopedActivationDescriptor activation_desc{parent_, activation_mode,
                                             dimensions.value_max()};
#else
  mlopenActivationMode_t mode;
  switch (activation_mode) {
    case dnn::ActivationMode::kRelu6:
      // TODO(leary) should probably do a post-pass to clip at 6?
      LOG(WARNING) << "user requested Relu6, but providing Relu instead";
      mode = mlopenActivationRELU;
      break;
    case dnn::ActivationMode::kReluX:
      // TODO(broune) should probably do a post-pass to clip at X?
      LOG(WARNING) << "user requested ReluX, but providing Relu instead";
      mode = mlopenActivationRELU;
      break;
    case dnn::ActivationMode::kRelu:
      mode = mlopenActivationRELU;
      break;
    case dnn::ActivationMode::kSigmoid:
      mode = mlopenActivationPATHTRU;
      break;
    case dnn::ActivationMode::kTanh:
      mode = mlopenActivationTANH;
      break;
    default:
      LOG(ERROR) << "unrecognized activation mode: "
                 << static_cast<int>(activation_mode);
      return false;
  }
#endif

  ScopedTensorDescriptor input_nd{parent_, dimensions, mlopenFloat};
  // Alpha is the input scaling factor.
  float alpha = 1.0;
  // Beta is the output scaling factor.
  float beta = 0.0;
  status = dynload::mlopenActivationForward(
      parent_, ToHandle(dnn_handle_),
#if CUDNN_VERSION >= 5000
      activation_desc.handle(),
#else
      mode,
#endif
      &alpha, input_nd.handle(), input_data.opaque(), &beta, input_nd.handle(),
      output_data->opaque());
  if (status != mlopenStatusSuccess) {
    LOG(ERROR) << "stream " << stream
               << " could not enqueue activation: " << ToString(status);
    return false;
  }

  return true;
#else
  LOG(FATAL) << "Activation not yet implemented";
  return false;

#endif
}

bool CudnnSupport::DoPoolForward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<float>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    DeviceMemory<float>* output_data) {
  mutex_lock lock{dnn_handle_mutex_};
/*  auto status = dynload::mlopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != mlopenStatusSuccess) {
    LOG(ERROR) << "failed to set stream for mlopen handle: " << ToString(status);
    return false;
  }*/

  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  void* t = NULL;

  ScopedTensorDescriptor src_desc{parent_, input_dimensions, mlopenFloat};
  ScopedTensorDescriptor dest_desc{parent_, output_dimensions,
                                   mlopenFloat};
  ScopedPoolingDescriptor pooling_desc{parent_, pooling_dimensions};

  auto status = dynload::mlopenPoolingForward(
      parent_, ToHandle(dnn_handle_), pooling_desc.handle(), &alpha,
      src_desc.handle(), input_data.opaque(), &beta, dest_desc.handle(),
      output_data->opaque(), 0, t, 0);
  if (status != mlopenStatusSuccess) {
    LOG(ERROR) << "failed to enqueue forward pooling on stream: "
               << ToString(status);
    return false;
  }
  return true;
}

bool CudnnSupport::DoPoolForward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<Eigen::half>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    DeviceMemory<Eigen::half>* output_data) {
  mutex_lock lock{dnn_handle_mutex_};
/*  auto status = dynload::mlopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != mlopenStatusSuccess) {
    LOG(ERROR) << "failed to set stream for mlopen handle: " << ToString(status);
    return false;
  }*/

  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  void* t = NULL;

  ScopedTensorDescriptor src_desc{parent_, input_dimensions, mlopenHalf};
  ScopedTensorDescriptor dest_desc{parent_, output_dimensions, mlopenHalf};
  ScopedPoolingDescriptor pooling_desc{parent_, pooling_dimensions};
  auto status = dynload::mlopenPoolingForward(
      parent_, ToHandle(dnn_handle_), pooling_desc.handle(), &alpha,
      src_desc.handle(), input_data.opaque(), &beta, dest_desc.handle(),
      output_data->opaque(), 0, t, 0);
  if (status != mlopenStatusSuccess) {
    LOG(ERROR) << "failed to enqueue forward pooling on stream: "
               << ToString(status);
    return false;
  }
  return true;
}

bool CudnnSupport::DoPoolBackward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<float>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    const DeviceMemory<float>& output_data,
    const DeviceMemory<float>& input_diff_data,
    DeviceMemory<float>* output_diff_data) {
  mutex_lock lock{dnn_handle_mutex_};
/*  auto status = dynload::mlopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != mlopenStatusSuccess) {
    LOG(ERROR) << "failed to set stream for mlopen handle: " << ToString(status);
    return false;
  }*/

  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  const void* t = NULL;

  ScopedTensorDescriptor src_desc{parent_, input_dimensions, mlopenFloat};
  ScopedTensorDescriptor dest_desc{parent_, output_dimensions,
                                   mlopenFloat};
  ScopedPoolingDescriptor pooling_desc{parent_, pooling_dimensions};
  auto status = dynload::mlopenPoolingBackward(
      parent_, ToHandle(dnn_handle_), pooling_desc.handle(), &alpha,
      dest_desc.handle(), output_data.opaque(), dest_desc.handle(),
      input_diff_data.opaque(), src_desc.handle(), input_data.opaque(), &beta,
      src_desc.handle(), output_diff_data->opaque(), t);
  if (status != mlopenStatusSuccess) {
    LOG(ERROR) << "failed to enqueue backward pooling on stream: "
               << ToString(status);
    return false;
  }
  return true;
}

bool CudnnSupport::DoPoolBackward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<Eigen::half>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    const DeviceMemory<Eigen::half>& output_data,
    const DeviceMemory<Eigen::half>& input_diff_data,
    DeviceMemory<Eigen::half>* output_diff_data) {
  mutex_lock lock{dnn_handle_mutex_};
/*  auto status = dynload::mlopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != mlopenStatusSuccess) {
    LOG(ERROR) << "failed to set stream for mlopen handle: " << ToString(status);
    return false;
  }*/

  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  void* t = NULL;

  ScopedTensorDescriptor src_desc{parent_, input_dimensions, mlopenHalf};
  ScopedTensorDescriptor dest_desc{parent_, output_dimensions, mlopenHalf};
  ScopedPoolingDescriptor pooling_desc{parent_, pooling_dimensions};
  auto status = dynload::mlopenPoolingBackward(
      parent_, ToHandle(dnn_handle_), pooling_desc.handle(), &alpha,
      dest_desc.handle(), output_data.opaque(), dest_desc.handle(),
      input_diff_data.opaque(), src_desc.handle(), input_data.opaque(), &beta,
      src_desc.handle(), output_diff_data->opaque(), t);
  if (status != mlopenStatusSuccess) {
    LOG(ERROR) << "failed to enqueue backward pooling on stream: "
               << ToString(status);
    return false;
  }
  return true;
}

bool CudnnSupport::DoNormalize(
    Stream* stream, const dnn::NormalizeDescriptor& normalize_descriptor,
    const DeviceMemory<float>& input_data, DeviceMemory<float>* output_data) {
  LOG(FATAL) << "not yet implemented";  // TODO(leary)
}

bool CudnnSupport::DoNormalizeWithDimensions(
    Stream* stream, const dnn::NormalizeDescriptor& normalize_descriptor,
    const dnn::BatchDescriptor& dimensions,
    const DeviceMemory<float>& input_data, DeviceMemory<float>* output_data) {
  // Check for unsupported modes.
  if (normalize_descriptor.wrap_around()) {
    LOG(ERROR) << "CUDA LRN does not support wrap-around mode";
    return false;
  }
  if (normalize_descriptor.segment_size()) {
    LOG(ERROR) << "CUDA LRN does not support segmentation";
    return false;
  }

  // Launch the normalization.
  mutex_lock lock{dnn_handle_mutex_};
/*  auto status = dynload::mlopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != mlopenStatusSuccess) {
    LOG(ERROR) << "failed to set stream for mlopen handle: " << ToString(status);
    return false;
  }*/

  ScopedTensorDescriptor dims{parent_, dimensions, mlopenFloat};
  ScopedNormalizeDescriptor normalize{parent_, normalize_descriptor};

  // Alpha is the scaling factor for input.
  float alpha = 1.0f;
  // Beta is the scaling factor for output.
  float beta = 0.0f;

  void* t = NULL;

  auto status = dynload::mlopenLRNForward(
      parent_, ToHandle(dnn_handle_), normalize.handle(),
      &alpha, dims.handle(), input_data.opaque(),
      &beta, dims.handle(), output_data->opaque(), 0, t);
  if (status != mlopenStatusSuccess) {
    LOG(ERROR) << "failed to run mlopenLRNCrossChannelForward";
    return false;
  }
  return true;
}

bool CudnnSupport::DoNormalizeBackwardWithDimensions(
    Stream* stream, const dnn::NormalizeDescriptor& normalize_descriptor,
    const dnn::BatchDescriptor& dimensions, const DeviceMemory<float>& raw_data,
    const DeviceMemory<float>& normalized_data,
    const DeviceMemory<float>& normalized_variable_gradient,
    DeviceMemory<float>* raw_variable_gradient) {
  // Check for unsupported modes.
  if (normalize_descriptor.wrap_around()) {
    LOG(ERROR) << "CUDA LRN does not support wrap-around mode";
    return false;
  }
  if (normalize_descriptor.segment_size()) {
    LOG(ERROR) << "CUDA LRN does not support segmentation";
    return false;
  }

  mutex_lock lock{dnn_handle_mutex_};
/*  auto status = dynload::mlopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != mlopenStatusSuccess) {
    LOG(ERROR) << "failed to set stream for mlopen handle: " << ToString(status);
    return false;
  }*/

  ScopedTensorDescriptor dims{parent_, dimensions, mlopenFloat};
  ScopedNormalizeDescriptor normalize{parent_, normalize_descriptor};

  float alpha = 1.0f;
  float beta = 0.0f;

  void* t = NULL;

  auto status = dynload::mlopenLRNBackward(
      parent_, ToHandle(dnn_handle_), normalize.handle(),
      &alpha, dims.handle(),
      normalized_data.opaque(), dims.handle(),
      normalized_variable_gradient.opaque(), dims.handle(), raw_data.opaque(),
      &beta, dims.handle(), raw_variable_gradient->opaque(), t);
  if (status != mlopenStatusSuccess) {
    LOG(ERROR) << "failed to run mlopenLRNCrossChannelBackward";
    return false;
  }
  return true;
}

bool CudnnSupport::DoDepthConcatenate(
    Stream* stream, port::ArraySlice<dnn::BatchDescriptor> input_dimensions,
    port::ArraySlice<const DeviceMemory<float>*> input_data,
    DeviceMemory<float>* output_data) {
  CHECK_EQ(input_dimensions.size(), input_data.size());

  for (const auto& dimensions : input_dimensions) {
    if (dimensions.layout() != dnn::DataLayout::kBatchDepthYX) {
      LOG(ERROR) << "CudnnSupport::DoDepthConcatenate currently only "
                    "supports the kBatchDepthYX layout.";
      return false;
    }
  }

  if (input_dimensions.empty()) {
    return true;  // Nothing to do.
  }

  dnn::BatchDescriptor output_dimensions =
      dnn::BatchDescriptor::DepthConcatenateOutputDescriptor(input_dimensions);

  const int64 area = output_dimensions.width() * output_dimensions.height();
  const auto index = [area](int64 batch, int64 depth, int64 yx,
                            int64 max_depth) {
    return (batch * max_depth + depth) * area + yx;
  };

  std::vector<float> output_host(output_dimensions.ElementCount());
  std::vector<float> tmp;
  int64 depth_sum = 0;
  for (size_t i = 0; i < input_data.size(); ++i) {
    const auto& dimensions = input_dimensions[i];
    tmp.resize(dimensions.ElementCount());
    stream->ThenMemcpyD2H<float>(*input_data[i], &tmp).BlockHostUntilDone();

    for (int64 batch = 0; batch < output_dimensions.count(); ++batch) {
      for (int64 yx = 0; yx < area; ++yx) {
        for (int64 depth = 0; depth < dimensions.feature_map_count(); ++depth) {
          LOG(INFO) << output_dimensions.ElementCount() << ' ' << batch << ' '
                    << yx << ' ' << depth;
          output_host[index(batch, depth + depth_sum, yx,
                            output_dimensions.feature_map_count())] =
              tmp[index(batch, depth, yx, dimensions.feature_map_count())];
        }
      }
    }
    depth_sum += dimensions.feature_map_count();
  }
  stream->ThenMemcpyH2D<float>(output_host, output_data);
  return true;
}

bool CudnnSupport::DoElementwiseOperate(
    Stream* stream, dnn::ElementwiseOperation operation,
    port::ArraySlice<dnn::BatchDescriptor> input_dimensions,
    port::ArraySlice<const DeviceMemory<float>*> input_data,
    const dnn::BatchDescriptor& output_dimensions,
    DeviceMemory<float>* output_data) {
  LOG(FATAL) << "not yet implemented";  // TODO(leary)
  return false;
}

bool CudnnSupport::DoXYPad(Stream* stream,
                           const dnn::BatchDescriptor& dimensions,
                           const DeviceMemory<float>& input_data,
                           int64 left_pad, int64 right_pad, int64 top_pad,
                           int64 bottom_pad, DeviceMemory<float>* output_data) {
  LOG(FATAL) << "not yet implemented";  // TODO(leary)
  return false;
}

bool CudnnSupport::DoXYSlice(Stream* stream,
                             const dnn::BatchDescriptor& dimensions,
                             const DeviceMemory<float>& input_data,
                             int64 left_trim, int64 right_trim, int64 top_trim,
                             int64 bottom_trim,
                             DeviceMemory<float>* output_data) {
  LOG(FATAL) << "not yet implemented";  // TODO(leary)
  return false;
}

bool CudnnSupport::DoMemcpyD2HQuantized(
    Stream* stream, const DeviceMemory<float>& gpu_unquantized_src,
    dnn::QuantizedActivationMode mode, void* host_dst, int64 size) {
  LOG(ERROR) << "quantized memcpy not supported";
  return false;
}

bool CudnnSupport::DoMemcpyH2DQuantized(
    Stream* stream, const void* host_src, int64 size,
    dnn::QuantizedActivationMode mode,
    DeviceMemory<float>* gpu_unquantized_dst) {
  LOG(ERROR) << "quantized memcpy not supported";
  return false;
}

bool CudnnSupport::DeriveOutputBatchDescriptor(
    const BatchDescriptor& batch_descriptor,
    const FilterDescriptor& filter_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    dnn::BatchDescriptor* output_batch_descriptor) {
  ScopedTensorDescriptor input_nd{parent_, batch_descriptor, mlopenFloat};
  ScopedFilterDescriptor filter{parent_, filter_descriptor, batch_descriptor,
                                mlopenFloat};
  ScopedConvolutionDescriptor conv{parent_, convolution_descriptor,
                                   mlopenFloat};

  int dn = batch_descriptor.ndims() + 2;
  std::vector<int> dims(dn);  // in BDYX
  auto status = dynload::mlopenGetConvolutionForwardOutputDim(
      parent_, conv.handle(), input_nd.handle(), filter.handle(), &dn,
      &dims[0], &dims[1], &dims[2]);
  if (status != mlopenStatusSuccess) {
    LOG(ERROR) << "could not get output tensor for convolution: "
               << ToString(status);
    return false;
  }

  output_batch_descriptor->set_count(dims[0])
      .set_feature_map_count(dims[1])
      .set_layout(batch_descriptor.layout());

  for (int i = 0; i < batch_descriptor.ndims(); i++) {
    output_batch_descriptor->set_spatial_dim(static_cast<dnn::DimIndex>(i),
                                             dims.rbegin()[i]);
  }

  return true;
}

}  // namespace cuda
#if 0
namespace gpu = ::perftools::gputools;

void initialize_mlopen() {
  gpu::port::Status status =
      gpu::PluginRegistry::Instance()
          ->RegisterFactory<gpu::PluginRegistry::DnnFactory>(
              gpu::cuda::kCudaPlatformId, gpu::cuda::kCuDnnPlugin, "cuDNN",
              [](gpu::internal::StreamExecutorInterface*
                     parent) -> gpu::dnn::DnnSupport* {
                gpu::cuda::CUDAExecutor* cuda_executor =
                    dynamic_cast<gpu::cuda::CUDAExecutor*>(parent);
                if (cuda_executor == nullptr) {
                  LOG(ERROR)
                      << "Attempting to initialize an instance of the cuBLAS "
                      << "support library with a non-CUDA StreamExecutor";
                  return nullptr;
                }

                gpu::cuda::CudnnSupport* dnn =
                    new gpu::cuda::CudnnSupport(cuda_executor);
                if (!dnn->Init().ok()) {
                  // Note: Init() will log a more specific error.
                  delete dnn;
                  return nullptr;
                }
                return dnn;
              });

  if (!status.ok()) {
    LOG(ERROR) << "Unable to register cuDNN factory: "
               << status.error_message();
  }

  // Prime the cuDNN DSO. The loader will log more information.
  auto statusor = gpu::internal::CachedDsoLoader::GetCudnnDsoHandle();
  if (!statusor.ok()) {
    LOG(INFO) << "Unable to load cuDNN DSO";
  }

  gpu::PluginRegistry::Instance()->SetDefaultFactory(gpu::cuda::kCudaPlatformId,
                                                     gpu::PluginKind::kDnn,
                                                     gpu::cuda::kCuDnnPlugin);
}

#endif
}  // namespace gputools
}  // namespace perftools

//REGISTER_MODULE_INITIALIZER(register_mlopen,
//                            { perftools::gputools::initialize_mlopen(); });

#endif
