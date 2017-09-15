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
#include "tensorflow/stream_executor/lib/stringpiece.h"
#include "tensorflow/stream_executor/lib/threadpool.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/scratch_allocator.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
// clang-format off
#include "cuda/include/miopen/miopen.h"
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
// For example, if miopen_version is 5107, the compatibility version
// number will be 5100.
/*size_t miopenCompatibilityVersion(size_t miopen_version) {
  return (miopen_version / 100) * 100;
}*/

}  // namespace

namespace perftools {
namespace gputools {

using dnn::BatchDescriptor;
using dnn::FilterDescriptor;
using dnn::ConvolutionDescriptor;
using dnn::PoolingDescriptor;
using dnn::NormalizeDescriptor;

namespace cuda {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kCuDnnPlugin);

string ToString(miopenStatus_t status) {
  switch (status) {
    case miopenStatusSuccess:
      return "miopenStatusSuccess";
    case miopenStatusNotInitialized:
      return "miopenStatusNotInitialized";
    case miopenStatusAllocFailed:
      return "miopenStatusAllocFailed";
    case miopenStatusBadParm:
      return "miopenStatusBadParm";
    case miopenStatusInternalError:
      return "miopenStatusInternalError";
    case miopenStatusInvalidValue:
      return "miopenStatusInvalidValue";
    case miopenStatusNotImplemented:
      return "miopenStatusNotImplemented";
    case miopenStatusUnknownError:
      return "miopenStatusUnknownError";
    default:
      return port::StrCat("<unknown miopen status: ", static_cast<int>(status),
                          ">");
  }
}

namespace dynload {

static port::ThreadPool* InitCudnnThreadpool() {
  port::ThreadPool* miopen_threadpool_;
  port::ThreadOptions options;
  // TBD(keveman): Conservatively setting the stack size and guard size to 2MB,
  // until we can get some guarantees from NVIDIA on the minimum stack space
  // they will work with.
  options.stack_size = 2 * 1024 * 1024;
  options.guard_size = 2 * 1024 * 1024;
  miopen_threadpool_ = new port::ThreadPool(port::Env::Default(), options,
                                           "miopen_threadpool", 1);
  CHECK(miopen_threadpool_);
  return miopen_threadpool_;
}

static mutex miopen_threadpool_mu(LINKER_INITIALIZED);
static port::ThreadPool* GetCudaThreadpool() {
  mutex_lock lock(miopen_threadpool_mu);
  static port::ThreadPool* miopen_threadpool = InitCudnnThreadpool();
  return miopen_threadpool;
}

// Retrieves the MIOPEN DSO, dies on failure.
void* GetDsoHandle() {
  static auto result = internal::CachedDsoLoader::GetCudnnDsoHandle();
  return result.ValueOrDie();
}

/*static void* DynLoadGetVersionOrDie() {
  void* f;
  port::Status s = port::Env::Default()->GetSymbolFromLibrary(
      GetDsoHandle(), "miopenGetVersion", &f);
  if (f == nullptr) {
    LOG(FATAL) << "could not find miopenGetVersion in miopen DSO; dlerror: "
               << s.error_message();
  }
  return f;
}

// Calls miopenGetVersion in the loaded DSO.
size_t miopenGetVersion() {
  static void* f = DynLoadGetVersionOrDie();
  auto callable = reinterpret_cast<size_t (*)(void)>(f);
  return callable();
}*/

#define PERFTOOLS_GPUTOOLS_MIOPEN_WRAP(__name)                           \
  struct DynLoadShim__##__name {                                         \
    static const char* kName;                                            \
    typedef std::add_pointer<decltype(::__name)>::type FuncPointerT;     \
    static FuncPointerT LoadOrDie() {                                    \
      void* f;                                                           \
      port::Status s = port::Env::Default()->GetSymbolFromLibrary(       \
          GetDsoHandle(), kName, &f);                                    \
      CHECK(s.ok()) << "could not find " << kName                        \
                    << " in miopen DSO; dlerror: " << s.error_message(); \
      return reinterpret_cast<FuncPointerT>(f);                          \
    }                                                                    \
    static FuncPointerT DynLoad() {                                      \
      static FuncPointerT f = LoadOrDie();                               \
      return f;                                                          \
    }                                                                    \
    template <typename... Args>                                          \
    miopenStatus_t operator()(CUDAExecutor* parent, Args... args) {      \
      cuda::ScopedActivateExecutorContext sac{parent};                   \
      miopenStatus_t retval = DynLoad()(args...);                        \
      return retval;                                                     \
    }                                                                    \
  } __name;                                                              \
  const char* DynLoadShim__##__name::kName = #__name;

// clang-format off
#define MIOPEN_DNN_ROUTINE_EACH(__macro)                   \
  __macro(miopenBatchNormalizationBackward)                \
  __macro(miopenBatchNormalizationForwardInference)        \
  __macro(miopenBatchNormalizationForwardTraining)         \
  __macro(miopenGetConvolutionForwardOutputDim)            \
  __macro(miopenFindConvolutionForwardAlgorithm)           \
  __macro(miopenCreateTensorDescriptor)                    \
  __macro(miopenDestroyTensorDescriptor)                   \
  __macro(miopenSet2dPoolingDescriptor)                    \
  __macro(miopenSetLRNDescriptor)                          \
  __macro(miopenLRNGetWorkSpaceSize)                       \
  __macro(miopenCreateConvolutionDescriptor)               \
  __macro(miopenCreatePoolingDescriptor)                   \
  __macro(miopenDestroyPoolingDescriptor)                  \
  __macro(miopenCreateLRNDescriptor)                       \
  __macro(miopenDestroyLRNDescriptor)                      \
  __macro(miopenDestroyConvolutionDescriptor)              \
  __macro(miopenCreateWithStream)                          \
  __macro(miopenDestroy)                                   \
  __macro(miopenSetStream)                                 \
  __macro(miopenActivationForward)                         \
  __macro(miopenConvolutionForward)                        \
  __macro(miopenConvolutionBackwardBias)                   \
  __macro(miopenConvolutionForwardGetWorkSpaceSize)        \
  __macro(miopenInitConvolutionDescriptor)                 \
  __macro(miopenSet4dTensorDescriptor)                     \
  __macro(miopenSetTensorDescriptor)                       \
  __macro(miopenGetTensorDescriptorSize)                   \
  __macro(miopenPoolingForward)                            \
  __macro(miopenPoolingGetWorkSpaceSize)                   \
  __macro(miopenPoolingBackward)                           \
  __macro(miopenLRNForward)                                \
  __macro(miopenLRNBackward)                               \
  __macro(miopenOpTensor)                                  \
  __macro(miopenConvolutionBackwardData)                   \
  __macro(miopenConvolutionBackwardWeights)                \
  __macro(miopenConvolutionBackwardWeightsGetWorkSpaceSize)\
  __macro(miopenFindConvolutionBackwardDataAlgorithm)      \
  __macro(miopenFindConvolutionBackwardWeightsAlgorithm)   \
  __macro(miopenConvolutionBackwardDataGetWorkSpaceSize)
// clang-format on

MIOPEN_DNN_ROUTINE_EACH(PERFTOOLS_GPUTOOLS_MIOPEN_WRAP)

#define MIOPEN_DNN_ROUTINE_EACH_R5(__macro)                    \
  __macro(miopenCreateActivationDescriptor)                    \
  __macro(miopenSetActivationDescriptor)                       \
  __macro(miopenGetActivationDescriptor)                       \
  __macro(miopenDestroyActivationDescriptor)                   \

MIOPEN_DNN_ROUTINE_EACH_R5(PERFTOOLS_GPUTOOLS_MIOPEN_WRAP)
#undef MIOPEN_DNN_ROUTINE_EACH_R5

#undef MIOPEN_DNN_ROUTINE_EACH

}  // namespace dynload

namespace {

miopenHandle_t ToHandle(void* opaque_handle) {
  return static_cast<miopenHandle_t>(opaque_handle);
}

miopenConvFwdAlgorithm_t ToConvForwardAlgo(dnn::AlgorithmType algorithm) {
  miopenConvFwdAlgorithm_t algo = miopenConvFwdAlgorithm_t(algorithm);
  switch (algo) {
    case miopenConvolutionFwdAlgoGEMM:
    case miopenConvolutionFwdAlgoDirect:
    case miopenConvolutionFwdAlgoFFT:
    case miopenConvolutionFwdAlgoWinograd:
      return algo;
    default:
      LOG(FATAL) << "Unsupported Cudnn convolution forward algorithm: "
                 << algorithm;
  }
}

miopenConvBwdDataAlgorithm_t ToConvBackwardDataAlgo(
    dnn::AlgorithmType algorithm) {
  miopenConvBwdDataAlgorithm_t algo = miopenConvBwdDataAlgorithm_t(algorithm);
  switch (algo) {
    case miopenConvolutionBwdDataAlgoGEMM:
    case miopenConvolutionBwdDataAlgoDirect:
    case miopenConvolutionBwdDataAlgoFFT:
    case miopenConvolutionBwdDataAlgoWinograd:
    case miopenTransposeBwdDataAlgoGEMM:
      return algo;
    default:
      LOG(FATAL)
          << "Unsupported Cudnn convolution backward algorithm for data: "
          << algorithm;
  }
}

miopenConvBwdWeightsAlgorithm_t ToConvBackwardFilterAlgo(
    dnn::AlgorithmType algorithm) {
  miopenConvBwdWeightsAlgorithm_t algo =
      miopenConvBwdWeightsAlgorithm_t(algorithm);
  switch (algo) {
    case miopenConvolutionBwdWeightsAlgoGEMM:
    case miopenConvolutionBwdWeightsAlgoDirect:
      return algo;
    default:
      LOG(FATAL)
          << "Unsupported Cudnn convolution backward algorithm for filter: "
          << algorithm;
  }
}

}  // namespace

CudnnSupport::CudnnSupport(CUDAExecutor* parent)
    : parent_(parent), dnn_handle_(nullptr) {}

CudnnSupport::~CudnnSupport() {
  auto status = dynload::miopenDestroy(parent_, ToHandle(dnn_handle_));
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "could not destroy miopen handle: " << ToString(status);
  }
}

port::Status CudnnSupport::Init() {
  auto status = dynload::miopenCreateWithStream(
      parent_, reinterpret_cast<miopenHandle_t*>(&dnn_handle_), (hipStream_t)(0));
  if (status == miopenStatusSuccess) {
    // Check whether loaded version of CuDNN matches what the source
    // was built with.
    /*size_t loaded_version = dynload::miopenGetVersion();
    size_t loaded_compat_version = miopenCompatibilityVersion(loaded_version);
    size_t compiled_compat_version = miopenCompatibilityVersion(MIOPEN_VERSION);
    bool library_loaded_matches_source =
        (loaded_compat_version == compiled_compat_version);
    if (!library_loaded_matches_source) {
      const string error =
          port::StrCat("Loaded runtime CuDNN library: ", loaded_version,
                       " (compatibility version ", loaded_compat_version,
                       ") but source was compiled with ", MIOPEN_VERSION,
                       " (compatibility version ", compiled_compat_version,
                       ").  If using a binary install, upgrade your CuDNN "
                       "library to match.  If building from sources, "
                       "make sure the library loaded at runtime matches a "
                       "compatible version specified during compile "
                       "configuration.");
      LOG(ERROR) << error;
      return port::Status{port::error::INTERNAL, error};
    }*/

    return port::Status::OK();
  }

  LOG(ERROR) << "could not create miopen handle: " << ToString(status);
  if (status == miopenStatusNotInitialized) {
    // This is the error code that the driver returns when we're not running a
    // sufficient CUDA driver -- miopen requires 6.5+ compatibility, which
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
            << "miopen library is only supported on 340.XX+ driver versions";
      }
#endif
    }
  }

  return port::Status{port::error::INTERNAL,
                      port::StrCat("miopen library could not create a handle: ",
                                   ToString(status))};
}

// Turns a BatchDescriptor structure into a miopen tensor handle within a scope.
class ScopedTensorDescriptor {
 public:
  ScopedTensorDescriptor(CUDAExecutor* parent,
                         const BatchDescriptor& batch_descriptor,
                         miopenDataType_t elem_type)
      : parent_(parent), handle_(nullptr) {
    miopenStatus_t status =
        dynload::miopenCreateTensorDescriptor(parent_, &handle_);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not create miopen tensor descriptor: "
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
    if (nd != 4) {
      LOG(FATAL) << "miopen only supports 4D tensors, dim=" << nd << " not allowed";
    }

    // cuDNN requires the strides and dims to be ordered as BDYX.
    std::vector<int64> strides64 =
        batch_descriptor.full_strides(dnn::DataLayout::kBatchDepthYX);
    std::vector<int64> dims64 =
        batch_descriptor.full_dims(dnn::DataLayout::kBatchDepthYX);

    // miopen does not use strides and must have 4D tensor.
    std::vector<int> strides(nd);
    std::vector<int> dims(nd);

    std::transform(strides64.cbegin(), strides64.cend(), strides.begin(),
                   &CheckedNarrowing<int64, int>);
    std::transform(dims64.cbegin(), dims64.cend(), dims.begin(),
                   &CheckedNarrowing<int64, int>);
    //LOG(INFO)<< "dims=" << dims[0] << "," << dims[1] << "," << dims[2] << "," << dims[3] << "," << "\n";
    //LOG(INFO)<< "strides=" << strides[0] << "," << strides[1] << "," << strides[2] << "," << strides[3] << "," << "\n";
    //assert (nd==4); // FIXME-MIOPEN : modify when MIOpen supports strides.
    status = dynload::miopenSet4dTensorDescriptor(
        parent_, handle_, elem_type, dims[0], dims[1], dims[2], dims[3]);

    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not set miopen tensor descriptor: "
                 << ToString(status);
    }
  }

  ~ScopedTensorDescriptor() {
    miopenStatus_t status =
        dynload::miopenDestroyTensorDescriptor(parent_, handle_);
    if (status != miopenStatusSuccess) {
      LOG(ERROR) << "could not destroy miopen tensor descriptor: "
                 << ToString(status);
    }
  }

  miopenTensorDescriptor_t handle() const { return handle_; }

 private:
  CUDAExecutor* parent_;            // Parent executor. Not owned.
  miopenTensorDescriptor_t handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedTensorDescriptor);
};

// Turns a FilterDescriptor structure into a miopen filter handle within a scope.
class ScopedFilterDescriptor {
 public:
  ScopedFilterDescriptor(CUDAExecutor* parent,
                         const FilterDescriptor& filter_descriptor,
                         const BatchDescriptor& batch_descriptor,
                         miopenDataType_t elem_type)
      : parent_(parent), handle_(nullptr) {
    miopenStatus_t status =
        dynload::miopenCreateTensorDescriptor(parent_, &handle_);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not create miopen filter descriptor: "
                 << ToString(status);
    }

    const int nd = batch_descriptor.ndims() + 2;

    if (nd != 4) {
      LOG(FATAL) << "miopen only supports 4D filters, dim=" << nd << "not allowed"
                 << ToString(status);
    }
    
    std::vector<int> dims(nd);
    dims[0] = filter_descriptor.output_feature_map_count();
    dims[1] = filter_descriptor.input_feature_map_count();
    const auto& spatial_dims = filter_descriptor.input_filter_dims();
    std::copy(spatial_dims.begin(), spatial_dims.end(), dims.begin() + 2);


    // should be miopenSetTensorDescriptor
    status = dynload::miopenSet4dTensorDescriptor(parent_, handle_, elem_type,
                                                 dims[0], dims[1], dims[2], dims[3]);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not set miopen filter descriptor: "
                 << ToString(status);
    }
  }

  ~ScopedFilterDescriptor() {
    miopenStatus_t status =
        dynload::miopenDestroyTensorDescriptor(parent_, handle_);
    if (status != miopenStatusSuccess) {
      LOG(ERROR) << "could not destroy miopen filter descriptor: "
                 << ToString(status);
    }
  }

  miopenTensorDescriptor_t handle() const { return handle_; }

 private:
  // Parent executor object. Not owned.
  CUDAExecutor* parent_;

  // miopen filter descriptor this object creates. Owned.
  miopenTensorDescriptor_t handle_;

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedFilterDescriptor);
};

// Turns a ConvolutionDescriptor structure into a miopen convolution handle
// within a scope.
class ScopedConvolutionDescriptor {
 public:
  ScopedConvolutionDescriptor(
      CUDAExecutor* parent, const ConvolutionDescriptor& convolution_descriptor,
      miopenDataType_t data_type)
      : parent_(parent), handle_(nullptr) {
    miopenStatus_t status =
        dynload::miopenCreateConvolutionDescriptor(parent_, &handle_);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not create miopen convolution descriptor: "
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

    status = dynload::miopenInitConvolutionDescriptor(
        parent_, handle_, miopenConvolution, padding[0], padding[1],
        strides[0], strides[1], upscale[0], upscale[1]);

    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not set miopen convolution descriptor: "
                 << ToString(status);
    }
  }

  ~ScopedConvolutionDescriptor() {
    miopenStatus_t status =
        dynload::miopenDestroyConvolutionDescriptor(parent_, handle_);
    if (status != miopenStatusSuccess) {
      LOG(ERROR) << "could not destroy miopen convolution descriptor: "
                 << ToString(status);
    }
  }

  miopenConvolutionDescriptor_t handle() const { return handle_; }

 private:
  CUDAExecutor* parent_;                 // Parent executor. Not owned.
  miopenConvolutionDescriptor_t handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedConvolutionDescriptor);
};

// Turns a PoolingDescriptor structure into a miopen pooling descriptor handle
// within a scope.
class ScopedPoolingDescriptor {
 public:
  ScopedPoolingDescriptor(CUDAExecutor* parent,
                          const PoolingDescriptor& pooling_descriptor)
      : parent_(parent), handle_(nullptr) {
    miopenStatus_t status =
        dynload::miopenCreatePoolingDescriptor(parent_, &handle_);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not create miopen pooling descriptor: "
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

    if (nd != 2) {
      LOG(FATAL) << "miopen requires pooling dimensions==2"
                 << ToString(status);
    }


    status = dynload::miopenSet2dPoolingDescriptor(
        parent_, handle_,
        (pooling_descriptor.mode() == dnn::PoolingMode::kMaximum
             ? miopenPoolingMax
             : miopenPoolingAverage),
#if 0//MIOPEN_VERSION >= 5000
        // Always propagate nans.
        MIOPEN_PROPAGATE_NAN,
#endif
        shape[0], shape[1], padding[0], padding[1], strides[0], strides[1]);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not set miopen pooling descriptor: "
                 << ToString(status);
    }
  }
  ~ScopedPoolingDescriptor() {
    miopenStatus_t status =
        dynload::miopenDestroyPoolingDescriptor(parent_, handle_);
    if (status != miopenStatusSuccess) {
      LOG(ERROR) << "could not destroy miopen pooling descriptor: "
                 << ToString(status);
    }
  }

  miopenPoolingDescriptor_t handle() const { return handle_; }

 private:
  CUDAExecutor* parent_;             // Parent executor. Not owned.
  miopenPoolingDescriptor_t handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedPoolingDescriptor);
};

// Turns a NormalizeDescriptor structure into a miopen LRN descriptor handle.
class ScopedNormalizeDescriptor {
 public:
  ScopedNormalizeDescriptor(CUDAExecutor* parent,
                            const NormalizeDescriptor& normalize_descriptor)
      : parent_(parent), handle_(nullptr) {
    miopenStatus_t status = dynload::miopenCreateLRNDescriptor(parent_, &handle_);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not create miopen LRN descriptor: "
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
    status = dynload::miopenSetLRNDescriptor(parent_, handle_, miopenLRNCrossChannel, lrnN, lrnAlpha,
                                            lrnBeta, lrnK);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not set miopen LRN descriptor: " << ToString(status);
    }
  }

  ~ScopedNormalizeDescriptor() {
    miopenStatus_t status = dynload::miopenDestroyLRNDescriptor(parent_, handle_);
    if (status != miopenStatusSuccess) {
      LOG(ERROR) << "could not destroy miopen LRN descriptor: "
                 << ToString(status);
    }
  }

  miopenLRNDescriptor_t handle() const { return handle_; }

 private:
  CUDAExecutor* parent_;         // Parent executor. Not owned.
  miopenLRNDescriptor_t handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedNormalizeDescriptor);
};

#if 0//MIOPEN_VERSION >= 5000
// Turns a ActivationDescriptor structure into a miopen activation
// descriptor handle within a scope.
class ScopedActivationDescriptor {
 public:
  ScopedActivationDescriptor(CUDAExecutor* parent,
                             dnn::ActivationMode activation_mode,
                             double value_max)
      : parent_(parent), handle_(nullptr) {
    miopenStatus_t status =
        dynload::miopenCreateActivationDescriptor(parent_, &handle_);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not create miopen activation descriptor: "
                 << ToString(status);
    }

    double relu_ceiling = 0.0;
    miopenActivationMode_t mode;
    switch (activation_mode) {
      case dnn::ActivationMode::kRelu6:
        relu_ceiling = 6.0;
        mode = MIOPEN_ACTIVATION_CLIPPED_RELU;
        break;
      case dnn::ActivationMode::kReluX:
        relu_ceiling = value_max;
        mode = MIOPEN_ACTIVATION_CLIPPED_RELU;
        break;
      case dnn::ActivationMode::kRelu:
        mode = MIOPEN_ACTIVATION_RELU;
        break;
      case dnn::ActivationMode::kSigmoid:
        mode = MIOPEN_ACTIVATION_SIGMOID;
        break;
      case dnn::ActivationMode::kTanh:
        mode = MIOPEN_ACTIVATION_TANH;
        break;
      default:
        LOG(FATAL) << "unrecognized activation mode: "
                   << static_cast<int>(activation_mode);
    }

    // Always propagate nans.
    miopenNanPropagation_t nan_propagation = MIOPEN_PROPAGATE_NAN;
    status = dynload::miopenSetActivationDescriptor(
        parent_, handle_,
        mode, nan_propagation, relu_ceiling);
    if (status != miopenStatusSuccess) {
      LOG(FATAL) << "could not set miopen activation descriptor: "
                 << ToString(status);
    }
  }

  ~ScopedActivationDescriptor() {
    miopenStatus_t status =
        dynload::miopenDestroyActivationDescriptor(parent_, handle_);
    if (status != miopenStatusSuccess) {
      LOG(ERROR) << "could not destroy miopen activation descriptor: "
                 << ToString(status);
    }
  }

  miopenActivationDescriptor_t handle() const { return handle_; }

 private:
  CUDAExecutor* parent_;                // Parent executor. Not owned.
  miopenActivationDescriptor_t handle_;  // Owned.

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedActivationDescriptor);
};
#endif

namespace {
miopenDataType_t ToCudnnDataType(dnn::DataType data_type) {
  switch (data_type) {
    case dnn::DataType::kFloat:
    case dnn::DataType::kDouble:
    case dnn::DataType::kHalf:
      return static_cast<miopenDataType_t>(data_type);
    default:
      LOG(FATAL) << "Invalid DNN data type: " << static_cast<int>(data_type);
  }
}

#if 0//MIOPEN_VERSION >= 5000

miopenRNNInputMode_t ToCudnnRnnInputMode(dnn::RnnInputMode input_mode) {
  switch (input_mode) {
    case dnn::RnnInputMode::kRnnLinearSkip:
    case dnn::RnnInputMode::kRnnSkipInput:
      return static_cast<miopenRNNInputMode_t>(input_mode);
    default:
      LOG(FATAL) << "Invalid RNN input mode: " << static_cast<int>(input_mode);
  }
}

miopenDirectionMode_t ToCudnnRnnDirectionMode(
    dnn::RnnDirectionMode direction_mode) {
  switch (direction_mode) {
    case dnn::RnnDirectionMode::kRnnUnidirectional:
    case dnn::RnnDirectionMode::kRnnBidirectional:
      return static_cast<miopenDirectionMode_t>(direction_mode);
    default:
      LOG(FATAL) << "Invalid RNN direction mode: "
                 << static_cast<int>(direction_mode);
  }
}

miopenRNNMode_t ToCudnnRnnMode(dnn::RnnMode rnn_mode) {
  switch (rnn_mode) {
    case dnn::RnnMode::kRnnRelu:
    case dnn::RnnMode::kRnnTanh:
    case dnn::RnnMode::kRnnLstm:
    case dnn::RnnMode::kRnnGru:
      return static_cast<miopenRNNMode_t>(rnn_mode);
    default:
      LOG(FATAL) << "Invalid RNN Mode: " << static_cast<int>(rnn_mode);
  }
}

int CudnnDataTypeToByteSize(miopenDataType_t data_type) {
  switch (data_type) {
    case miopenFloat:
      return sizeof(float);
    case MIOPEN_DATA_DOUBLE:
      return sizeof(double);
    case miopenHalf:
      return sizeof(Eigen::half);
    default:
      LOG(FATAL) << "Invalid DNN data type: " << static_cast<int>(data_type);
  }
}

#endif  // MIOPEN_VERSION

template <typename Base>
class MixinBase : public Base {};
template <>
class MixinBase<void> {};

}  // namespace

#if 0//MIOPEN_VERSION >= 5000

#define MIOPEN_RETURN_IF_FAIL(STATUS, ...)                                \
  if (!SE_PREDICT_TRUE((STATUS) == miopenStatusSuccess)) {              \
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
  CudnnDropoutDescriptor(CUDAExecutor* parent, miopenHandle_t miopen_handle,
                         float dropout, uint64 seed,
                         ScratchAllocator* state_allocator)
      : parent_(parent), handle_(nullptr) {
    miopenStatus_t status;
    status = dynload::miopenCreateDropoutDescriptor(parent_, &handle_);
    MIOPEN_RETURN_IF_FAIL(status, "Failed to create dropout descriptor");

    if (dropout == 0.f) {
      return;
    }

    DeviceMemory<uint8> state_memory;
    if (state_allocator) {
      size_t state_sizes_in_bytes = 0;
      status = dynload::miopenDropoutGetStatesSize(parent_, miopen_handle,
                                                  &state_sizes_in_bytes);
      MIOPEN_RETURN_IF_FAIL(status, "Failed to query dropout state sizes");

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
    status = dynload::miopenSetDropoutDescriptor(parent_, handle_, miopen_handle,
                                                dropout, state_memory.opaque(),
                                                state_memory.size(), seed);
    MIOPEN_RETURN_IF_FAIL(status, "Failed to set dropout descriptor");
  }

  ~CudnnDropoutDescriptor() {
    if (handle_) {
      miopenStatus_t status =
          dynload::miopenDestroyDropoutDescriptor(parent_, handle_);
      MIOPEN_RETURN_IF_FAIL(status, "Failed to destroy Cudnn dropout handle: ");
    }
  }

  miopenDropoutDescriptor_t handle() const {
    if (!ok()) return nullptr;
    return handle_;
  }

 private:
  CUDAExecutor* parent_;
  miopenDropoutDescriptor_t handle_;
  float dropout_;
  uint64 seed_;
  port::Status status_;
  SE_DISALLOW_COPY_AND_ASSIGN(CudnnDropoutDescriptor);
};

class CudnnRnnParamsDescriptor : public CudnnDescriptorCommon<void> {
 public:
  typedef dnn::RnnDescriptor::ParamsRegion ParamsRegion;
  typedef dnn::RnnDescriptor::ParamsRegions ParamsRegions;
  CudnnRnnParamsDescriptor(CUDAExecutor* parent, miopenHandle_t miopen_handle,
                           const CudnnRnnDescriptor& rnn_desc);
  ~CudnnRnnParamsDescriptor() {
    miopenStatus_t status =
        dynload::miopenDestroyFilterDescriptor(parent_, handle_);
    MIOPEN_RETURN_IF_FAIL(status, "Failed to destroy RNN filter desciptor");
  }
  miopenFilterDescriptor_t handle() const {
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
  miopenFilterDescriptor_t handle_;
  const CudnnRnnDescriptor* rnn_desc_;
  int64 params_size_in_bytes_;
  ParamsRegions weights_;
  ParamsRegions biases_;
  port::Status status_;
  SE_DISALLOW_COPY_AND_ASSIGN(CudnnRnnParamsDescriptor);
};

class CudnnRnnDescriptor : public CudnnDescriptorCommon<dnn::RnnDescriptor> {
 public:
  CudnnRnnDescriptor(CUDAExecutor* parent, miopenHandle_t miopen_handle,
                     int num_layers, int hidden_size, int input_size,
                     miopenRNNInputMode_t input_mode,
                     miopenDirectionMode_t direction_mode,
                     miopenRNNMode_t rnn_mode, miopenDataType_t data_type,
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
    miopen_dropout_desc_.reset(new CudnnDropoutDescriptor(
        parent, miopen_handle, dropout, seed, state_allocator));
    if (!miopen_dropout_desc_->ok()) {
      SetFailure(miopen_dropout_desc_->Status());
      return;
    }

    // Create the RNN handle
    miopenStatus_t status =
        dynload::miopenCreateRNNDescriptor(parent_, &rnn_desc_);
    MIOPEN_RETURN_IF_FAIL(status, "Unable to create RNN descriptor");
    status = dynload::miopenSetRNNDescriptor(
        parent, rnn_desc_ /*rnnDesc*/, hidden_size /*hiddenSize*/,
        num_layers /*numLayers*/, dropout_handle() /*dropoutDesc*/,
        input_mode /*inputMode*/, direction_mode /*direction*/,
        rnn_mode /*mode*/, data_type /*dataType*/);
    MIOPEN_RETURN_IF_FAIL(status, "Unable to update RNN descriptor");

    // Create the params handle.
    miopen_params_desc_.reset(
        new CudnnRnnParamsDescriptor(parent, miopen_handle, *this));
    if (!miopen_params_desc_->ok()) {
      SetFailure(miopen_params_desc_->Status());
      return;
    }
  }
  ~CudnnRnnDescriptor() override {
    if (rnn_desc_) {
      miopenStatus_t status =
          dynload::miopenDestroyRNNDescriptor(parent_, rnn_desc_);
      MIOPEN_RETURN_IF_FAIL(status, "Unable to destroy RNN descriptor");
    }
  }
  miopenRNNDescriptor_t handle() const {
    if (!ok()) return nullptr;
    return rnn_desc_;
  }
  int num_layers() const { return num_layers_; }
  int hidden_size() const { return hidden_size_; }
  int input_size() const { return input_size_; }
  miopenRNNInputMode_t input_mode() const { return input_mode_; }
  miopenDirectionMode_t direction_mode() const { return direction_mode_; }
  miopenRNNMode_t rnn_mode() const { return rnn_mode_; }
  miopenDataType_t data_type() const { return data_type_; }
  int64 ParamsSizeInBytes() const override {
    return miopen_params_desc_->params_size_in_bytes();
  }
  miopenDropoutDescriptor_t dropout_handle() const {
    if (!miopen_dropout_desc_) return nullptr;
    return miopen_dropout_desc_->handle();
  }
  miopenFilterDescriptor_t params_handle() const {
    if (!miopen_params_desc_) return nullptr;
    return miopen_params_desc_->handle();
  }
  ParamsRegions ParamsWeightRegions() const override {
    if (!ok()) return ParamsRegions();
    return miopen_params_desc_->params_weights();
  }
  ParamsRegions ParamsBiasRegions() const override {
    if (!ok()) return ParamsRegions();
    return miopen_params_desc_->params_biases();
  }

 private:
  CUDAExecutor* parent_;
  miopenRNNDescriptor_t rnn_desc_;
  int num_layers_;
  int hidden_size_;
  int input_size_;
  miopenRNNInputMode_t input_mode_;
  miopenDirectionMode_t direction_mode_;
  miopenRNNMode_t rnn_mode_;
  miopenDataType_t data_type_;
  port::Status status_;
  std::unique_ptr<CudnnDropoutDescriptor> miopen_dropout_desc_;
  std::unique_ptr<CudnnRnnParamsDescriptor> miopen_params_desc_;
  SE_DISALLOW_COPY_AND_ASSIGN(CudnnRnnDescriptor);
};

CudnnRnnParamsDescriptor::CudnnRnnParamsDescriptor(
    CUDAExecutor* parent, miopenHandle_t miopen_handle,
    const CudnnRnnDescriptor& rnn_desc)
    : parent_(parent),
      handle_(nullptr),
      rnn_desc_(&rnn_desc),
      params_size_in_bytes_(0) {
  miopenTensorDescriptor_t input_desc = nullptr;
  {
    // Query the params size.
    auto status = dynload::miopenCreateTensorDescriptor(parent, &input_desc);
    MIOPEN_RETURN_IF_FAIL(status, "Cudnn fails to create tensor descriptor");
    int dims[] = {1, rnn_desc.input_size(), 1};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    status = dynload::miopenSetTensorDescriptor(
        parent, input_desc /*tensorDesc*/, rnn_desc.data_type() /*dataType*/,
        sizeof(dims) / sizeof(dims[0]) /*nbDims*/, dims /*dimA*/,
        strides /*strideA*/);
    MIOPEN_RETURN_IF_FAIL(status, "Cudnn fails to set tensor descriptor");

    size_t params_size = 0;
    status = dynload::miopenGetRNNParamsSize(
        parent, miopen_handle /*handle*/, rnn_desc.handle() /*rnnDesc*/,
        input_desc /*xDesc*/, &params_size /*sizeInBytes*/,
        rnn_desc.data_type() /*dataType*/);
    MIOPEN_RETURN_IF_FAIL(status, "Cudnn fails to get RNN parameter size");
    params_size_in_bytes_ = static_cast<int64>(params_size);
  }

  {
    // Create the params descriptor.
    auto status = dynload::miopenCreateFilterDescriptor(parent, &handle_);
    MIOPEN_RETURN_IF_FAIL(status, "Cudnn fails to create RNN filter descriptor");
    int dims[] = {static_cast<int>(params_size_in_bytes_), 1, 1};
    status = dynload::miopenSetFilterNdDescriptor(
        parent, handle_ /*filterDesc*/, rnn_desc.data_type() /*dataType*/,
        MIOPEN_TENSOR_NCHW /*format*/, sizeof(dims) / sizeof(dims[0]) /*nbDims*/,
        dims /*filterDimA*/);
    MIOPEN_RETURN_IF_FAIL(status, "Cudnn fails to update RNN filter descriptor");
  }

  {
    // Create the weights and biases into the params buffer
    int region_count_per_layer = GetRegionCountPerLayer();
    miopenFilterDescriptor_t region_desc_handle = nullptr;
    auto status =
        dynload::miopenCreateFilterDescriptor(parent, &region_desc_handle);
    MIOPEN_RETURN_IF_FAIL(status, "Cudnn fails to create filter descriptor");
    for (int layer = 0; layer < rnn_desc.num_layers(); layer++) {
      for (int region = 0; region < region_count_per_layer; region++) {
        for (int type = 0; type < 2; type++) {
          void* offset = nullptr;
          if (type == 0) {
            status = dynload::miopenGetRNNLinLayerMatrixParams(
                parent, miopen_handle /*handle*/, rnn_desc.handle() /*rnnDesc*/,
                layer /*layer*/, input_desc /*xDesc*/, handle_ /*wDesc*/,
                nullptr /*w*/, region /*linLayerID*/,
                region_desc_handle /*linLayerMatDesc*/,
                &offset /*linLayerMat*/);
            MIOPEN_RETURN_IF_FAIL(
                status, "Cudnn fails to call miopenGetRNNLinLayerMatrixParams");
          } else {
            status = dynload::miopenGetRNNLinLayerBiasParams(
                parent, miopen_handle /*rnnDesc*/, rnn_desc.handle() /*rnnDesc*/,
                layer /*layer*/, input_desc /*xDesc*/, handle_ /*wDesc*/,
                nullptr /*w*/, region /*linLayerID*/,
                region_desc_handle /*linLayerBiasDesc*/,
                &offset /*linLayerBias*/);
            MIOPEN_RETURN_IF_FAIL(
                status, "Cudnn fails to call miopenGetRNNLinLayerBiasParams");
          }
          int dims[] = {1, 1, 1};
          miopenDataType_t data_type;
          miopenTensorFormat_t tensor_format;
          int n_dims;
          status = dynload::miopenGetFilterNdDescriptor(
              parent, region_desc_handle /*filterDesc*/,
              sizeof(dims) / sizeof(dims[0]) /*nbDimsRequested*/,
              &data_type /*dataType*/, &tensor_format /*format*/,
              &n_dims /*nbDims*/, dims /*filterDimA*/);
          MIOPEN_RETURN_IF_FAIL(status, "Cudnn fails to get filter description");
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
    status = dynload::miopenDestroyFilterDescriptor(parent, region_desc_handle);
    MIOPEN_RETURN_IF_FAIL(status, "Cudnn fails to destroy filter descriptor");
  }

  {
    // Release the dummy input tensor descriptor.
    auto status = dynload::miopenDestroyTensorDescriptor(parent, input_desc);
    MIOPEN_RETURN_IF_FAIL(status, "Cudnn fails to destroy tensor descriptor");
  }
}

int CudnnRnnParamsDescriptor::GetRegionCountPerLayer() const {
  auto rnn_mode = rnn_desc_->rnn_mode();
  switch (rnn_mode) {
    case MIOPEN_RNN_RELU:
    case MIOPEN_RNN_TANH:
      return 2;
    case MIOPEN_LSTM:
      return 8;
    case MIOPEN_GRU:
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
                                   miopenDataType_t data_type)
      : parent_(parent),
        seq_length_(seq_length),
        batch_size_(batch_size),
        data_size_(data_size),
        data_type_(data_type) {
    miopenTensorDescriptor_t handle = nullptr;
    if (seq_length <= 0) {
      string error_msg =
          port::StrCat("sequence length must be positive: ", seq_length);
      LOG(ERROR) << error_msg;
      SetFailure(port::Status(port::error::UNKNOWN, error_msg));
      return;
    }
    miopenStatus_t status =
        dynload::miopenCreateTensorDescriptor(parent, &handle);
    MIOPEN_RETURN_IF_FAIL(status, "Failed to create tensor descriptor");
    int dims[] = {batch_size, data_size, 1};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    status = dynload::miopenSetTensorDescriptor(
        parent, handle /*tensorDesc*/, data_type /*dataType*/,
        sizeof(dims) / sizeof(dims[0]) /*nbDims*/, dims /*dimA*/,
        strides /*strideA*/);
    MIOPEN_RETURN_IF_FAIL(status, "Failed to update tensor descriptor");
    // Replicate handle across the number of steps.
    handles_.assign(seq_length, handle);
  }

  ~CudnnRnnSequenceTensorDescriptor() override {
    // Only the first one needs to be destroyed. All others are the same.
    miopenStatus_t status =
        dynload::miopenDestroyTensorDescriptor(parent_, handles_[0]);
    MIOPEN_RETURN_IF_FAIL(status, "Failed to destroy sequence tensor desciptor");
  }

  const miopenTensorDescriptor_t* handles() const {
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
  miopenDataType_t data_type_;
  std::vector<miopenTensorDescriptor_t> handles_;
  port::Status status_;
  SE_DISALLOW_COPY_AND_ASSIGN(CudnnRnnSequenceTensorDescriptor);
};

class CudnnRnnStateTensorDescriptor
    : public CudnnDescriptorCommon<dnn::RnnStateTensorDescriptor> {
 public:
  CudnnRnnStateTensorDescriptor(CUDAExecutor* parent, int num_layers,
                                int batch_size, int data_size,
                                miopenDataType_t data_type)
      : parent_(parent),
        handle_(nullptr),
        num_layers_(num_layers),
        batch_size_(batch_size),
        data_size_(data_size),
        data_type_(data_type) {
    miopenStatus_t status =
        dynload::miopenCreateTensorDescriptor(parent, &handle_);
    MIOPEN_RETURN_IF_FAIL(status, "Failed to create tensor descriptor");
    int dims[] = {num_layers, batch_size, data_size};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    status = dynload::miopenSetTensorDescriptor(
        parent, handle_ /*tensorDesc*/, data_type /*dataType*/,
        sizeof(dims) / sizeof(dims[0]) /*nbDims*/, dims /*dimA*/,
        strides /*strideA*/);
    MIOPEN_RETURN_IF_FAIL(status, "Failed to update tensor descriptor");
  }

  ~CudnnRnnStateTensorDescriptor() override {
    if (!handle_) {
      miopenStatus_t status =
          dynload::miopenDestroyTensorDescriptor(parent_, handle_);
      MIOPEN_RETURN_IF_FAIL(status, "Unable to destroy RNN state tensor");
    }
  }

  miopenTensorDescriptor_t handle() const {
    if (!ok()) return nullptr;
    return handle_;
  }
  int num_layers() const { return num_layers_; }
  int batch_size() const { return batch_size_; }
  int data_size() const { return data_size_; }

 private:
  CUDAExecutor* parent_;
  miopenTensorDescriptor_t handle_;
  int num_layers_;
  int batch_size_;
  int data_size_;
  port::Status status_;
  miopenDataType_t data_type_;
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
      (rnn_desc.direction_mode() == MIOPEN_BIDIRECTIONAL) ? 2 : 1;

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

bool CheckRNNParameterSize(CUDAExecutor* parent, miopenHandle_t miopen_handle,
                           const CudnnRnnDescriptor& rnn_desc,
                           const CudnnRnnSequenceTensorDescriptor& input_desc) {
  size_t params_size_in_bytes = 0;
  miopenStatus_t status = dynload::miopenGetRNNParamsSize(
      parent, miopen_handle /*handle*/, rnn_desc.handle() /*rnnDesc*/,
      input_desc.handles()[0] /*xDesc*/, &params_size_in_bytes /*sizeInBytes*/,
      rnn_desc.data_type() /*dataType*/);
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "Unable to check RNN param size: " << ToString(status);
    return false;
  }
  return static_cast<int64>(params_size_in_bytes) ==
         rnn_desc.ParamsSizeInBytes();
}

bool CreateRnnWorkspace(Stream* stream, CUDAExecutor* parent,
                        miopenHandle_t miopen_handle,
                        const CudnnRnnDescriptor& rnn_desc,
                        const CudnnRnnSequenceTensorDescriptor& input_desc,
                        ScratchAllocator* workspace_allocator,
                        DeviceMemory<uint8>* workspace) {
  // Query the workspace size.
  size_t workspace_size_in_bytes = 0;
  miopenStatus_t status = dynload::miopenGetRNNWorkspaceSize(
      parent, miopen_handle /*handle*/, rnn_desc.handle() /*rnnDesc*/,
      input_desc.seq_length() /*seqLength*/, input_desc.handles() /*xDesc*/,
      &workspace_size_in_bytes /*sizeInBytes*/);
  if (status != miopenStatusSuccess) {
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
    miopenStatus_t status = dynload::miopenGetRNNTrainingReserveSize(
        parent_, ToHandle(dnn_handle_) /*handle*/,
        rnn_desc.handle() /*rnnDesc*/, model_dims.seq_length /*seqLength*/,
        input_desc.handles() /*xDesc*/,
        &reserve_space_size_in_bytes /*sizeInBytes*/);
    if (status != miopenStatusSuccess) {
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
    miopenStatus_t status = dynload::miopenRNNForwardInference(
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
    if (status != miopenStatusSuccess) {
      LOG(ERROR) << "Failed to call miopenRNNForwardInference: "
                 << ToString(status);
      return false;
    }
  } else {
    miopenStatus_t status = dynload::miopenRNNForwardTraining(
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
    if (status != miopenStatusSuccess) {
      LOG(ERROR) << "Failed to call miopenRNNForwardTraining"
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
  miopenStatus_t status = dynload::miopenRNNBackwardData(
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
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "Failed to call miopenRNNBackwardData: " << ToString(status);
    return false;
  }

  if (params_backprop_data != nullptr) {
    // Clear the dw to zeros.
    stream->ThenMemZero(params_backprop_data, params_backprop_data->size());
    // make the backward weight call
    status = dynload::miopenRNNBackwardWeights(
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
    if (status != miopenStatusSuccess) {
      LOG(ERROR) << "Failed to call miopenRNNBackwardWeights: "
                 << ToString(status);
      return false;
    }
  }

  return true;
}

#endif  // MIOPEN_VERSION

port::StatusOr<std::unique_ptr<dnn::RnnDescriptor>>
CudnnSupport::createRnnDescriptor(int num_layers, int hidden_size,
                                  int input_size, dnn::RnnInputMode input_mode,
                                  dnn::RnnDirectionMode direction_mode,
                                  dnn::RnnMode rnn_mode,
                                  dnn::DataType data_type, float dropout,
                                  uint64 seed,
                                  ScratchAllocator* state_allocator) {
#if 0//MIOPEN_VERSION >= 5000
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
      port::StrCat("createRnnDescriptor not supported ");
  LOG(ERROR) << error_msg;
  return port::Status{port::error::UNIMPLEMENTED, error_msg};
#endif  // MIOPEN_VERSION
}

port::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
CudnnSupport::createRnnSequenceTensorDescriptor(int seq_length, int batch_size,
                                                int data_size,
                                                dnn::DataType data_type) {
#if 0//MIOPEN_VERSION >= 5000
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
      "createRnnSequenceTensorDescriptor not supported");
  LOG(ERROR) << error_msg;
  return port::Status{port::error::UNIMPLEMENTED, error_msg};
#endif  // MIOPEN_VERSION
}

port::StatusOr<std::unique_ptr<dnn::RnnStateTensorDescriptor>>
CudnnSupport::createRnnStateTensorDescriptor(int num_layer, int batch_size,
                                             int data_size,
                                             dnn::DataType data_type) {
#if 0//MIOPEN_VERSION >= 5000
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
      "createRnnStateTensorDescriptor not supported");
  LOG(ERROR) << error_msg;
  return port::Status{port::error::UNIMPLEMENTED, error_msg};
#endif  // MIOPEN_VERSION
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
#if 0//MIOPEN_VERSION >= 5000
  const CudnnRnnDescriptor& miopen_rnn_desc =
      static_cast<const CudnnRnnDescriptor&>(rnn_desc);
  const CudnnRnnSequenceTensorDescriptor& miopen_input_desc =
      static_cast<const CudnnRnnSequenceTensorDescriptor&>(input_desc);
  const CudnnRnnStateTensorDescriptor& miopen_input_h_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(input_h_desc);
  const CudnnRnnStateTensorDescriptor& miopen_input_c_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(input_c_desc);
  const CudnnRnnSequenceTensorDescriptor& miopen_output_desc =
      static_cast<const CudnnRnnSequenceTensorDescriptor&>(output_desc);
  const CudnnRnnStateTensorDescriptor& miopen_output_h_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(output_h_desc);
  const CudnnRnnStateTensorDescriptor& miopen_output_c_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(output_c_desc);

  return DoRnnForwardImpl<float>(
      stream, miopen_rnn_desc, miopen_input_desc, input_data, miopen_input_h_desc,
      input_h_data, miopen_input_c_desc, input_c_data, params, miopen_output_desc,
      output_data, miopen_output_h_desc, output_h_data, miopen_output_c_desc,
      output_c_data, is_training, reserve_space_allocator, workspace_allocator);
#else
  return false;
#endif  // MIOPEN_VERSION
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
#if 0//MIOPEN_VERSION >= 5000
  const CudnnRnnDescriptor& miopen_rnn_desc =
      static_cast<const CudnnRnnDescriptor&>(rnn_desc);
  const CudnnRnnSequenceTensorDescriptor& miopen_input_desc =
      static_cast<const CudnnRnnSequenceTensorDescriptor&>(input_desc);
  const CudnnRnnStateTensorDescriptor& miopen_input_h_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(input_h_desc);
  const CudnnRnnStateTensorDescriptor& miopen_input_c_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(input_c_desc);
  const CudnnRnnSequenceTensorDescriptor& miopen_output_desc =
      static_cast<const CudnnRnnSequenceTensorDescriptor&>(output_desc);
  const CudnnRnnStateTensorDescriptor& miopen_output_h_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(output_h_desc);
  const CudnnRnnStateTensorDescriptor& miopen_output_c_desc =
      static_cast<const CudnnRnnStateTensorDescriptor&>(output_c_desc);

  return DoRnnBackwardImpl<float>(
      stream, miopen_rnn_desc, miopen_input_desc, input_data, miopen_input_h_desc,
      input_h_data, miopen_input_c_desc, input_c_data, params, miopen_output_desc,
      output_data, miopen_output_h_desc, output_h_data, miopen_output_c_desc,
      output_c_data, output_backprop_data, output_h_backprop_data,
      output_c_backprop_data, input_backprop_data, input_h_backprop_data,
      input_c_backprop_data, params_backprop_data, reserve_space_data,
      workspace_allocator);
#else
  return false;
#endif  // MIOPEN_VERSION
}

template <class T>
bool CudnnSupport::DoConvolveImpl(
    Stream* stream, int miopen_type,  // Actually miopenDataType_t.
    const BatchDescriptor& batch_descriptor, const DeviceMemory<T>& input_data,
    const FilterDescriptor& filter_descriptor,
    const DeviceMemory<T>& filter_data,
    const ConvolutionDescriptor& convolution_descriptor,
    const BatchDescriptor& output_descriptor, DeviceMemory<T>* output_data,
    ScratchAllocator* scratch_allocator,
    const dnn::AlgorithmConfig& algorithm_config,
    dnn::ProfileResult* output_profile_result) {
  ScopedTensorDescriptor input_nd{parent_, batch_descriptor,
      static_cast<miopenDataType_t>(miopen_type)};
  ScopedTensorDescriptor output_nd{parent_, output_descriptor,
      static_cast<miopenDataType_t>(miopen_type)};
  ScopedFilterDescriptor filter{parent_, filter_descriptor, batch_descriptor,
      static_cast<miopenDataType_t>(miopen_type)};
  // TODO(sesse): Figure out under what circumstances cuDNN would
  // accept miopenHalf here; probably related to compute capability
  // and cuDNN version; at least cuDNN 4 on TITAN X only supports
  // miopenFloat even for half input.
  ScopedConvolutionDescriptor conv{parent_, convolution_descriptor,
      miopenFloat};

  mutex_lock lock{dnn_handle_mutex_};
  auto status = dynload::miopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != miopenStatusSuccess) {
    LOG(FATAL) << "failed to set stream for miopen handle: " << ToString(status);
  }
  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  const bool is_profiling = output_profile_result != nullptr;
  std::pair<miopenConvFwdAlgorithm_t,size_t> algo_sz;
  DeviceMemory<uint8> scratch;

  if (algorithm_config.algorithm() == dnn::kDefaultAlgorithm) {
    // With the default algorithm, use Cudnn's heuristics.
    auto get_algorithm = [&]()
        SHARED_LOCKS_REQUIRED(dnn_handle_mutex_) -> std::pair<miopenConvFwdAlgorithm_t, size_t> {

          assert (scratch_allocator);

          size_t size_in_bytes;
          status = dynload::miopenConvolutionForwardGetWorkSpaceSize(
              parent_, ToHandle(dnn_handle_), /*filterDesc=*/filter.handle(),
              /*srcDesc=*/input_nd.handle(), /*convDesc=*/conv.handle(),
              /*destDesc=*/output_nd.handle(), /*sizeInBytes=*/&size_in_bytes);
          if (status == miopenStatusSuccess && size_in_bytes != 0) {
            auto allocated =
                scratch_allocator->AllocateBytes(stream, size_in_bytes);
            if (allocated.ok()) {
              scratch = allocated.ValueOrDie();
            }
          }

          miopenConvAlgoPerf_t preference;
          int returnedAlgoCount;

          status = dynload::miopenFindConvolutionForwardAlgorithm(
              parent_, ToHandle(dnn_handle_), input_nd.handle(),
              input_data.opaque(), filter.handle(), filter_data.opaque(),
              conv.handle(), output_nd.handle(), output_data->opaque(),
              /*requestAlgoCount=*/1, &returnedAlgoCount,
              /*preference=*/&preference, /*workspace*/scratch.opaque(),
              /*WorkSpaceSize*/scratch.size(), /*exhaustiveSearch*/false);


          CHECK_EQ(status, miopenStatusSuccess)
              << "Unable to find a suitable "
                 "algorithm for doing forward "
                 "convolution";
          return std::pair<miopenConvFwdAlgorithm_t, size_t> (preference.fwd_algo, preference.memory);
        };

    algo_sz = get_algorithm();

    // MIOpen requires workspace:
    assert (scratch != nullptr) ;

  } else {
    // An algorithm has been specified.
    algo_sz.first = ToConvForwardAlgo(algorithm_config.algorithm());
    algo_sz.second = algorithm_config.algorithm_scratch_size();

    size_t size_in_bytes = algo_sz.second;
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
        algo_sz.first = ToConvForwardAlgo(algorithm_config.algorithm_no_scratch());
        algo_sz.second = 0;
      }
    }
  }

  std::unique_ptr<CUDATimer> timer;
  if (is_profiling) {
    timer.reset(new CUDATimer(parent_));
    if (!timer->Init()) {
      return false;
    }
    // The start and stop of the timer should be as close to the Cudnn call as
    // possible. It is still possible for other threads to issue workload on
    // to this stream. So it could take multiple profiling measurements.
    if (!timer->Start(AsCUDAStream(stream))) {
      timer->Destroy();
      return false;
    }
  }
  status = dynload::miopenConvolutionForward(
      parent_, ToHandle(dnn_handle_),
      /*alpha=*/&alpha, /*srcDesc=*/input_nd.handle(),
      /*srcData=*/input_data.opaque(), /*filterDesc=*/filter.handle(),
      /*filterData=*/filter_data.opaque(), /*convDesc=*/conv.handle(),
      /*algo=*/algo_sz.first, /*beta=*/&beta, /*destDesc=*/output_nd.handle(), 
      /*destData=*/output_data->opaque(), /*workSpace=*/scratch.opaque(), 
      /*workSpaceSizeInBytes=*/scratch.size());
  if (is_profiling) {
    if (!timer->Stop(AsCUDAStream(stream))) {
      timer->Destroy();
      return false;
    }
    output_profile_result->set_is_valid(true);
    output_profile_result->set_algorithm(algo_sz.first);
    output_profile_result->set_scratch_size(algo_sz.second);
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
    timer->Destroy();
  }

  if (status != miopenStatusSuccess) {
    // Silently return when we are profiling.
    if (!is_profiling) {
      LOG(FATAL) << "failed to enqueue convolution on stream: "
                 << ToString(status);
    }
    return false;
  }

  return true;
}

// A helper class to decide whether to enable the WINOGRAD_NONFUSED algorithms.
// Doing so by default make a few TensorFlow test cases to fail. Users can
// explicitly enable them through an env-var "TF_ENABLE_WINOGRAD_NONFUSED=1".
// https://github.com/tensorflow/tensorflow/pull/4901
class WinogradNonfused {
 public:
  static bool IsEnabled() {
    static bool is_enabled = IsEnabledImpl();
    return is_enabled;
  }

 private:
  static bool IsEnabledImpl() {
    const char* tf_env_var_val = getenv("TF_ENABLE_WINOGRAD_NONFUSED");
    if (tf_env_var_val != nullptr) {
      port::StringPiece tf_env_var_val_str(tf_env_var_val);
      if (tf_env_var_val_str == "0") {
        return false;
      }
      return true;
    }
    // TODO(zhengxq): turn the default to True when the test failure is
    // resolved.
    return false;
  }
};

bool CudnnSupport::GetConvolveAlgorithms(
    std::vector<dnn::AlgorithmType>* out_algorithms) {
  out_algorithms->assign({
      // clang-format off
      miopenConvolutionFwdAlgoGEMM,
      miopenConvolutionFwdAlgoDirect,
      miopenConvolutionFwdAlgoFFT,
      miopenConvolutionFwdAlgoWinograd,
      // clang-format on
  });
  return true;
}

bool CudnnSupport::GetConvolveBackwardDataAlgorithms(
    std::vector<dnn::AlgorithmType>* out_algorithms) {
  out_algorithms->assign({
      // clang-format off
      miopenConvolutionBwdDataAlgoGEMM,
      miopenConvolutionBwdDataAlgoDirect,
      miopenConvolutionBwdDataAlgoFFT,
      miopenConvolutionBwdDataAlgoWinograd,
      miopenTransposeBwdDataAlgoGEMM,
      // clang-format on
  });
  return true;
}

bool CudnnSupport::GetConvolveBackwardFilterAlgorithms(
    std::vector<dnn::AlgorithmType>* out_algorithms) {
  out_algorithms->assign({
      // clang-format off
      miopenConvolutionBwdWeightsAlgoGEMM,
      miopenConvolutionBwdWeightsAlgoDirect,
      // clang-format on
  });
  return true;
}

bool CudnnSupport::DoBatchNormalizationForward(
    Stream* stream, const DeviceMemory<float>& x,
    const DeviceMemory<float>& scale, const DeviceMemory<float>& offset,
    const DeviceMemory<float>& estimated_mean,
    const DeviceMemory<float>& estimated_variance,
    const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    DeviceMemory<float>* y, DeviceMemory<float>* batch_mean,
    DeviceMemory<float>* batch_var, DeviceMemory<float>* saved_mean,
    DeviceMemory<float>* saved_inv_var, bool is_training,
    std::function<const DeviceMemory<float>&()> var_to_inv_var,
    std::function<void()> inv_var_to_var) {
  return DoBatchNormalizationForwardImpl<float>(
      stream, dnn::DataType::kFloat, x, scale, offset, estimated_mean,
      estimated_variance, x_desc, scale_offset_desc, epsilon, y, batch_mean,
      batch_var, saved_mean, saved_inv_var, is_training,
      std::move(var_to_inv_var), std::move(inv_var_to_var));
}

template <class T>
bool CudnnSupport::DoBatchNormalizationForwardImpl(
    Stream* stream, dnn::DataType data_type, const DeviceMemory<T>& x,
    const DeviceMemory<T>& scale, const DeviceMemory<T>& offset,
    const DeviceMemory<T>& estimated_mean,
    const DeviceMemory<T>& estimated_variance,
    const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    DeviceMemory<T>* y, DeviceMemory<T>* batch_mean, DeviceMemory<T>* batch_var,
    DeviceMemory<T>* saved_mean, DeviceMemory<T>* saved_inv_var,
    bool is_training, std::function<const DeviceMemory<T>&()> var_to_inv_var,
    std::function<void()> inv_var_to_var) {
  mutex_lock lock{dnn_handle_mutex_};
  auto status = dynload::miopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to set stream for miopen handle: " << ToString(status);
    return false;
  }

  ScopedTensorDescriptor x_descriptor{parent_, x_desc,
                                      ToCudnnDataType(data_type)};
  ScopedTensorDescriptor scale_offset_descriptor{parent_, scale_offset_desc,
                                                 ToCudnnDataType(data_type)};
  miopenBatchNormMode_t mode = miopenBNSpatial;
  float one = 1.0;
  float zero = 0.0;

  if (is_training) {
    stream->ThenMemZero(batch_mean, batch_mean->size());
    stream->ThenMemZero(batch_var, batch_var->size());
    status = dynload::miopenBatchNormalizationForwardTraining(
        parent_, ToHandle(dnn_handle_), mode, &one, &zero,
        x_descriptor.handle(), x.opaque(), x_descriptor.handle(), y->opaque(),
        scale_offset_descriptor.handle(), const_cast<void*>(scale.opaque()), 
        const_cast<void*>(offset.opaque()), 1.0,
        batch_mean->opaque(), batch_var->opaque(), epsilon,
        saved_mean->opaque(), saved_inv_var->opaque());
#if 1//MIOPEN_VERSION < 5000
    CHECK(inv_var_to_var);
    inv_var_to_var();
#endif
  } else {
#if 1//MIOPEN_VERSION < 5000
    CHECK(var_to_inv_var);
    const void* maybe_inv_var = var_to_inv_var().opaque();
#else
    const void* maybe_inv_var = estimated_variance.opaque();
#endif
    status = dynload::miopenBatchNormalizationForwardInference(
        parent_, ToHandle(dnn_handle_), mode, &one, &zero,
        x_descriptor.handle(), x.opaque(), x_descriptor.handle(), y->opaque(),
        scale_offset_descriptor.handle(), const_cast<void*>(scale.opaque()), const_cast<void*>(offset.opaque()),
        const_cast<void*>(estimated_mean.opaque()), const_cast<void*>(maybe_inv_var), epsilon);
  }
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to enqueue forward batch normalization on stream: "
               << ToString(status);
    return false;
  }
  return true;
}

bool CudnnSupport::DoBatchNormalizationBackward(
    Stream* stream, const DeviceMemory<float>& y_backprop,
    const DeviceMemory<float>& x, const DeviceMemory<float>& scale,
    const DeviceMemory<float>& mean, const DeviceMemory<float>& variance,
    const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    DeviceMemory<float>* x_backprop, DeviceMemory<float>* scale_backprop,
    DeviceMemory<float>* offset_backprop) {
  return DoBatchNormalizationBackwardImpl(
      stream, miopenFloat, y_backprop, x, scale, mean, variance, x_desc,
      scale_offset_desc, epsilon, x_backprop, scale_backprop, offset_backprop);
}

template <class T>
bool CudnnSupport::DoBatchNormalizationBackwardImpl(
    Stream* stream, int miopen_type, const DeviceMemory<T>& y_backprop,
    const DeviceMemory<T>& x, const DeviceMemory<T>& scale,
    const DeviceMemory<T>& mean, const DeviceMemory<T>& variance,
    const dnn::BatchDescriptor& x_desc,
    const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
    DeviceMemory<T>* x_backprop, DeviceMemory<T>* scale_backprop,
    DeviceMemory<T>* offset_backprop) {
  mutex_lock lock{dnn_handle_mutex_};
  auto status = dynload::miopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to set stream for miopen handle: " << ToString(status);
    return false;
  }

  ScopedTensorDescriptor x_descriptor{parent_, x_desc,
                                      static_cast<miopenDataType_t>(miopen_type)};
  ScopedTensorDescriptor scale_offset_descriptor{
      parent_, scale_offset_desc, static_cast<miopenDataType_t>(miopen_type)};
  miopenBatchNormMode_t mode = miopenBNSpatial;
  float one = 1.0;
  float zero = 0.0;

  status = dynload::miopenBatchNormalizationBackward(
      parent_, ToHandle(dnn_handle_), mode, &one, &zero, &one, &zero,
      x_descriptor.handle(), x.opaque(), x_descriptor.handle(),
      y_backprop.opaque(), x_descriptor.handle(), x_backprop->opaque(),
      scale_offset_descriptor.handle(), scale.opaque(),
      scale_backprop->opaque(), offset_backprop->opaque(), epsilon,
      mean.opaque(), variance.opaque());
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to enqueue backward batch normalization on stream: "
               << ToString(status);
    return false;
  }
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
      stream, miopenFloat, batch_descriptor, input_data, filter_descriptor,
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
  LOG(ERROR) << "miopen does not support half type";
  return DoConvolveImpl<Eigen::half>(
      stream, miopenHalf, batch_descriptor, input_data, filter_descriptor,
      filter_data, convolution_descriptor, output_descriptor, output_data,
      scratch_allocator, algorithm_config, output_profile_result);
}

template<class T>
DeviceMemory<T> CudnnSupport::MaybeTransformLayout(
    Stream* stream,
    int miopen_type,  // Actually miopenDataType_t.
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
      parent_, *output_descriptor, static_cast<miopenDataType_t>(miopen_type)};
  ScopedTensorDescriptor transformed_out_back_nd{
      parent_, transformed_output_descriptor,
      static_cast<miopenDataType_t>(miopen_type)};

  float alpha1 = 1.0f;
  float alpha2 = 0.0f;
  float beta = 0.0f;
  auto status = dynload::miopenOpTensor(
      parent_, ToHandle(dnn_handle_), miopenTensorOpAdd, &alpha1, orig_out_back_nd.handle(),
      backward_output_data.opaque(), &alpha2, orig_out_back_nd.handle(), 
      backward_output_data.opaque(), &beta, transformed_out_back_nd.handle(),
      (*transform_scratch)->mutable_device_memory()->opaque());

  if (status != miopenStatusSuccess) {
    LOG(FATAL) << "Failed to transform the data layout.";
  }
  output_descriptor->set_layout(dnn::DataLayout::kBatchDepthYX);
  return (*transform_scratch)->device_memory();
}

template <class T>
bool CudnnSupport::DoConvolveBackwardDataImpl(
    Stream* stream,
    int miopen_type,  // Actually miopenDataType_t.
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
  auto status = dynload::miopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != miopenStatusSuccess) {
    LOG(FATAL) << "failed to set stream for miopen handle: " << ToString(status);
  }

  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  // TBD(keveman): remove once cuDNN supports kBatchYXDepth for backward pass.
  BatchDescriptor output_descriptor;
  output_descriptor.CloneFrom(output_descriptor_in);
  std::unique_ptr<TemporaryDeviceMemory<T>> transform_scratch;
  backward_output_data = MaybeTransformLayout(
      stream, miopen_type, &output_descriptor, backward_output_data,
      &transform_scratch);

  ScopedTensorDescriptor out_back_nd{parent_, output_descriptor,
                                     static_cast<miopenDataType_t>(miopen_type)};
  ScopedTensorDescriptor in_back_nd{parent_, input_descriptor,
                                    static_cast<miopenDataType_t>(miopen_type)};
  ScopedFilterDescriptor filter{parent_, filter_descriptor, input_descriptor,
                                static_cast<miopenDataType_t>(miopen_type)};
  // TODO(sesse): Figure out under what circumstances cuDNN would
  // accept miopenHalf here; probably related to compute capability
  // and cuDNN version; at least cuDNN 4 on TITAN X only supports
  // miopenFloat even for half input.
  ScopedConvolutionDescriptor conv{parent_, convolution_descriptor,
                                   miopenFloat};

  const bool is_profiling = output_profile_result != nullptr;
  std::pair<miopenConvBwdDataAlgorithm_t, size_t> algo_sz;
  DeviceMemory<uint8> scratch;

  if (algorithm_config.algorithm() == dnn::kDefaultAlgorithm) {
    // With the default algorithm, use Cudnn's heuristics.
    auto get_algorithm = [&]() SHARED_LOCKS_REQUIRED(
        dnn_handle_mutex_) -> std::pair<miopenConvBwdDataAlgorithm_t, size_t> {

      assert (scratch_allocator);

      size_t size_in_bytes;
      status = dynload::miopenConvolutionBackwardDataGetWorkSpaceSize(
          parent_, ToHandle(dnn_handle_),
          /*diffDesc=*/out_back_nd.handle(),
          /*filterDesc=*/filter.handle(),
          /*convDesc=*/conv.handle(),
          /*gradDesc=*/in_back_nd.handle(),
          /*sizeInBytes=*/&size_in_bytes);
      if (status == miopenStatusSuccess && size_in_bytes != 0) {
        auto allocated =
            scratch_allocator->AllocateBytes(stream, size_in_bytes);
        if (allocated.ok()) {
          scratch = allocated.ValueOrDie();
        }
      }

      miopenConvAlgoPerf_t preference;
      int returnedAlgoCount;

      miopenStatus_t status = dynload::miopenFindConvolutionBackwardDataAlgorithm(
          parent_, ToHandle(dnn_handle_),
          /*diffDesc=*/out_back_nd.handle(), backward_output_data.opaque(),
          /*filterDesc=*/filter.handle(), filter_data.opaque(),
          /*convDesc=*/conv.handle(),
          /*gradDesc=*/in_back_nd.handle(), backward_input_data->opaque(),
          /*requestCount=*/1, /*returnedAlgoCount=*/&returnedAlgoCount,
          /*preference=*/&preference, /*WorkSpace=*/scratch.opaque(),
          /*WorkSpaceSize=*/scratch.size(), /*exhaustiveSearch=*/false);
      CHECK_EQ(status, miopenStatusSuccess) << "Unable to find a suitable "
                                                "algorithm for doing backward "
                                                "filter convolution";
      return std::pair<miopenConvBwdDataAlgorithm_t , size_t> (preference.bwd_data_algo, preference.memory);
    };

    algo_sz = get_algorithm();

    // MIOpen requires workspace:
    assert (scratch != nullptr) ;

  } else {
    // An algorithm has been specified.
    algo_sz.first  = ToConvBackwardDataAlgo(algorithm_config.algorithm());
    algo_sz.second = algorithm_config.algorithm_scratch_size();

    size_t size_in_bytes = algo_sz.second;
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
        algo_sz.first = ToConvBackwardDataAlgo(algorithm_config.algorithm_no_scratch());
        algo_sz.second = 0;
      }
    }
  }

  std::unique_ptr<CUDATimer> timer;
  if (is_profiling) {
    timer.reset(new CUDATimer(parent_));
    timer->Init();
    // The start and stop of the timer should be as close to the Cudnn call as
    // possible. It is still possible for other threads to issue workload on
    // to this stream. So it could take multiple profiling measurements.
    timer->Start(AsCUDAStream(stream));
  }

  status = dynload::miopenConvolutionBackwardData(
      parent_, ToHandle(dnn_handle_),
      /*alpha=*/&alpha,
      /*diffDesc=*/out_back_nd.handle(),
      /*diffData=*/backward_output_data.opaque(),
      /*filterDesc=*/filter.handle(),
      /*filterData=*/filter_data.opaque(),
      /*convDesc=*/conv.handle(),
      /*algo=*/algo_sz.first,
      /*beta=*/&beta,
      /*gradDesc=*/in_back_nd.handle(),
      /*gradData=*/backward_input_data->opaque(),
      /*workSpace=*/scratch.opaque(),
      /*workSpaceSizeInBytes=*/scratch.size());

  if (is_profiling) {
    timer->Stop(AsCUDAStream(stream));
    output_profile_result->set_is_valid(true);
    output_profile_result->set_algorithm(algo_sz.first);
    output_profile_result->set_scratch_size(algo_sz.second);
    //output_profile_result->set_scratch_size(algo_sz.second);
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
    timer->Destroy();
  }
  if (status != miopenStatusSuccess) {
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
      stream, miopenFloat, filter_descriptor, filter_data,
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
  LOG(ERROR) << "miopen does not support half type";
  return DoConvolveBackwardDataImpl(
      stream, miopenHalf, filter_descriptor, filter_data,
      output_descriptor_in, backward_output_data, convolution_descriptor,
      input_descriptor, backward_input_data, scratch_allocator,
      algorithm_config, output_profile_result);
}

template <class T>
bool CudnnSupport::DoConvolveBackwardFilterImpl(
    Stream* stream, int miopen_type,  // Actually miopenDataType_t.
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
  auto status = dynload::miopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != miopenStatusSuccess) {
    LOG(FATAL) << "failed to set stream for miopen handle: " << ToString(status);
  }

  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  // TBD(keveman): remove once cuDNN supports kBatchYXDepth for backward pass.
  BatchDescriptor output_descriptor;
  output_descriptor.CloneFrom(output_descriptor_in);
  std::unique_ptr<TemporaryDeviceMemory<T>> transform_scratch;
  backward_output_data = MaybeTransformLayout(
      stream, static_cast<miopenDataType_t>(miopen_type),
      &output_descriptor, backward_output_data,
      &transform_scratch);

  ScopedTensorDescriptor out_back_nd{parent_, output_descriptor,
        static_cast<miopenDataType_t>(miopen_type)};
  ScopedTensorDescriptor input_nd{parent_, input_descriptor,
          static_cast<miopenDataType_t>(miopen_type)};
  ScopedFilterDescriptor filter{parent_, filter_descriptor, input_descriptor,
        static_cast<miopenDataType_t>(miopen_type)};
  // TODO(sesse): Figure out under what circumstances cuDNN would
  // accept miopenHalf here; probably related to compute capability
  // and cuDNN version; at least cuDNN 4 on TITAN X only supports
  // miopenFloat even for half input.
  ScopedConvolutionDescriptor conv{parent_, convolution_descriptor,
      miopenFloat};

  const bool is_profiling = output_profile_result != nullptr;
  std::pair<miopenConvBwdWeightsAlgorithm_t, size_t> algo_sz;
  DeviceMemory<uint8> scratch;

  if (algorithm_config.algorithm() == dnn::kDefaultAlgorithm) {
    // With the default algorithm, use Cudnn's heuristics.
    auto get_algorithm = [&]() SHARED_LOCKS_REQUIRED(
        dnn_handle_mutex_) -> std::pair<miopenConvBwdWeightsAlgorithm_t, size_t> {

      assert (scratch_allocator);

      size_t size_in_bytes;
      status = dynload::miopenConvolutionBackwardWeightsGetWorkSpaceSize(
          parent_, ToHandle(dnn_handle_), /*diffDesc=*/out_back_nd.handle(),
          /*srcDesc=*/input_nd.handle() ,/*convDesc=*/conv.handle(),
          /*gradDesc=*/filter.handle(), /*sizeInBytes=*/&size_in_bytes);
      if (status == miopenStatusSuccess && size_in_bytes != 0) {
        auto allocated =
            scratch_allocator->AllocateBytes(stream, size_in_bytes);
        if (allocated.ok()) {
          scratch = allocated.ValueOrDie();
        }
      }

      miopenConvAlgoPerf_t preference;
      int returnedAlgoCount;

      miopenStatus_t status =
          dynload::miopenFindConvolutionBackwardWeightsAlgorithm(
              parent_, ToHandle(dnn_handle_),
              /*diffDesc=*/out_back_nd.handle(), backward_output_data.opaque(),
              /*srcDesc=*/input_nd.handle(), input_data.opaque(),
              /*convDesc=*/conv.handle(),
              /*gradDesc=*/filter.handle(), backward_filter_data->opaque(),
              /*requestAlgoCount=*/1, /*returnedAlgoCount=*/&returnedAlgoCount,
              /*preference=*/&preference, /*WorkSpace=*/scratch.opaque(),
              /*WorkSpaceSize=*/scratch.size(), /*exhaustiveSearch=*/false);
      CHECK_EQ(status, miopenStatusSuccess) << "Unable to find a suitable "
                                                "algorithm for doing backward "
                                                "filter convolution";
      return std::pair<miopenConvBwdWeightsAlgorithm_t , size_t> (preference.bwd_weights_algo, preference.memory);
    };

    algo_sz = get_algorithm();

    // MIOpen requires workspace:
    assert (scratch != nullptr) ;

  } else {
    // An algorithm has been specified.
    algo_sz.first = ToConvBackwardFilterAlgo(algorithm_config.algorithm());
    algo_sz.second = algorithm_config.algorithm_scratch_size();

    size_t size_in_bytes = algo_sz.second;

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
        algo_sz.first =
            ToConvBackwardFilterAlgo(algorithm_config.algorithm_no_scratch());
        algo_sz.second = 0;
      }
    }
  }

  std::unique_ptr<CUDATimer> timer;
  if (is_profiling) {
    timer.reset(new CUDATimer(parent_));
    timer->Init();
    // The start and stop of the timer should be as close to the Cudnn call as
    // possible. It is still possible for other threads to issue workload on
    // to this stream. So it could take multiple profiling measurements.
    timer->Start(AsCUDAStream(stream));
  }

  status = dynload::miopenConvolutionBackwardWeights(
      parent_, ToHandle(dnn_handle_), /*alpha=*/&alpha,
      /*diffDesc=*/out_back_nd.handle(),
      /*diffData=*/backward_output_data.opaque(),
      /*srcDesc=*/input_nd.handle(),
      /*srcData=*/input_data.opaque(),
      /*convDesc=*/conv.handle(),
      /*algo=*/algo_sz.first,
      /*beta=*/&beta,
      /*gradDesc=*/filter.handle(),
      /*gradData=*/backward_filter_data->opaque(),
      /*workSpace=*/scratch.opaque(),
      /*workSpaceSizeInBytes=*/scratch.size());
  if (is_profiling) {
    timer->Stop(AsCUDAStream(stream));
    output_profile_result->set_is_valid(true);
    output_profile_result->set_algorithm(algo_sz.first);
    output_profile_result->set_scratch_size(algo_sz.second);
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
    timer->Destroy();
  }
  if (status != miopenStatusSuccess) {
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
      stream, miopenFloat, input_descriptor, input_data,
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
  LOG(ERROR) << "miopen does not support half type";
  return DoConvolveBackwardFilterImpl(
      stream, miopenHalf, input_descriptor, input_data,
      output_descriptor_in, backward_output_data, convolution_descriptor,
      filter_descriptor, backward_filter_data, scratch_allocator,
      algorithm_config, output_profile_result);
}

template <class T>
bool CudnnSupport::DoConvolveBackwardBiasImpl(
    Stream* stream, int miopen_type,  // Actually miopenDataType_t.
    const dnn::BatchDescriptor& input_descriptor,
    const DeviceMemory<T>& input_data,
    const dnn::BatchDescriptor& bias_descriptor,
    DeviceMemory<T>* backward_bias_data) {
  mutex_lock lock{dnn_handle_mutex_};
  auto status = dynload::miopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != miopenStatusSuccess) {
    LOG(FATAL) << "failed to set stream for miopen handle: " << ToString(status);
  }

  ScopedTensorDescriptor input_nd{parent_, input_descriptor,
                                  static_cast<miopenDataType_t>(miopen_type)};
  ScopedTensorDescriptor bias_nd{parent_, bias_descriptor,
                                 static_cast<miopenDataType_t>(miopen_type)};

  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  status = dynload::miopenConvolutionBackwardBias(
      parent_, ToHandle(dnn_handle_), &alpha, input_nd.handle(),
      input_data.opaque(), &beta, bias_nd.handle(),
      backward_bias_data->opaque());
  if (status != miopenStatusSuccess) {
    LOG(FATAL) << "failed to enqueue backward convolution on stream: "
               << ToString(status);
    return false;
  }
  return true;
}

bool CudnnSupport::DoConvolveBackwardBias(
    Stream* stream, const BatchDescriptor& input_descriptor,
    const DeviceMemory<double>& input_data,
    const BatchDescriptor& bias_descriptor,
    DeviceMemory<double>* backward_bias_data) {
  return false;/*DoConvolveBackwardBiasImpl(stream, MIOPEN_DATA_DOUBLE, input_descriptor,
                                    input_data, bias_descriptor,
                                    backward_bias_data);*/
}

bool CudnnSupport::DoConvolveBackwardBias(
    Stream* stream, const BatchDescriptor& input_descriptor,
    const DeviceMemory<float>& input_data,
    const BatchDescriptor& bias_descriptor,
    DeviceMemory<float>* backward_bias_data) {
  return DoConvolveBackwardBiasImpl(stream, miopenFloat, input_descriptor,
                                    input_data, bias_descriptor,
                                    backward_bias_data);
}

bool CudnnSupport::DoConvolveBackwardBias(
    Stream* stream, const BatchDescriptor& input_descriptor,
    const DeviceMemory<Eigen::half>& input_data,
    const BatchDescriptor& bias_descriptor,
    DeviceMemory<Eigen::half>* backward_bias_data) {
  LOG(ERROR) << "miopen does not support half type";
  return DoConvolveBackwardBiasImpl(stream, miopenHalf, input_descriptor,
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
                                          miopenFloat};

  BatchDescriptor bias_dimensions;
  bias_dimensions.set_count(1)
      .set_feature_map_count(dimensions.feature_map_count())
      .set_height(1)
      .set_width(1)
      .set_layout(dnn::DataLayout::kBatchYXDepth);
  ScopedTensorDescriptor bias_descriptor{parent_, bias_dimensions,
                                         miopenFloat};

  // miopenAddTensor after R3 is in-place, so we need to copy input_data to
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
  auto status = dynload::miopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to set stream for miopen handle: " << ToString(status);
    return false;
  }

  const float alpha1 = 1.0f;
  const float alpha2 = 0.0f;
  const float beta = 1.0f;

  status = dynload::miopenOpTensor(
      parent_, ToHandle(dnn_handle_), miopenTensorOpAdd, &alpha1, bias_descriptor.handle(),
      biases.opaque(), &alpha2, bias_descriptor.handle(), biases.opaque(), 
      &beta, input_descriptor.handle(), output_data->opaque());

  if (status != miopenStatusSuccess) {
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
  mutex_lock lock{dnn_handle_mutex_};
  auto status = dynload::miopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to set stream for miopen handle: " << ToString(status);
    return false;
  }

#if 0// MIOPEN_VERSION >= 5000
  ScopedActivationDescriptor activation_desc{parent_, activation_mode,
                                             dimensions.value_max()};
#if 0//#else
  miopenActivationMode_t mode;
  switch (activation_mode) {
    case dnn::ActivationMode::kRelu6:
      // TODO(leary) should probably do a post-pass to clip at 6?
      LOG(WARNING) << "user requested Relu6, but providing Relu instead";
      mode = miopenActivationRELU;
      break;
    case dnn::ActivationMode::kReluX:
      // TODO(broune) should probably do a post-pass to clip at X?
      LOG(WARNING) << "user requested ReluX, but providing Relu instead";
      mode = miopenActivationRELU;
      break;
    case dnn::ActivationMode::kRelu:
      mode = miopenActivationRELU;
      break;
    case dnn::ActivationMode::kSigmoid:
      mode = miopenActivationLOGISTIC;
      break;
    case dnn::ActivationMode::kTanh:
      mode = miopenActivationTANH;
      break;
    default:
      LOG(ERROR) << "unrecognized activation mode: "
                 << static_cast<int>(activation_mode);
      return false;
  }
#endif

  ScopedTensorDescriptor input_nd{parent_, dimensions, miopenFloat};
  // Alpha is the input scaling factor.
  float alpha = 1.0;
  // Beta is the output scaling factor.
  float beta = 0.0;
  status = dynload::miopenActivationForward(
      parent_, ToHandle(dnn_handle_),
      activation_desc.handle(),
      &alpha, input_nd.handle(), input_data.opaque(), &beta, input_nd.handle(),
      output_data->opaque());
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "stream " << stream
               << " could not enqueue activation: " << ToString(status);
    return false;
  }
#endif
  return true;
}

bool CudnnSupport::DoPoolForward(
    Stream* stream, const dnn::PoolingDescriptor& pooling_dimensions,
    const dnn::BatchDescriptor& input_dimensions,
    const DeviceMemory<float>& input_data,
    const dnn::BatchDescriptor& output_dimensions,
    DeviceMemory<float>* output_data,
    ScratchAllocator* workspace_allocator) {
    
  mutex_lock lock{dnn_handle_mutex_};
  auto status = dynload::miopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to set stream for miopen handle: " << ToString(status);
    return false;
  }

  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  ScopedTensorDescriptor src_desc{parent_, input_dimensions, miopenFloat};
  ScopedTensorDescriptor dest_desc{parent_, output_dimensions,
                                   miopenFloat};
  ScopedPoolingDescriptor pooling_desc{parent_, pooling_dimensions};


  DeviceMemory<uint8> workspace;
  size_t workspace_size_in_bytes = 0;
  status = dynload::miopenPoolingGetWorkSpaceSize(parent_, dest_desc.handle(),
                                                  &workspace_size_in_bytes);

  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to obtain workspace size for backward pooling on stream: "
               << ToString(status);
    return false;
  }

  // Allocate the workspace.
  if (workspace_size_in_bytes > 0) {
    assert(workspace_allocator);
    auto allocated =
        workspace_allocator->AllocateBytes(stream, workspace_size_in_bytes);
    if (!allocated.ok() || (workspace = allocated.ValueOrDie()) == nullptr) {
      LOG(ERROR) << "Failed to allocate backward pooling workspace";
      return false;
    }
  }

  status = dynload::miopenPoolingForward(
      parent_, ToHandle(dnn_handle_), pooling_desc.handle(), &alpha,
      src_desc.handle(), input_data.opaque(), &beta, dest_desc.handle(),
      output_data->opaque(), true, workspace.opaque(), workspace_size_in_bytes);
  if (status != miopenStatusSuccess) {
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
    DeviceMemory<Eigen::half>* output_data,
    ScratchAllocator* workspace_allocator) {
  mutex_lock lock{dnn_handle_mutex_};
  auto status = dynload::miopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to set stream for miopen handle: " << ToString(status);
    return false;
  }
  LOG(ERROR) << "miopen does not support half type";

  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  ScopedTensorDescriptor src_desc{parent_, input_dimensions, miopenHalf};
  ScopedTensorDescriptor dest_desc{parent_, output_dimensions, miopenHalf};
  ScopedPoolingDescriptor pooling_desc{parent_, pooling_dimensions};

  DeviceMemory<uint8> workspace;
  size_t workspace_size_in_bytes = 0;
  status = dynload::miopenPoolingGetWorkSpaceSize(parent_, dest_desc.handle(),
                                                  &workspace_size_in_bytes);

  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to obtain workspace size for backward pooling on stream: "
               << ToString(status);
    return false;
  }

  // Allocate the workspace.
  if (workspace_size_in_bytes > 0) {
    assert(workspace_allocator);
    auto allocated =
        workspace_allocator->AllocateBytes(stream, workspace_size_in_bytes);
    if (!allocated.ok() || (workspace = allocated.ValueOrDie()) == nullptr) {
      LOG(ERROR) << "Failed to allocate backward pooling workspace";
      return false;
    }
  }


  status = dynload::miopenPoolingForward(
      parent_, ToHandle(dnn_handle_), pooling_desc.handle(), &alpha,
      src_desc.handle(), input_data.opaque(), &beta, dest_desc.handle(),
      output_data->opaque(), true, workspace.opaque(), workspace_size_in_bytes);
  if (status != miopenStatusSuccess) {
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
    DeviceMemory<float>* output_diff_data,
    ScratchAllocator* workspace_allocator) {
  mutex_lock lock{dnn_handle_mutex_};
  auto status = dynload::miopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to set stream for miopen handle: " << ToString(status);
    return false;
  }

  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  ScopedTensorDescriptor src_desc{parent_, input_dimensions, miopenFloat};
  ScopedTensorDescriptor dest_desc{parent_, output_dimensions,
                                   miopenFloat};
  ScopedPoolingDescriptor pooling_desc{parent_, pooling_dimensions};

  DeviceMemory<uint8> workspace;
  size_t workspace_size_in_bytes = 0;
  status = dynload::miopenPoolingGetWorkSpaceSize(parent_, dest_desc.handle(),
                                                  &workspace_size_in_bytes);

  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to obtain workspace size for backward pooling on stream: "
               << ToString(status);
    return false;
  }

  // Allocate the workspace.
  if (workspace_size_in_bytes > 0) {
    assert(workspace_allocator);
    auto allocated =
        workspace_allocator->AllocateBytes(stream, workspace_size_in_bytes);
    if (!allocated.ok() || (workspace = allocated.ValueOrDie()) == nullptr) {
      LOG(ERROR) << "Failed to allocate backward pooling workspace";
      return false;
    }
  }

  DeviceMemory<uint8> dest2; // duplicated dest from forward:
  int dest2_size = 0;

  // miopen requires the strides and dims to be ordered as BDYX.
  std::vector<int64> dims64 =
      output_dimensions.full_dims(dnn::DataLayout::kBatchDepthYX);

  // miopen does not use strides and must have 4D tensor.
  std::vector<int> dims(4);

  std::transform(dims64.cbegin(), dims64.cend(), dims.begin(),
                 &CheckedNarrowing<int64, int>);

  dest2_size = dims[0] * dims[1] * dims[2] * dims[3] * sizeof(float);

  if (dest2_size > 0) {
    assert(workspace_allocator);
    auto allocated = workspace_allocator->AllocateBytes(stream, dest2_size);
    if (!allocated.ok() || (dest2 = allocated.ValueOrDie()) == nullptr) {
      LOG(ERROR) << "Failed to allocate backward pooling workspace";
      return false;
    }
  } else {
    LOG(ERROR) << "Failed to calcuate tensor size to chain forward and backward pooling";
  }

  status = dynload::miopenPoolingForward(
      parent_, ToHandle(dnn_handle_), pooling_desc.handle(), &alpha,
      src_desc.handle(), input_data.opaque(), &beta, dest_desc.handle(),
      dest2.opaque(), true, workspace.opaque(), workspace_size_in_bytes);

  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to enqueue forward pooling (before backward) on stream: "
               << ToString(status);
    return false;
  }

  status = dynload::miopenPoolingBackward(
      parent_, ToHandle(dnn_handle_), pooling_desc.handle(), &alpha,
      dest_desc.handle(), dest2.opaque(), dest_desc.handle(),
      input_diff_data.opaque(), src_desc.handle(), input_data.opaque(), &beta,
      src_desc.handle(), output_diff_data->opaque(), workspace.opaque());

  if (status != miopenStatusSuccess) {
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
    DeviceMemory<Eigen::half>* output_diff_data,
    ScratchAllocator* workspace_allocator) {
  mutex_lock lock{dnn_handle_mutex_};
  auto status = dynload::miopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to set stream for miopen handle: " << ToString(status);
    return false;
  }

  // Alpha is the scaling factor for input.
  float alpha = 1.0;
  // Beta is the scaling factor for output.
  float beta = 0.0;

  LOG(ERROR) << "miopen does not support half type";
  ScopedTensorDescriptor src_desc{parent_, input_dimensions, miopenHalf};
  ScopedTensorDescriptor dest_desc{parent_, output_dimensions, miopenHalf};
  ScopedPoolingDescriptor pooling_desc{parent_, pooling_dimensions};

  DeviceMemory<uint8> workspace;
  size_t workspace_size_in_bytes = 0;
  status = dynload::miopenPoolingGetWorkSpaceSize(parent_, dest_desc.handle(),
                                                  &workspace_size_in_bytes);

  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to obtain workspace size for backward pooling on stream: "
               << ToString(status);
    return false;
  }

  // Allocate the workspace.
  if (workspace_size_in_bytes > 0) {
    assert(workspace_allocator);
    auto allocated =
        workspace_allocator->AllocateBytes(stream, workspace_size_in_bytes);
    if (!allocated.ok() || (workspace = allocated.ValueOrDie()) == nullptr) {
      LOG(ERROR) << "Failed to allocate backward pooling workspace";
      return false;
    }
  }

  status = dynload::miopenPoolingBackward(
      parent_, ToHandle(dnn_handle_), pooling_desc.handle(), &alpha,
      dest_desc.handle(), output_data.opaque(), dest_desc.handle(),
      input_diff_data.opaque(), src_desc.handle(), input_data.opaque(), &beta,
      src_desc.handle(), output_diff_data->opaque(), workspace.opaque());

  if (status != miopenStatusSuccess) {
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
  // TODO - should this be miopenLRNWithinChannel?  Would need to reset miopenLRN handle here.
  return false;
}

bool CudnnSupport::DoNormalizeWithDimensions(
    Stream* stream, const dnn::NormalizeDescriptor& normalize_descriptor,
    const dnn::BatchDescriptor& dimensions,
    const DeviceMemory<float>& input_data, DeviceMemory<float>* output_data) {
  // Check for unsupported modes.
  if (normalize_descriptor.wrap_around()) {
    LOG(ERROR) << "miopen LRN does not support wrap-around mode";
    return false;
  }
  if (normalize_descriptor.segment_size()) {
    LOG(ERROR) << "miopen LRN does not support segmentation";
    return false;
  }

  // Launch the normalization.
  mutex_lock lock{dnn_handle_mutex_};
  auto status = dynload::miopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to set stream for miopen handle: " << ToString(status);
    return false;
  }

  ScopedTensorDescriptor dims{parent_, dimensions, miopenFloat};
  ScopedNormalizeDescriptor normalize{parent_, normalize_descriptor};

  // Alpha is the scaling factor for input.
  float alpha = 1.0f;
  // Beta is the scaling factor for output.
  float beta = 0.0f;

  DeviceMemory<uint8> workspace;
  size_t workspace_size_in_bytes = 0;
  status = dynload::miopenLRNGetWorkSpaceSize(parent_, dims.handle(),
                                              &workspace_size_in_bytes);


  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to obtain workspace size for miopenLRNBackward";
    return false;
  }

#if 0 //  enable when we pipe through the workspace allocator
  // Allocate the workspace.
  if (workspace_size_in_bytes > 0) {
    assert(workspace_allocator);
    auto allocated =
        workspace_allocator->AllocateBytes(stream, workspace_size_in_bytes);
    if (!allocated.ok() || (workspace = allocated.ValueOrDie()) == nullptr) {
      LOG(ERROR) << "Failed to allocate backward pooling workspace";
      return false;
    }
  }
#endif


  status = dynload::miopenLRNForward(
      parent_, ToHandle(dnn_handle_), normalize.handle(),
      &alpha, dims.handle(), input_data.opaque(),
      &beta, dims.handle(), output_data->opaque(), false, nullptr); // TODO - enable
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to run miopenLRNForward";
    return false;
  }
  return true;
}

bool CudnnSupport::DoNormalizeBackwardWithDimensions(
    Stream* stream, const dnn::NormalizeDescriptor& normalize_descriptor,
    const dnn::BatchDescriptor& dimensions, const DeviceMemory<float>& raw_data,
    const DeviceMemory<float>& normalized_data,
    const DeviceMemory<float>& normalized_variable_gradient,
    DeviceMemory<float>* raw_variable_gradient,
    ScratchAllocator* workspace_allocator) {
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
  auto status = dynload::miopenSetStream(parent_, ToHandle(dnn_handle_),
                                        AsCUDAStreamValue(stream));
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to set stream for miopen handle: " << ToString(status);
    return false;
  }

  ScopedTensorDescriptor dims{parent_, dimensions, miopenFloat};
  ScopedNormalizeDescriptor normalize{parent_, normalize_descriptor};

  float alpha = 1.0f;
  float beta = 0.0f;

  DeviceMemory<uint8> workspace;
  size_t workspace_size_in_bytes = 0;
  status = dynload::miopenLRNGetWorkSpaceSize(parent_, dims.handle(),
                                              &workspace_size_in_bytes);

  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to obtain workspace size for miopenLRNBackward";
    return false;
  }

  // Allocate the workspace.
  if (workspace_size_in_bytes > 0) {
    assert(workspace_allocator);
    auto allocated =
        workspace_allocator->AllocateBytes(stream, workspace_size_in_bytes);
    if (!allocated.ok() || (workspace = allocated.ValueOrDie()) == nullptr) {
      LOG(ERROR) << "Failed to allocate backward pooling workspace";
      return false;
    }
  }

  status = dynload::miopenLRNBackward(
      parent_, ToHandle(dnn_handle_), normalize.handle(),
      &alpha, dims.handle(),
      normalized_data.opaque(), dims.handle(),
      normalized_variable_gradient.opaque(), dims.handle(), raw_data.opaque(),
      &beta, dims.handle(), raw_variable_gradient->opaque(), workspace.opaque());

  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "failed to run miopenLRNBackward";
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
  LOG(ERROR) << "quantized memcpy not supported by cuDNN";
  return false;
}

bool CudnnSupport::DoMemcpyH2DQuantized(
    Stream* stream, const void* host_src, int64 size,
    dnn::QuantizedActivationMode mode,
    DeviceMemory<float>* gpu_unquantized_dst) {
  LOG(ERROR) << "quantized memcpy not supported by cuDNN";
  return false;
}

bool CudnnSupport::DeriveOutputBatchDescriptor(
    const BatchDescriptor& batch_descriptor,
    const FilterDescriptor& filter_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor,
    dnn::BatchDescriptor* output_batch_descriptor) {
  ScopedTensorDescriptor input_nd{parent_, batch_descriptor, miopenFloat};
  ScopedFilterDescriptor filter{parent_, filter_descriptor, batch_descriptor,
                                miopenFloat};
  ScopedConvolutionDescriptor conv{parent_, convolution_descriptor,
                                   miopenFloat};

  int dn = batch_descriptor.ndims() + 2;
  std::vector<int> dims(dn);  // in BDYX
  auto status = dynload::miopenGetConvolutionForwardOutputDim(
      parent_, conv.handle(), input_nd.handle(), filter.handle(),
      &dims[0], &dims[1], &dims[2], &dims[3]);
  if (status != miopenStatusSuccess) {
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

namespace gpu = ::perftools::gputools;

void initialize_miopen() {
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

}  // namespace gputools
}  // namespace perftools

REGISTER_MODULE_INITIALIZER(register_miopen,
                            { perftools::gputools::initialize_miopen(); });
#endif
