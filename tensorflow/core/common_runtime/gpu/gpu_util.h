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

#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_GPU_UTIL_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_GPU_UTIL_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor.h"

#if GOOGLE_CUDA
#include <tuple>
#include <unordered_map>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/hash/hash.h"
#endif // GOOGLE_CUDA

namespace tensorflow {

#if GOOGLE_CUDA
template <typename T>
inline perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory,
                                                           uint64 size) {
  perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory),
                                                size * sizeof(T));
  perftools::gputools::DeviceMemory<T> typed(wrapped);
  return typed;
}

// Get the Cudnn workspace limit from the environment variable, which is in MB.
// Return the workspace memory limit in bytes. If no value is set, return the
// default value.
int64 GetCudnnWorkspaceLimit(const string& envvar_in_mb,
                             int64 default_value_in_bytes);

// A class to provide scratch-space allocator for Stream-Executor Cudnn
// callback. TensorFlow is responsible for releasing the temporary buffers after
// the kernel finishes.
class CudnnScratchAllocator : public perftools::gputools::ScratchAllocator {
 public:
  virtual ~CudnnScratchAllocator() {}
  CudnnScratchAllocator(int64 memory_limit, OpKernelContext* context)
      : memory_limit_(memory_limit), total_byte_size_(0), context_(context) {}
  virtual int64 GetMemoryLimitInBytes(
      perftools::gputools::Stream* stream) override {
    return memory_limit_;
  }
  virtual perftools::gputools::port::StatusOr<
      perftools::gputools::DeviceMemory<uint8>>
  AllocateBytes(perftools::gputools::Stream* stream, int64 byte_size) override {
    Tensor temporary_memory;
    if (byte_size > memory_limit_) {
      return perftools::gputools::port::StatusOr<
          perftools::gputools::DeviceMemory<uint8>>();
    }
    AllocationAttributes allocation_attr;
    allocation_attr.no_retry_on_failure = true;
    Status allocation_status(context_->allocate_temp(
        DT_UINT8, TensorShape({byte_size}), &temporary_memory,
        AllocatorAttributes(), allocation_attr));
    if (!allocation_status.ok()) {
      return perftools::gputools::port::StatusOr<
          perftools::gputools::DeviceMemory<uint8>>();
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    total_byte_size_ += byte_size;
    return perftools::gputools::port::StatusOr<
        perftools::gputools::DeviceMemory<uint8>>(
        AsDeviceMemory(temporary_memory.flat<uint8>().data(),
                       temporary_memory.flat<uint8>().size()));
  }
  int64 TotalByteSize() { return total_byte_size_; }

 private:
  int64 memory_limit_;
  int64 total_byte_size_;
  OpKernelContext* context_;
  std::vector<Tensor> allocated_tensors_;
};
#endif // GOOGLE_CUDA


class RecvTensorResponse;
class TensorProto;

namespace gpu = ::perftools::gputools;

class GPUUtil {
 public:
  // "tensor" is GPU-local.  "dev" is the hosting GPU.
  // "device_context" should be the context of the GPU "_Send" op
  // which provides the Tensor.
  // Sets all necessary fields of "proto" by transferring value
  // bytes from GPU to CPU RAM. "is_dead" indicates that the
  // tensor is dead with an uninit value.
  static void SetProtoFromGPU(const Tensor& tensor, Device* dev,
                              const DeviceContext* device_context,
                              TensorProto* proto, bool is_dead,
                              StatusCallback done);

  // Copies the data in 'gpu_tensor' into 'cpu_tensor'.
  // 'gpu_tensor''s backing memory must be on 'gpu_device' and
  // 'cpu_tensor' must be allocated to be of the same size as
  // 'gpu_tensor'. Synchronous: may block.
  static void CopyGPUTensorToCPU(Device* gpu_device,
                                 const DeviceContext* device_context,
                                 const Tensor* gpu_tensor, Tensor* cpu_tensor,
                                 StatusCallback done);

  // Blocks until all operations queued on the stream associated with
  // "gpu_device" at the time of the call have completed.  Returns any
  // error pending on the stream at completion.
  static Status Sync(Device* gpu_device);

  // Blocks until all operations queued on all streams associated with the
  // corresponding GPU device at the time of call have completed.
  // Returns any error pending on the stream at completion.
  static Status SyncAll(Device* gpu_device);

  // For debugging purpose, given a "device" and a "tensor" allocated
  // on the device, return a string printing each byte in the tensor
  // (up to a limit).  "device" can be either a CPU or a GPU device.
  static string MemoryDebugString(const Device* device, Tensor* tensor);

  // Map a Tensor as a DeviceMemory object wrapping the given typed
  // buffer.
  //
  // NOTE: will be removed soon, see StreamExecutorUtil::AsDeviceMemory
  // instead.
  template <typename T>
  static perftools::gputools::DeviceMemory<T> AsDeviceMemory(const Tensor& t) {
    T* ptr = reinterpret_cast<T*>(const_cast<void*>(DMAHelper::base(&t)));
    return perftools::gputools::DeviceMemory<T>(
        perftools::gputools::DeviceMemoryBase(ptr, t.TotalBytes()));
  }

  // Computes a checksum over the contents of "tensor", which is allocated
  // on "gpu_device".
  static uint64 Checksum(Device* gpu_device,
                         const DeviceContext* device_context,
                         const Tensor& tensor);

  // Computes a checksum over the contents of "tensor", which is allocated
  // in local CPU RAM.
  static uint64 Checksum(const Tensor& tensor);

  static void CopyCPUTensorToGPU(const Tensor* cpu_tensor,
                                 const DeviceContext* device_context,
                                 Device* gpu_device, Tensor* gpu_tensor,
                                 StatusCallback done);

  static void DeviceToDeviceCopy(DeviceContext* send_dev_context,
                                 DeviceContext* recv_dev_context, Device* src,
                                 Device* dst,
                                 AllocatorAttributes src_alloc_attr,
                                 AllocatorAttributes dst_alloc_attr,
                                 const Tensor* input, Tensor* output,
                                 StatusCallback done);

  // Deep-copying of GPU tensor on the same device.
  // 'src_gpu_tensor''s and 'dst_gpu_tensor''s backing memory must be on
  // 'gpu_device' and 'dst_cpu_tensor' must be allocated to be of the same
  // size as 'src_gpu_tensor'.
  static void CopyGPUTensorToSameGPU(Device* gpu_device,
                                     const DeviceContext* device_context,
                                     const Tensor* src_gpu_tensor,
                                     Tensor* dst_gpu_tensor,
                                     StatusCallback done);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_GPU_UTIL_H_
