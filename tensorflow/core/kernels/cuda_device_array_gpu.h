/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Contains structs and functions to be included in device code.

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_CUDA_DEVICE_ARRAY_GPU_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_CUDA_DEVICE_ARRAY_GPU_H_

#if GOOGLE_CUDA

namespace tensorflow {

static constexpr int kMaxInlineCudaPointers = 8;
// To decode on the device side, use GetCudaDeviceArrayOnDevice.
// To encode on the host side, use CudaDeviceArrayOnHost.
template <typename ValueType, int MaxInlineValues = 8>
struct CudaDeviceArrayStruct {
  int32 size;
  // used if size <= MaxInlineValues;
  ValueType inline_values[MaxInlineValues];
  ValueType* out_of_line_values = nullptr;  // used if size > MaxInlineValues;

#ifdef __HCC__
  __attribute__((annotate("user_deserialize")))
  CudaDeviceArrayStruct(int32 vs,
                        ValueType v0, ValueType v1, ValueType v2, ValueType v3,
                        ValueType v4, ValueType v5, ValueType v6, ValueType v7,
                        ValueType* v8) [[cpu]][[hc]] {
    size = vs;
    inline_values[0] = v0;
    inline_values[1] = v1;
    inline_values[2] = v2;
    inline_values[3] = v3;
    inline_values[4] = v4;
    inline_values[5] = v5;
    inline_values[6] = v6;
    inline_values[7] = v7;
    out_of_line_values = v8;
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    s.Append(sizeof(int32), &size);
    for (int i = 0; i < MaxInlineValues; ++i) {
      s.Append(sizeof(ValueType), &inline_values[i]);
    }
    s.Append(sizeof(ValueType*), &out_of_line_values);
  }
#endif
};

template <typename ValueType, int MaxInlineValues = 8>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ValueType* GetCudaDeviceArrayOnDevice(
    CudaDeviceArrayStruct<ValueType, MaxInlineValues>* data) {
  if (data->size <= MaxInlineValues) {
    return data->inline_values;
  } else {
    return data->out_of_line_values;
  }
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_CUDA_DEVICE_ARRAY_GPU_H_
