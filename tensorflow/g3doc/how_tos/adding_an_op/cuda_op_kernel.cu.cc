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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

__global__ void AddOneKernel(hipLaunchParm lp, const int* in, const int N, int* out) {
  for (int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; i < N;
       i += hipBlockDim_x * hipGridDim_x) {
    out[i] = in[i] + 1;
  }
}

void AddOneKernelLauncher(const int* in, const int N, int* out) {
  hipLaunchKernel(HIP_KERNEL_NAME(AddOneKernel), dim3(32), dim3(256), 0, 0, in, N, out);
}

#endif
