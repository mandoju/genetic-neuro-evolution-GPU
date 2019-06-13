#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/util/cuda_launch_config.h"

__global__ void MutationKernel(const float* in, const int N, float* out) {
  std::default_random_engine generator;
  std::default_random_engine generator_biased;
  std::uniform_real_distribution<double> distribution(0.0,1.0);
  std::uniform_real_distribution<double> distribution_biased(0.9,1.1);
  
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    out[i] = in[i];
    if (distribution(generator) < 0.01) {
        out[i] *= distribution_biased(generator_biased);
    }
  }
}

void MutationKernelLauncher(const float* in, const int N, float* out) {
  TF_CHECK_OK(::tensorflow::CudaLaunchKernel(MutationKernel, 32, 256, 0, nullptr,
                                             in, N, out));
}

#endif