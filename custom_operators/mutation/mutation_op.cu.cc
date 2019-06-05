// kernel_example.cu.cc
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "mutation_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void MutationCudaKernel(const int size, const T* in, T* out) {
  
  std::uniform_real_distribution<double> distribution(0.0,1.0)
  std::uniform_real_distribution<double> distribution_biased(0.9,1.1);
  
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    out[i] = ldg(in + i);
    if (distribution(generator) < 0.01) {
        out[i] *= distribution_biased(generator_biased);
    }
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void MutationFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, int size, const T* in, T* out) {
  // Launch the cuda kernel.
  //
  // See core/util/gpu_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  int block_count = 1024;
  int thread_per_block = 20;
  MutationCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct MutationFunctor<GPUDevice, float>;
template struct MutationFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA