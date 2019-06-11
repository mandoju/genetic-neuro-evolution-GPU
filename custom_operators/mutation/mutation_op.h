// kernel_example.h
#ifndef KERNEL_MUTATION_H_
#define KERNEL_MUTATION_H_

template <typename Device, typename T>
struct MutationFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename Eigen::GpuDevice, typename T>
struct MutationFunctor {
  void operator()(const Eigen::GpuDevice& d, int size, const T* in, T* out);
};
#endif

#endif KERNEL_MUTATION_H_