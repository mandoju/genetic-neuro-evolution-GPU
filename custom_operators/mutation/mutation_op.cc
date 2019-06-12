#include "mutation_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"


using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


REGISTER_OP("Mutation")
    .Input("input: float32")
    .Output("mutated: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });


// CPU specialization of actual computation.
template <typename T>
struct MutationFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, const T* in, T* out) {
     // Set all but the first element of the output tensor to 0.
    std::default_random_engine generator;
    std::default_random_engine generator_biased;

    std::uniform_real_distribution<double> distribution(0.0,1.0);
    std::uniform_real_distribution<double> distribution_biased(0.9,1.1);

    for (int i = 0; i < size; i++) {
      out[i] = in[i];
      if (distribution(generator) < 0.01) {
        out[i] *= distribution_biased(generator_biased);
      }
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class MutationOp : public OpKernel {
 public:
  explicit MutationOp(OpKernelConstruction* context) : OpKernel(context) 
  {
    
  }

  void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& input_tensor = context->input(0);

      // Create an output tensor
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                      &output_tensor));

      // Do the computation.
      OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                  errors::InvalidArgument("Too many elements in tensor"));
      MutationFunctor<Device, T>()(
          context->eigen_device<Device>(),
          static_cast<int>(input_tensor.NumElements()),
          input_tensor.flat<T>().data(),
          output_tensor->flat<T>().data());
  }
};


// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Mutation").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      MutationOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);


// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template MutationFunctor<GPUDevice, T>;                  \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Mutation").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      MutationOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA















