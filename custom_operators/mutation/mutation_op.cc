//#include "mutation_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"


using namespace tensorflow;

REGISTER_OP("Mutation")
    .Input("input: float")
    .Output("mutated: float");

void MutationKernelLauncher(const float* in, const int N, float* out);

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
class MutationOp : public OpKernel {
 public:
  explicit MutationOp(OpKernelConstruction* context) : OpKernel(context) 
  {
    
  }

  void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& input_tensor = context->input(0);
      auto input = input_tensor.flat<float>();

      // Create an output tensor
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                      &output_tensor));
      auto output = output_tensor->template flat<float>();
     
      // Set all but the first element of the output tensor to 0.
      const int N = input.size();

      MutationKernelLauncher(input.data(),N,output.data());
      // MutationFunctor<Device, T>()(
      //     context->eigen_device<Device>(),
      //     static_cast<int>(input_tensor.NumElements()),
      //     input_tensor.flat<T>().data(),
      //     output_tensor->flat<T>().data());
  }
};


REGISTER_KERNEL_BUILDER(Name("Mutation").Device(DEVICE_GPU), MutationOp);















