#!/usr/bin/env bash
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
echo $TF_INC

CUDA_PATH=/usr/local/cuda/

cd mutation

nvcc -std=c++11 -c -o mutation_op.cu.o mutation_op.cu.cc \
	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_61 -DNDEBUG --expt-relaxed-constexpr

g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o mutation_op.so mutation_op.cc \
	mutation_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64 -L $TF_LIB -ltensorflow_framework
