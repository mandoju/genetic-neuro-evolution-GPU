scl run devtoolset-7 bash
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )

nvcc  -std=c++11 -c -o mutation_op.o  mutation_op.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG -expt-relaxed-constexpr -w
g++ -std=c++11 -shared -o mutation_op.so  mutation_op.cc   mutation_op.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}