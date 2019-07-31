import tensorflow as tf
import numpy as np


# Todos aleatorios
def generate_child_by_all(mother_tensor, father_tensor):
    with tf.name_scope('Passagem_Genes'):
        temp_neural_network = []

        shape_size = tf.shape(mother_tensor)

        # Criação do array binário para definir quais são os genes que irão receber a mistura do  mãe
        random_array_binary = tf.random_uniform(
            dtype=tf.float32, minval=0, maxval=1, shape=shape_size)
        random_array_select = tf.random_uniform(
            dtype=tf.float32, minval=0, maxval=1, shape=shape_size)
        random_array_select = tf.math.round(random_array_select)

        # Criando o array inverso para definir o número ao contrário para criar a quantidade recebida pelo pai
        # random_array_inverse = tf.map_fn(lambda x: (1 - x), random_array_binary, dtype=tf.float32)
        random_array_inverse = tf.scalar_mul(
            -1, random_array_binary) + tf.ones_like(random_array_binary)

        crossoved = tf.multiply(father_tensor, random_array_binary) + \
            tf.multiply(mother_tensor, random_array_inverse)

        return crossoved


def generate_child_by_two_points(mother_tensor, father_tensor):
    with tf.name_scope('Passagem_Genes'):

        father_flat = tf.reshape(father_tensor, [-1])
        mother_flat = tf.reshape(mother_tensor, [-1])
        tamanho_tensor = tf.size(father_flat)
        indice_1 = tf.random_uniform(
            [], minval=0, maxval=tamanho_tensor-1, dtype=tf.int32)
        indice_2 = tf.random_uniform(
            [], minval=indice_1+1, maxval=tamanho_tensor, dtype=tf.int32)
        child = tf.concat([mother_flat[0:indice_1], father_flat[indice_1:indice_2],
                           mother_flat[indice_2:tamanho_tensor]], axis=0)
        child = tf.reshape(filho, tf.shape(pai))

    return child


def generate_child_by_division(mother_tensor, father_tensor):
    with tf.name_scope('Passagem_Genes'):
        temp_neural_network = []

        shape_size = tf.shape(mother_tensor)

        # Criação do array binário para definir quais são os genes que irão receber a mistura do  mãe
        random_array_binary = tf.random_uniform(
            dtype=tf.float32, minval=0, maxval=1, shape=shape_size)
        random_array_select = tf.random_uniform(
            dtype=tf.float32, minval=0, maxval=1, shape=shape_size)
        random_array_select = tf.math.round(random_array_select)

        # Criando o array inverso para definir o número ao contrário para criar a quantidade recebida pelo pai
        # random_array_inverse = tf.map_fn(lambda x: (1 - x), random_array_binary, dtype=tf.float32)
        random_array_inverse = tf.scalar_mul(
            -1, random_array_binary) + tf.ones_like(random_array_binary)

        crossoved = tf.multiply(father_tensor, random_array_binary) + \
            tf.multiply(mother_tensor, random_array_inverse)

        return crossoved


def generate_child_by_mixed(mother_tensor, father_tensor, mutationRate):
    with tf.name_scope('Passagem_Genes'):
        temp_neural_network = []

        shape_size = tf.shape(mother_tensor)

        # Criação do array binário para definir quais são os genes que irão receber a mistura do  mãe
        random_array_binary = tf.random_uniform(
            dtype=tf.float32, minval=0, maxval=1, shape=shape_size)

        random_array_select = tf.random_uniform(
            dtype=tf.float32, minval=0, maxval=1, shape=[shape_size[0]])
        random_array_select = tf.math.round(random_array_select)

        random_array_binary = tf.multiply(
            random_array_select[:, tf.newaxis], random_array_binary)
        # Criando o array inverso para definir o número ao contrário para criar a quantidade recebida pelo pai
        # random_array_inverse = tf.map_fn(lambda x: (1 - x), random_array_binary, dtype=tf.float32)
        random_array_inverse = tf.scalar_mul(
            -1, random_array_binary) + tf.ones_like(random_array_binary)

        crossoved = tf.multiply(father_tensor, random_array_binary) + \
            tf.multiply(mother_tensor, random_array_inverse)

        return crossoved


# Apenas as layers
# def generate_child_by_layer(mother_tensor, father_tensor, mutationRate, layers):
#     with tf.name_scope('Passagem_Genes'):
#         temp_neural_network = []
#
#         shape_size = tf.shape(mother_tensor)
#
#         # Criação do array binário para definir quais são os genes que irão receber a mistura do  mãe
#         random_array_binary = tf.random_uniform(dtype=tf.float32, minval=0, maxval=1, shape=[shape_size[0]])
#         random_array_select = tf.random_uniform(dtype=tf.float32, minval=0, maxval=1, shape=[shape_size[0]])
#         random_array_select = tf.math.round(random_array_select)
#
#         # Criando o array inverso para definir o número ao contrário para criar a quantidade recebida pelo pai
#         # random_array_inverse = tf.map_fn(lambda x: (1 - x), random_array_binary, dtype=tf.float32)
#         random_array_inverse = 1 - random_array_binary
#         # Criação o array de taxa de mistura para ambos
#         # random_array_start = tf.cast(
#         #    tf.random_uniform(dtype=tf.int32, minval=0, maxval=1, shape=[shape_size[0]]), tf.float32)
#
#         for weight_idx_range in range(layers - 1):
#             weight_idx = weight_idx_range - 1
#             father_tensor_process = mother_tensor[weight_idx]
#             mother_tensor_process = father_tensor[weight_idx]
#
#             shape_size = tf.shape(mother_tensor[weight_idx])
#
#
#             crossoved = tf.multiply(father_tensor_process, random_array_binary[weight_idx]) + tf.multiply(
#                 mother_tensor_process, random_array_inverse[weight_idx])
#             temp_neural_network.append(mutation(crossoved, mutationRate))
#
#         return tf.stack(temp_neural_network)
