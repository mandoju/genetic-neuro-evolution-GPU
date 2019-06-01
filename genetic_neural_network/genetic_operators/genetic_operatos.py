import tensorflow as tf
import numpy as np

from genetic_neural_network.geneticLayer import GeneticLayer
from .mutation import mutation, mutation_unbiased
from typing import List
from itertools import chain
from collections import defaultdict


def select_operator_and_apply(genetic_operator, genetic_operator_param, genetic_operator_population_size, elite_size,
                              mutatioRate, genetic_layers):
    if (genetic_operator == 'crossover'):
        return crossover_operator(genetic_layers, elite_size, genetic_operator_population_size)
    elif (genetic_operator == 'mutation'):
        return mutation_operator(genetic_layers, elite_size, mutatioRate, genetic_operator_param,
                                 genetic_operator_population_size)
    elif (genetic_operator == 'mutation_unbiased'):
        return mutation_unbiased_operator(genetic_layers, elite_size, mutatioRate, genetic_operator_param,
                                          genetic_operator_population_size)


def apply_genetic_operatos(genetic_operators, genetic_operators_size, elite_size, genetic_layers: List[GeneticLayer],
                           mutationRate,
                           tournamentSize, layers):
    conv_operators_results = []
    bias_operators_results = []
    conv_result_dict = {}
    bias_result_dict = {}
    assigns_conv = []
    assigns_bias = []

    # TODO: Consertar esta parte para usar a classe genetic layer
    # conv_operators_results, bias_operators_results = zip(*[
    #     select_operator_and_apply(genetic_operator[0], genetic_operator[1],
    #                               tf.cast(genetic_operators_size[idx], dtype=tf.int32), elite_size, mutationRate,
    #                               genetic_layers) for idx, genetic_operator in
    #     enumerate(genetic_operators)])

    assigns_weights = []
    assigns_biases = []
    for genetic_layer in genetic_layers:
        for idx in range(len(genetic_operators)):
            print(idx)
            genetic_layer.add_operator(
                *select_operator_and_apply(genetic_operators[idx][0], genetic_operators[idx][1],
                                           tf.cast(genetic_operators_size[idx], dtype=tf.int32), elite_size,
                                           mutationRate, genetic_layer))
        # print(genetic_layer.best_weights)
        # print(genetic_layer.operators_weights)

        assigns_weights.append(tf.assign(genetic_layer.layer.weight,
                                         tf.concat([genetic_layer.best_weights] + genetic_layer.operators_weights,
                                                   axis=0)))
        assigns_biases.append(tf.assign(genetic_layer.layer.bias,
                                        tf.concat([genetic_layer.best_biases] + genetic_layer.operators_biases,
                                                  axis=0)))
    return assigns_weights, assigns_biases



##Todos aleatorios
def generate_child_by_all(mother_tensor, father_tensor):
    with tf.name_scope('Passagem_Genes'):
        temp_neural_network = []

        shape_size = tf.shape(mother_tensor)

        # Criação do array binário para definir quais são os genes que irão receber a mistura do  mãe
        random_array_binary = tf.random_uniform(dtype=tf.float32, minval=0, maxval=1, shape=shape_size)
        random_array_select = tf.random_uniform(dtype=tf.float32, minval=0, maxval=1, shape=shape_size)
        random_array_select = tf.math.round(random_array_select)

        # Criando o array inverso para definir o número ao contrário para criar a quantidade recebida pelo pai
        # random_array_inverse = tf.map_fn(lambda x: (1 - x), random_array_binary, dtype=tf.float32)
        random_array_inverse = tf.scalar_mul(-1, random_array_binary) + tf.ones_like(random_array_binary)

        crossoved = tf.multiply(father_tensor, random_array_binary) + tf.multiply(mother_tensor, random_array_inverse)


        return crossoved


def generate_child_by_mixed(mother_tensor, father_tensor, mutationRate):
    with tf.name_scope('Passagem_Genes'):
        temp_neural_network = []

        shape_size = tf.shape(mother_tensor)

        # Criação do array binário para definir quais são os genes que irão receber a mistura do  mãe
        random_array_binary = tf.random_uniform(dtype=tf.float32, minval=0, maxval=1, shape=shape_size)

        random_array_select = tf.random_uniform(dtype=tf.float32, minval=0, maxval=1, shape=[shape_size[0]])
        random_array_select = tf.math.round(random_array_select)

        random_array_binary = tf.multiply(random_array_select[:, tf.newaxis], random_array_binary)
        # Criando o array inverso para definir o número ao contrário para criar a quantidade recebida pelo pai
        # random_array_inverse = tf.map_fn(lambda x: (1 - x), random_array_binary, dtype=tf.float32)
        random_array_inverse = tf.scalar_mul(-1, random_array_binary) + tf.ones_like(random_array_binary)

        crossoved = tf.multiply(father_tensor, random_array_binary) + tf.multiply(mother_tensor, random_array_inverse)


        return crossoved


##Apenas as layers
def generate_child_by_layer(mother_tensor, father_tensor, mutationRate, layers):
    with tf.name_scope('Passagem_Genes'):
        temp_neural_network = []

        shape_size = tf.shape(mother_tensor)

        # Criação do array binário para definir quais são os genes que irão receber a mistura do  mãe
        random_array_binary = tf.random_uniform(dtype=tf.float32, minval=0, maxval=1, shape=[shape_size[0]])
        random_array_select = tf.random_uniform(dtype=tf.float32, minval=0, maxval=1, shape=[shape_size[0]])
        random_array_select = tf.math.round(random_array_select)

        # Criando o array inverso para definir o número ao contrário para criar a quantidade recebida pelo pai
        # random_array_inverse = tf.map_fn(lambda x: (1 - x), random_array_binary, dtype=tf.float32)
        random_array_inverse = 1 - random_array_binary
        # Criação o array de taxa de mistura para ambos
        # random_array_start = tf.cast(
        #    tf.random_uniform(dtype=tf.int32, minval=0, maxval=1, shape=[shape_size[0]]), tf.float32)

        for weight_idx_range in range(layers - 1):
            weight_idx = weight_idx_range - 1
            father_tensor_process = mother_tensor[weight_idx]
            mother_tensor_process = father_tensor[weight_idx]

            shape_size = tf.shape(mother_tensor[weight_idx])


            crossoved = tf.multiply(father_tensor_process, random_array_binary[weight_idx]) + tf.multiply(
                mother_tensor_process, random_array_inverse[weight_idx])
            temp_neural_network.append(mutation(crossoved, mutationRate))

        return tf.stack(temp_neural_network)


def crossover_operator(genetic_layer: GeneticLayer, tamanhoElite, tamanhoCrossover):
    with tf.name_scope('Crossover'):
        permutations = tf.range(tamanhoCrossover * 2)
        permutations = permutations % tamanhoElite
        # permutations = tf.random_shuffle(permutations)
        permutations = tf.reshape(permutations, [-1, 2])
        permutations = tf.transpose(permutations)

        i = tf.constant(0)

        # for i in tf.range(tamanhoCrossover):
        # def add_crossover(permutation):
        #    print(genetic_layer)
        father_tensor = tf.gather(genetic_layer.best_weights, permutations[0])
        mother_tensor = tf.gather(genetic_layer.best_weights, permutations[1])

        finish_conv = generate_child_by_all(father_tensor, mother_tensor)
        father_tensor = tf.gather(genetic_layer.best_biases, permutations[0])
        mother_tensor = tf.gather(genetic_layer.best_biases, permutations[1])
        finish_bias = generate_child_by_all(father_tensor, mother_tensor)


        return finish_conv, finish_bias


def mutation_operator(genetic_layer: GeneticLayer, tamanhoElite, mutationRate, mutationPercent, tamanhoMutacoes):
    with tf.name_scope('Mutation_operator'):
        shape_module = tf.shape(genetic_layer.best_weights)

        elite_key = tf.strided_slice(genetic_layer.best_weights, [0], [tamanhoElite], name="Stride_elite_mutation")
        times_to_repeat = (tamanhoMutacoes // tamanhoElite) + 1
        shape_with_ones = tf.ones_like(tf.shape(elite_key[0]))
        saida_shape = tf.concat([[times_to_repeat], shape_with_ones], axis=0, name="concat")
        tensors_to_mutate = tf.strided_slice(tf.tile(elite_key, saida_shape), [0], [tamanhoMutacoes],
                                             name="stride_tensor_to_mutate")

        finish_conv = mutation(tensors_to_mutate, mutationRate, mutationPercent)

        elite_key = genetic_layer.best_biases[0: tamanhoElite]
        times_to_repeat = (tamanhoMutacoes // tamanhoElite) + 1
        shape_with_ones = tf.ones_like(tf.shape(elite_key[0]))
        saida_shape = tf.concat([[times_to_repeat], shape_with_ones], axis=0)
        tensors_to_mutate = tf.tile(elite_key, saida_shape)[0:tamanhoMutacoes]

        finish_bias = mutation(tensors_to_mutate, mutationRate, mutationPercent)

    return finish_conv, finish_bias


def mutation_unbiased_operator(best_conv, best_bias, tamanhoElite, mutationRate, mutationPercent, tamanhoMutacoes):
    with tf.name_scope('Mutatio_unbiased'):

        finish = []
        finish_conv = {}
        finish_bias = {}
        tamanhoCrossover = tamanhoElite
        permutations = tf.range(tamanhoElite)
        permutations = tf.reshape(permutations, [tamanhoElite // 2, 2])
        keys = best_conv.keys()

        for key in best_conv:
            shape_module = tf.shape(best_conv[key])[0]
            finish_conv[key] = tf.map_fn(
                lambda x: mutation_unbiased(best_conv[key][x % shape_module], mutationRate, mutationPercent),
                tf.range(tamanhoMutacoes), dtype=tf.float32)

        for key in best_bias:
            shape_module = tf.shape(best_bias[key])[0]
            finish_bias[key] = tf.map_fn(
                lambda x: mutation_unbiased(best_bias[key][x % shape_module], mutationRate, mutationPercent),
                tf.range(tamanhoMutacoes), dtype=tf.float32)

    return finish_conv, finish_bias
