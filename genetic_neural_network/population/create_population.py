import tensorflow as tf
import numpy as np
import itertools

from genetic_neural_network.geneticLayer import GeneticLayer
from typing import List


def create_population(population_size: int, layers: List[GeneticLayer]):
    # tf.reset_default_graph()

    # convulations_weights = {}
    initializer = tf.contrib.layers.variance_scaling_initializer(uniform=False, factor=2.0, mode='FAN_IN',
                                                                 dtype=tf.float32)
    for layer in layers:
        layer.set_layer(population_size, initializer=initializer)
    # for key, val in in_weights.items():
    #    convulations_weights[key] = tf.Variable(initial_value=tf.stack(
    #        tf.map_fn(lambda x: initializer(list(val)), tf.range(populationSize), dtype=tf.float32)), dtype=tf.float32,
    #        name='w' + key)
    # biases = {}
    # for key, val in in_biases.items():
    #    print(val)
    #    biases[key] = tf.get_variable(key, shape=(populationSize, val), initializer=tf.random_normal_initializer())

    #return convulations_weights, biases
