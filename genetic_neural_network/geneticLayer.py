import tensorflow as tf
from .neural_network.layer import Layer
from typing import Tuple, Callable


class GeneticLayer:

    def __init__(self, weight_size: Tuple[int, ...], bias_size: Tuple[int, ...], type: str, activation, name: str):
        self.weight_size = weight_size
        self.bias_size = bias_size
        self.type = type
        self.activation = activation
        self.name = name
        self.layer = None
        self.best_weights = None
        self.best_biases = None
        self.operators_weights = []
        self.operators_biases = []

    def set_layer(self, population_size: int, initializer):
        weight = tf.Variable(initial_value=tf.stack(
            tf.map_fn(lambda x: initializer(list(self.weight_size)), tf.range(population_size), dtype=tf.float32)),
            dtype=tf.float32,
            name='w' + self.name)
        print(tf.map_fn(lambda x: initializer(list(self.weight_size)), tf.range(population_size), dtype=tf.float32))
        bias = tf.get_variable('b' + self.name, shape=(population_size, self.bias_size),
                               initializer=tf.random_normal_initializer())
        self.layer = Layer(population_size, weight, bias, self.type, self.activation)

    def set_best(self,weights,biases):
        self.best_weights = weights
        self.best_biases = biases

    def add_operator(self,weight,bias):
        self.operators_weights.append(weight)
        self.operators_biases.append(bias)
