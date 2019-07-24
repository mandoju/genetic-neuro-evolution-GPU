import tensorflow as tf
from typing import List
import numpy as np
import copy

from genetic_neural_network.geneticLayer import GeneticLayer


def create_constants(neural_networks):
    neural_networs_output = []

    for current_neural_network in neural_networks:
        temp_neural_network = []
        # print("NEURAL NETWORK")
        i = 0
        for weight in current_neural_network:
            # if (type(weight) != tf.Tensor):
            # print(type(weight))
            temp_neural_network.append(tf.constant(weight))
            i += 1
        neural_networs_output.append(temp_neural_network[:])

    return neural_networs_output


def choose_best_tensor(neural_networks, fitnesses):
    with tf.name_scope('Choose_best') as scope:
        top_values, top_indices = tf.math.top_k(
            tf.reshape(fitnesses, (-1,)), 4)

        neural_networks_output = tf.stack([neural_networks[top_indices[0]], neural_networks[top_indices[1]],
                                           neural_networks[top_indices[2]], neural_networks[top_indices[3]]])
        return neural_networks_output


def choose_best_tensor_conv(genetic_layers: List[GeneticLayer], fitnesses, chooseNumber):
    with tf.name_scope('Choose_best') as scope:

        print(chooseNumber)
        fitnesses = tf.squeeze(fitnesses)
        top_values, top_indices = tf.math.top_k(
            fitnesses, chooseNumber)
        # tf.reshape(fitnesses, (-1,)), 4)

        print('-----')
        print(fitnesses)
        print(top_indices)
        top_mutate_values, top_mutate_indices = tf.math.top_k(
            fitnesses, chooseNumber)

        for genetic_layer in genetic_layers:
            print(tf.gather(genetic_layer.layer.weight,top_indices))
            genetic_layer.set_best(tf.gather(genetic_layer.layer.weight,top_indices),tf.gather(genetic_layer.layer.bias,top_indices))
        #print(top_indices)
        #return convulation_weights_output, biases_output  # , convulation_weights_best_output, biases_output_best, convulation_weights_mutate_output, biases_output_mutate


def tournament(fitnesses, indexes):
    # convulation_weights_output[key] = tf.gather(convulations[key],top_indices)
    tournament_fitnesses = tf.gather(fitnesses, indexes)
    get_best_value, get_best_index = tf.math.top_k(tournament_fitnesses)
    return indexes[get_best_index[0]]


def choose_best_tensor_tournament(genetic_layers: List[GeneticLayer], fitnesses, chooseNumber):
    with tf.name_scope('Choose_best') as scope:

        tournamentSize = tf.shape(fitnesses)[0] // chooseNumber


        numbers_to_tournament = tf.range(tf.shape(fitnesses)[0])
        numbers_to_tournament = tf.random.shuffle(numbers_to_tournament)
        numbers_to_tournament = tf.reshape(numbers_to_tournament,
                                           [tf.shape(numbers_to_tournament)[0] // tournamentSize, tournamentSize])

        fitnesses = tf.squeeze(fitnesses)
        top_indices = tf.map_fn(lambda x: tournament(fitnesses, x), numbers_to_tournament)
        top_indices = tf.stack(top_indices)
        for genetic_layer in genetic_layers:
            print(tf.gather(genetic_layer.layer.weight,top_indices))
            genetic_layer.set_best(tf.gather(genetic_layer.layer.weight,top_indices),tf.gather(genetic_layer.layer.bias,top_indices))



def choose_best(chooseType, genetic_layers, fitnesses, chooseNumber):
    if(chooseType == 'tournament'):
       return choose_best_tensor_tournament(genetic_layers, fitnesses, chooseNumber)
    else:
        return choose_best_tensor_conv(genetic_layers, fitnesses, chooseNumber)