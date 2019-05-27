import tensorflow as tf
from typing import List
from genetic_neural_network.geneticLayer import GeneticLayer

import numpy as np
import time
# from utils import variable_summaries
from .layer import Layer


class Neural_network:

    def __init__(self, genetic_layers: List[GeneticLayer], train_x, train_y, test_x, test_y, logdir):
        # self.neural_networks = neural_networks
        self.geneticLayers = genetic_layers
        self.logdir = logdir
        # self.train_x, self.test_x, self.train_y,self.test_y = get_mnist_data()
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        # self.convulations = convulations
        # self.biases = biases
        # self.populationSize = populationSize
        self.classification = False

    def run_neural_structure(self, X):

        # weights = self.convulations
        # biases = self.biases

        # layer_1 = Layer(self.populationSize,weights['wd1'], biases['bc1'],'wd',tf.math.tanh)
        # layer_2 = Layer(self.populationSize,weights['wd2'], biases['bc2'],'wd',tf.math.tanh)
        # layer_3 = Layer(self.populationSize,weights['out'], biases['out'],'wd')

        layer_output = self.geneticLayers[0].layer.run_fist_layer(X)

        for geneticLayer in self.geneticLayers[1:]:
            layer_output = geneticLayer.layer.run(layer_output)

        # layer_1_out = layer_1.run_fist_layer(X)
        # layer_2_out = layer_2.run(layer_1_out)
        # layer_3_out = layer_3.run(layer_2_out)

        return layer_output

    def get_accuracies(self, predict):

        if (self.classification):
            correct_prediction = tf.equal(
                tf.argmax(predict, 1), tf.argmax(self.Y, 1))

            # calculate accuracy across all the given images and average them out.
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return accuracy
        else:
            correct_prediction = tf.equal(
                predict, self.Y)

            # calculate accuracy across all the given images and average them out.
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return accuracy

    def get_square_mean_error(self, predict):

        with tf.name_scope('calculo_square_mean_error') as scope:
            label_test = tf.cast(tf.argmax(
                self.Y, axis=1, name="label_test_argmax_sme"), tf.float32)
            square_mean_error = tf.metrics.mean_squared_error(
                labels=label_test, predictions=predict)
            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=self.Y))

        return square_mean_error[0]
        # return cost

    def get_cost_functions(self, predict, train, test):
        with tf.name_scope('calculo_da_acuracia') as scope:
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=train, logits=predict))

            # return test_cost

    def run(self):
        # print("Rodando rede neural")

        # mnist
        with tf.name_scope('Fitness') as scope:

            print(self.train_y.shape)
            y_size = self.train_y.shape[1]

            # self.X = tf.placeholder("float", shape=[None, 28, 28, 1], name="X")
            # self.Y = tf.placeholder("float", shape=[None, y_size], name="Y")

            self.X = tf.placeholder("float", shape=[None, 1], name="X")
            self.Y = tf.placeholder("float", shape=[None, y_size], name="Y")

            X = self.X
            Y = self.Y

            with tf.name_scope('predicts') as scope:

                self.predicts = self.run_neural_structure(X)
                predicts = tf.stack(self.predicts)
                print(predicts)

            with tf.name_scope('accuracies') as scope:

                train_accuracies = tf.map_fn(
                    lambda x: self.get_accuracies(x), predicts)

                # train_accuracies = self.get_accuracies(predicts[0])
            with tf.name_scope('cost') as cost:

                cost = tf.map_fn(lambda pred: tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=self.Y),
                                 predicts)
                print(cost)
                # cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicts, labels=self.Y, axis=1)
                if (self.classification):
                    cost = tf.reduce_mean(cost, 1)

                # cost = tf.map_fn(lambda pred: -tf.reduce_sum(self.Y * tf.log(pred)), predicts)

            with tf.name_scope('square_mean_error') as scope:

                 square_mean_error = tf.map_fn(lambda pred: tf.reduce_mean(tf.squared_difference(tf.cast(tf.argmax(
                    pred, axis=1, name="label_test_argmax_sme"), tf.float32),
                    tf.cast(tf.argmax(self.Y, axis=1, name="label_test_argmax_sme"), tf.float32))), predicts)
                # argmax_y = tf.cast(tf.argmax(self.Y, axis=1, name="label_test_argmax_sme"), dtype=tf.float32)
                # argmax_pred = tf.cast(tf.argmax(predicts, axis=2, name="label_test_argmax_sme"), dtype=tf.float32)
                # squared_differences = tf.squared_difference(argmax_pred, argmax_y)
                # square_mean_error = tf.reduce_mean(squared_differences, axis=1)
            with tf.name_scope('root_square_mean_error') as scope:

                root_square_mean_error = tf.map_fn(lambda pred: tf.sqrt(
                   tf.reduce_mean(tf.square(tf.subtract(Y, pred)))), predicts, dtype=tf.float32)
                # print(root_square_mean_error)
                # squared_differences = tf.square(tf.subtract(Y, predicts))
                # print(squared_differences)
                # root_square_mean_error = tf.reduce_mean(squared_differences, axis=1)
                # print(root_square_mean_error)
            # Utilizacao das acuracias e predicts como tensores
            # self.predicts = predicts
            if (self.classification):
                self.argmax_predicts = tf.argmax(predicts[0], 1)
            else:
                self.argmax_predicts = predicts[0]
            self.accuracies = train_accuracies
            self.cost = cost
            self.square_mean_error = square_mean_error
            self.root_square_mean_error = root_square_mean_error
            self.label_argmax = tf.cast(tf.argmax(
                self.Y, axis=1, name="label_test_argmax_sme"), tf.float32)

            # writer.close()
            # sess.close()

            # self.predicts = predicts_session
            # self.accuracies = train_accuracies_session

            # return train_accuracies_session
            return self.accuracies

#
# def calculate_fitness(neural_networks, layers, logdir):
#     # return nn_cube(neural_networks,layers)
#     neural_structure = neural_network(neural_networks, layers, logdir)
#     return neural_structure.run()
