from .neural_network.neural_network import Neural_network
from .population.create_population import create_population
from .population.choose_best import choose_best
from tensorflow.python.client import timeline
from .genetic_operators.genetic_operatos import apply_genetic_operatos
from .debug.graph import Graph
from statistics import mean
from tensorflow.python import debug as tf_debug
import tkinter as tk
import numpy as np
import tensorflow as tf
import time
import matplotlib
import pickle
import sys

# matplotlib.use('Agg')

import matplotlib.pyplot as plt


class GeneticNeuralNetwork:

    def __init__(self, geneticSettings):
        self.populationSize = geneticSettings['populationSize']
        self.layers = geneticSettings['layers']
        self.mutationRate = geneticSettings['mutationRate']
        self.geneticLayers = geneticSettings['geneticLayers']
        create_population(self.populationSize, self.geneticLayers)
        self.neural_networks = Neural_network(self.geneticLayers,
                                              geneticSettings['train_x'], geneticSettings['train_y'],
                                              geneticSettings['test_x'],
                                              geneticSettings['test_y'], './log/')
        self.geneticSettings = geneticSettings
        self.current_epoch = 0
        self.eliteSize = int(geneticSettings['elite'] * self.populationSize)
        self.slice_sizes = [self.populationSize * x for x in geneticSettings['genetic_operators_size']]
        self.genetic_operators_size = geneticSettings['genetic_operators_size']
        self.fineTuningRate = geneticSettings['fineTuningRate']
        self.fineTuningBoolean = geneticSettings['fineTuning']

        # self.fitnesses = tf.Variable(shape=[self.populationSize],initial_value=tf.zeros_initializer(), dtype=tf.float32, name="fitnesses")
        self.fitnesses = tf.get_variable("fitnesses", shape=(self.populationSize), initializer=tf.zeros_initializer())

    def run_epoch(self):

        start = time.time()

        self.mutationRate = tf.placeholder(tf.float32, shape=[])
        self.operatorSize = tf.placeholder(tf.float32, shape=[len(self.geneticSettings['genetic_operators_size'])])
        self.neural_networks.run()

        if (self.geneticSettings['fitness'] == 'cross_entropy'):
            fitness = -self.neural_networks.cost
        elif (self.geneticSettings['fitness'] == 'square_mean_error'):
            fitness = -self.neural_networks.square_mean_error
        elif (self.geneticSettings['fitness'] == 'root_square_mean_error'):
            fitness = -self.neural_networks.root_square_mean_error
        elif (self.geneticSettings['fitness'] == 'cross_entropy_mix_accuracies'):
            fitness = self.neural_networks.accuracies + self.neural_networks.cost
        else:
            fitness = self.neural_networks.accuracies

        print("choose best")
        choose_best(
            self.geneticSettings['selection'], self.geneticLayers, fitness,
            self.eliteSize)

        print("apply_genetic_operators")
        assigns_weights, assigns_biases = apply_genetic_operatos(self.geneticSettings['genetic_operators'],
                                                                 self.operatorSize,
                                                                 self.eliteSize, self.geneticLayers, self.mutationRate,
                                                                 2, len(self.layers))

        print('passei por aqui')
        merged = tf.summary.merge_all()

        self.current_epoch += 1
        sess = tf.Session()

        writer = tf.summary.FileWriter(self.neural_networks.logdir, sess.graph)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        start = time.time()
        sess.run(tf.global_variables_initializer())
        print("Global variables:", time.time() - start)

        start = time.time()
        sess.run(tf.local_variables_initializer())
        print("local variables:", time.time() - start)

        train_x = self.neural_networks.train_x
        train_y = self.neural_networks.train_y

        acuracias = []
        fitnesses = []
        validation_fitnesses = []
        validation_acuracias = []
        tempos = []
        tempos_validation = []
        fine_tuning_graph = []

        print(assigns_weights)

        print(assigns_biases)
        print("batchs: " + str(len(train_x) // 125))
        mutate = self.geneticSettings['mutationRate']
        print(mutate)

        start_time = time.time()
        for i in range(self.geneticSettings['epochs']):

            print("época: " + str(i))
            start_generation = time.time()

            batch_size = 10000

            for batch in range(1):
                # for batch in range( (len(train_x)//batch_size ) - 1 ):
                # for j in range(self.geneticSettings['inner_loop']):
                print("batch: " + str(batch))
                start_batch = time.time()
                batch_x = train_x[batch * batch_size:min((batch + 1) * batch_size, len(train_x))]
                batch_y = train_y[batch * batch_size:min((batch + 1) * batch_size, len(train_y))]

                # print("Mutação atual: " + str(mutate) )
                print(self.slice_sizes)
                fine_tuning_graph.append(self.slice_sizes[:])

                session_time = time.time()


                predicts, label_argmax, cost, finished_conv, finished_bias = sess.run(
                    [self.neural_networks.predicts, self.neural_networks.label_argmax,
                     fitness, assigns_weights, assigns_biases], feed_dict={
                        self.neural_networks.X: batch_x, self.neural_networks.Y: batch_y, self.mutationRate: mutate,
                        self.operatorSize: self.slice_sizes})#, options=run_options, run_metadata=run_metadata)
                print("sessao demorou: " + str(time.time() - session_time))
                writer.add_run_metadata(run_metadata, 'step%s' % (str(batch) + '_' + str(i)))
                msg = "Batch: " + str(batch)
                # np.savetxt('weights_save.txt',finished_conv[0])
                # np.savetxt('predicts_save.txt',predicts)
                # np.savetxt('Y.txt',label_argmax)
                # print("Accuracy: ")
                # print(accuracies)
                print("Cost: ")
                print(cost[0:])
                print("tempo atual: " + str(time.time() - start_time))
                # if(max(cost) < 3):
                fitnesses.append(max(cost))
                # acuracias.append(max(accuracies))
                tempos.append(time.time() - start_time)
                if (self.fineTuningBoolean):
                    # if(max(accuracies) <= last_accuracy):
                    #     mutate += 0.1
                    #     if(mutate > 0.7):
                    #         mutate = 0.7
                    # else:
                    #     mutate -= 0.1
                    #     if(mutate < 0.1):
                    #         mutate = 0.1
                    last_cost = max(cost)

                    last_population_slice = self.eliteSize
                    operators_max = []
                    for population_slice in self.slice_sizes:
                        slice_finish = int(last_population_slice + population_slice - 1)
                        if (population_slice > 1):
                            operators_max.append(np.mean(cost[last_population_slice:slice_finish]))
                        else:
                            operators_max.append(cost[last_population_slice])
                        last_population_slice += int(population_slice)

                    max_fitness_operator_index = operators_max.index(max(operators_max))

                    # possible_slices_remove = self.slice_sizes
                    # minimum_not_one = possible_slices[operators_max.index(min(operators_max))]
                    slice_with_operator = np.column_stack(
                        (self.slice_sizes, operators_max, range(len(self.slice_sizes))))
                    slice_with_operator = list(filter(lambda x: x[0] > self.populationSize * 0.05, slice_with_operator))
                    min_fitness_slice = min(slice_with_operator, key=lambda x: x[1])
                    min_fitness_operator_index = int(min_fitness_slice[2])
                    if (self.slice_sizes[min_fitness_operator_index] > 1):
                        self.slice_sizes[max_fitness_operator_index] += 1
                        self.slice_sizes[min_fitness_operator_index] -= 1
                    # if(batch % 30 == 0):
                    #     if(self.slice_sizes[1] > 18 or self.slice_sizes[2] > 18):
                    #         mutate = mutate * 10
                    #         print('mutei')
                    #     if(self.slice_sizes[0] > 18 or self.slice_sizes[4] > 18):
                    #         mutate = mutate / 10
                    #         print('mutei')
                    #     last_best_cost = max(cost)

            # batch = (len(train_x)//batch_size ) - 1
            # batch_x = train_x[batch*batch_size:min((batch+1)*batch_size,len(train_x))]
            # batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]
            final_predict, accuracies, cost = sess.run(
                [self.neural_networks.predicts, self.neural_networks.accuracies, fitness], feed_dict={
                    self.neural_networks.X: train_x, self.neural_networks.Y: train_y})
            print("acuracia:" + str(accuracies[0]))
            validation_acuracias.append(accuracies[0])
            print("fitness:" + str(cost[0]))
            validation_fitnesses.append(cost[0])
            tempos_validation.append(time.time() - start_time)

            # mutate = mutate * 2
        sess.close()
        file_string = []
        # if (len(sys.argv) > 2):
        #     file_string = './debug/graphs_logs/' + str(self.populationSize) + '_' + sys.argv[2] + '.pckl'
        # else:
        #     file_string = './debug/graphs_logs/' + str(self.populationSize) + '_10.pckl'
        # with open(file_string, 'wb') as save_graph_file:
        #     save_graph = Graph(tempos, fitnesses, acuracias, tempos_validation, validation_fitnesses,
        #                        validation_acuracias, fine_tuning_graph)
        #     pickle.dump(save_graph, save_graph_file)
        #     print('salvei em: ' + '.debug/graphs_logs/' + str(self.populationSize) + '.pckl')

        # plt.plot(tempos, acuracias, '-', lw=2)
        # plt.grid(True)
        # plt.savefig('acuracias.png')
        plt.plot(train_x, train_y, '-', label="seno")
        plt.plot(train_x, final_predict[0], '-', label="neural_network")
        plt.grid(True)
        plt.show()
