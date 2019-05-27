import tensorflow as tf
import numpy as np
import time
from datasets.datasets import get_mnist_data_reshape, get_sine_data
# from win10toast import ToastNotifier
from genetic_neural_network.geneticLayer import GeneticLayer
from genetic_neural_network.genetic_neural_network import GeneticNeuralNetwork
from packs import get_biases, get_weight_convolution, get_weight_dense, get_biases_dense
import traceback
import sys

# if(len(sys.argv) > 2):
#    weights_convulation_input = get_weight_dense(int(sys.argv[2]),0)
#    biases_input = get_biases_dense(int(sys.argv[2]))
# else:
#    print("por favor utilize dois argumentos")
#    sys.exit()

layers = [
    GeneticLayer((1, 40), 40, 'wd', tf.math.tanh, 'd1'),
    GeneticLayer((40, 12), 12, 'wd', tf.math.tanh, 'd2'),
    GeneticLayer((12, 1), 1, 'wd', None, 'out')
]

train_x, train_y, test_x, test_y = get_sine_data()
geneticSettings = {
    'train_x': train_x,
    'train_y': train_y,
    'test_x': test_x,
    'test_y': test_y,
    'populationSize': int(sys.argv[1]),
    'epochs': 1000,
    'inner_loop': 10,
    # 'weights_convulation': weights_convulation_input,
    # 'biases': biases_input,
    'geneticLayers': layers,
    'fitness': 'root_square_mean_error',
    'selection': 'tournament',
    'elite': 0.20,
    'genetic_operators': [['crossover', 0.10], ['mutation', 0.1], ['mutation', 0.01], ['mutation', 0.001],
                          ['mutation', 0.0001], ['mutation', 0.00001], ['mutation', 0.000001], ['mutation', 0.0000001]],
    'genetic_operators_size': [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
    'fineTuningRate': 0.05,
    'layers': [785, 10],
    'mutationRate': 0.001,
    'logdir': './log/',
    'fineTuning': True
}

# populationSize = geneticSettings['populationSize']
# epochs = geneticSettings['epochs']
# layers = geneticSettings['layers']
# mutationRate = geneticSettings['mutationRate']
# logdir = geneticSettings['logdir']
# weights_convulation = geneticSettings['weights_convulation'],
# biases = geneticSettings['biases']

start_time = time.time()
begin_time = start_time

# toaster = ToastNotifier()
# toaster.show_toast("Programa iniciado","Rodando programa")

try:

    genetic = GeneticNeuralNetwork(geneticSettings)
    genetic.run_epoch()
    print(genetic.neural_networks.accuracies)

    # toaster.show_toast("Sucesso!","Programa finalizado com sucesso")
except Exception as e:
    print(traceback.format_exc())
    print(e)
    # toaster.show_toast("Erro!","Ocorreu um erro")
