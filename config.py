import tensorflow as tf
from genetic_neural_network.geneticLayer import GeneticLayer
import sys

def getLayer():
    if(sys.argv[4] == 'tanh'):
        activation = tf.math.tanh
    else:
        activation = tf.math.sigmoid
    if (sys.argv[3] == 'feed_foward_6'):
        return [
            GeneticLayer((784, 500), 500, 'wd', activation, 'd1'),
            GeneticLayer((500, 500), 500, 'wd', activation, 'd2'),
            GeneticLayer((500, 500), 500, 'wd', activation, 'd3'),
            GeneticLayer((500, 500), 500, 'wd', activation, 'd4'),

            GeneticLayer((500, 10), 10, 'wd', None, 'out')]
    elif (sys.argv[3] == 'feed_foward_3'):
        return [
            GeneticLayer((784, 500), 500, 'wd', activation, 'd1'),
            GeneticLayer((500, 500), 500, 'wd', activation, 'd2'),
            GeneticLayer((500, 500), 500, 'wd', activation, 'd3'),
            GeneticLayer((500, 500), 500, 'wd', activation, 'd4'),
            GeneticLayer((500, 500), 500, 'wd', activation, 'd5'),
            GeneticLayer((500, 500), 500, 'wd', activation, 'd6'),
            GeneticLayer((500, 500), 500, 'wd', activation, 'd7'),

            GeneticLayer((500, 10), 10, 'wd', None, 'out')]
    elif(sys.argv[3] == 'cnn_3'):
        return [
            GeneticLayer((3,3,1,36), 36, 'wc', tf.nn.relu,'c1'),
            GeneticLayer((3, 3, 36, 36), 36, 'wc', tf.nn.relu, 'c2'),
            GeneticLayer((3, 3, 36, 36), 36, 'wcd', tf.nn.relu, 'c3'),
            GeneticLayer((4*4*36, 500), 500, 'wd', activation, 'd1'),
            GeneticLayer((500, 10), 10, 'wd', None, 'out')
        ]
    elif (sys.argv[3] == 'cnn_6'):
        return [
            GeneticLayer((3, 3, 1, 36), 36, 'wc', tf.nn.relu, 'c1'),
            GeneticLayer((3, 3, 36, 36), 36, 'wc', tf.nn.relu, 'c2'),
            GeneticLayer((3, 3, 36, 36), 36, 'wc', tf.nn.relu, 'c3'),
            GeneticLayer((3, 3, 36, 36), 36, 'wc', tf.nn.relu, 'c4'),
            GeneticLayer((3, 3, 36, 36), 36, 'wc', tf.nn.relu, 'c5'),
            GeneticLayer((3, 3, 36, 36), 36, 'wcd', tf.nn.relu, 'c6'),
            GeneticLayer((36, 500), 500, 'wd', activation, 'd1'),
            GeneticLayer((500, 10), 10, 'wd', None, 'out')
        ]
    else:
        raise ValueError("Layer não setada")

def getMnistFlag():
    if (sys.argv[3] == 'cnn_3' or sys.argv[3] == 'cnn_6'):
        return True
    elif (sys.argv[3] == 'feed_foward_3' or sys.argv[3] == 'feed_foward_6'):
        return False
    else:
        return True
