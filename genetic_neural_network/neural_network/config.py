import sys
import tensorflow as tf


def getLayerDimension():
    if (sys.argv[3] == 'cnn_3'):
        return 4 * 4 * 36
    else:
        return 36


def getPlaceHolderType():
    if (sys.argv[3] == 'cnn_3' or sys.argv[3] == 'cnn_6'):
        return tf.placeholder("float", shape=[None, 28, 28, 1], name="X")
    elif (sys.argv[3] == 'feed_foward_3' or sys.argv[3] == 'feed_foward_6'):
        return tf.placeholder("float", shape=[None, 784], name="X")
    else:
        return tf.placeholder("float", shape=[None, 784], name="X")

def getExperimento():
    if(sys.argv[5] == 'exp1'):
        return 1
    else:
        return 2
