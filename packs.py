import tensorflow as tf


def get_weight_convolution(number_layer):
    if (number_layer == 10):
        return {
            # ('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
            'wc1': (3, 3, 1, 2),
            # ('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()),
            'wc2': (3, 3, 2, 4),
            # ('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
            'wc3': (3, 3, 4, 16),
            'wc4': (3, 3, 16, 32),
            'wc5': (3, 3, 32, 64),
            'wc6': (3, 3, 64, 128),
            'wc7': (3, 3, 128, 256),
            'wc8': (3, 3, 256, 256),
            'wc9': (3, 3, 256, 256),
            'wc10': (3, 3, 256, 256),

            'wd1': (256, 16),
            'out': (16, 10)
        }
    elif (number_layer == 5):
        return {
            # ('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
            'wc1': (3, 3, 1, 2),
            # ('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()),
            'wc2': (3, 3, 2, 4),
            # ('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
            'wc3': (3, 3, 4, 16),
            'wc4': (3, 3, 16, 32),
            'wc5': (3, 3, 32, 64),
            'wd1': (64, 16),
            'out': (16, 10)}
    else:
        return {
            # ('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
            'wc1': (3, 3, 1, 32),
            # ('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()),
            'wc2': (3, 3, 32, 64),
            # ('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
            'wc3': (3, 3, 64, 128),
            'wd1': (4 * 4 * 128, 128),
            'out': (128, 10)
        }


def get_weight_dense(number_layer, input):
    return {
        'wd1': (1, 40),
        'wd2': (40, 12),
        'out': (12, 1)
    }


def get_biases_dense(number_layer):
    return {
        'bc1': (40),
        'bc2': (12),
        'out': (1),
    }


def get_biases(number_layer):
    if (number_layer == 10):
        return {
            'bc1': (2),
            'bc2': (4),
            'bc3': (16),
            'bc4': (32),
            'bc5': (64),
            'bc6': (128),
            'bc7': (256),
            'bc8': (256),
            'bc9': (256),
            'bc10': (256),
            'bd1': (16),
            'out': (10)}
    elif (number_layer == 5):
        return {
            'bc1': (2),
            'bc2': (4),
            'bc3': (16),
            'bc4': (32),
            'bc5': (64),
            'bd1': (16),
            'out': (10), }
    else:
        return {
            'bc1': (32),
            'bc2': (64),
            'bc3': (128),
            'bd1': (128),
            'out': (10), }


def get_gradient_convolution(number_layer):
    if (number_layer == 10):
        return {

            'wc1': tf.get_variable('W0', shape=(3, 3, 1, 2), initializer=tf.keras.initializers.he_normal()),
            'wc2': tf.get_variable('W1', shape=(3, 3, 2, 4), initializer=tf.keras.initializers.he_normal()),
            'wc3': tf.get_variable('W2', shape=(3, 3, 4, 16), initializer=tf.keras.initializers.he_normal()),
            'wc4': tf.get_variable('W3', shape=(3, 3, 16, 32), initializer=tf.keras.initializers.he_normal()),
            'wc5': tf.get_variable('W4', shape=(3, 3, 32, 64), initializer=tf.keras.initializers.he_normal()),
            'wc6': tf.get_variable('W5', shape=(3, 3, 64, 128), initializer=tf.keras.initializers.he_normal()),
            'wc7': tf.get_variable('W6', shape=(3, 3, 128, 256), initializer=tf.keras.initializers.he_normal()),
            'wc8': tf.get_variable('W7', shape=(3, 3, 256, 256), initializer=tf.keras.initializers.he_normal()),
            'wc9': tf.get_variable('W8', shape=(3, 3, 256, 256), initializer=tf.keras.initializers.he_normal()),
            'wc10': tf.get_variable('W9', shape=(3, 3, 256, 256), initializer=tf.keras.initializers.he_normal()),

            'wd1': tf.get_variable('Wd1', shape=(256, 16), initializer=tf.keras.initializers.he_normal()),
            'out': tf.get_variable('Wout', shape=(16, 10), initializer=tf.keras.initializers.he_normal()),
        }
    elif (number_layer == 5):
        return {

            'wc1': tf.get_variable('W0', shape=(3, 3, 1, 2), initializer=tf.keras.initializers.he_normal()),
            'wc2': tf.get_variable('W1', shape=(3, 3, 2, 4), initializer=tf.keras.initializers.he_normal()),
            'wc3': tf.get_variable('W2', shape=(3, 3, 4, 16), initializer=tf.keras.initializers.he_normal()),
            'wc4': tf.get_variable('W3', shape=(3, 3, 16, 32), initializer=tf.keras.initializers.he_normal()),
            'wc5': tf.get_variable('W4', shape=(3, 3, 32, 64), initializer=tf.keras.initializers.he_normal()),

            'wd1': tf.get_variable('Wd1', shape=(64, 16), initializer=tf.keras.initializers.he_normal()),
            'out': tf.get_variable('Wout', shape=(16, 10), initializer=tf.keras.initializers.he_normal()),
        }
    else:
        return {

            'wc1': tf.get_variable('W0', shape=(3, 3, 1, 2), initializer=tf.keras.initializers.he_normal()),
            'wc2': tf.get_variable('W1', shape=(3, 3, 2, 4), initializer=tf.keras.initializers.he_normal()),
            'wc3': tf.get_variable('W2', shape=(3, 3, 4, 16), initializer=tf.keras.initializers.he_normal()),

            'wd1': tf.get_variable('Wd1', shape=(256, 16), initializer=tf.keras.initializers.he_normal()),
            'out': tf.get_variable('Wout', shape=(16, 10), initializer=tf.keras.initializers.he_normal()),
        }


def get_gradient_biases(number_layer):
    if (number_layer == 10):
        return {
            'bc1': tf.get_variable('B0', shape=(2), initializer=tf.keras.initializers.he_normal()),
            'bc2': tf.get_variable('B1', shape=(4), initializer=tf.keras.initializers.he_normal()),
            'bc3': tf.get_variable('B2', shape=(16), initializer=tf.keras.initializers.he_normal()),
            'bc4': tf.get_variable('B3', shape=(32), initializer=tf.keras.initializers.he_normal()),
            'bc5': tf.get_variable('B4', shape=(64), initializer=tf.keras.initializers.he_normal()),
            'bc6': tf.get_variable('B5', shape=(128), initializer=tf.keras.initializers.he_normal()),
            'bc7': tf.get_variable('B6', shape=(256), initializer=tf.keras.initializers.he_normal()),
            'bc8': tf.get_variable('B7', shape=(256), initializer=tf.keras.initializers.he_normal()),
            'bc9': tf.get_variable('B8', shape=(256), initializer=tf.keras.initializers.he_normal()),
            'bc10': tf.get_variable('B9', shape=(256), initializer=tf.keras.initializers.he_normal()),

            'bd1': tf.get_variable('Bd1', shape=(16), initializer=tf.keras.initializers.he_normal()),
            'out': tf.get_variable('Bout', shape=(10), initializer=tf.keras.initializers.he_normal()),
        }
    if (number_layer == 10):
        return {
            'bc1': tf.get_variable('B0', shape=(2), initializer=tf.keras.initializers.he_normal()),
            'bc2': tf.get_variable('B1', shape=(4), initializer=tf.keras.initializers.he_normal()),
            'bc3': tf.get_variable('B2', shape=(16), initializer=tf.keras.initializers.he_normal()),
            'bc4': tf.get_variable('B3', shape=(32), initializer=tf.keras.initializers.he_normal()),
            'bc5': tf.get_variable('B4', shape=(64), initializer=tf.keras.initializers.he_normal()),

            'bd1': tf.get_variable('Bd1', shape=(16), initializer=tf.keras.initializers.he_normal()),
            'out': tf.get_variable('Bout', shape=(10), initializer=tf.keras.initializers.he_normal()),
        }
    else:
        return {
            'bc1': tf.get_variable('B0', shape=(2), initializer=tf.keras.initializers.he_normal()),
            'bc2': tf.get_variable('B1', shape=(4), initializer=tf.keras.initializers.he_normal()),
            'bc3': tf.get_variable('B2', shape=(16), initializer=tf.keras.initializers.he_normal()),

            'bd1': tf.get_variable('Bd1', shape=(16), initializer=tf.keras.initializers.he_normal()),
            'out': tf.get_variable('Bout', shape=(10), initializer=tf.keras.initializers.he_normal()),
        }
