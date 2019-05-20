import tensorflow as tf


class Layer:

    def __init__(self, populationSize: int, weight, bias, layer_type: str, activation=None):
        self.weight = weight
        self.bias = bias
        self.type = layer_type
        self.activation = activation
        self.populationSize = populationSize

    def run_fist_layer(self, input):
        out = []
        for i in range(self.populationSize):
            out.append(self.run_slice(input, i))
        return out

    def run(self, input):
        out = []
        for i in range(self.populationSize):
            out.append(self.run_slice(input[i], i))
        return out

    def run_slice(self, input, slice):
        if (self.type == 'wc'):
            out = self.conv2d(input, self.weight[slice], self.bias[slice])
        if (self.type == 'wd'):
            out = tf.add(tf.matmul(input, self.weight[slice]), self.bias[slice])
        # if(self.activation == 'relu'):
        #     out = tf.nn.relu(out)
        # if(self.activation == 'sigmoid'):
        #     out = tf.nn.sigmoid(out)
        if (self.activation):
            out = self.activation(out)
        return out

    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(
            x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return x

    def maxpool2d(self, input, k=2):
        return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
