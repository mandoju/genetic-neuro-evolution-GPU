import tensorflow as tf


class Layer:

    def __init__(self, populationSize: int, weight, bias, bias_size, layer_type: str, activation=None):
        self.weight = weight
        self.bias = bias
        self.bias_size = bias_size
        self.type = layer_type
        self.activation = activation
        self.populationSize = populationSize

    def run_fist_layer(self, input):
        out = []
        # for i in range(self.populationSize):
        #    out.append(self.run_slice(input, i))
        if (self.type == 'wd'):
            input_replicated = tf.tile(tf.expand_dims(input, 0), [self.populationSize, 1, 1])
            #out = tf.add(tf.matmul(input_replicated, self.weight), self.bias)
            multiplication = tf.matmul(input_replicated, self.weight)
            bias_reshaped = tf.reshape(self.bias, [self.populationSize,1,self.bias_size])
            out = tf.add(multiplication,bias_reshaped)
            #out = tf.map_fn(lambda x: tf.add(tf.gather(multiplication,x), tf.gather(self.bias,x)), tf.range(self.populationSize),dtype=tf.float32)
            if (self.activation):
                out = self.activation(out)
        elif (self.type == 'wc'):
            # out = tf.map_fn(lambda x: self.conv2d(input, self.weight[x], self.bias[x]),
            #                 tf.range(self.populationSize), dtype=tf.float32)
            out = tf.map_fn(lambda x: self.conv2d(input, x[0], x[1]),
                            (self.weight,self.bias), dtype=tf.float32)
            out = tf.map_fn(lambda x: self.maxpool2d(x, k=2), out, dtype=tf.float32)

        else:
            for i in range(self.populationSize):
                out.append(self.run_slice(input, i))
        return out

    def run(self, input):
        out = []
        print(self.type)
        if (self.type == 'wd'):
            multiplication = tf.matmul(input, self.weight)
            #out = tf.map_fn(lambda x: tf.add(multiplication[x], self.bias[x]), tf.range(self.populationSize),dtype=tf.float32)
            bias_reshaped = tf.reshape(self.bias, [self.populationSize, 1, self.bias_size])
            out = tf.add(multiplication, bias_reshaped)
            if (self.activation):
                out = self.activation(out)

        else:
            for i in range(self.populationSize):
                out.append(self.run_slice(input[i], i))
            if(self.type == 'wcd'):
                out = tf.stack(out)
        return out

    def run_slice(self, input, slice):
        if (self.type == 'wc' or self.type == 'wcd'):
            out = self.conv2d(input, self.weight[slice], self.bias[slice])
        if (self.type == 'wd' ):
            out = tf.add(tf.matmul(input, self.weight[slice]), self.bias[slice])
        # if(self.activation == 'relu'):
        #     out = tf.nn.relu(out)
        # if(self.activation == 'sigmoid'):
        #     out = tf.nn.sigmoid(out)
        if (self.activation):
            out = self.activation(out)
        if (self.type == 'wc' or self.type == 'wcd'):
            #out = tf.add(tf.matmul(input, self.weight[slice]), self.bias[slice])
            out = self.maxpool2d(out)
        if ( self.type == 'wcd'):
            out = tf.reshape(out, [-1, 36])
        return out

    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(
            x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return x

    def maxpool2d(self, input, k=2):
        return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
