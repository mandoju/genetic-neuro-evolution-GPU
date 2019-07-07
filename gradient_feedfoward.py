import tensorflow as tf
import numpy as np
import time
from datasets.datasets import get_mnist_data_reshape, get_sine_data


with tf.name_scope('placeholders'):
    x = tf.placeholder('float', [None, 1])
    y = tf.placeholder('float', [None, 1])

with tf.name_scope('neural_network'):
    x1 = tf.contrib.layers.fully_connected(x, 500, activation_fn=tf.tanh)
    x2 = tf.contrib.layers.fully_connected(x1, 250, activation_fn=tf.tanh)
    x3 = tf.contrib.layers.fully_connected(x2, 100, activation_fn=tf.tanh)
    x4 = tf.contrib.layers.fully_connected(x3, 50, activation_fn=tf.tanh)

    result = tf.contrib.layers.fully_connected(x4, 1,
                                               activation_fn=None)

    loss = tf.reduce_mean(tf.squared_difference(result, y))#tf.nn.l2_loss(result - y)

with tf.name_scope('optimizer'):
    train_op = tf.train.AdamOptimizer().minimize(loss)


train_x, train_y, test_x, test_y = get_sine_data()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Train the network
    for i in range(1000):
        batch_size = 1000
        for batch in range(1):
            batch_x = train_x[batch * batch_size:min((batch + 1) * batch_size, len(train_x))]
            batch_y = train_y[batch * batch_size:min((batch + 1) * batch_size, len(train_y))]

            # xpts = np.random.rand(100) * 10
            # ypts = np.sin(xpts)

            session_time = time.time()

            _, loss_result = sess.run([train_op, loss],
                                      feed_dict={x: batch_x,
                                                 y: batch_y})

            print('iteration {}, loss={}'.format(i, loss_result))
            print("sessao demorou: " + str(time.time() - session_time))
