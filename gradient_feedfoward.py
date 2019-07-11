import tensorflow as tf
import numpy as np
import time
import dill
from datasets.datasets import get_mnist_data_reshape, get_sine_data
from genetic_neural_network.debug.graph import Graph

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

    acuracias = []
    fitnesses = []
    validation_fitnesses = []
    validation_acuracias = []
    tempos = []
    tempos_validation = []
    fine_tuning_graph = []

    # Train the network
    start_time = time.time()
    for i in range(1000):
        batch_size = 10000
        for batch in range(1):
            batch_x = train_x[batch * batch_size:min((batch + 1) * batch_size, len(train_x))]
            batch_y = train_y[batch * batch_size:min((batch + 1) * batch_size, len(train_y))]

            # xpts = np.random.rand(100) * 10
            # ypts = np.sin(xpts)

            session_time = time.time()

            _, loss_result = sess.run([train_op, loss],
                                      feed_dict={x: batch_x,
                                                 y: batch_y})

            fitnesses.append(loss_result)
            tempos.append(time.time() - session_time)
            #print('iteration {}, loss={}'.format(i, loss_result))
            #print("sessao demorou: " + str(time.time() - session_time))

        loss_result = sess.run(loss,
                               feed_dict={x: test_x,
                                          y: test_y})

        print(loss_result)
        validation_fitnesses.append(loss_result)
        tempos_validation.append(time.time() - start_time)
        #print('iteration {}, loss={}'.format(i, loss_result))
        #print("sessao demorou: " + str(time.time() - session_time))


    file_string = './debug/graphs_logs/gradient.pckl'
    with open(file_string, 'wb') as save_graph_file:
        save_graph = Graph(tempos, fitnesses, acuracias, tempos_validation, validation_fitnesses,
                           validation_acuracias, fine_tuning_graph)
        dill.dump(save_graph, save_graph_file)
        print('salvei em: ' + './debug/graphs_logs/gradient.pckl')