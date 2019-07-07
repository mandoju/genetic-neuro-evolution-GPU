# Import libraries
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#from test_packs import get_gradient_convolution, get_gradient_biases
from datasets.datasets import get_mnist_data_reshape, get_sine_data

#from graph import Graph
import time
import pickle
import sys


# import matplotlib.pyplot as plt

train_x, train_y, test_x, test_y = get_sine_data()
#train_x = train_x.reshape(-1, 28, 28, 1)
#test_x = test_x.reshape(-1, 28, 28, 1)
training_iters = 20
learning_rate = 0.0001
batch_size = 1000
n_classes = 10
x_size = train_x.shape[1]
y_size = train_y.shape[1]  # Number of outcomes (3 iris flowers)
# Symbols
X = tf.placeholder("float", shape=[None, 28, 28, 1], name="X")
y = tf.placeholder("float", shape=[None, y_size], name="Y")


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(x, weights, biases):
    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.

    convs = []
    convs.append(conv1)
    for i in range(len(weights.keys()) - 3):
        # conv = conv2d(convs[i], weights['wc' + str(i+2)], biases['bc' + str(i+2)])
        # conv = maxpool2d(conv, k=2)
        convs.append(maxpool2d(conv2d(convs[i], weights['wc' + str(i + 2)], biases['bc' + str(i + 2)]), k=2))
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    last_conv = convs.pop()

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    # conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    # conv2 = maxpool2d(conv2, k=2)

    # conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    # conv3 = maxpool2d(conv3, k=2)

    # conv7 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    # conv7 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(last_conv, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

if (len(sys.argv) > 1):
    weights = get_gradient_convolution(sys.argv[1])
    biases = get_gradient_biases(sys.argv[1])
else:
    weights = {

        'wc1': tf.get_variable('W0', shape=(3, 3, 1, 32), initializer=tf.keras.initializers.he_normal()),
        'wc2': tf.get_variable('W1', shape=(3, 3, 32, 64), initializer=tf.keras.initializers.he_normal()),
        'wc3': tf.get_variable('W2', shape=(3, 3, 64, 128), initializer=tf.keras.initializers.he_normal()),

        'wd1': tf.get_variable('Wd1', shape=(4 * 4 * 128, 128), initializer=tf.keras.initializers.he_normal()),
        'out': tf.get_variable('Wout', shape=(128, 10), initializer=tf.keras.initializers.he_normal()),
    }
    biases = {
        'bc1': tf.get_variable('B0', shape=(32), initializer=tf.keras.initializers.he_normal()),
        'bc2': tf.get_variable('B1', shape=(64), initializer=tf.keras.initializers.he_normal()),
        'bc3': tf.get_variable('B2', shape=(128), initializer=tf.keras.initializers.he_normal()),

        'bd1': tf.get_variable('Bd1', shape=(128), initializer=tf.keras.initializers.he_normal()),
        'out': tf.get_variable('Bout', shape=(10), initializer=tf.keras.initializers.he_normal()),
    }
# pred = conv_net(X, weights, biases)
# print(pred)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

pred = conv_net(X, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# calculate accuracy across all the given images and average them out.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Initializing the variables
init = tf.global_variables_initializer()

""" with tf.Session() as sess:
    sess.run(init) 
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    tempos = []
    #fig, ax = plt.subplots()
    #summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    iter_time = time.time()
    for i in range(training_iters):
        for batch in range(len(train_x)//batch_size):

            batch_x = train_x[batch*batch_size:min((batch+1)*batch_size,len(train_x))]
            batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]    
            # Run optimization op (backprop).
            # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={X: batch_x,
                                                              y: batch_y})
            predict,loss, acc = sess.run([pred,cost, accuracy], feed_dict={X: batch_x,
                                                              y: batch_y})
        #print("predict = ")

        # Calculate accuracy for all 10000 mnist test images
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={X: test_x,y : test_y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        time_passed = time.time() - iter_time
        tempos.append(time_passed)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc))
        print("Iter " + str(i) + ", Loss= " + \
                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    "{:.5f}".format(acc))
        time_passed = time.time() - iter_time
        print("tempo atual: " + str(time_passed) )
        print("Batch Finished!")

        # summary_writer.close()
        # plt.plot(tempos, test_accuracy, '-', lw=2)
    file_string = ''
    if(len(sys.argv) > 1):
        file_string = './graphs/gradient_' + sys.argv[1] + '.pckl'
    else:
        file_string = './graphs/gradient_10.pckl'
    with open(file_string, 'wb') as save_graph_file:
        save_graph = Graph(tempos,test_loss)
        pickle.dump(save_graph,save_graph_file)
        print('salvei em: ./graphs/gradient.pckl')
    # plt.grid(True)
    # plt.show()
 """
time_to_stop = 291
time_begin = time.time()
with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    tempos = []
    tempos_validation = []
    iter_time = time.time()

    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    # for i in range(training_iters):
    i = 0
    while (time.time() - time_begin < time_to_stop):
        for batch in range((len(train_x) // batch_size) - 1):
            print(batch)
            batch_x = train_x[batch * batch_size:min((batch + 1) * batch_size, len(train_x))]
            batch_y = train_y[batch * batch_size:min((batch + 1) * batch_size, len(train_y))]
            # Run optimization op (backprop).
            # Calculate batch loss and accuracy
            session_time = time.time()
            opt = sess.run(optimizer, feed_dict={X: batch_x,
                                                 y: batch_y})
            print("sessao demorou:" + str(time.time() - session_time))

            loss, acc = sess.run([cost, accuracy], feed_dict={X: batch_x,
                                                              y: batch_y})
            train_loss.append(loss)
            train_accuracy.append(acc)
            time_passed = time.time() - iter_time
            tempos.append(time_passed)
            if (time.time() - time_begin < time_to_stop):
                break
        print("Iter " + str(i) + ", Loss= " + \
              "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        batch = (len(train_x) // batch_size) - 1
        validate_x = train_x[batch * batch_size:min((batch + 1) * batch_size, len(train_x))]
        validate_y = train_y[batch * batch_size:min((batch + 1) * batch_size, len(train_y))]
        test_acc, valid_loss = sess.run([accuracy, cost], feed_dict={X: validate_x, y: validate_y})
        # train_loss.append(loss)
        time_passed = time.time() - iter_time
        tempos_validation.append(time_passed)
        test_loss.append(valid_loss)
        # train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:", "{:.5f}".format(test_acc))
        print("ficamos com:", time.time() - iter_time)
        print(time.time())

        i += 1
    summary_writer.close()
    file_string = ''
    if (len(sys.argv) > 1):
        file_string = './graphs/gradient_' + sys.argv[1] + '.pckl'
    else:
        file_string = './graphs/gradient_10.pckl'
    with open(file_string, 'wb') as save_graph_file:
        save_graph = Graph(tempos, train_loss, train_accuracy, tempos_validation, test_loss, test_accuracy)
        pickle.dump(save_graph, save_graph_file)
        print('salvei em: ./graphs/gradient.pckl')
    # plt.grid(True)
    # plt.show()
