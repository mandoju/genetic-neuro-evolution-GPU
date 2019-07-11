import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def get_mnist_data():
    mnist = tf.keras.datasets.mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = np.array(train_X).reshape(len(train_X), 784)
    # Prepend the column of 1s for bias
    N, M = train_X.shape
    all_X = np.ones((N, M))
    all_X[:, :] = train_X

    num_labels = len(np.unique(train_y))
    train_y_eye = np.eye(num_labels)[train_y]  # One liner trick!
    test_y_eye = np.eye(num_labels)[test_y]  # One liner trick!
    # a,b,c,d = train_test_split(all_X, all_Y, test_size=0.00, random_state=0)
    # return (all_X, all_X, all_Y, all_Y)
    return train_X, train_y_eye, test_X, test_y_eye

def get_mnist_data_reshape():
    mnist = tf.keras.datasets.mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = np.array(train_X).reshape(len(train_X), 784)
    # Prepend the column of 1s for bias
    N, M = train_X.shape
    all_X = np.ones((N, M))
    all_X[:, :] = train_X

    num_labels = len(np.unique(train_y))
    train_y_eye = np.eye(num_labels)[train_y]  # One liner trick!
    test_y_eye = np.eye(num_labels)[test_y]  # One liner trick!
    # a,b,c,d = train_test_split(all_X, all_Y, test_size=0.00, random_state=0)
    # return (all_X, all_X, all_Y, all_Y)
    train_X = train_X.reshape(-1, 28, 28, 1)
    test_X = test_X.reshape(-1, 28, 28, 1)
    return train_X, train_y_eye, test_X, test_y_eye    

def get_sine_data():
    
    x = np.arange(100000).reshape(-1,1) / 1000
    y = np.sin(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return X_train , y_train , X_test , y_test