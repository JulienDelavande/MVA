from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys
import warnings


def data_mnist(datadir='/tmp/', train_start=0, train_end=60000, test_start=0,
               test_end=10000):
    """
    Load and preprocess MNIST dataset
    :param datadir: path to folder where data should be stored
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :return: tuple of four arrays containing training data, training labels,
             testing data and testing labels.
    """
    assert isinstance(train_start, int)
    assert isinstance(train_end, int)
    assert isinstance(test_start, int)
    assert isinstance(test_end, int)
    ''' old
    from tensorflow.examples.tutorials.mnist import input_data
    
    mnist = input_data.read_data_sets(datadir, one_hot=True, reshape=False)
    X_train = np.vstack((mnist.train.images, mnist.validation.images))
    Y_train = np.vstack((mnist.train.labels, mnist.validation.labels))
    X_test = mnist.test.images
    Y_test = mnist.test.labels

    X_train = X_train[train_start:train_end]
    Y_train = Y_train[train_start:train_end]
    X_test = X_test[test_start:test_end]
    Y_test = Y_test[test_start:test_end]'''

    from keras.datasets import fashion_mnist

    # Load data from keras
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

    # Normalize data to [0, 1] range
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Add channel dimension (28x28 -> 28x28x1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # One-hot encode labels
    from keras.utils import to_categorical
    Y_train = to_categorical(Y_train, 10)
    Y_test = to_categorical(Y_test, 10)

    # Apply slicing
    X_train = X_train[train_start:train_end]
    Y_train = Y_train[train_start:train_end]
    X_test = X_test[test_start:test_end]
    Y_test = Y_test[test_start:test_end]

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    return X_train, Y_train, X_test, Y_test

