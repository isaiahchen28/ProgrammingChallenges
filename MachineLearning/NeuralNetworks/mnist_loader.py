'''
A library to load the MNIST image data. In practice, load_data_wrapper() is
the function usually called by our neural network code, as it is in a more
convenient format to use.
'''
import gzip
import pickle
import numpy as np


def load_data():
    '''
    Return the MNIST data as a tuple containing the training data, the
    validation data, and the test data.

    The training data is returned as a tuple with two entries. The first entry
    contains the actual training images - this is a NumPy array with 50,000
    entries. Each entry is, in turn, a NumPy array with 784 values,
    representing the 28 x 28 array of pixels in a single MNIST image.

    The second entry in the training data tuple is a NumPy array containing
    50,000 entries. Those entries are just the digit values (0, 1, ..., 9) for
    the corresponding images contained in the first entry of the tuple.

    The validation data and test data values are similar, except each contains
    only 10,000 images.
    '''
    f = gzip.open('mnist.pkl.gz', 'rb')
    training, validation, test = pickle.load(f, encoding="latin1")
    f.close()
    return (training, validation, test)


def load_data_wrapper():
    '''
    Return a tuple containing the training data, validation data, and
    test data. This format is more convenient for use in our implementation of
    neural networks.

    In particular, the training data is a list containing 50,000 2-tuples
    (x, y), where x is a 784-dimensional NumPy array containing the input
    image and y is a 10-dimensional NumPy array representing the unit vector
    corresponding to the correct digit for x.

    The validation data and test data are lists containing 10,000 2-tuples
    (x, y). In each case, x is a 784-dimensional NumPy array containing the
    input image and y is the corresponding classification, corresponding to x.
    '''
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    '''
    Return a 10-dimensional unit vector with a 1.0 in the jth position and
    zeroes elsewhere. This is used to convert a digit into a corresponding
    desired output from the neural network.
    '''
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
