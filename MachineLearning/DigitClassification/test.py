import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    fig = plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(X_train[i], cmap='gray', interpolation='none')
        plt.title("Digit: {}".format(y_train[i]))
        plt.xticks([])
        plt.yticks([])
    fig
