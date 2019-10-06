"""
Isaiah Chen
EN.553.740 Machine Learning
Project 1: Kernel Prediction and Cross-Validation

Programs written using Python 2.7
Uses various data science modules from Anaconda
(https://www.anaconda.com/distribution/)

Question 2
"""
import warnings
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd


def warnings_off():
    warnings.warn("depreciated", DeprecationWarning)


with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings_off()


def K(x):
    """ Returns the value of the kernel of the Bayes estimator """
    d = x.ndim
    if d == 1:
        return np.exp(-1 * (abs(x) ** 2) / 2) / ((2.0 * np.pi) ** (d / 2))
    else:
        return np.exp(-1 * (np.linalg.norm(x) ** 2) / 2) / (
            (2.0 * np.pi) ** (d / 2))


def Kh(x, h):
    """ Returns the value of the kernel of the Bayes estimator """
    d = x.ndim
    return K(x / h) / (h ** d)


def compute_kernel_regression_estimator(x, X, Y, h):
    """ Computes the kernel regression estimator based on a training set """
    total1, total2 = 0, 0
    for k in range(len(X)):
        total1 += Y[k] * Kh(x - X[k], h)
        total2 += Kh(x - X[k], h)
    return total1 / total2


def k_fold_cross_validation(X, Y, h, k):
    """ Performs k-fold cross-validation for a given set of data """
    kf = KFold(k, True)
    kf.get_n_splits(X)
    errors = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        kre = []
        for count, i in enumerate(X_test):
            kre.append(compute_kernel_regression_estimator(
                X_test[i], X_train, Y_train, h))
        total = 0
        for i in range(len(X_test)):
            total += (Y_test[i] - kre[i]) ** 2
        errors.append(total / len(X_test))
    return np.sum(errors) / k


if __name__ == '__main__':

    # Read training set and testing set data files
    training_set = pd.read_csv('project1_S19_1_train.csv')
    X_train = np.column_stack((
        np.array(training_set.X1, dtype=np.int8), np.array(
            training_set.X2, dtype=np.int8)))
    Y_train = np.array(training_set.Y, dtype=np.int8)
    testing_set = pd.read_csv('project1_S19_1_test.csv')
    X_test = np.column_stack((
        np.array(testing_set.X1, dtype=np.int8), np.array(
            testing_set.X2, dtype=np.int8)))
    Y_test = np.array(testing_set.Y, dtype=np.int8)
    # Define array of h vales
    M = 1000
    k = 10
    h = np.linspace(0.01, 1, M, endpoint=True)
    ecv_list = []
    # Run k-fold cross-validation for all the values of h
    for count, i in enumerate(h):
        Ecv = k_fold_cross_validation(X_train, Y_train, i, k)
        ecv_list.append(Ecv)
    # Plot figure
    plt.figure()
    plt.plot(h, ecv_list, 'k-')
    plt.xlabel("h")
    plt.ylabel("$\epsilon_{cv}$")
    # Report minimum error and corresponding h value
    min_error = min(ecv_list)
    h0 = h[np.argmin(ecv_list)]
    print("Kernel Regression Estimator:\n")
    print("h0 = %.4f" % h0)
    print("minimum Ecv = %.4f" % min_error)
    # Retrain f_h0 on the whole training set
    kre = []
    for count, i in enumerate(X_test):
        kre.append(compute_kernel_regression_estimator(
            X_test[i], X_train, Y_train, h0))
    total = 0
    for i in range(len(Y_test)):
        total += (Y_test[i] - kre[i]) ** 2
    error = total / len(Y_test)
    print("error evaluated on the test set = %.4f\n" % error)
    plt.show()
