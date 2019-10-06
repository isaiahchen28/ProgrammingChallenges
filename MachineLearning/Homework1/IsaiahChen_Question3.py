"""
Isaiah Chen
EN.553.740 Machine Learning
Project 1: Kernel Prediction and Cross-Validation

Programs written using Python 2.7
Uses various data science modules from Anaconda
(https://www.anaconda.com/distribution/)

Question 3
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


def phi_h(x, X, Y, h, g):
    total, count = 0, 0
    for k in range(len(Y)):
        if Y[k] == g:
            total += Kh(x - X[k], h)
            count += 1
    return total / count


def pi_h(x, X, Y, h):
    total = 0
    for g in [0, 1]:
        total += phi_h(x, X, Y, h, g)
    return phi_h(x, X, Y, h, g) / total


def compute_kernel_classification_estimator(U, X, Y, h):
    MAP, post = [], []
    for i in range(len(U)):
        post.append(pi_h(U[i], X, Y, h))
        MAP.append(np.argmax(pi_h(U[i], X, Y, h)))
    return MAP, post


def k_fold_cross_validation(X, Y, h, k):
    kf = KFold(k, True)
    kf.get_n_splits(X)
    errors = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        kce, _ = compute_kernel_classification_estimator(
            X_test, X_train, Y_train, h)
        total = 0
        for i in range(len(Y_test)):
            if Y_test[i] != kce[i]:
                total += 1
        errors.append(total / len(X_test))
    return np.sum(errors) / k


def k_fold_cross_validation2(X, Y, h, k):
    kf = KFold(k, True)
    kf.get_n_splits(X)
    errors = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train = Y[train_index]
        kce, post_dist = compute_kernel_classification_estimator(
            X_test, X_train, Y_train, h)
        total = 0
        for i in range(len(post_dist)):
            total += np.log10(post_dist)
        errors.append(-total / np.linalg.norm(X_train))
    return np.sum(errors) / k


if __name__ == '__main__':

    # Read training set and testing set data files
    training_set = pd.read_csv('project1_S19_2_train.csv')
    X_train = np.column_stack((
        np.array(training_set.X1), np.array(training_set.X2)))
    Y_train = np.array(training_set.Y)
    testing_set = pd.read_csv('project1_S19_2_test.csv')
    X_test = np.column_stack((
        np.array(testing_set.X1), np.array(testing_set.X2)))
    Y_test = np.array(testing_set.Y)
    ecv = []
    # Run k-fold cross-validation for all the values of h
    M = 1000
    k = 10
    h = np.linspace(0.01, 1, M, endpoint=True)
    for count, i in enumerate(h):
        ecv.append(k_fold_cross_validation(X_train, Y_train, i, k))
    # Plot figure
    plt.figure()
    plt.plot(h, ecv, 'k-')
    plt.xlabel("h")
    plt.ylabel("$\epsilon_{cv}$")
    # Report minimum error and corresponding h value
    min_error = min(ecv)
    h1 = h[np.argmin(ecv)]
    print("Kernel Classification Estimator (Part 1):")
    print("h1 = %.4f" % h1)
    print("minimum Ecv = %.4f" % min_error)
    # Retrain f_h0 on the whole training set
    kce, _ = compute_kernel_classification_estimator(
        X_train, X_train, Y_train, h1)
    total_err = 0
    for i in range(len(Y_train)):
        if Y_train[i] != kce[i]:
            total_err += 1
    error = total_err / len(Y_train)
    print("error evaluated on the training set = %.4f" % np.mean(error))
    kce, _ = compute_kernel_classification_estimator(
        X_test, X_train, Y_train, h1)
    total_err = 0
    for i in range(len(Y_test)):
        if Y_test[i] != kce[i]:
            total_err += 1
    error = total_err / len(Y_test)
    print("error evaluated on the test set = %.4f\n" % np.mean(error))
    # Modify cross-validation and repeat
    ecv = []
    for count, i in enumerate(h):
        ecv.append(k_fold_cross_validation2(X_train, Y_train, i, k))
    # Plot figure
    plt.figure()
    plt.plot(h, ecv, 'k-')
    plt.xlabel("h")
    plt.ylabel("$\epsilon_{cv}$")
    # Report minimum error and corresponding h value
    min_error = min(ecv)
    h2 = h[np.argmin(ecv)]
    print("Kernel Classification Estimator (Part 2):")
    print("h2 = %.4f" % h2)
    print("minimum Ecv = %.4f" % min_error)
    # Retrain f_h0 on the whole training set
    kce, _ = compute_kernel_classification_estimator(
        X_train, X_train, Y_train, h2)
    total_err = 0
    for i in range(len(Y_train)):
        if Y_train[i] != kce[i]:
            total_err += 1
    error = total_err / len(Y_train)
    print("error evaluated on the training set = %.4f" % np.mean(error))
    kce, _ = compute_kernel_classification_estimator(
        X_test, X_train, Y_train, h2)
    total_err = 0
    for i in range(len(Y_test)):
        if Y_test[i] != kce[i]:
            total_err += 1
    error = total_err / len(Y_test)
    print("error evaluated on the test set = %.4f\n" % np.mean(error))
    plt.show()
