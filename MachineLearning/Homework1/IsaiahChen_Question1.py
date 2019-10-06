"""
Isaiah Chen
EN.553.740 Machine Learning
Project 1: Kernel Prediction and Cross-Validation

Programs written using Python 2.7
Uses various data science modules from Anaconda
(https://www.anaconda.com/distribution/)

Question 1
"""
import warnings
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def warnings_off():
    warnings.warn("depreciated", DeprecationWarning)


with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings_off()


def f(x):
    """ Returns the mean of the given function """
    return 2 * (x ** 3) - 0.5 * x + 1


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


def e1(Xn, Yn, h):
    """
    Computes error regarding the kernel regression estimator and the
    training data
    """
    total = 0
    for k in range(len(Yn)):
        total += (Yn[k] - compute_kernel_regression_estimator(
            Xn[k], Xn, Yn, h)) ** 2
    return total / len(Yn)


def e2(U, X, Y, h):
    """
    Computes error regarding the kernel regression estimator of the
    testing data and the mean function
    """
    total = 0
    for j in range(len(U)):
        total += (f(U[j]) - compute_kernel_regression_estimator(
            U[j], X, Y, h)) ** 2
    return total / len(U)


def e3(X, Y, h):
    """
    Computes error regarding the kernel regression estimator of the
    testing data
    """
    total = 0
    for j in range(len(Y)):
        total += (Y[j] - compute_kernel_regression_estimator(
            X[j], X, Y, h)) ** 2
    return total / len(Y)


def k_fold_cross_validation(X, Y, h, k):
    """ Performs k-fold cross-validation for a given set of data """
    kf = KFold(k, True)
    kf.get_n_splits(X)
    errors = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        kre = compute_kernel_regression_estimator(X_test, X_train, Y_train, h)
        total = 0
        for i in range(len(X_test)):
            total += (Y_test[i] - kre[i]) ** 2
        errors.append(total / len(X_test))
    return np.sum(errors) / k


if __name__ == '__main__':

    # Generate M-sample to be used as the testing data
    M = 1000
    X = np.array(np.random.uniform(-1, 1, M))
    Y = np.array(np.random.normal(f(X), 1, M))
    # Generate deterministic vector U
    U = np.linspace(-1, 1, M, endpoint=True)
    # Define N and h values of interest, as well as the dimension and the
    # desired number of folds
    N = [100, 250]
    h = [0.05, 0.1, 0.25, 0.5]
    k = 10
    # Initialize empty lists for calculating error values
    e1_list, e2_list, e3_list, ecv_list = [], [], [], []
    for count1, n in enumerate(N):
        # For N = 100 and N = 250, generate the N-sample of training set data,
        # which will be used to determine the kernel regression estimator
        Xn = np.random.choice(X, n, replace=False)
        Yn = np.random.choice(Y, n, replace=False)
        for count2, i in enumerate(h):
            # For each value of h, compute the kernel regression estimator
            # and make figures for plotting the functions of interest
            kre = compute_kernel_regression_estimator(U, Xn, Yn, i)
            plt.figure()
            plt.plot(U, f(U), 'b-', U, kre, 'r--', Xn, Yn, 'k.')
            plt.title("(N, h) = (%d, %.2f)" % (n, i))
            plt.legend([
                "f(U)",
                "$\hat{f}_{h}(U)$",
                "$(x_{k}, y_{k})$"],
                loc='upper left')
            # Evaluate errors for each case
            e1_list.append(e1(Xn, Yn, i))
            e2_list.append(e2(U, X, Y, i))
            e3_list.append(e3(X, Y, i))
            # Calulate errors using 10-fold cross-validation
            ecv_list.append(k_fold_cross_validation(Xn, Yn, i, k))
    # Output values and figures for Table 1
    N_print = [100, 100, 100, 100, 250, 250, 250, 250]
    h_print = [0.05, 0.1, 0.25, 0.5, 0.05, 0.1, 0.25, 0.5]
    print(" N    h    e1     e2     e3     Ecv")
    for i in range(len(N_print)):
        print("%d %.2f %.4f %.4f %.4f %.4f" % (
            N_print[i],
            h_print[i],
            e1_list[i],
            e2_list[i],
            e3_list[i],
            ecv_list[i]))
    print("\n")
    plt.show()
