"""
Isaiah Chen
EN.553.740 Machine Learning
Project 2: Penalized Regression

Programs written using Python 2.7
Uses various data science modules from Anaconda
(https://www.anaconda.com/distribution/)
"""
import numpy as np
from scipy.linalg import solve_sylvester
from scipy.sparse import diags
import pandas as pd
import matplotlib.pyplot as plt


def ridge_regression(Y, X, D, L):
    """
    Takes into input a matrix Y containing each yk as row vectors, and X
    containing each xk as row vectors, the matrix D and lambda (L) > 0.
    Returns the estimated optimal parameters for beta0 and b.
    """
    # Define N, q, and d
    N = Y.shape[0]
    q = Y.shape[1]
    d = X.shape[1]
    # Calculate average of y1,..,yN
    total_Y = np.zeros(q)
    for i, j in enumerate(Y):
        total_Y += j
    y_bar = total_Y / N
    # Calculate average of x1,...,xN
    total_X = np.zeros(d)
    for i, j in enumerate(X):
        total_X += j
    x_bar = total_X / N
    # Calculate Yc matrix
    Yc = np.transpose(Y[0] - y_bar)
    for i in range(1, N):
        Yc = np.vstack((Yc, np.transpose(Y[i] - y_bar)))
    # Calculate Xc matrix
    Xc = np.transpose(X[0] - x_bar)
    for i in range(1, N):
        Xc = np.vstack((Xc, np.transpose(X[i] - x_bar)))
    # Calculate the optimal parameters for b
    b_opt = np.matmul((np.matmul((np.linalg.inv(np.matmul(np.transpose(
        Xc), Xc) + (L * D))), np.transpose(Xc))), Yc)
    # Calculate the optimal parameters for beta0
    beta_opt = y_bar - np.matmul(np.transpose(b_opt), x_bar)
    return beta_opt, b_opt


def prediction_error(Y, X, D, L, beta, b):
    """ Computes prediction error as given by F(beta0, b) """
    N = Y.shape[0]
    total = 0
    for k in range(N):
        total += np.linalg.norm(Y[k] - beta - np.matmul(
            np.transpose(b), X[k])) ** 2
    return total + (L * np.trace(np.matmul(np.matmul(np.transpose(b), D), b)))


def ridge_regression_modified(X, Y, L, D):
    """
    Computes the solution of the multivariate problem given in question 1.4.
    Takes as input X and Y arrays, the parameter lambda, and the penalty
    matrix D. Returns the optimal values for beta0 and b.
    """
    # Define N, d, and q
    N = X.shape[1]
    d = X.shape[0]
    q = Y.shape[0]
    # Convert X and Y arrays
    X = np.transpose(np.vstack((np.ones(N), X)))
    Y = np.transpose(Y)
    # Calculate the optimal parameters for b
    b_opt = solve_sylvester(np.matmul(
        np.transpose(X), X), L * D, np.matmul(np.transpose(X), Y))
    # Calculate average of x1,...,xN
    total_X = np.zeros(d + 1)
    for i, j in enumerate(X):
        total_X += j
    x_bar = total_X / N
    # Calculate average of y1,..,yN
    total_Y = np.zeros(q)
    for i, j in enumerate(Y):
        total_Y += j
    y_bar = total_Y / N
    # Calculate the optimal parameters for beta0
    beta_opt = y_bar - np.matmul(np.transpose(b_opt), x_bar)
    return beta_opt, b_opt


def gaussian_kernel(x, y, s):
    """ Computes the Gaussian kernel """
    return np.exp((-1 * (np.linalg.norm(x - y) ** 2)) / (2 * (s ** 2)))


def polynomial_kernel(x, y, h):
    """ Computes the polynomial kernel """
    total = 0
    A = np.inner(x, y)
    for i in range(1, h):
        total += A ** h
    return total


def kernel_ridge_regression(X, Y, kp, L, case):
    """ Computes kernel version of ridge regression for two cases """
    # Define N and q
    N = X.shape[0]
    q = Y.shape[1]
    # Compute kernel
    if case == "Gaussian":
        K = np.array([[gaussian_kernel(x1, x2, kp) for x1 in X] for x2 in X])
    elif case == "Polynomial":
        K = np.array([[polynomial_kernel(x1, x2, kp) for x1 in X] for x2 in X])
    else:
        return -1
    # Compute P matrix
    P = np.identity(N) - ((
        np.matmul(np.transpose(np.ones(N)), np.ones(N))) / N)
    # Calculate average of y1,..,yN
    total_Y = np.zeros(q)
    for i, j in enumerate(Y):
        total_Y += j
    y_bar = total_Y / N
    # Calculate Yc matrix
    Yc = np.transpose(Y[0] - y_bar)
    for i in range(1, N):
        Yc = np.vstack((Yc, np.transpose(Y[i] - y_bar)))
    # Use Cholesky decomposition to calculate the optimal values for alpha
    alpha_opt = np.matmul(np.linalg.pinv(
        np.matmul(P, K) + (L * np.identity(N))), Yc)
    # Calculate the optimal values for beta0
    beta_opt = y_bar - np.matmul(np.matmul(np.transpose(
        alpha_opt), K), np.transpose(np.ones(N)) / N)
    return beta_opt, alpha_opt


def kernel_prediction_error(X, Y, kp, beta, alpha, case):
    """ Computes prediction error using the kernel """
    # Define N and M
    N = len(alpha)
    M = X.shape[0]
    # Use beta, alpha, and kernel to calculate the predicted values of y
    y_pred = []
    for j in range(M):
        total = 0
        for k in range(N):
            if case == "Gaussian":
                total += alpha[k] * gaussian_kernel(X[j], X[k], kp)
            elif case == "Polynomial":
                total += alpha[k] * polynomial_kernel(X[j], X[k], kp)
            else:
                break
        y_pred.append(total + beta)
    # Compute the error between the predicted value and the actual value
    total = 0
    for j in range(M):
        total += np.linalg.norm(Y[j] - y_pred[j]) ** 2
    return total


if __name__ == "__main__":

    """ Question 1.3 """

    print("--- Question 1.3 ---\n")
    # Read in training data
    train = pd.read_csv('project2_S2019_Q1Train.csv')
    X = np.transpose([train.x1, train.x2, train.x3, train.x4])
    Y = np.transpose([train.y1, train.y2, train.y3, train.y4])
    # Define positive semi-definite symmetric matrix D
    d = X.shape[1]
    D = np.identity(d)
    # Define values of lambda
    L = np.arange(0.01, 2.01, 0.01)
    # Initialize lists for parameters
    beta_opt_m, b_opt_m = [], []
    # For each value of lambda, calculate the optimal beta0 and b
    for i in range(L.size):
        beta_opt, b_opt = ridge_regression(Y, X, D, i)
        beta_opt_m.append(beta_opt)
        b_opt_m.append(b_opt)
    # Read in testing data
    test = pd.read_csv('project2_S2019_Q1Test.csv')
    X_test = np.transpose([test.x1, test.x2, test.x3, test.x4])
    Y_test = np.transpose([test.y1, test.y2, test.y3, test.y4])
    # Initialize list for prediction error
    errors = []
    # For each value of lambda, compute prediction error
    for i, j in enumerate(L):
        errors.append(prediction_error(
            Y_test, X_test, D, j, beta_opt_m[i], b_opt_m[i]))
    # Plot figure of error as a function of lambda
    plt.figure()
    plt.plot(L, errors)
    plt.xlabel("$\lambda$")
    plt.ylabel("Prediction Error")
    # Retrieve beta0 and b for lambda = 1
    index_1 = np.where(L == 1.0)[0]
    print("For lambda = 1, the optimal parameters are calculated.\n")
    print("Optimal parameters for beta0:")
    print(beta_opt_m[index_1[0]])
    print("")
    print("Optimal parameters for b:")
    print(b_opt_m[index_1[0]])

    """ Question 1.5 """

    # Read in dataset
    Xdata = np.transpose(pd.read_csv(
        'project2_S2019_Q1.2.csv', usecols=range(0, 3)).values.tolist())
    Ydata = np.transpose(pd.read_csv(
        'project2_S2019_Q1.2.csv', usecols=range(3, 203)).values.tolist())
    # Define D for Case 1
    q = Ydata.shape[0]
    D1 = np.identity(q)
    # Calculate optimal beta0 and b for Case 1
    beta_opt_1, b_opt_1 = ridge_regression_modified(Xdata, Ydata, 10, D1)
    j = range(1, 201)
    plt.figure()
    plt.plot(j, beta_opt_1, '-', j, b_opt_1[0], '--',
             j, b_opt_1[1], '-.', j, b_opt_1[2], ':')
    plt.xlabel("j")
    plt.legend([
        "$\\beta_{0}(j)$",
        "$b(1, j)$",
        "$b(2, j)$",
        "$b(3, j)$"],
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=4)
    # Define D for Case 2
    two_diag = 2 * np.ones(q)
    one_diag = -1 * np.ones(q - 1)
    one_and_two = np.array([two_diag, one_diag, one_diag])
    D2 = diags(one_and_two, [0, -1, 1]).toarray()
    D2[0, 0] = 1
    D2[q - 1, q - 1] = 1
    # Calculate optimal beta0 and b for Case 2
    beta_opt_2, b_opt_2 = ridge_regression_modified(Xdata, Ydata, 1000, D2)
    plt.figure()
    plt.plot(j, beta_opt_2, '-', j, b_opt_2[0], '--',
             j, b_opt_2[1], '-.', j, b_opt_2[2], ':')
    plt.xlabel("j")
    plt.legend([
        "$\\beta_{0}(j)$",
        "$b(1, j)$",
        "$b(2, j)$",
        "$b(3, j)$"],
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=4)

    """ Question 2.1 """

    print("\n--- Question 2.1 ---\n")
    # Read in training data
    train2 = pd.read_csv('project2_S2019_Q2Train.csv')
    X2 = np.transpose([train2.X1, train2.X2, train2.X3, train2.X4, train2.X5,
                      train2.X6, train2.X7, train2.X8, train2.X9, train2.X10])
    Y2 = np.transpose([train2.Y])
    # Define positive semi-definite symmetric matrix D
    d2 = X2.shape[1]
    d_rand_2 = np.tril(np.random.rand(d2, d2))
    D22 = np.matmul(d_rand_2, np.transpose(d_rand_2))
    # Define values of lambda
    L2 = np.arange(1, 101, 1)
    # Initialize lists for parameters
    beta_opt_m2, b_opt_m2 = [], []
    # For each value of lambda, calculate the optimal beta0 and b
    for i in range(L2.size):
        beta_opt, b_opt = ridge_regression(Y2, X2, D22, i)
        beta_opt_m2.append(beta_opt)
        b_opt_m2.append(b_opt)
    # Read in testing data
    test2 = pd.read_csv('project2_S2019_Q2Test.csv')
    X_test2 = np.transpose([test2.X1, test2.X2, test2.X3, test2.X4, test2.X5,
                           test2.X6, test2.X7, test2.X8, test2.X9, test2.X10])
    Y_test2 = np.transpose([test2.Y])
    # Initialize list for prediction error
    errors2 = []
    # For each value of lambda, compute prediction error
    for i, j in enumerate(L2):
        errors2.append(prediction_error(
            Y_test2, X_test2, D22, j, beta_opt_m2[i], b_opt_m2[i]))
    # Plot figure of error as a function of lambda
    plt.figure()
    plt.plot(L2, errors2)
    plt.xlabel("$\lambda$")
    plt.ylabel("Prediction Error")
    # Return minimum error and corresponding value of lambda
    min_err = min(errors2)
    Lmin = L2[np.argmin(errors2)]
    print("Minimum prediction error = %d, where lambda = %d" % (min_err, Lmin))

    """ Question 2.2 """

    print("\n--- Question 2.2 ---")
    print("(may take approximately 12 minutes to solve and plot figures)\n")
    # Define new lambda range for the Gaussian kernel
    L3 = np.arange(0.001, 0.1001, 0.001)
    # Initialize lists for parameters for the Gaussian case
    beta_opt_mg, alpha_opt_mg = [], []
    s = 2.5
    # For each value of lambda, calculate the optimal beta0 and alpha using
    # the Gaussian kernel
    for i in range(L3.size):
        beta_opt, alpha_opt = kernel_ridge_regression(X2, Y2, s, i, "Gaussian")
        beta_opt_mg.append(beta_opt)
        alpha_opt_mg.append(alpha_opt)
    # Initialize lists for parameters for the polynomial case
    beta_opt_fin, alpha_opt_fin = [], []
    h = [1, 2, 3, 4]
    # For each value of lambda and h, calculate the optimal beta0 and alpha
    # using the polynomial kernel
    for i, j in enumerate(h):
        beta_opt_mp, alpha_opt_mp = [], []
        for k in range(L2.size):
            beta_opt, alpha_opt = kernel_ridge_regression(
                X2, Y2, j, k, "Polynomial")
            beta_opt_mp.append(beta_opt)
            alpha_opt_mp.append(alpha_opt)
        beta_opt_fin.append(beta_opt_mp)
        alpha_opt_fin.append(alpha_opt_mp)
    # Initialize list for prediction error
    errors3 = []
    # For each value of lambda, compute prediction error
    for i, j in enumerate(L3):
        errors3.append(kernel_prediction_error(
            X_test2, Y_test2, s, beta_opt_mg[i], alpha_opt_mg[i], "Gaussian"))
    # Plot figure of error as a function of lambda
    plt.figure()
    plt.plot(L3, errors3)
    plt.xlabel("$\lambda$")
    plt.ylabel("Prediction Error")
    # Return minimum error and corresponding value of lambda
    min_err_g = min(errors3)
    Lmin_g = L3[np.argmin(errors3)]
    print("For the Gaussian case,")
    print("minimum prediction error = %d, where lambda = %0.3f\n" % (
        min_err_g, Lmin_g))
    # For each value of lambda and h, compute prediction error
    for i, j in enumerate(h):
        errors4 = []
        for k, l in enumerate(L2):
            errors4.append(kernel_prediction_error(
                X_test2, Y_test2, j,
                beta_opt_fin[i][k], alpha_opt_fin[i][k], "Polynomial"))
        # Plot figure of error as a function of lambda
        plt.figure()
        plt.plot(L2, errors4)
        plt.xlabel("$\lambda$")
        plt.ylabel("Prediction Error")
        # Return minimum error and corresponding value of lambda
        min_err_p = min(errors4)
        Lmin_p = L2[np.argmin(errors4)]
        print("For the polynomial case where h = %d," % (j))
        print("minimum prediction error = %d, where lambda = %0.3f\n" % (
            min_err_p, Lmin_p))
    plt.show()
