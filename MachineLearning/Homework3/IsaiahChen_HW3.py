"""
Isaiah Chen
EN.553.740 Machine Learning
Project 3: Optimal Scoring and Lasso

Programs written using Python 2.7
Uses various data science modules from Anaconda
(make sure pandas 0.24.2 is installed)
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt


def F_lambda(theta, b, q, d, r, M, C, sigma_xx, L):
    """
    Function to be minimzed as defined in Question 2
    """
    theta = np.reshape(theta, (q, r))
    b = np.reshape(b, (d, r))
    total = 0.0
    for k in range(d):
        for l in range(r):
            total += np.linalg.norm(b[k][l])
    t1 = -2.0 * np.trace(
        np.matmul(np.matmul(np.matmul(b.T, M.T), C), theta))
    t2 = np.trace(np.matmul(np.matmul(b.T, sigma_xx), b))
    return t1 + t2 + L * total


def constraint_1(theta, q, r, C):
    """
    theta^T * C * theta = Id
    """
    theta = np.reshape(theta, (q, r))
    c = np.matmul(np.matmul(theta.T, C), theta) - np.identity(r)
    return c.flatten()


def constraint_2(theta, q, r, C):
    """
    theta^T * C * ones = 0
    """
    theta = np.reshape(theta, (q, r))
    c = np.matmul(np.matmul(theta.T, C), np.ones((q, r)))
    return c.flatten()


def minimization(X, Y, r, L, max_iter):
    """
    This function returns the optimal b and theta such that F_lambda is
    minimized.
    """
    # Define the number of data points and dimension
    N = float(X.shape[0])
    d = X.shape[1]
    # Calculate the number of classes, q
    classes, q = [], 0
    for i in Y:
        if i not in classes:
            classes.append(i)
            q += 1
    # Calculate averages and M matrix
    classes_X = [[] for i in range(q)]
    for i, h in enumerate(classes):
        for j, k in enumerate(Y):
            if k == i:
                classes_X[i].append(X[j])
    mu_g = []
    for i, j in enumerate(classes_X):
        mu_g.append(np.mean(j, axis=0))
    mu = np.mean(X, axis=0)
    M = np.zeros((q, d))
    for i in range(q):
        M[i] = mu_g[i] - mu
    # Calculate sigma_xx
    Xc = np.transpose(X[0] - mu)
    for i in range(1, int(N)):
        Xc = np.vstack((Xc, np.transpose(X[i] - mu)))
    sigma_xx = np.matmul(Xc.T, Xc) / N
    # Calculate the size of each class
    Ng = []
    for i in classes:
        Ng.append((Y == i).sum())
    # Calculate the C matrix
    C = np.zeros((q, q))
    for i in range(q):
        C[i, i] = Ng[i] / N
    # Calculate C^(-1/2)
    C12 = np.zeros((q, q))
    for i in range(q):
        C12[i, i] = C[i, i] ** -0.5
    # Calculate initial theta
    Id = np.identity(r)
    z = np.zeros((q - r, r))
    theta = np.matmul(C12, np.vstack((Id, z)))
    # Initialize flags and error tolerance
    CONVERGED = False
    MAXITER_REACHED = False
    eps = 1e-06
    # Define constraints
    cons = ({'type': 'eq', 'fun': lambda x: constraint_1(x, q, r, C)},
            {'type': 'eq', 'fun': lambda x: constraint_2(x, q, r, C)})
    # Initialize error
    b = np.random.random((d, r))
    error = F_lambda(theta, b, q, d, r, M, C, sigma_xx, L)
    cost = []
    # Initialize counters for individual algorithm iterations
    b_it, theta_it = 0, 0
    # Run minimization algorithm
    for i in range(max_iter):
        # Save value of error from the previous iteration
        prev_error = error
        if i % 2 == 0:
            # Minimize b with respect to fixed theta using BFGS
            b_flat = b.flatten()
            b_min = minimize(lambda x: F_lambda(
                theta, x, q, d, r, M, C, sigma_xx, L), b_flat, method='BFGS')
            b = np.reshape(b_min.x, (d, r))
            b_it += b_min.nit
            # Check if rank(Mb) >= r
            test = np.linalg.matrix_rank(np.matmul(M, b))
            if test < r:
                raise Exception("Error: rank(Mb) < r")
        else:
            # Minimize theta with respect to fixed b using SLSQP
            theta_flat = theta.flatten()
            theta_min = minimize(lambda x: F_lambda(
                x, b, q, d, r, M, C, sigma_xx, L), theta_flat, method='SLSQP',
                constraints=cons)
            theta = np.reshape(theta_min.x, (q, r))
            theta_it += theta_min.nit
        # Check if function has been minimized
        error = F_lambda(theta, b, q, d, r, M, C, sigma_xx, L)
        cost.append(error)
        CONVERGED = abs(error - prev_error) < eps
        if CONVERGED:
            break
    # Check if the maximum number of iterations has been reached
    MAXITER_REACHED = i == max_iter - 1
    if CONVERGED:
        print("Overall algorithm iterations: %d" % i)
        print("Combined number of iterations for BFGS: %d" % b_it)
        print("Combined number of iterations for SLSQP: %d" % theta_it)
        print("Total number of iterations: %d" % (b_it + theta_it))
    elif MAXITER_REACHED:
        print("The maximum number of iterations has been reached.")
    it = np.arange(0, i + 1)
    return b, theta, mu, it, cost


def classification(X, Y, b, theta):
    """
    Use parameters to predict what class each data point belongs to and
    calculate the prediction error
    """
    # Predict the value of Y for each data point
    Y_pred = []
    for i, j in enumerate(X):
        Y_pred.append(np.argmax(np.matmul(np.matmul(j.T, b), theta.T)))
    # Calculate the fraction of incorrectly-classified samples
    return accuracy_score(Y.flatten(), Y_pred)


if __name__ == "__main__":

    """ Question 3.1 """
    print("-- Question 3.1 --\n")
    # Read training data for part 1
    train = pd.read_csv("Train_project3_Q3.csv")
    Xtrain = np.transpose([train.x1, train.x2, train.x3, train.x4, train.x5])
    Ytrain = np.transpose([train.Y])
    # Normalize the X data
    stdev_train = np.std(Xtrain, axis=0)
    Xtrain_norm = Xtrain / stdev_train
    # Define lambda
    N = float(Xtrain_norm.shape[0])
    Lambda = 5.0 / np.sqrt(N)
    # Run minimization algorithm for training data
    b, theta, mu, _, _ = minimization(Xtrain_norm, Ytrain,
                                      r=1, L=Lambda, max_iter=1000000)
    # Unnormalize results
    for i, j in enumerate(b):
        b[i] = j / stdev_train[i]
    # Use parameters to classify X values and calculate accuracy/error
    train_error = classification(Xtrain, Ytrain, b, theta)
    # Read test data for part 1
    test = pd.read_csv("Test_project3_Q3.csv")
    Xtest = np.transpose([test.x1, test.x2, test.x3, test.x4, test.x5])
    Ytest = np.transpose([test.Y])
    # Classify test data and calculate error
    test_error = classification(Xtest, Ytest, b, theta)
    # Print results
    print("Training error: %0.3f" % (1 - train_error))
    print("Testing error: %0.3f" % (1 - test_error))
    print("Optimal b:")
    for i in b:
        print(i)
    print("\n")

    """ Question 3.2 """
    print("-- Question 3.2 --\n")
    # Read training data for part 2
    train2 = pd.read_csv("Train_project3_Q3_2.csv").to_numpy()
    d = train2.shape[1] - 1
    Xtrain2 = np.delete(train2, -1, axis=1)
    Ytrain2 = np.delete(train2, np.arange(0, d), axis=1)
    # Normalize the X data
    stdev_train2 = np.std(Xtrain2, axis=0)
    Xtrain_norm2 = Xtrain2 / stdev_train2
    # Read test data for part 2
    test2 = pd.read_csv("Test_project3_Q3_2.csv").to_numpy()
    Xtest2 = np.delete(test2, -1, axis=1)
    Ytest2 = np.delete(test2, np.arange(0, d), axis=1)
    # Define lambda values
    N = float(Xtrain_norm2.shape[0])
    l_values = [0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 30.0]
    Lambda2 = l_values / np.sqrt(N)
    # For each value of lambda, run the minimization algorithm
    for i, j in enumerate(Lambda2):
        print("For sqrt(N) * Lambda = %2.1f:" % l_values[i])
        b, theta, mu, it, cost = minimization(Xtrain_norm2, Ytrain2,
                                              r=2, L=j, max_iter=1000000)
        # Unnormalize results
        for k, l in enumerate(b):
            b[k] = l / stdev_train2[k]
        # Compute rates of correct classification
        e1 = classification(Xtrain2, Ytrain2, b, theta)
        e2 = classification(Xtest2, Ytest2, b, theta)
        print("Rate of correct classification for training set: %0.3f" % e1)
        print("Rate of correct classification for testing set: %0.3f" % e2)
        # Print optimal b and the number of non-zero indices in each column
        print("Optimal b:")
        for k in b:
            print(k)
        counts = []
        for k in b.T:
            count = 0
            for l in k:
                if abs(l) > 1e-04:
                    count += 1
            counts.append(count)
        print("Number of non-zero indices in each column of b:")
        print("(%d, %d)" % (counts[0], counts[1]))
        # Plot cost function for specific values of lambda
        if i == 0 or i == 4 or i == 6:
            plt.plot(it, cost)
            plt.xlabel("Iterations")
            plt.ylabel("Cost")
            fig = plt.gcf()
            fig.savefig("Figure.png")
        print("\n")
