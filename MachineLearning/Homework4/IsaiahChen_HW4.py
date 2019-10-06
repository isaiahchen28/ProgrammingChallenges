import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC


def visualize_data(data, tick_range, tick_values, figure_name):
    # Plot the data and save figure
    plt.matshow(data)
    plt.xticks(tick_range, tick_values)
    plt.yticks(tick_range, tick_values)
    plt.set_cmap("gray")
    fig = plt.gcf()
    fig.savefig(figure_name)


def reshape_data(data, dim, figure_name):
    # Define the number of columns with x data
    cols = data.shape[1] - 1
    # Extract the distinct x values in the data
    x = [[] for i in range(cols)]
    for i in range(cols):
        for j in data[:, i]:
            if j not in x[i]:
                x[i].append(j)
    # Reshape the data to a matrix with specified dimensions
    re_data = np.reshape(data[:, cols], (dim, dim))
    # Define the steps and numbers to display
    steps = []
    for i, j in enumerate(x[0]):
        if i % 10 == 0:
            steps.append(str(round(j, 1)))
    step_num = np.arange(0, 100, 10)
    # Use previous function to plot data
    visualize_data(re_data, step_num, steps, figure_name)


def train_SVM(train_data, test_data, sigma):

    def cauchy_kernel(x, y):
        # Calculate the kernel matrix for a given value of sigma
        temp = []
        for i, j in enumerate(x):
            for k, l in enumerate(y):
                temp.append(1.0 / (1.0 + ((
                    np.linalg.norm(j - l) ** 2) / (sigma ** 2))))
        return np.reshape(np.array(temp), (len(x), len(y)))

    # Extract x data from overall training dataset
    last_col = train_data.shape[1] - 1
    x = [[] for i in range(last_col)]
    for i in range(last_col):
        for j in train_data[:, i]:
            x[i].append(j)
    Xtrain = np.transpose(np.array(x))
    # Extract y data from overall training dataset
    Ytrain = np.array(train_data[:, last_col])
    # Train the SVM
    clf = SVC(kernel=cauchy_kernel)
    clf.fit(Xtrain, Ytrain)
    # Extract x and y data from the overall test dataset
    last_col = test_data.shape[1] - 1
    x = [[] for i in range(last_col)]
    for i in range(last_col):
        for j in test_data[:, i]:
            x[i].append(j)
    Xtest = np.transpose(np.array(x))
    Ytest = np.array(test_data[:, last_col])
    # Predict classes on test data and return mean accuracy of SVM
    return clf.predict(Xtest), clf.score(Xtest, Ytest)


def train_SVM_pendigits(train_data, validation_data, sigma, m, k):

    def my_kernel(x, y):
        # Calculate the kernel matrix for a given value of sigma
        temp = []
        for i, j in enumerate(x):
            for k, l in enumerate(y):
                temp.append((1.0 + (np.linalg.norm(j - l) / sigma) + ((
                    np.linalg.norm(j - l) ** 2) / (3.0 * (
                        sigma ** 2)))) * np.exp(-1.0 * (np.linalg.norm(
                            j - l) / sigma)))
        return np.reshape(np.array(temp), (len(x), len(y)))

    def cauchy_kernel(x, y):
        # Calculate the kernel matrix for a given value of sigma
        temp = []
        for i, j in enumerate(x):
            for k, l in enumerate(y):
                temp.append(1.0 / (1.0 + ((
                    np.linalg.norm(j - l) ** 2) / (sigma ** 2))))
        return np.reshape(np.array(temp), (len(x), len(y)))

    # Extract x data from overall training dataset
    last_col = train_data.shape[1] - 1
    x = [[] for i in range(last_col)]
    for i in range(last_col):
        for j in train_data[:, i]:
            x[i].append(j)
    Xtrain = np.transpose(np.array(x))
    # Extract y data from overall training dataset
    Ytrain = np.array(train_data[:, last_col])
    # Train the SVM
    if k == "my_kernel":
        clf = SVC(C=m, kernel=my_kernel)
    elif k == "cauchy_kernel":
        clf = SVC(C=m, kernel=cauchy_kernel)
    else:
        raise Exception("Invalid kernel.")
    clf.fit(Xtrain, Ytrain)
    # Extract x and y data from the validation set
    last_col = validation_data.shape[1] - 1
    x = [[] for i in range(last_col)]
    for i in range(last_col):
        for j in validation_data[:, i]:
            x[i].append(j)
    Xvalid = np.transpose(np.array(x))
    Yvalid = np.array(validation_data[:, last_col])
    # Compute classification error on validation set
    return 1.0 - clf.score(Xvalid, Yvalid)


def test_SVM_pendigits(train_data, test_data, sigma, m, k):

    def my_kernel(x, y):
        # Calculate the kernel matrix for a given value of sigma
        temp = []
        for i, j in enumerate(x):
            for k, l in enumerate(y):
                temp.append((1.0 + (np.linalg.norm(j - l) / sigma) + ((
                    np.linalg.norm(j - l) ** 2) / (3.0 * (
                        sigma ** 2)))) * np.exp(-1.0 * (np.linalg.norm(
                            j - l) / sigma)))
        return np.reshape(np.array(temp), (len(x), len(y)))

    def cauchy_kernel(x, y):
        # Calculate the kernel matrix for a given value of sigma
        temp = []
        for i, j in enumerate(x):
            for k, l in enumerate(y):
                temp.append(1.0 / (1.0 + ((
                    np.linalg.norm(j - l) ** 2) / (sigma ** 2))))
        return np.reshape(np.array(temp), (len(x), len(y)))

    # Extract x data from overall training dataset
    last_col = train_data.shape[1] - 1
    x = [[] for i in range(last_col)]
    for i in range(last_col):
        for j in train_data[:, i]:
            x[i].append(j)
    Xtrain = np.transpose(np.array(x))
    # Extract y data from overall training dataset
    Ytrain = np.array(train_data[:, last_col])
    # Train the SVM
    if k == "my_kernel":
        clf = SVC(C=m, kernel=my_kernel)
    elif k == "cauchy_kernel":
        clf = SVC(C=m, kernel=cauchy_kernel)
    else:
        raise Exception("Invalid kernel.")
    clf.fit(Xtrain, Ytrain)
    # Extract x and y data from the overall test dataset
    last_col = test_data.shape[1] - 1
    x = [[] for i in range(last_col)]
    for i in range(last_col):
        for j in test_data[:, i]:
            x[i].append(j)
    Xtest = np.transpose(np.array(x))
    Ytest = np.array(test_data[:, last_col])
    # Predict classes on test data
    Ypred = clf.predict(Xtest)
    # Calculate confusion matrix
    a = np.empty((10, 10))
    for g in range(10):
        for go in range(10):
            sum1, sum2 = 0.0, 0.0
            for i, j in enumerate(Ytest):
                if j == go:
                    sum2 += 1
                    if Ypred[i] == g:
                        sum1 += 1
            a[g, go] = sum1 / sum2
    return a


if __name__ == '__main__':

    """ Question 1.1 """
    kernel1Test = pd.read_csv("kernel1Test.csv").to_numpy()
    reshape_data(kernel1Test, 100, "Figure_1.png")

    """ Question 1.2 """
    print("--- Question 1.2 ---\n")
    kernel1Train = pd.read_csv("kernel1Train.csv").to_numpy()
    Xtest = np.delete(kernel1Test, -1, 1)
    s = [0.01, 0.025, 0.05, 0.1, 0.5, 1.0, 2.5]
    for i, j in enumerate(s):
        Ypred, acc = train_SVM(kernel1Train, kernel1Test, j)
        print("For sigma = %1.3f, error = %1.4f" % (j, 1.0 - acc))
        pred_data = np.insert(Xtest, Xtest.shape[1], Ypred, 1)
        reshape_data(pred_data, 100, "Figure_2_" + str(i + 1) + ".png")

    """ Question 2.1 """
    pendigitsTrain = pd.read_csv("pendigitsTrain.csv").to_numpy()
    pendigitsValid = pd.read_csv("pendigitsValid.csv").to_numpy()
    pendigitsTest = pd.read_csv("pendigitsTest.csv").to_numpy()
    s2 = [1.0, 5.0, 10.0, 50.0, 100.0, 150.0, 200.0]
    pendigitsTrainValid = np.vstack((pendigitsTrain, pendigitsValid))
    # Run for m = 10
    print("\n--- Question 2.1 ---\n")
    errors1 = []
    for i in s2:
        errors1.append(train_SVM_pendigits(
            pendigitsTrain, pendigitsValid, i, 10.0, "my_kernel"))
    min_sigma1 = s2[errors1.index(min(errors1))]
    cm1 = test_SVM_pendigits(pendigitsTrainValid,
                             pendigitsTest, min_sigma1, 10.0, "my_kernel")
    print("For m = 10:")
    print(cm1.T)
    """ Question 2.2 """
    print("\n--- Question 2.2 ---\n")
    errors2 = []
    for i in s2:
        errors2.append(train_SVM_pendigits(
            pendigitsTrain, pendigitsValid, i, 20.0, "my_kernel"))
    min_sigma2 = s2[errors2.index(min(errors2))]
    cm2 = test_SVM_pendigits(pendigitsTrainValid,
                             pendigitsTest, min_sigma2, 20.0, "my_kernel")
    print("For m = 20:")
    print(cm2.T)
    """ Question 2.3 """
    print("\n--- Question 2.3 ---\n")
    errors3 = []
    for i in s2:
        errors3.append(train_SVM_pendigits(
            pendigitsTrain, pendigitsValid, i, 10.0, "cauchy_kernel"))
    min_sigma3 = s2[errors3.index(min(errors3))]
    cm3 = test_SVM_pendigits(pendigitsTrainValid,
                             pendigitsTest, min_sigma3, 10.0, "cauchy_kernel")
    print("For m = 10:")
    print(cm3.T)
    errors4 = []
    for i in s2:
        errors4.append(train_SVM_pendigits(
            pendigitsTrain, pendigitsValid, i, 20.0, "cauchy_kernel"))
    min_sigma4 = s2[errors4.index(min(errors4))]
    cm4 = test_SVM_pendigits(pendigitsTrainValid,
                             pendigitsTest, min_sigma4, 20.0, "cauchy_kernel")
    print("\nFor m = 20:")
    print(cm4.T)
