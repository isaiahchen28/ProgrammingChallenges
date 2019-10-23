# http://neuralnetworksanddeeplearning.com/chap1.html
import mnist_loader
import network

if __name__ == "__main__":
    # Load in MNIST data using mnist_loader.py
    training, validation, test = mnist_loader.load_data_wrapper()
    training = list(training)
    # Set up a network with 100 hidden neurons
    net = network.Network([784, 100, 10])
    # Use stochastic gradient descent to learn over 30 epochs
    # with a mini-batch size of 10 and a learning rate of 3.0
    net.SGD(training, 30, 10, 3.0, test_data=test)
