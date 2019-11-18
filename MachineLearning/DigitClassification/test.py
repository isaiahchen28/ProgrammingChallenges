import matplotlib.pyplot as plt
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils


def show_examples(x, y):
    # Use matplotlib to show nine examples of digits
    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(x_train[i], cmap='gray', interpolation='none')
        plt.title("Digit: {}".format(y_train[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def show_pixel_distribution(x, y):
    # Show pixel distribution for a given example
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(x[0], cmap='gray', interpolation='none')
    plt.title("Digit: {}".format(y[0]))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 1, 2)
    plt.hist(x_train[0].reshape(784))
    plt.title("Pixel Value Distribution")
    plt.show()


def build_network(x_train, y_train, x_test, y_test,
                  model_name='keras_mnist.h5'):
    # building a linear stack of layers with the sequential model
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))
    # compiling the sequential model
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'], optimizer='adam')

    # training the model and saving metrics in history
    history = model.fit(x_train, y_train,
                        batch_size=128, epochs=10,
                        verbose=2,
                        validation_data=(x_test, y_test))
    # Save model
    save_dir = os.getcwd()
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    return model, history


def plot_performance(model, history):
    # plotting the metrics
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.tight_layout()

    loss_and_metrics = model.evaluate(x_test, y_test, verbose=2)
    print("Test Loss", loss_and_metrics[0])
    print("Test Accuracy", loss_and_metrics[1])
    plt.show()


if __name__ == '__main__':
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Load MNIST data into appropriate training and testing variables
    # x_train = array of 60,000 different 28x28 images
    # y_train = array of 60,000 classes corresponding to training images
    # x_test = array of 10,000 different 28x28 images
    # y_test = array of 10,000 classes corresponding to test images
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Show examples and pixel distribution
    show_examples(x_train, y_train)
    show_pixel_distribution(x_train, y_train)
    # Flatten the input vectors
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalize the data to help with the training
    x_train /= 255
    x_test /= 255
    # Input data has now been normalized and flattened into 784-element arrays

    # one-hot encoding using keras' numpy-related utilities
    n_classes = 10
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)
    # Build network
    model, history = build_network(x_train, y_train, x_test, y_test)
    plot_performance(model, history)
