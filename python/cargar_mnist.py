from mnist import MNIST
import numpy as np
from mlp import MLP, MSE, Sigmoid


def one_hot(labels, num_classes=10):
    y = np.zeros((len(labels), num_classes), dtype=np.float32)

    for i, label in enumerate(labels):
        y[i, label] = 1.0

    return y


def cargar_mnist():
    mndata = MNIST("data_mnist")
    mndata.gz = False

    images_train, labels_train = mndata.load_training()
    images_test, labels_test = mndata.load_testing()

    X_train = np.array(images_train, dtype=np.float32) / 255.0
    X_test = np.array(images_test, dtype=np.float32) / 255.0

    Y_train = one_hot(labels_train)
    Y_test = one_hot(labels_test)

    return X_train, Y_train, X_test, Y_test


def evaluate_accuracy(model, X, Y):
    correct = 0

    for x, y in zip(X, Y):

        output = model.predict(x)

        pred = np.argmax(output)
        true = np.argmax(y)

        if pred == true:
            correct += 1

    return correct / len(X)

