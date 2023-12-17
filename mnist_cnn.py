import array
import random

from src.activations.sigmoid_activation import SigmoidActivations
from src.core.matrix import Matrix
from src.datasets.mnist import Dataset
from src.datasets.mnist_sample import MnistSample
from src.layers.activation_layer import ActivationLayer
from src.layers.convolutional_layer import ConvolutionalLayer
from src.layers.fully_connected_layer import FullyConnectedLayer
from src.layers.reshape_layer import ReshapeLayer
from src.losses.bin_cross_entropy import BinaryCrossEntropy
from src.losses.mean_square_error import MeanSquareError
from src.network import Network
import numpy as np


def preprocess_data(x: list[Matrix], y: Matrix, limit):
    zero_indexes = y.get_indexes(lambda el: el == 0)[:limit]
    one_indexes = y.get_indexes(lambda el: el == 1)[:limit]
    all_indexes = Matrix.hstack((zero_indexes, one_indexes))

    all_indexes = Matrix.permute(all_indexes)

    x_indexes: list[Matrix] = []
    y_indexes: list[int] = []

    for j in range(len(all_indexes.array)):
        x_indexes.append(x[int(all_indexes.array[j])].resize(28, 28))
        y_indexes.append(int(y.array[int(all_indexes.array[j])]))

    y_bin = Matrix.create_binary_matrix(y_indexes)

    return x_indexes, y_bin


if __name__ == '__main__':
    dataset = Dataset.load()

    (x_train, y_train) = MnistSample.get_data(dataset[0])
    (x_test, y_test) = MnistSample.get_data(dataset[1])

    y_train_m = Matrix(1, len(y_train), array.array('f', y_train))
    y_test_m = Matrix(1, len(y_test), array.array('f', y_test))

    x_train, y_train = preprocess_data(x_train, y_train_m, 100)
    x_test, y_test = preprocess_data(x_test, y_test_m, 100)

    net = Network(BinaryCrossEntropy.bce, BinaryCrossEntropy.bce_prime)
    net.add(ConvolutionalLayer((1, 28, 28), 3, 5))
    net.add(ActivationLayer(SigmoidActivations.sigmoid, SigmoidActivations.sigmoid_prime))
    net.add(ReshapeLayer((5, 26, 26)))
    net.add(FullyConnectedLayer(5 * 26 * 26, 100))
    net.add(ActivationLayer(SigmoidActivations.sigmoid, SigmoidActivations.sigmoid_prime))
    net.add(FullyConnectedLayer(100, 2))
    net.add(ActivationLayer(SigmoidActivations.sigmoid, SigmoidActivations.sigmoid_prime))

    net.fit_mnist_cnn(x_train, y_train, 20)

    output = net.predict_mnist(x_test)
    for i in range(len(output)):
        print(f"true: {y_test.array[i]} pred: {Matrix.argmax(output[i])}, probability: {output[i].array[Matrix.argmax(output[i])]}")

