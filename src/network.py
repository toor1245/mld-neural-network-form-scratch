from typing import Callable

from src.core.matrix import Matrix
from src.layers.layer import Layer


class Network:
    def __init__(self, loss, loss_prime):
        self._layers: list[Layer] = []
        self._loss: Callable = loss
        self._loss_prime: Callable = loss_prime

    def add(self, layer):
        self._layers.append(layer)

    def predict(self, input_data: Matrix):
        samples = input_data
        result = []

        for i in range(samples.rows):
            output = Matrix.to_matrix(input_data.get_row(i))
            for layer in self._layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit(self, x_train: Matrix, y_train: Matrix, epochs, learning_rate):
        samples = x_train.rows

        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = Matrix.to_matrix(x_train.get_row(j))
                for layer in self._layers:
                    output = layer.forward_propagation(output)

                y_true = Matrix.to_matrix(y_train.get_row(j))
                y_pred = output
                err = err + self._loss(y_true, y_pred)

                grad = self._loss_prime(y_true, y_pred)
                for layer in reversed(self._layers):
                    grad = layer.backward_propagation(grad, learning_rate)

            err /= samples
            print('epoch %d/%d   error=%f' % (i + 1, epochs, err))

    def predict_mnist(self, input_data):
        samples = input_data
        result = []

        for i in range(len(samples)):
            output = input_data[i]
            for layer in self._layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit_mnist(self, x_train, y_train, epochs=1000, learning_rate=0.1):
        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]

                for layer in self._layers:
                    output = layer.forward_propagation(output)

                y_true = Matrix.to_matrix(y_train.get_row(j))
                y_pred = output
                err = err + self._loss(y_true, y_pred)

                grad = self._loss_prime(y_true, y_pred)
                for layer in reversed(self._layers):
                    grad = layer.backward_propagation(grad, learning_rate)

            err /= samples
            print('epoch %d/%d   error=%f' % (i + 1, epochs, err))
