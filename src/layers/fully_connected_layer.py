from src.layers.layer import Layer
from src.core.matrix import Matrix


class FullyConnectedLayer(Layer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weights: Matrix = Matrix.random(output_size, input_size)
        self.bias: Matrix = Matrix.random(output_size, 1)

    def forward_propagation(self, input: Matrix):
        self.input = input
        self.output = self.weights * self.input + self.bias
        return self.output

    def backward_propagation(self, output_error: Matrix, learning_rate: float):
        weights_error = Matrix.dot(output_error, self.input.transpose())
        input_error = Matrix.dot(self.weights.transpose(), output_error)
        self.weights = Matrix.sub(self.weights, weights_error * learning_rate)
        self.bias = Matrix.sub(self.bias, output_error * learning_rate)
        return input_error
