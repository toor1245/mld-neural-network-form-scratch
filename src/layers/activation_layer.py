from typing import Callable

from src.core.matrix import Matrix
from src.layers.layer import Layer


class ActivationLayer(Layer):
    def __init__(self, activation: Callable, activation_prime: Callable):
        super().__init__()
        self.activation: Callable = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input: Matrix):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return Matrix.hadamard_product(self.activation_prime(self.input), output_error)
