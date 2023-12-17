import array
import copy
from src.layers.layer import Layer

from src.core.matrix import Matrix


class ReshapeLayer(Layer):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def forward_propagation(self, input: list[Matrix]):
        length = 0
        arr = []
        for i in range(len(input)):
            arr += input[i].array.tolist()
            length += input[i].length

        matrix = Matrix(length, 1, array.array('f', arr))
        return matrix

    def backward_propagation(self, output_gradient: Matrix, learning_rate):
        num_matrices, m, n = self.input_shape

        matrices: list[Matrix] = []

        for i in range(num_matrices):
            span = output_gradient.array[i * (m * n): (i + 1) * (m * n)]
            matrices.append(Matrix(m, n, copy.deepcopy(span)))

        return matrices
