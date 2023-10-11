from src.core.matrix import Matrix


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input: Matrix):
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
