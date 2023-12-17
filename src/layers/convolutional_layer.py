from src.core.matrix import Matrix
from src.layers.layer import Layer
from scipy import signal
import numpy as np


class ConvolutionalLayer(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        super().__init__()
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (input_depth, kernel_size, kernel_size, depth)

        self.kernels: list[list[Matrix]] = Matrix.random_conv_kernels(*self.kernels_shape)
        self.biases: list[Matrix] = Matrix.random_conv_biases(*self.output_shape)

    def forward_propagation(self, input: Matrix):
        self.input = input
        self.output = Matrix.conv_copy_biases(self.biases)

        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] = self.output[i] + Matrix.correlate2d_valid(self.input, self.kernels[i][j])
        return self.output

    def backward_propagation(self, output_error: list[Matrix], learning_rate):
        depth, m, n, input_depth = self.kernels_shape
        kernels_gradient = Matrix.zeros(input_depth, m, n, depth)[0]
        input_gradient = Matrix.zeros(*self.input_shape)[0]

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i] = Matrix.correlate2d_valid(self.input, output_error[i])
                input_gradient[j] = input_gradient[j] + Matrix.convolve2d_full(output_error[i], self.kernels[i][j])

        for i in range(len(self.kernels)):
            for j in range(len(self.kernels[i])):
                self.kernels[i][j] = Matrix.sub(self.kernels[i][j], kernels_gradient[i] * learning_rate)

        for i in range(len(self.biases)):
            self.biases[i] = Matrix.sub(self.biases[i], output_error[i] * learning_rate)

        return input_gradient
