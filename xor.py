from src.activations.tanh_activation import TanHyperbolicActivations
from src.core.matrix import Matrix
from src.core.nn_backend import NnBackend
from src.core.nn_types import NnMatrix
from src.layers.activation_layer import ActivationLayer
from src.layers.fully_connected_layer import FullyConnectedLayer
from src.losses.mean_square_error import MeanSquareError
from src.network import Network
from ctypes import *
import numpy as np

def xor_solve():
    x_train = Matrix(4, 2)
    x_train.set_matrix_arr([0, 0, 0, 1, 1, 0, 1, 1])

    y_train = Matrix(4, 1)
    y_train.set_matrix_arr([0, 1, 1, 0])

    net = Network(MeanSquareError.mse, MeanSquareError.mse_prime)
    net.add(FullyConnectedLayer(2, 3))
    net.add(ActivationLayer(TanHyperbolicActivations.tanh, TanHyperbolicActivations.tanh_prime))
    net.add(FullyConnectedLayer(3, 1))
    net.add(ActivationLayer(TanHyperbolicActivations.tanh, TanHyperbolicActivations.tanh_prime))

    # train
    net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

    # test
    print("\nOut: \n")
    out = net.predict(x_train)
    for i in range(len(out)):
        print(out[i])


# xor_solve()

# (depth, input_depth, kernel_size, kernel_size)
kernels_shape = (5, 3, 3)

# Creating the kernels array with random values
kernels = np.random.randn(*kernels_shape)
print(np.zeros(kernels_shape))

# Displaying the result
#print(kernels)


kernels_shape2 = (4, 3, 3)

# Creating the kernels array with random values
#kernels2 = np.random.randn(*kernels_shape2)

# Displaying the result
#print(kernels2)

#nn_provider = NnBackend()

#matrix = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#kernel = [9, 8, 7, 6, 5, 4, 3, 2, 1]


#ma = NnMatrix(matrix, c_uint32(3), c_uint32(3))
#kern = NnMatrix(kernel, c_uint32(3), c_uint32(3))
#nn_provider.cross_correlate2d_valid(ma, mb)

