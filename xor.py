from src.activations.tanh_activation import TanHyperbolicActivations
from src.core.matrix import Matrix
from src.layers.activation_layer import ActivationLayer
from src.layers.fully_connected_layer import FullyConnectedLayer
from src.losses.mean_square_error import MeanSquareError
from src.network import Network


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


xor_solve()
