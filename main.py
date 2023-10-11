from src.activations.tanh_activation import TanHyperbolicActivations
from src.core.matrix import Matrix
from src.datasets.mnist import Dataset
from src.datasets.mnist_sample import MnistSample
from src.layers.activation_layer import ActivationLayer
from src.layers.fully_connected_layer import FullyConnectedLayer
from src.losses.mean_square_error import MeanSquareError
from src.network import Network

if __name__ == '__main__':
    dataset = Dataset.load()

    (x_train, y_train) = MnistSample.get_data(dataset[0], 1000)
    (x_test, y_test) = MnistSample.get_data(dataset[1], 20)

    y_train = Matrix.create_binary_matrix(y_train)
    y_test = Matrix.create_binary_matrix(y_test)

    net = Network(MeanSquareError.mse, MeanSquareError.mse_prime)
    net.add(FullyConnectedLayer(28 * 28, 40))
    net.add(ActivationLayer(TanHyperbolicActivations.tanh, TanHyperbolicActivations.tanh_prime))
    net.add(FullyConnectedLayer(40, 10))
    net.add(ActivationLayer(TanHyperbolicActivations.tanh, TanHyperbolicActivations.tanh_prime))

    net.fit_mnist(x_train, y_train, 100)

    output = net.predict_mnist(x_test)
    for i in range(len(output)):
        print(output[i])

    print(y_test)

