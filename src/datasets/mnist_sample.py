import array

from src.core.matrix import Matrix


class MnistSample:
    @property
    def matrix(self):
        return self._matrix

    @property
    def label(self):
        return self._label

    def __init__(self, label: str | int, pixels: list[float]):
        self._label: int = int(label)
        self._matrix = Matrix(28 * 28, 1)

        for i in range(len(pixels)):
            pixels[i] = pixels[i] / 255.0

        self._matrix.set_matrix_arr(array.array('f', pixels))

    @staticmethod
    def get_data(samples: list['MnistSample'], num_samples=None):
        x: list[Matrix] = []
        y: list[int] = []

        if num_samples is None:
            num_samples = len(samples)

        if num_samples > len(samples):
            raise Exception

        for i in range(num_samples):
            x.append(samples[i]._matrix)
            y.append(samples[i]._label)

        return x, y


class MnistCNNSample:
    @property
    def matrix(self):
        return self._matrix

    @property
    def label(self):
        return self._label

    def __init__(self, label: str | int, pixels: list[float]):
        self._label: int = int(label)
        self._matrix = Matrix(28 * 28, 1)

        self._matrix.set_matrix_arr(array.array('f', pixels))

    @staticmethod
    def get_data(samples: list['MnistSample'], num_samples=None):
        x = []
        y = []

        if num_samples is None:
            num_samples = len(samples)

        if num_samples > len(samples):
            raise Exception

        for i in range(num_samples):
            x.append(samples[i].matrix)
            y.append(samples[i].label)

        return x, y
