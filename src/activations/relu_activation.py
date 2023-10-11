from src.core.matrix import Matrix


class ReLUActivation:
    @staticmethod
    def _relu(x):
        return max(0, x)

    @staticmethod
    def _relu_prime(x):
        if x < 0:
            return 0
        else:
            return 1

    @staticmethod
    def relu(x: Matrix):
        res = Matrix(x.rows, x.columns)

        for i in range(res.rows):
            for j in range(res.columns):
                res[i, j] = ReLUActivation._relu(x[i, j])

    @staticmethod
    def relu_prime(x: Matrix):
        res = Matrix(x.rows, x.columns)

        for i in range(res.rows):
            for j in range(res.columns):
                res[i, j] = ReLUActivation._relu_prime(x[i, j])
