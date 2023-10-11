from src.core.matrix import Matrix


class MeanSquareError:
    @staticmethod
    def mse(y_true: Matrix, y_pred: Matrix):
        return (y_true - y_pred).power(2).mean()

    @staticmethod
    def mse_prime(y_true: Matrix, y_pred: Matrix):
        return (y_pred - y_true) * (2 / y_true.length)
