import math

from src.core.matrix import Matrix


class TanHyperbolicActivations:
    @staticmethod
    def tanh(x: Matrix) -> Matrix:
        res = Matrix(x.rows, x.columns)

        for i in range(x.rows):
            for j in range(x.columns):
                res[i, j] = math.tanh(x[i, j])

        return res

    @staticmethod
    def tanh_prime(x: Matrix) -> Matrix:
        res = Matrix(x.rows, x.columns)

        for i in range(x.rows):
            for j in range(x.columns):
                res[i, j] = 1 - math.tanh(x[i, j]) ** 2

        return res
