from src.core.matrix import Matrix


class SigmoidActivations:
    @staticmethod
    def sigmoid(x):
        return Matrix.sigmoid(x)

    @staticmethod
    def sigmoid_prime(x):
        s = SigmoidActivations.sigmoid(x)
        return Matrix.dot(s, Matrix.sub_num(1.0, s))
