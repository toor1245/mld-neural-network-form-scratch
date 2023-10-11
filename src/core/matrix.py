import array
import array as arr
import math
import random


class Matrix:
    @property
    def rows(self) -> int:
        return self._rows

    @property
    def columns(self) -> int:
        return self._columns

    @property
    def length(self) -> int:
        return len(self._matrix)

    @property
    def array(self) -> arr.array:
        return self._matrix

    @property
    def typecode(self):
        return self._typecode

    def __init__(self, rows: int, columns: int, typecode='f'):
        self._rows: int = rows
        self._columns: int = columns
        self._typecode = typecode
        self._matrix = arr.array(typecode, [0] * (rows * columns))

    def __setitem__(self, key: tuple, value):
        row, column = key
        self._matrix[row * self.columns + column] = value

    def __getitem__(self, key: tuple):
        row, column = key
        return self._matrix[row * self.columns + column]

    def __add__(self, other: 'Matrix') -> 'Matrix':
        left: Matrix = self
        right: Matrix = other

        if left.rows != right.rows or left.columns != right.columns:
            raise Exception('left.rows != right.Rows or left.columns != right.columns')

        res = Matrix(left.rows, left.columns)

        for i in range(left.rows):
            for j in range(left.columns):
                res[i, j] = left[i, j] + right[i, j]

        return res

    def __mul__(self, other) -> 'Matrix':
        if isinstance(other, Matrix):
            return self._mul_matrix(other)
        if isinstance(other, array.array):
            pass
        elif isinstance(other, int) or isinstance(other, float):
            return self._mul_num(other)
        else:
            raise TypeError

    def __sub__(self, other) -> 'Matrix':
        if isinstance(other, Matrix):
            return self._sub_matrix(other)
        elif isinstance(other, int) or isinstance(other, float):
            return self._sub_num(other)
        else:
            raise TypeError

    def __len__(self):
        return self.length

    def __str__(self) -> str:
        # ohh python, where is string builder...
        res = str()

        for i in range(self.rows):
            for j in range(self.columns):
                res += str(self[i, j]) + " "
            res += '\n'

        return res

    def get(self, index: int):
        return self._matrix[index]

    def get_row(self, index: int) -> arr.array:
        res = arr.array(self.typecode, [0] * self.columns)

        for i in range(self.columns):
            res[i] = self[index, i]

        return res

    def set_matrix_arr(self, input: arr.array | list):
        if len(input) != self.length:
            raise Exception

        for i in range(self.length):
            self._matrix[i] = input[i]

    def _mul_matrix(self, other: 'Matrix') -> 'Matrix':
        left: Matrix = self
        right: Matrix = other

        if left.columns != right.rows:
            raise Exception('left.columns != right.rows')

        res = Matrix(left.rows, right.columns)

        for i in range(left.rows):
            for j in range(right.columns):
                for k in range(left.columns):
                    res[i, j] = res[i, j] + left[i, k] * right[k, j]

        return res

    def _mul_num(self, other) -> 'Matrix':
        left: Matrix = self
        right = other

        res = Matrix(left.rows, left.columns)

        for i in range(left.rows):
            for j in range(left.columns):
                res[i, j] = left[i, j] * right

        return res

    def _sub_num(self, other) -> 'Matrix':
        left: Matrix = self
        right = other

        res = Matrix(left.rows, left.columns)

        for i in range(left.rows):
            for j in range(left.columns):
                res[i, j] = left[i, j] - right

        return res

    def _sub_matrix(self, other: 'Matrix') -> 'Matrix':
        left: Matrix = self
        right: Matrix = other

        if left.rows != right.rows or left.columns != right.columns:
            raise Exception('left.rows != right.Rows or left.columns != right.columns')

        res = Matrix(left.rows, left.columns)

        for i in range(left.rows):
            for j in range(left.columns):
                res[i, j] = left[i, j] - right[i, j]

        return res

    def transpose(self) -> 'Matrix':
        res = Matrix(self.columns, self.rows)

        for i in range(res.rows):
            for j in range(res.columns):
                res[i, j] = self[j, i]

        return res

    def power(self, value: int) -> 'Matrix':
        res = Matrix(self.rows, self.columns)

        for i in range(res.rows):
            for j in range(res.columns):
                res[i, j] = math.pow(self[i, j], value)

        return res

    def sum(self):
        res = 0

        for i in range(self.length):
            res += self.get(i)

        return res

    def mean(self) -> float:
        return self.sum() / self.length

    def argmax(self):
        index = 0
        max_value = self._matrix[0]

        for i in range(self.length):
            if max_value < self._matrix[i]:
                max_value = self._matrix[i]
                index = i

        return index

    @staticmethod
    def _dot_value_to_column(value, matrix):
        res = Matrix(1, matrix.columns)
        for i in range(matrix.columns):
            res._matrix[i] = value * matrix.get(i)
        return res

    @staticmethod
    def dot(x: 'Matrix', y: 'Matrix') -> 'Matrix':
        if x.length == 1 and y.rows == 1:
            return Matrix._dot_value_to_column(x.get(0), y)

        if y.length == 1 and x.rows == 1:
            return Matrix._dot_value_to_column(y.get(0), x)

        return x * y

    @staticmethod
    def _sub_vector(x: 'Matrix', y: 'Matrix') -> 'Matrix':
        res = Matrix(1, x.columns)

        for i in range(x.columns):
            res._matrix[i] = x._matrix[i] - y._matrix[i]

        return res

    @staticmethod
    def sub(x: 'Matrix', y: 'Matrix') -> 'Matrix':
        if x.rows == 1 and y.rows == 1 and x.columns == y.columns:
            return Matrix._sub_vector(x, y)

        return x - y

    @staticmethod
    def to_matrix(input: arr.array | list):
        length = len(input)
        res = Matrix(length, 1)

        res.set_matrix_arr(input)

        return res

    @staticmethod
    def hadamard_product(x: 'Matrix', y: 'Matrix'):
        if x.rows != y.rows or x.columns != y.columns:
            raise Exception('x.rows != y.rows or x.columns != y.columns')

        res = Matrix(x.rows, x.columns)

        for i in range(x.length):
            res._matrix[i] = x._matrix[i] * y._matrix[i]

        return res

    @staticmethod
    def div(matrix: 'Matrix', value):
        res = Matrix(matrix.rows, matrix.columns)

        for i in range(res.rows):
            for j in range(res.columns):
                res[i, j] = res[i, j] / value

        return res

    @staticmethod
    def create_binary_matrix(vector) -> 'Matrix':
        res = Matrix(len(vector), max(vector) + 1)

        for i, value in enumerate(vector):
            res[i, value] = 1

        return res

    @staticmethod
    def random(rows: int, columns: int) -> 'Matrix':
        res = Matrix(rows, columns)

        for i in range(rows):
            for j in range(columns):
                res[i, j] = random.gauss(0, 1)

        return res
