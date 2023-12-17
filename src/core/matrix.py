import array
import array as arr
import copy
import math
import random
from ctypes import *
from typing import Tuple, Union, List

from src.core.nn_types import *
import src.core.nn_backend as nn


class Matrix:
    @property
    def rows(self) -> int:
        return self._rows

    @property
    def columns(self) -> int:
        return self._columns

    @property
    def length(self) -> int:
        return self.rows * self.columns

    @property
    def array(self) -> arr.array:
        return self._matrix

    @property
    def typecode(self):
        return self._typecode

    def __init__(self, rows: int, columns: int, matrix: arr.array = None, typecode='f'):
        self._rows: int = rows
        self._columns: int = columns
        self._typecode = typecode

        if matrix is None:
            self._matrix = arr.array(typecode, [0] * (rows * columns))
        else:
            self.set_matrix_arr(matrix)

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

        if left.length < 4 and right.length < 4:
            res = Matrix(left.rows, left.columns)

            for i in range(res.length):
                res.array[i] = left.array[i] + right.array[i]

            return res

        lm = nn_matrix_new(left.array, c_uint32(left.rows), c_uint32(left.columns))
        rm = nn_matrix_new(right.array, c_uint32(right.rows), c_uint32(right.columns))

        mc = nn.nn_provider.add(lm, rm)
        res = Matrix(left.rows, left.columns, mc)
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
                res += str(round(self[i, j], 2)) + " "
            res += '\n'

        return res

    def get(self, index: int):
        return self._matrix[index]

    def get_row(self, index: int) -> arr.array:
        res = arr.array(self.typecode, [0] * self.columns)

        for i in range(self.columns):
            res[i] = self[index, i]

        return res

    def get_column(self, index: int) -> arr.array:
        res = arr.array(self.typecode, [0] * self.rows)

        for i in range(self.rows):
            res[i] = self[i, index]

        return res

    def set_matrix_arr(self, input: arr.array):
        if len(input) != self.length:
            raise Exception

        self._matrix = input

    def _mul_matrix(self, other: 'Matrix') -> 'Matrix':
        left: Matrix = self
        right: Matrix = other

        if left.columns != right.rows:
            raise Exception('left.columns != right.rows')

        nn_ma = nn_matrix_new(left._matrix, c_uint32(left.rows), c_uint32(left.columns))
        nn_mb = nn_matrix_new(right._matrix, c_uint32(right.rows), c_uint32(right.columns))

        nn_output, output_arr = nn.nn_provider.multiply(nn_ma, nn_mb)
        return Matrix(nn_output.rows, nn_output.columns, output_arr)

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

    def resize(self, rows: int, columns: int) -> 'Matrix':
        if rows * columns != self.length:
            raise Exception

        self._rows = rows
        self._columns = columns
        return self

    def copy(self) -> 'Matrix':
        src_ptr = cast(self.array.buffer_info()[0], POINTER(c_void_p))
        dst = Matrix(self.rows, self.columns)
        dst_ptr = cast(dst.array.buffer_info()[0], POINTER(c_void_p))
        nn.nn_provider.copy(dst_ptr, src_ptr, len(dst) * 4)
        return dst

    def set_column(self, column_index: int, data):
        for i in range(self.rows):
            self[i, column_index] = data[i]

    def set_row(self, row_index: int, data):
        for j in range(self.columns):
            self[row_index, j] = data[j]

    def get_indexes(self, func):
        res = []

        for i in range(self.length):
            if func(self.array[i]):
                res.append(i)

        return res

    def to_list_by_columns(self):
        res: list = []

        for j in range(self.columns):
            for item in self.get_column(j):
                res.append(item)

        return res

    def to_array_by_columns(self):
        res: list = self.to_list_by_columns()

        return arr.array('f', res)

    def __neg__(self):
        res = Matrix(self.rows, self.columns)

        for i in range(res.rows):
            for j in range(res.columns):
                res[i, j] = -self[i, j]

        return res

    @staticmethod
    def _dot_value_to_column(value, matrix):
        res = Matrix(1, matrix.columns)
        for i in range(matrix.columns):
            res._matrix[i] = value * matrix.get(i)
        return res

    @staticmethod
    def _dot(x: 'Matrix', y: 'Matrix') -> 'Matrix':
        if x.length == 1 and y.rows == 1:
            return Matrix._dot_value_to_column(x.get(0), y)

        if y.length == 1 and x.rows == 1:
            return Matrix._dot_value_to_column(y.get(0), x)

        if y.columns == 1 and y.length == x.length:
            return Matrix.hadamard_product(x, y)

        return x * y

    @staticmethod
    def dot(x, y):
        if isinstance(x, Matrix) and isinstance(y, Matrix):
            return Matrix._dot(x, y)

        if isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                raise Exception

            res: list['Matrix'] = []
            for i in range(len(x)):
                res.append(Matrix._dot(x[i], y[i]))

            return res

        raise Exception

    @staticmethod
    def _sub_vector(x: 'Matrix', y: 'Matrix') -> 'Matrix':
        res = Matrix(1, x.columns)

        for i in range(x.columns):
            res._matrix[i] = x._matrix[i] - y._matrix[i]

        return res

    @staticmethod
    def _sub_num2(x: float, y: 'Matrix'):
        res = Matrix(y.rows, y.columns)

        for i in range(res.length):
            res.array[i] = x - y.array[i]

        return res

    @staticmethod
    def sub_num(x: float, y):
        if isinstance(y, Matrix):
            return Matrix._sub_num2(x, y)

        if isinstance(y, list):
            res: list['Matrix'] = []

            for i in range(len(y)):
                res.append(Matrix._sub_num2(x, y[i]))
            return res

        raise Exception

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
    def _hadamard_product(x: 'Matrix', y: 'Matrix'):
        if x.length != y.length:
            raise Exception

        res = Matrix(x.rows, x.columns)

        for i in range(x.length):
            res._matrix[i] = x._matrix[i] * y._matrix[i]

        return res

    @staticmethod
    def hadamard_product(x, y):
        if isinstance(x, Matrix) and isinstance(y, Matrix):
            return Matrix._hadamard_product(x, y)

        if isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                raise Exception

            res: list['Matrix'] = []
            for i in range(len(x)):
                res.append(Matrix._hadamard_product(x[i], y[i]))

            return res

        raise Exception

    @staticmethod
    def div(matrix: 'Matrix', value):
        res = Matrix(matrix.rows, matrix.columns)

        for i in range(res.rows):
            for j in range(res.columns):
                res[i, j] = matrix[i, j] / value

        return res

    @staticmethod
    def div_mt(left: 'Matrix', right: 'Matrix'):

        if left.rows != right.rows or left.columns != right.columns:
            raise Exception

        res = Matrix(left.rows, left.columns)

        for i in range(res.length):
            res.array[i] = left.array[i] / right.array[i]

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

    @staticmethod
    def random_conv_kernels(input_depth, kernel_size_m, kernel_size_n, depth=1) -> list[list['Matrix']]:
        kernels: list[list['Matrix']] = []

        for i in range(depth):
            kernels.append([])
            for j in range(input_depth):
                kernels[i].append(Matrix.random(kernel_size_m, kernel_size_n))

        return kernels

    @staticmethod
    def random_conv_biases(input_depth, kernel_size_m, kernel_size_n) -> list['Matrix']:
        return Matrix.random_conv_kernels(input_depth, kernel_size_m, kernel_size_n)[0]

    @staticmethod
    def conv_copy_biases(biases: list['Matrix']) -> list['Matrix']:
        result: list['Matrix'] = []

        for i in range(len(biases)):
            result.append(biases[i].copy())

        return result

    @staticmethod
    def zeros(num_matrices_in_list, rows, columns, depth=1) -> list[list['Matrix']]:
        res: list[list['Matrix']] = []

        for i in range(depth):
            res.append([])
            for j in range(num_matrices_in_list):
                res[i].append(Matrix(rows, columns))

        return res

    @staticmethod
    def correlate2d_valid(input: 'Matrix', kernel: 'Matrix'):
        nn_input = nn_matrix_new(input._matrix, c_uint32(input.rows), c_uint32(input.columns))
        nn_kernel = nn_matrix_new(kernel._matrix, c_uint32(kernel.rows), c_uint32(kernel.columns))

        nn_output, output_arr = nn.nn_provider.cross_correlate2d_valid(nn_input, nn_kernel)
        return Matrix(nn_output.rows, nn_output.columns, output_arr)

    @staticmethod
    def convolve2d_full(input: 'Matrix', kernel: 'Matrix'):
        nn_input = nn_matrix_new(input._matrix, c_uint32(input.rows), c_uint32(input.columns))
        nn_kernel = nn_matrix_new(kernel._matrix, c_uint32(kernel.rows), c_uint32(kernel.columns))

        nn_output, output_arr = nn.nn_provider.convolve2d_full(nn_input, nn_kernel)
        return Matrix(nn_output.rows, nn_output.columns, output_arr)

    @staticmethod
    def hstack(tup: Tuple) -> 'Matrix':
        res = Matrix(len(tup[0]), len(tup))

        for i in range(len(tup)):
            res.set_column(i, tup[i])

        return res

    @staticmethod
    def permute(matrix: 'Matrix') -> 'Matrix':
        arr_copy = copy.deepcopy(matrix.array)
        random.shuffle(arr_copy)

        return Matrix(matrix.rows, matrix.columns, arr_copy)

    @staticmethod
    def create_by_indexes(matrix: 'Matrix', indexes: arr, size: Tuple[int, int] = None) -> 'Matrix':
        if size is None:
            res = Matrix(1, len(indexes))
            for i in range(res.length):
                res.array[i] = matrix.array[indexes[i]]

            return res

        rows, columns = size
        length = rows * columns
        if length != len(indexes) or length > len(indexes):
            raise Exception('size(rows * columns) must be equal or less than length of indexes')

        res = Matrix(rows, columns)
        for i in range(res.length):
            res.array[i] = matrix.array[indexes[i]]

        return res

    @staticmethod
    def log(matrix: 'Matrix') -> 'Matrix':
        res = Matrix(matrix.rows, matrix.columns)

        for i in range(res.length):
            res.array[i] = math.log(matrix.array[i])

        return res

    @staticmethod
    def _sigmoid(matrix: 'Matrix'):
        res = Matrix(matrix.rows, matrix.columns)

        for i in range(matrix.length):
            res.array[i] = 1 / (1 + math.exp(-matrix.array[i]))

        return res

    @staticmethod
    def sigmoid(matrix):
        if isinstance(matrix, list):
            res: list['Matrix'] = []
            for i in range(len(matrix)):
                res.append(Matrix._sigmoid(matrix[i]))
            return res

        if isinstance(matrix, Matrix):
            return Matrix._sigmoid(matrix)

        raise Exception('unsupported type')
