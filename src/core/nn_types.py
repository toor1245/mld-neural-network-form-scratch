import array
from ctypes import *


class NnMemoryType(Structure):
    _fields_ = [("property_flags", c_uint32),
                ("heap_index", c_uint32)]


class NnMemoryProps(Structure):
    _fields_ = [("memory_type_count", c_uint32),
                ("memory_types", POINTER(NnMemoryType))]


class NnComputeInfo(Structure):
    _fields_ = [("device", POINTER(c_void_p)),
                ("queue_compute_index", c_uint32),
                ("memory_props", NnMemoryProps),
                ("queue", POINTER(c_void_p)),
                ("pipeline_cache", POINTER(c_void_p))]


class NnMatrix(Structure):
    _fields_ = [("ptr", POINTER(c_float)),
                ("rows", c_uint32),
                ("columns", c_uint32)]


def nn_matrix_new(ptr: array.array, rows: c_uint32, columns: c_uint32):
    nn_matrix = NnMatrix()
    nn_matrix.ptr = cast(ptr.buffer_info()[0], POINTER(c_float))
    nn_matrix.rows = rows
    nn_matrix.columns = columns
    return nn_matrix
