import sys
from typing import Tuple

from src.core.nn_types import *


class NnBackend:
    def __init__(self):
        # TODO: replace to bin/library/libnn_backend-linux-amd64-avx2.so
        self._lib = CDLL("/home/mhohsadze/CLionProjects/nn-backend/cmake-build-release/libnn_backend.so")

        self._instance_p = POINTER(c_void_p)()
        self._debug_messenger_p = POINTER(c_void_p)()
        self._phys_devices_p = POINTER(POINTER(c_void_p))()
        self._num_physical_devices = c_uint32(0)
        self._phys_device = POINTER(c_void_p)()
        self._queue_compute_index = c_int(0)
        self._memory_props = None
        self._device = None
        self._queue = None
        self._pipeline_cache = POINTER(c_void_p)()

        self.call()

    def _get_vk_phys_devices(self):
        self._lib.nnGetVkPhysicalDevices.restype = POINTER(POINTER(c_void_p))
        return self._lib.nnGetVkPhysicalDevices(self._instance_p, byref(self._num_physical_devices))

    def call(self):
        self._lib.nnCreateDefaultVkInstance(byref(self._instance_p))
        self._lib.nnCreateVkDebugUtilsMessenger(self._instance_p, byref(self._debug_messenger_p))

        self._lib.nnGetVkPhysicalDevices.restype = POINTER(POINTER(c_void_p))
        self._phys_devices_p = self._lib.nnGetVkPhysicalDevices(self._instance_p, byref(self._num_physical_devices))

        self._lib.nnGetVkPhysicalDeviceIndexByExtensionName.argtypes = POINTER(POINTER(c_void_p)), c_uint32, c_char_p
        self._lib.nnGetVkPhysicalDeviceIndexByExtensionName.restype = POINTER(c_void_p)

        self._phys_device = self._lib.nnGetVkPhysicalDeviceIndexByExtensionName(self._phys_devices_p,
                                                                                self._num_physical_devices,
                                                                                c_char_p(b"VK_NV_cooperative_matrix"))

        self._lib.nnGetVkQueueComputeIndex.restype = c_int
        self._queue_compute_index = self._lib.nnGetVkQueueComputeIndex(self._phys_device)

        if self._queue_compute_index == -1:
            print("Can't find physical device with queue compute.")
            sys.exit(1)

        self._lib.nnGetMemoryProperties.restype = NnMemoryProps
        self._memory_props = self._lib.nnGetMemoryProperties(self._phys_device)

        self._lib.nnCreateVkDevice.restype = POINTER(c_void_p)
        self._device = self._lib.nnCreateVkDevice(self._phys_device, self._queue_compute_index)

        self._lib.nnGetVkDeviceQueue.restype = POINTER(c_void_p)
        self._queue = self._lib.nnGetVkDeviceQueue(self._device, self._queue_compute_index)

        self._lib.nnCreateVkPipelineCache.restype = POINTER(c_void_p)
        self._pipeline_cache = self._lib.nnCreateVkPipelineCache(self._device,
                                                                 c_char_p(b"bin/pipeline_caches/pipeline_cache.data"))

    def _run_3layouts(self, nn_ma: NnMatrix, nn_mb: NnMatrix, shader: c_char_p) -> array.array:
        self._lib.nnCreateVkComputePipeline2MatricesAndOutput.restype = POINTER(c_void_p)
        pipeline = self._lib.nnCreateVkComputePipeline2MatricesAndOutput(self._device, self._pipeline_cache, shader)

        compute_info = NnComputeInfo()
        compute_info.device = self._device
        compute_info.pipeline_cache = pipeline
        compute_info.queue_compute_index = self._queue_compute_index
        compute_info.memory_props = self._memory_props
        compute_info.queue = self._queue

        mc_arr = array.array('f', [0] * (nn_ma.rows * nn_ma.columns))
        mc = nn_matrix_new(mc_arr, nn_ma.rows, nn_ma.columns)

        self._lib.nnRunTwoMatricesAndOutput.argtypes = [POINTER(NnComputeInfo), POINTER(NnMatrix), POINTER(NnMatrix),
                                                        POINTER(NnMatrix)]
        self._lib.nnRunTwoMatricesAndOutput.restype = None

        self._lib.nnRunTwoMatricesAndOutput(byref(compute_info), byref(nn_ma), byref(nn_mb), byref(mc))
        return mc_arr

    def add(self, nn_ma: NnMatrix, nn_mb: NnMatrix):
        return self._run_3layouts(nn_ma, nn_mb, c_char_p(b"bin/shaders/add.spv"))

    def hadamard(self, nn_ma: NnMatrix, nn_mb: NnMatrix):
        return self._run_3layouts(nn_ma, nn_mb, c_char_p(b"bin/shaders/hadamard.spv"))

    def cross_correlate2d_valid(self, nn_mi: NnMatrix, nn_mk: NnMatrix):
        output_rows = c_uint32(nn_mi.rows - nn_mk.rows + 1)
        output_columns = c_uint32(nn_mi.columns - nn_mk.columns + 1)
        output_len = output_rows.value * output_columns.value

        output = array.array('f', [0] * output_len)
        nn_mo = nn_matrix_new(output, output_rows, output_columns)

        self._lib.nnValidCrossCorrelationCpu.argtypes = [POINTER(NnMatrix), POINTER(NnMatrix), POINTER(NnMatrix)]
        self._lib.nnValidCrossCorrelationCpu.restype = None
        self._lib.nnValidCrossCorrelationCpu(byref(nn_mi), byref(nn_mk), byref(nn_mo))

        return nn_mo, output

    def cross_correlate2d_valid_gpu(self, nn_mi: NnMatrix, nn_mk: NnMatrix):
        self._lib.nnCreateVkComputePipelineCorrelate2dValid.restype = POINTER(c_void_p)
        pipeline = self._lib.nnCreateVkComputePipelineCorrelate2dValid(self._device, self._pipeline_cache)

        compute_info = NnComputeInfo()
        compute_info.device = self._device
        compute_info.pipeline_cache = pipeline
        compute_info.queue_compute_index = self._queue_compute_index
        compute_info.memory_props = self._memory_props
        compute_info.queue = self._queue

        output_rows = c_uint32(nn_mi.rows - nn_mk.rows + 1)
        output_columns = c_uint32(nn_mi.columns - nn_mk.columns + 1)
        output_len = output_rows.value * output_columns.value

        output = array.array('f', [0] * output_len)
        nn_mo = nn_matrix_new(output, output_rows, output_columns)

        self._lib.nnValidCrossCorrelationGpu.argtypes = [POINTER(NnComputeInfo), POINTER(NnMatrix), POINTER(NnMatrix),
                                                         POINTER(NnMatrix)]
        self._lib.nnValidCrossCorrelationGpu.restype = None
        self._lib.nnValidCrossCorrelationGpu(byref(compute_info), byref(nn_mi), byref(nn_mk), byref(nn_mo))

        return nn_mo

    def convolve2d_full(self, nn_mi: NnMatrix, nn_mk: NnMatrix):
        self._lib.nnFullCrossCorrelationCpu.restype = None

        self._lib.nnFullCrossCorrelationCpu.argtypes = [POINTER(NnMatrix), POINTER(NnMatrix), POINTER(NnMatrix)]

        output_rows = c_uint32(nn_mi.rows + nn_mk.rows - 1)
        output_columns = c_uint32(nn_mi.columns + nn_mk.columns - 1)
        output_len = output_rows.value * output_columns.value

        output = array.array('f', [0] * output_len)
        nn_mo = nn_matrix_new(output, output_rows, output_columns)

        self._lib.nnFullCrossCorrelationCpu(byref(nn_mi), byref(nn_mk), byref(nn_mo))
        return nn_mo, output

    def multiply(self, nn_ma: NnMatrix, nn_mb: NnMatrix):
        self._lib.nnMultiply.restype = None
        self._lib.nnMultiply.argtypes = [POINTER(NnMatrix), POINTER(NnMatrix), POINTER(NnMatrix)]

        mc_rows = c_uint32(nn_ma.rows)
        mc_columns = c_uint32(nn_mb.columns)
        mc_len = mc_rows.value * mc_columns.value

        output = array.array('f', [0] * mc_len)
        nn_mc = nn_matrix_new(output, mc_rows, mc_columns)

        self._lib.nnMultiply(byref(nn_ma), byref(nn_mb), byref(nn_mc))
        return nn_mc, output

    def copy(self, dst: POINTER(c_void_p), src: POINTER(c_void_p), count: c_size_t):
        self._lib.nnMemoryCopy.restype = None
        self._lib.nnMemoryCopy(dst, src, count)

    def save(self):
        self._lib.nnSaveVkPipelineCache(self._device, self._pipeline_cache)


nn_provider = NnBackend()
