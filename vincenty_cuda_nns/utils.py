import math
import numpy as np
import numba
from numba import cuda
from cuda_friendly_vincenty import vincenty


wrap = cuda.jit('float32(float32, float32, float32, float32)', device=True)
get_dist = wrap(vincenty)


@cuda.jit('int32(int32)', device=True)
def node_to_level(node):
    return math.floor(math.log(np.float32(node + 1)) / math.log(np.float32(2)))


@cuda.jit('int32(int32, int32)', device=True)
def node_range_start(node, n):
    level = node_to_level(node)
    step = n / (2**level)
    pos = node - 2**level + 1
    return math.floor(pos * step)


@cuda.jit('int32(int32, int32)', device=True)
def node_range_end(node, n):
    level = node_to_level(node)
    step = n / (2**level)
    pos = node - 2**level + 1
    return math.floor((pos + 1) * step)


@cuda.jit('int32(int32, int32, int32)', device=True)
def point_id_to_node(point_id, n_points, n_nodes):
    step = n_points / ((n_nodes + 1) // 2)
    level = node_to_level(n_nodes - 1)
    return 2**level + (math.ceil((point_id + 1) / step) - 2)


@numba.njit
def map_idx(array, idx_array):
    result = np.zeros(array.shape, dtype=idx_array.dtype)

    for i in numba.prange(len(result)):
        result[i] = idx_array[array[i]]

    return result
