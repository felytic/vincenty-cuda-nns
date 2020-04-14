import math

import numpy as np
import numba
from numba import cuda
from cuda_friendly_vincenty import vincenty

wrap = cuda.jit('float32(float32, float32, float32, float32)', device=True)
vincenty = wrap(vincenty)


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


@cuda.jit('int32(int32)', device=True)
def next_right(node):
    """
    :param node: node index
    :return: index of next node
    """
    i = node

    # Go up left as far, as we can
    while i % 2 == 0:
        i = (i - 2) // 2

    # Node to the right
    return i + 1


@numba.njit
def shuffle(points, idx_array):
    result = np.zeros(points.shape, dtype=np.float32)

    for i in numba.prange(len(result)):
        result[i] = points[idx_array[i]]

    return result


@numba.njit
def unshuffle(points, idx_array):
    result = np.zeros(points.shape, dtype=np.float32)

    for i in numba.prange(len(result)):
        result[idx_array[i]] = points[i]

    return result


@numba.njit
def map_idx(array, idx_array):
    result = np.zeros(array.shape, dtype=np.float32)

    for i in numba.prange(len(result)):
        result[i] = idx_array[array[i]]

    return result


@cuda.jit('float32(float32[:], int32, float32[:,:], float32[:])', device=True)
def distance_to_node(point, node, centroids, radiuses):
    distance = vincenty(point[0], point[1],
                        centroids[node][0], centroids[node][1])

    distance = distance - radiuses[node]

    # 1e-4 meters is less than Vincenty's formula accuracy
    return distance if distance > 1e-4 else 0


@cuda.jit('void(int32, int32, float32[:,:], float32[:,:], int32[:,:])',
          device=True)
def process_node(i, node, points, distances, indices):
    n_points = points.shape[0]
    n_neighbors = distances.shape[1]

    start = node_range_start(node, n_points)
    end = node_range_end(node, n_points)

    for j in range(start, end):
        p1 = points[i]
        p2 = points[j]

        dist = vincenty(p1[0], p1[1], p2[0], p2[1])

        if distances[i][0] > dist:
            # replace farthest neighbor with current point
            distances[i][0] = dist
            indices[i][0] = j

            # bubble sort neighbors
            for k in range(1, n_neighbors):
                if distances[i][k - 1] < distances[i][k]:

                    # swap distances
                    temp = distances[i][k]
                    distances[i][k] = distances[i][k - 1]
                    distances[i][k - 1] = temp

                    # swap indices
                    temp = indices[i][k]
                    indices[i][k] = indices[i][k - 1]
                    indices[i][k - 1] = temp


@cuda.jit('void(float32[:,:], float32[:,:], float32[:], float32[:,:],'
          'int32[:,:])')
def query(points, centroids, radiuses, distances, indices):
    i = cuda.grid(1)

    if i >= points.shape[0]:
        return

    point = points[i]
    n_nodes = centroids.shape[0]
    n_points = points.shape[0]

    home_node = point_id_to_node(i, n_points, n_nodes)
    node = home_node

    while node < n_nodes:
        distance = distance_to_node(point, node, centroids, radiuses)
        left_child = node * 2 + 1

        # distance to the node more tan current min distance
        if distance > distances[i][0]:
            node = next_right(node)

        # this node has children
        elif left_child < n_nodes:
            node = left_child

        # find distances to all points in the node
        else:
            process_node(i, node, points, distances, indices)
            node = next_right(node)

        # we walked all nodes in the tree and came back
        if node == home_node:
            break
