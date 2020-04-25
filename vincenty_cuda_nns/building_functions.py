import math
from numba import cuda
from .utils import node_range_start, node_range_end, get_dist


@cuda.jit('void(float32[:,:], float32[:,:], float32[:], int32[:], int32,'
          'int32)')
def process_nodes(points, centroids, radiuses, idx_array, n_nodes, leaf_size):
    node = cuda.grid(1)
    if node >= n_nodes:
        return

    n_points = points.shape[0]
    start = node_range_start(node, n_points)
    end = node_range_end(node, n_points)

    for ii in range(start, end):
        i = idx_array[ii]
        centroids[node, 0] += points[i][0]
        centroids[node, 1] += points[i][1]

    centroids[node, 0] /= (end - start)
    centroids[node, 1] /= (end - start)

    for ii in range(start, end):
        i = idx_array[ii]
        dist = get_dist(centroids[node, 0], centroids[node, 1],
                        points[i, 0], points[i, 1])

        if dist > radiuses[node]:
            radiuses[node] = dist


@cuda.jit('int32(int32, int32)', device=True)
def get_node(n_points, level):
    i = cuda.grid(1)

    n_nodes = int(2 ** (level + 1)) - 1
    step = n_points / ((n_nodes + 1) // 2)
    return 2**level + (math.ceil((i + 1) / step) - 2)


@cuda.jit('void(float32[:,:], int32, int32[:], int32[:])')
def sort_level(points, level, idx_array, result):
    ii = cuda.grid(1)
    n_points = points.shape[0]

    if ii >= n_points:
        return

    i = idx_array[ii]
    node = get_node(n_points, level)
    start = node_range_start(node, n_points)
    end = node_range_end(node, n_points)

    min_x, min_y = 180, 90
    max_x, max_y = -180, -90
    pos_x, pos_y = start, start

    for jj in range(start, end):
        j = idx_array[jj]
        # x
        max_x = max(max_x, points[j, 0])
        min_x = min(min_x, points[j, 0])

        if (
            points[j, 0] < points[i, 0] or
            (
                points[j, 0] == points[i, 0]
                and j < i
            )
        ):
            pos_x += 1

        # y
        max_y = max(max_y, points[j, 1])
        min_y = min(min_y, points[j, 1])

        if (
            points[j, 1] < points[i, 1] or
            (
                points[j, 1] == points[i, 1]
                and j < i
            )
        ):
            pos_y += 1

    pos = pos_x if (max_x - min_x) > (max_y - min_y) else pos_y
    result[pos] = i
