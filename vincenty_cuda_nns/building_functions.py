import math
import numba
from cuda_friendly_vincenty import vincenty


vincenty = numba.jit(vincenty)


@numba.njit
def node_id_to_range(node_id, n):
    """
    :param node_id: index of the node
    :param n: number of points

    :return: indexes of first and last points in the node
    """

    level = math.floor(math.log(node_id + 1) / math.log(2))
    step = n / (2 ** level)
    pos = node_id - 2 ** level + 1
    idx_start = math.floor(pos * step)
    idx_end = math.floor((pos + 1) * step)

    return idx_start, idx_end


@numba.njit
def recursive_build(i_node, data, node_centroids, node_radius, idx_array,
                    node_idx, n_nodes, leaf_size):
    idx_start, idx_end = node_id_to_range(i_node, data.shape[0])

    # determine Node centroid
    node_centroids[i_node] = [0, 0]

    for i in range(idx_start, idx_end):
        node_centroids[i_node] += data[idx_array[i]]

    node_centroids[i_node] /= (idx_end - idx_start)

    # determine Node radius
    radius = 0.0

    for i in range(idx_start, idx_end):
        dist = vincenty(node_centroids[i_node][0],
                        node_centroids[i_node][1],
                        data[idx_array[i]][0],
                        data[idx_array[i]][1])
        if dist > radius:
            radius = dist

    # set node properties
    node_radius[i_node] = radius
    node_idx[i_node] = idx_start, idx_end

    i_child = 2 * i_node + 1

    # recursively create subnodes
    if i_child + 1 >= n_nodes:
        if idx_end - idx_start > 2 * leaf_size:
            print('Memory layout is flawed: not enough nodes allocated')

    elif idx_end - idx_start < 2:
        print('Memory layout is flawed: not enough nodes allocated')

    else:
        partition_indices(data, idx_array, idx_start, idx_end)

        recursive_build(i_child, data, node_centroids, node_radius,
                        idx_array, node_idx, n_nodes, leaf_size)
        recursive_build(i_child + 1, data, node_centroids, node_radius,
                        idx_array, node_idx, n_nodes, leaf_size)


@numba.njit
def partition_indices(data, idx_array, idx_start, idx_end):

    min_x, min_y = 180, 90
    max_x, max_y = -180, -90

    for i in range(idx_start, idx_end):
        max_x = max(max_x, data[idx_array[i], 0])
        max_y = max(max_y, data[idx_array[i], 1])

        min_x = min(min_x, data[idx_array[i], 0])
        min_y = min(min_y, data[idx_array[i], 1])

    dim = 0 if ((max_x - min_x) > (max_y - min_y)) else 1

    mid = (idx_start + idx_end) // 2
    rr = idx_end - 1
    ll = idx_start

    while rr > ll:

        pivot = data[idx_array[(ll + rr) // 2], dim]
        left = ll
        right = rr

        while left < right:
            while data[idx_array[left], dim] < pivot:
                left += 1

            while data[idx_array[right], dim] > pivot:
                right -= 1

            if left <= right:
                idx_array[left], idx_array[right] = \
                    idx_array[right], idx_array[left]
                left += 1
                right -= 1

        if (ll < mid) and (right > mid):
            rr = right

        elif (left < mid) and (rr > mid):
            ll = left
        else:
            break
