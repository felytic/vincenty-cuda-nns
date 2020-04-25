from numba import cuda
from .utils import node_range_start, node_range_end, point_id_to_node, get_dist


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


@cuda.jit('float32(float32[:], int32, float32[:,:], float32[:])', device=True)
def distance_to_node(point, node, centroids, radiuses):
    distance = get_dist(point[0], point[1],
                        centroids[node, 0], centroids[node, 1])

    distance = distance - radiuses[node]

    # 1e-4 meters is less than Vincenty's formula accuracy
    return distance if distance > 1e-4 else 0


@cuda.jit('void(int32, float32[:], float32[:,:], int32[:], float32[:,:],'
          'int32[:,:])', device=True)
def process_node(node, point, tree_points, idx_array, distances, indices):
    i = cuda.grid(1)
    n_points = tree_points.shape[0]
    n_neighbors = distances.shape[1]

    start = node_range_start(node, n_points)
    end = node_range_end(node, n_points)

    for jj in range(start, end):
        j = idx_array[jj]
        tree_point = tree_points[j]

        dist = get_dist(point[0], point[1], tree_point[0], tree_point[1])

        if distances[i][0] > dist:
            # replace farthest neighbor with current point
            distances[i][0] = dist
            indices[i][0] = jj

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


@cuda.jit('int32(float32[:], float32[:,:], int32[:], float32[:,:],'
          'float32[:])', device=True)
def get_home_node(point, tree_points, idx_array, centroids, radiuses):
    i = cuda.grid(1)
    n_nodes = centroids.shape[0]
    n_points = tree_points.shape[0]

    if i < tree_points.shape[0]:
        ii = idx_array[i]
        # if this point belongs to the tree
        if point[0] == tree_points[ii][0] and point[1] == tree_points[ii][1]:
            return point_id_to_node(i, n_points, n_nodes)

    home_node = 0
    node = home_node

    # while node has childrens
    while node < n_nodes:
        distance = distance_to_node(point, node, centroids, radiuses)

        # if this point intersects with the node
        if distance == 0:
            home_node = node
            node = node * 2 + 1  # left child

        else:
            node = next_right(node)

            # walk around tree completed
            if node == 0:
                break

    return home_node


@cuda.jit('void(float32[:,:], float32[:,:], int32[:], float32[:,:],'
          'float32[:], float32[:,:], int32[:,:])')
def query(points, tree_points, idx_array, centroids, radiuses, distances,
          indices):
    i = cuda.grid(1)

    if i >= points.shape[0]:
        return

    point = points[i]
    n_nodes = centroids.shape[0]

    home_node = get_home_node(point, tree_points, idx_array, centroids,
                              radiuses)
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
            process_node(node, point, tree_points, idx_array, distances,
                         indices)
            node = next_right(node)

        # we walked all nodes in the tree and came back
        if node == home_node:
            break
