import numpy as np
from cuda_friendly_vincenty import vincenty


def brute_force(points, neighbors=None, n_neighbors=2):
    if neighbors is None:
        neighbors = points

    n = len(points)

    distances = np.zeros((n, n_neighbors), dtype=np.float32)
    distances[:] = np.inf

    indices = np.zeros((n, n_neighbors), dtype=np.int32)

    for i in range(n):
        for j in range(len(neighbors)):
            distance = vincenty(points[i, 0], points[i, 1],
                                neighbors[j, 0], neighbors[j, 1])

            if distance < distances[i][0]:
                distances[i][0] = distance
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

    distances = np.flip(distances, 1)
    indices = np.flip(indices, 1)
    return distances, indices
