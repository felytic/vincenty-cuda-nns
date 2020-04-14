import numpy as np
from cuda_friendly_vincenty import vincenty


def brute_force(points):
    n = len(points)

    distances = np.zeros(n, dtype=np.float32)
    distances[:] = np.inf

    indices = np.zeros(n, dtype=np.int32)

    for x in range(n):
        for y in range(x + 1, n):
            distance = vincenty(points[x, 0], points[x, 1],
                                points[y, 0], points[y, 1])

            if (distance < distances[x]):
                distances[x] = distance
                indices[x] = y

            if (distance < distances[y]):
                distances[y] = distance
                indices[y] = x

    return distances, indices
