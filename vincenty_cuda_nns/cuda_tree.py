import numpy as np

from .query_functions import query, map_idx
from .building_functions import recursive_build
from functools import partial


class CudaTree:
    def __init__(self, data, leaf_size=5):
        """
        Build Ball Tree for points clusters on Earth's ellipsoid
        with indexing like this:

                      0
                      |
              1----------------2
              |                |
          3------4        5--------6
          |      |        |        |
        7---8  9---10  11---12  13---14

        Each cluster (node) described as centroid [lng, lat] and
        it's radius in meters.

        :param data: array of points like (longitude, latitude)
        :param leaf_size: approximate size of tree's smallest nodes
        """
        self.data = np.array(data, dtype=np.float32)
        self.leaf_size = leaf_size

        # validate data
        if self.data.size == 0:
            raise ValueError('data is an empty array')

        if leaf_size < 1:
            raise ValueError('leaf_size must be greater than or equal to 1')

        n = self.data.shape[0]

        self.n_levels = int(1 + np.log2(max(1, ((n - 1) // self.leaf_size))))
        self.n_nodes = int(2 ** self.n_levels) - 1

        # allocate arrays for storage
        self.idx_array = np.arange(n, dtype=np.int32)
        self.node_radius = np.zeros(self.n_nodes, dtype=np.float32)
        self.node_idx = np.zeros((self.n_nodes, 2), dtype=np.int32)
        self.node_centroids = np.zeros((self.n_nodes, 2), dtype=np.float32)

        # build the tree
        recursive_build(0, self.data, self.node_centroids,
                        self.node_radius, self.idx_array, self.node_idx,
                        self.n_nodes, self.leaf_size)

        self.map_idx = partial(map_idx, idx_array=self.idx_array)

    def query(self, data, n_neighbors=2, threadsperblock=64):
        """
        Search nearest neighbors for each point inside the tree

        :param threadsperblock: GPU threads per block, see
        numba.pydata.org/numba-doc/dev/cuda/kernels.html#kernel-invocation

        :param n_neighbors: number of n_neighbors to search including itself
        :return: distances: each entry gives the list of distances to the
                            neighbors of the corresponding point
                 indices: each entry gives the list of indices of neighbors of
                          the corresponding point
        """
        data = np.array(data, dtype=np.float32)
        n = data.shape[0]

        # validate data
        if n == 0:
            raise ValueError('data is an empty array')

        distances = np.zeros((n, n_neighbors), dtype=np.float32)
        distances[:] = np.inf
        indices = np.zeros((n, n_neighbors), dtype=np.int32)

        blockspergrid = int(np.ceil(n / 64))
        query[blockspergrid, threadsperblock](data, self.data,
                                              self.node_centroids,
                                              self.node_radius, distances,
                                              indices)

        indices = np.apply_along_axis(self.map_idx, 0, indices)

        distances = np.flip(distances, 1)
        indices = np.flip(indices, 1)

        return distances, indices
