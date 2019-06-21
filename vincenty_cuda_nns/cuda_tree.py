import numpy as np

from .query_functions import query, shuffle, unshuffle, map_idx
from .building_functions import recursive_build
from functools import partial


class CudaTree:
    def __init__(self, data, leaf_size=5):
        """
        Build Ball Tree for points on Earth's ellipsoid

        :param data: array of points like (longitude, latitude)
        :param leaf_size: approximate size of tree's smallest nodes
        """
        self.data = data
        self.leaf_size = leaf_size

        # validate data
        if self.data.size == 0:
            raise ValueError('data is an empty array')

        if leaf_size < 1:
            raise ValueError('leaf_size must be greater than or equal to 1')

        self.n_samples = self.data.shape[0]
        self.n_features = self.data.shape[1]

        self.n_levels = int(
            1 + np.log2(max(1, ((self.n_samples - 1) // self.leaf_size)))
        )
        self.n_nodes = int(2 ** self.n_levels) - 1

        # allocate arrays for storage
        self.idx_array = np.arange(self.n_samples, dtype=np.int32)
        self.node_radius = np.zeros(self.n_nodes, dtype=np.float32)
        self.node_idx = np.zeros((self.n_nodes, 2), dtype=np.int32)
        self.node_centroids = np.zeros((self.n_nodes, self.n_features),
                                       dtype=np.float32)

        # build the tree
        recursive_build(0, self.data, self.node_centroids,
                        self.node_radius, self.idx_array, self.node_idx,
                        self.n_nodes, self.leaf_size)

        self.shuffle = partial(shuffle, idx_array=self.idx_array)
        self.unshuffle = partial(unshuffle, idx_array=self.idx_array)
        self.map_idx = partial(map_idx, idx_array=self.idx_array)

    def query(self, n_neighbors=2, threadsperblock=64):
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
        n = len(self.data)

        distances = np.zeros((n, n_neighbors), dtype=np.float32)
        distances[:] = np.inf
        indices = np.zeros((n, n_neighbors), dtype=np.int32)

        new_points = self.data
        new_points = self.shuffle(self.data)

        blockspergrid = int(np.ceil(n / 64))
        query[blockspergrid, threadsperblock](new_points, self.node_centroids,
                                              self.node_radius, distances,
                                              indices)

        distances = self.unshuffle(np.flip(distances, 1))

        indices = np.apply_along_axis(self.map_idx, 0, indices)
        indices = self.unshuffle(np.flip(indices, 1)).astype(np.int32)

        return distances, indices
