import pytest
import numpy as np
from vincenty_cuda_nns import CudaTree

from .utils import brute_force

DATASET_SIZES = [10**n for n in range(3)]
N_NEIGBORS = [*range(1, 4)]


def process_array(array, n_neighbors=2):
    distances, indices = brute_force(array, n_neighbors=n_neighbors)

    cuda_tree = CudaTree(array, leaf_size=5)
    cuda_dist, cuda_ind = cuda_tree.query(array, n_neighbors=n_neighbors)

    assert np.allclose(cuda_dist, distances, atol=1)

    if not (array[0] == array[:]).all():
        assert (cuda_ind == indices).all()


def process_two_arrays(array, size, n_neighbors=2):
    from_array = generate_points(size)

    distances, indices = brute_force(from_array, neighbors=array,
                                     n_neighbors=n_neighbors)

    cuda_tree = CudaTree(array, leaf_size=5)
    cuda_dist, cuda_ind = cuda_tree.query(from_array, n_neighbors=n_neighbors)

    assert np.allclose(cuda_dist, distances, atol=1)

    if not (array[0] == array[:]).all():
        assert (cuda_ind == indices).all()


def generate_points(size):
    # random points
    X = (np.random.random((size, 2)) * 180) - 90

    # some zero points
    zeros = np.zeros((6, 2))

    # some ±180 points
    sides = np.zeros((6, 2))
    sides[::2, 0] = 180
    sides[1::2, 0] = -180
    sides[:, 1] = (np.random.random(6) * 180) - 90

    # some ±180 points
    poles = np.zeros((6, 2))
    poles[::2, 1] = 90
    poles[1::2, 1] = -90
    poles[:, 0] = (np.random.random(6) * 180) - 90

    # edge cases
    edges = np.array([[-180, -90], [-180, 90], [180, -90], [180, 90]])

    result = np.concatenate((X, zeros, sides, poles, edges))
    np.random.shuffle(result)

    return result


@pytest.mark.parametrize('size', DATASET_SIZES)
@pytest.mark.parametrize('n_neighbors', N_NEIGBORS)
def test_distances(size, n_neighbors):
    X = (np.random.random((size, 2)) * 180) - 90
    process_array(X, n_neighbors=n_neighbors)


@pytest.mark.parametrize('size', DATASET_SIZES)
@pytest.mark.parametrize('source_size', DATASET_SIZES)
def test_two_arrays(size, source_size):
    X = (np.random.random((size, 2)) * 180) - 90
    process_two_arrays(X, source_size)


@pytest.mark.parametrize('size', DATASET_SIZES)
def test_zero_meredian(size):
    X = (np.random.random((size, 2)) * 180) - 90
    X[::2, 0] = 0
    process_array(X)


@pytest.mark.parametrize('size', DATASET_SIZES)
def test_180_meredian(size):
    X = (np.random.random((size, 2)) * 180) - 90
    X[::3, 0] = 180
    X[1::3, 0] = -180
    process_array(X)


@pytest.mark.parametrize('size', DATASET_SIZES)
def test_poles(size):
    X = (np.random.random((size, 2)) * 180) - 90
    X[::3, 0] = 90
    X[1::3, 0] = -90
    process_array(X)


@pytest.mark.parametrize('size', DATASET_SIZES)
def test_zeros(size):
    X = np.zeros((size, 2))
    process_array(X)
