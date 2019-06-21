# Vincenty nearest neighbor search using CUDA
Nearest neighbor search algorithm on Earth's surface that runs on a GPU and uses [Vincenty's formula](https://en.wikipedia.org/wiki/Vincenty%27s_formulae)

## Requirements
- CUDA-enabled GPU with compute capability 2.0 or above with an up-to-data Nvidia driver.
- [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html])


## Usage example
```python
import geopandas as gpd
from vincenty_cuda_nns import CudaTree

df = gpd.read_file('points.geojson')
X = np.stack(df['geometry']).astype(np.float32)

cuda_tree = CudaTree(X, leaf_size=4)
distances, indices = cuda_tree.query(n_neighbors=2)
```
