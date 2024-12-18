import umap
import numpy as np
from pathlib import Path
from utils.log import logger
import anndata
import matplotlib.pyplot as plt
import scienceplots
from pp.neighbor import fast_knn_from_precomputed
from matplotlib.colors import ListedColormap
import seaborn as sns
import h5py

plt.style.use(['science', 'nature'])

n_neighbors = 30
random_state = 42

adata = anndata.read_h5ad("./data/contours.h5ad")
with h5py.File('./data/gw_dist.h5', 'r') as f:
    X = f["/gw_dist"][...]

target = adata.obs['time']
knn_indices, knn_dists = fast_knn_from_precomputed(X, n_neighbors)
disconnected_index = knn_dists == np.inf
knn_indices[disconnected_index] = -1
umap = umap.UMAP(
    precomputed_knn=(knn_indices, knn_dists, None),
    n_neighbors=n_neighbors,
    random_state=random_state,
    n_components=2,
    min_dist=0.5,
    densmap=False,
)

embedding = umap.fit_transform(X)
embedding = np.mean(embedding, axis=0) - embedding  # transform for better visualization
adata.obsm['X_umap'] = embedding
adata.obsp['connectivities'] = umap.graph_
adata.obsp['distances'] = X
adata.obsp['metric'] = X
adata.uns['neighbors'] = {
    'connectivities_key': 'connectivities',
    'distances_key': 'distances',
    'params': {
        'method': 'umap',
        'metric': 'metric',
        'n_neighbors': n_neighbors,
        'random_state': random_state,
    }
}
adata.write_h5ad(Path("./data/contours.h5ad"), compression='gzip')

plt.figure(figsize=(10, 6))
color_label = np.unique(target, return_inverse=True)[1]

plt.scatter(*embedding.T, s=15, c=color_label, cmap="Spectral", alpha=0.8)
plt.colorbar(shrink=0.5)
plt.axis('off')
plt.savefig("./figures/umap.svg", dpi=600)
plt.close()


import scvelo
scvelo.pp.filter_and_normalize()