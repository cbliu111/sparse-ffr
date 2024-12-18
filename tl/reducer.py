import umap
from umap.umap_ import find_ab_params
import numpy as np
from scipy.sparse import issparse
from scipy.special import softmax
import warnings
from utils.log import logger


def umap_embed(
        adata,
        *,
        n_components: int = 2,
        use_obsp: str | None = 'distances',
        use_obsm: str | None = None,
        n_neighbors: int | None = 50,
        random_state: float | None = 0,  # random state did influence the final embed result
        min_dist: float = 0.5,
        spread: float = 1.0,
        **kwargs,
):
    if use_obsp:
        if use_obsm:
            logger.warn("Both obsm and obsp are provided. Use obsp instead.")
        dist = adata.obsp[use_obsp]
    else:
        dist = adata.obsm[use_obsm]

    if issparse(dist):
        dist = dist.toarray()
    else:
        dist = np.array(dist)

    if n_neighbors is None:
        n_neighbors = 15

    a, b = find_ab_params(spread, min_dist)
    if use_obsp:
        reducer = umap.UMAP(metric="precomputed", random_state=random_state, n_neighbors=n_neighbors,
                            min_dist=min_dist, spread=spread, n_components=n_components)

    else:
        reducer = umap.UMAP(metric="euclidean", random_state=random_state, n_neighbors=n_neighbors,
                            min_dist=min_dist, spread=spread, n_components=n_components)

    logger.info(f"Computing umap embedding...")
    reduced_features = reducer.fit_transform(dist)
    adata.obsm['X_umap'] = reduced_features
    adata.uns['X_umap'] = {
        'random_state': random_state,
        'min_dist': min_dist,
        'spread': spread,
        'n_components': n_components,
        'params': {'a': a, 'b': b}
    }
