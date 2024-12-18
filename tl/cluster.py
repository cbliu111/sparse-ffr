import sklearn
import scipy
import numpy as np
import pandas as pd

from pp.cajal.utilities import leiden_clustering


def kmeans(
        adata,
        *,
        use_obsm: str = 'X_pca',
        n_cluster: int = 10,
):
    _m = adata.obsm[use_obsm]
    norm_vectors = sklearn.preprocessing.normalize(_m)
    reducer = sklearn.cluster.KMeans(
        n_clusters=n_cluster,
        init='k-means++',
        n_init=3,
        max_iter=300,
        random_state=9
    ).fit(norm_vectors)  # init is plus,but orginally cluster, not available in sklearn
    cluster_centers = reducer.cluster_centers_
    distance_to_centers = scipy.spatial.distance.cdist(_m, cluster_centers, metric='euclidean')
    cluster_ids = np.argmin(distance_to_centers, axis=1)
    min_distances = np.around(np.amin(distance_to_centers, axis=1), decimals=2)
    goodness = scipy.special.softmax(1 - distance_to_centers, axis=1)  # probabilities of belonging to each cluster

    modes = []
    reduced_shape_modes = []
    _f = adata.obsm["realigned"]
    _rf = adata.obsm["X_pca"]
    for s in set(cluster_ids):
        idx = cluster_ids == s
        modes.append(np.mean(_f[idx, :], axis=0))
        reduced_shape_modes.append(np.mean(_rf[idx, :], axis=0))
    modes = np.array(modes)
    reduced_modes = np.array(reduced_shape_modes)

    adata.uns["kmean_cluster"] = {
        "centers": cluster_centers,
        "distances_to_centers": distance_to_centers,
        "min_distances": min_distances,
        "goodness": goodness,
        "modes": modes,
        "reduced_modes": reduced_modes,
    }
    adata.obsm["kmean_cluster_id"] = pd.Categorical(cluster_ids)
