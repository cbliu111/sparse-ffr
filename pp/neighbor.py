from anndata import AnnData
import numpy as np
from numpy.typing import NDArray
from pandas import Index
from scipy.sparse import coo_matrix, csr_matrix, issparse
import numba
from umap.umap_ import fuzzy_simplicial_set
from utils.log import logger


@numba.njit(parallel=True)
def fast_knn_indices_from_precomputed(X, n_neighbors):
    knn_indices = np.empty((X.shape[0], n_neighbors), dtype=np.int32)
    for row in numba.prange(X.shape[0]):
        # v = np.argsort(X[row])  # Need to call argsort this way for numba
        v = X[row].argsort(kind="quicksort")
        v = v[:n_neighbors]
        knn_indices[row] = v
    return knn_indices


def fast_knn_from_precomputed(X, n_neighbors):
    """A fast computation of knn indices and distances.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The input data to compute the k-neighbor indices of.

    n_neighbors: int
        The number of nearest neighbors to compute for each sample in ``X``.

    Returns
    -------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` the closest points in the dataset.
    knn_distances: array of shape (n_samples, n_neighbors)
    """
    knn_indices = fast_knn_indices_from_precomputed(X, n_neighbors)
    knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()
    return knn_indices, knn_dists


def connectivity_gauss(
        distances: NDArray[np.float32 | np.float64],
        n_neighbors: int,
        *,
        knn: bool
) -> csr_matrix:
    """
    Derive gaussian connectivities between data points from their distances.

    Parameters
    ----------
    distances
        The input matrix of distances between data points.
    n_neighbors
        The number of nearest neighbors to consider.
    knn
        Specify if the distances have been restricted to k nearest neighbors.
    """
    # init distances
    Dsq = np.power(distances, 2)
    indices, distances_sq = fast_knn_from_precomputed(
        Dsq, n_neighbors
    )

    # exclude the first point, the 0th neighbor
    indices = indices[:, 1:]
    distances_sq = distances_sq[:, 1:]

    # choose sigma, the heuristic here doesn't seem to make much of a difference,
    # but is used to reproduce the figures of Haghverdi et al. (2016)

    # the last item is already in its sorted position through argpartition
    # we have decay beyond the n_neighbors neighbors
    sigmas_sq = distances_sq[:, -1] / 4
    sigmas = np.sqrt(sigmas_sq)

    # compute the symmetric weight matrix
    Num = 2 * np.multiply.outer(sigmas, sigmas)
    Den = np.add.outer(sigmas_sq, sigmas_sq)
    W = np.sqrt(Num / Den) * np.exp(-Dsq / Den)
    # make the weight matrix sparse
    if not knn:
        mask = W > 1e-14
        W[~mask] = 0
    else:
        # restrict number of neighbors to ~k
        # build a symmetric mask
        mask = np.zeros(Dsq.shape, dtype=bool)
        for i, row in enumerate(indices):
            mask[i, row] = True
            for j in row:
                if i not in set(indices[j]):
                    W[j, i] = W[i, j]
                    mask[j, i] = True
        # set all entries that are not nearest neighbors to zero
        W[~mask] = 0

    return csr_matrix(W)


def connectivity_umap(
        X: NDArray[np.float32 | np.float64],
        *,
        n_neighbors: int,
        set_op_mix_ratio: float = 1.0,
        local_connectivity: float = 1.0,
        rand_state: float = 0,
) -> csr_matrix:
    """\
    This is from umap.fuzzy_simplicial_set :cite:p:`McInnes2018`.

    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    """

    # Compute indices of n nearest neighbors
    knn_indices, knn_dists = fast_knn_from_precomputed(X, n_neighbors)
    # Prune any nearest neighbours that are infinite distance apart.
    disconnected_index = knn_dists == np.inf
    knn_indices[disconnected_index] = -1

    # use empty matrix to fit the api of umap
    Y = coo_matrix(([], ([], [])), shape=(X.shape[0], 1))
    connectivities, _sigmas, _rhos = fuzzy_simplicial_set(
        Y,
        n_neighbors,
        rand_state,
        None,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
    )

    return connectivities.tocsr()


def soft_max(x):
    x = np.exp(x)
    return x / np.sum(x)


@numba.njit(parallel=True)
def renomralize_dist(x, n_elements):
    # metric usually are very insensitive for close points
    # therefore only large distances are kept
    x = (x - np.min(x)) / np.std(x)
    for row in numba.prange(x.shape[0]):
        v = x[row].argsort(kind="quicksort")
        v = v[:x.shape[0] - n_elements]
        x[row, v] = 0
    return x


@numba.njit(parallel=True)
def compute_space_time_distances(x, time_vector, time_factor):
    n_row = x.shape[0]
    n_col = x.shape[1]
    for row in numba.prange(n_row):
        for col in numba.prange(n_col):
            t_scr = time_vector[row]
            t_des = time_vector[col]
            ds = x[row, col]
            dt = (t_des - t_scr) * time_factor
            x[row, col] = np.sqrt(ds ** 2 + dt ** 2)
    return x


def neighbors_precomputed(
        adata: AnnData,
        *,
        n_neighbors: int = 15,
        knn: bool = True,
        mask: np.ndarray | bool | Index = None,
        metric_key: str = 'metric',
        time_key: str | None = None,
        method: str = "umap",
        random_state: float = 0,
        time_factor: float | None = None,
):
    if mask is not None:
        # select masked samples to calculate connectivities
        x = adata[mask].obsp[metric_key].copy()
    else:
        x = adata.obsp[metric_key].copy()

    if issparse(x):
        x = x.toarray()

    if time_factor is not None and time_key is not None:
        t_vec = adata.obs[time_key].to_numpy()
        x = compute_space_time_distances(x, t_vec, time_factor)

    logger.info(f"Computing {n_neighbors} nearest neighbors")
    conn = None
    if method == 'umap':
        conn = connectivity_umap(x, n_neighbors=n_neighbors, rand_state=random_state)
    elif method == 'gauss':
        conn = connectivity_gauss(x, n_neighbors=n_neighbors, knn=knn)

    if conn is not None:
        if mask is None:
            adata.obsp['connectivities'] = conn
            adata.obsp['distances'] = x
            adata.uns['neighbors'] = {
                'connectivities_key': 'connectivities',
                'distances_key': 'distances',
                'params': {
                    'method': method,
                    'metric': metric_key,
                    'n_neighbors': n_neighbors,
                    'random_state': random_state,
                }
            }
        else:
            return conn
    else:
        raise ValueError("No connectivity matrix was computed.")
