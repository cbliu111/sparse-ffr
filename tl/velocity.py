import numpy as np
import scipy.sparse as sp


def compute_flows(
        adata,
        basis: str = "X_umap",
        t_basis: str = 'dpt_pseudotime',
):
    X = adata.obsm[basis]
    V = np.zeros_like(X)
    tmat = adata.obsp['transition_matrix']
    conn = adata.obsp['connectivities']
    t = adata.obs[t_basis].to_numpy()
    for row_id, row in enumerate(tmat):
        conn_idxs = conn[row_id, :].indices
        # conn_idxs = row.toarray()[0] > 0  # using transition matrix to obtain the neighbors
        dX = X[conn_idxs] - X[row_id, None]
        dt = np.abs(t[conn_idxs] - t[row_id, None])

        if np.any(np.isnan(dX)):
            V[row_id, :] = np.nan
        else:
            probs = row[:, conn_idxs].A.squeeze() if sp.issparse(row) else row[conn_idxs]
            dV = dX / dt[:, None]
            dV = np.nan_to_num(dV)
            dV /= np.linalg.norm(dV, axis=1)[:, None]
            if dV.shape[0] > 1:
                V[row_id, :] = probs.dot(dV) - dV.sum(0) / dV.shape[0]

    adata.obsm['flows_umap'] = V


def quiver_autoscale(X_emb, V_emb):
    """TODO."""
    import matplotlib.pyplot as pl

    scale_factor = np.abs(X_emb).max()  # just so that it handles very large values
    fig, ax = pl.subplots()
    Q = ax.quiver(
        X_emb[:, 0] / scale_factor,
        X_emb[:, 1] / scale_factor,
        V_emb[:, 0],
        V_emb[:, 1],
        angles="xy",
        scale_units="xy",
        scale=None,
    )
    Q._init()
    fig.clf()
    pl.close(fig)
    return Q.scale / scale_factor