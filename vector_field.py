import anndata
from pathlib import Path
import tl
import dynamo as dyn
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
from utils import logger
import pandas as pd
import scienceplots
import pl
plt.style.use(['science', 'nature'])



def default_quiver_args(arrow_size, arrow_len=None):
    if isinstance(arrow_size, (list, tuple)) and len(arrow_size) == 3:
        head_w, head_l, ax_l = arrow_size
    elif type(arrow_size) in [int, float]:
        head_w, head_l, ax_l = 10 * arrow_size, 12 * arrow_size, 8 * arrow_size
    else:
        head_w, head_l, ax_l = 10, 12, 8

    scale = 1 / arrow_len if arrow_len is not None else 1 / arrow_size

    return head_w, head_l, ax_l, scale


def quiver_autoscaler(X_emb: np.ndarray, V_emb: np.ndarray) -> float:
    """Function to automatically calculate the value for the scale parameter of quiver plot, adapted from scVelo

    Args:
        X_emb: X, Y-axis coordinates
        V_emb: Velocity (U, V) values on the X, Y-axis

    Returns:
        The scale for quiver plot
    """

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    scale_factor = np.ptp(X_emb, 0).mean()
    X_emb = X_emb - X_emb.min(0)

    if len(V_emb.shape) == 3:
        Q = ax.quiver(
            X_emb[0] / scale_factor,
            X_emb[1] / scale_factor,
            V_emb[0],
            V_emb[1],
            angles="xy",
            scale_units="xy",
            scale=None,
        )
    else:
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
    plt.close(fig)

    return Q.scale / scale_factor * 2


times = np.array([
    1.0,
    1.333,
    1.666,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
])

# Normalize the values
norm = Normalize(vmin=min(times), vmax=max(times))
colormap = matplotlib.colormaps['viridis']

quiver_size = 1
head_w, head_l, ax_l, scale = default_quiver_args(quiver_size, 6)
quiver_kwargs = {
    "angles": "xy",
    "scale": scale,
    "scale_units": "xy",
    "width": 0.0005,
    "headwidth": head_w,
    "headlength": head_l,
    "headaxislength": ax_l,
    "minshaft": 1,
    "minlength": 1,
    "pivot": "tail",
    "linewidth": 0.1,
    "edgecolors": "black",
    "alpha": 1,
    "zorder": 10,
}

adata = anndata.read_h5ad(Path("./data/contours.h5ad"))
logger.info(adata)
values = adata.obs['time']
color_list = colormap(norm(values))

############################################################################################################
#                                     compute flow                                                         #
#                                                                                                          #
############################################################################################################

tl.compute_flows(adata, basis='X_umap', t_basis='dpt_pseudotime')

X = adata.obsm['X_umap']
flow = adata.obsm['flows_umap']
print(f"range of flow : {flow.max()}, {flow.min()}")

plt.figure(figsize=(10, 6))
flow /= 3 * quiver_autoscaler(X, flow)
plt.quiver(
    X[:, 0],
    X[:, 1],
    flow[:, 0],
    flow[:, 1],
    color=color_list,
    facecolors=color_list,
    **quiver_kwargs,
)
plt.axis('off')
plt.savefig(f"./figures/flows.svg", dpi=600)

############################################################################################################
#                                     compute score                                                        #
#                                                                                                          #
############################################################################################################

score = np.zeros_like(X)
dx = dy = 1e-6

gmm_dicts = adata.uns['gmm_dicts']
for i, k in enumerate(gmm_dicts.keys()):
    d = gmm_dicts[k]
    indices = d['indices']
    means = d['means']
    covariances = d['covariances']
    weights = d['weights']
    precisions_cholesky = d['precisions_cholesky']
    gmm = GaussianMixture(n_components=means.shape[0], covariance_type='full', random_state=0)
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.weights_ = weights
    gmm.precisions_cholesky_ = precisions_cholesky
    obs = X[indices]
    dsx = (gmm.score_samples(obs + np.array([dx, 0])) - gmm.score_samples(obs)) / dx
    dsy = (gmm.score_samples(obs + np.array([0, dy])) - gmm.score_samples(obs)) / dy
    score[indices, 0] = dsx
    score[indices, 1] = dsy

# score /= np.linalg.norm(score, axis=1)[:, None]
adata.obsm['scores_umap'] = score

print(f"range of score : {score.max()}, {score.min()}")

plt.figure(figsize=(10, 6))
score /= 3 * quiver_autoscaler(X, score)
plt.quiver(
    X[:, 0],
    X[:, 1],
    score[:, 0],
    score[:, 1],
    color=color_list,
    facecolors=color_list,
    **quiver_kwargs,
)
plt.axis('off')
plt.savefig(f"./figures/scores.svg", dpi=600)

############################################################################################################
#                                     vector field                                                         #
#                                                                                                          #
############################################################################################################

score = adata.obsm['scores_umap']
flow = adata.obsm['flows_umap']
vf = np.zeros_like(X)
gmm_dicts = adata.uns['gmm_dicts']
gs = [1.414, 1.414]  # equal contribution of flow and score, 1.414
for i, k in enumerate(gmm_dicts.keys()):
    d = gmm_dicts[k]
    indices = d['indices']
    vf[indices] = 0.5 * gs[i] ** 2 * score[indices] + flow[indices]
    # vf[indices] = score[indices]
    # vf[indices] = flow[indices]
# g = 2
# vf = 0.5 * g ** 2 * score + flow
adata.obsm['vector_field'] = vf
plt.figure(figsize=(10, 6))
plt.quiver(
    X[:, 0],
    X[:, 1],
    vf[:, 0],
    vf[:, 1],
    color=color_list,
    facecolors=color_list,
    **quiver_kwargs,
)
plt.axis('off')
plt.savefig(f"./figures/cell_wise_vector_field.svg", dpi=600)

# calculate vector field based on velocities
vf = tl.VectorField(adata, coord_basis='X_umap', velo_basis='vector_field', dims=2, M=1000)
vf.train()  # parameters for reconstruct the vf is stored in uns['vf_dict'] after training

adata.write_h5ad(Path("./data/contours.h5ad"), compression='gzip')


def func(x):
    return vf.vector_field_function(x, adata.uns['vf_dict'])


pl.scatter(adata, c='dpt_pseudotime', cmap='viridis', save=False, show=False)
pl.flow_field(adata, func, save="./figures/vector_field.svg")
