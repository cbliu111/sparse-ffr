import tl
import kernels
import anndata
import scanpy as sc
import scvelo as scv
import pandas as pd
import wot
from scanpy.tools._dpt import DPT
import matplotlib.pyplot as plt
from pathlib import Path
from utils.log import logger
import numpy as np
import scienceplots

plt.style.use(['science', 'nature'])


def get_symmetric_transition_matrix(transition_matrix):
    """Symmetrize transition matrix."""
    sym_mat = (transition_matrix + transition_matrix.T) / 2

    # normalise transition matrix
    row_sums = sym_mat.sum(axis=1).A1
    sym_mat.data = sym_mat.data / row_sums[sym_mat.nonzero()[0]]

    return sym_mat


adata = anndata.read_h5ad(Path("./data/contours.h5ad"))
logger.info(adata)
time = adata.obs['time'].to_numpy()

# compute real-time informed pseudo-time (RTIPT) as the latent time
ot_model = wot.ot.OTModel(adata, day_field="time")
ot_model.compute_all_transport_maps(tmap_out="./data/tmaps/")

logger.info(f"Computing transport matrix using wot...")
tl.compute_transport_maps(adata, time_field='time', metric_kw='metric', tmap_out='./data/tmaps/')
logger.info(f"Running real time kernel...")
rtk = kernels.RealTimeKernel.from_wot(adata, path='./data/tmaps/', time_key="time")
logger.info(f"Computing transtion matrix...")
rtk = rtk.compute_transition_matrix(self_transitions='all', conn_weight=0.5)
adata.obsp['transition_matrix'] = rtk.transition_matrix

dpt = DPT(adata=adata, neighbors_key="metric")
transitions_sym = get_symmetric_transition_matrix(rtk.transition_matrix)
connectivities = adata.obsp["connectivities"]
evals, evecs = tl.compute_eigen(transitions_sym, connectivities, n_comps=15)

adata.obsm["X_diffmap"] = evecs
adata.uns["diffmap_evals"] = evals

sc.tl.diffmap(adata)

df = (
    pd.DataFrame(
        {
            "diff_comp": adata.obsm["X_diffmap"][:, 5],
            "time": adata.obs["state"].values,
        }
    )
    .reset_index()
    .rename({"index": "obs_id"}, axis=1)
)

df = df.loc[df['time'] == 1.0, "diff_comp"]
root_idx = df.index[df.argmax()]

adata.uns["iroot"] = root_idx
logger.info(f"Computing latent time...")
sc.tl.dpt(adata, neighbors_key="neighbors", n_dcs=6)

logger.info(adata)
adata.write_h5ad(Path("./data/contours.h5ad"), compression='gzip')

# visualize latent time
plt.figure(figsize=(10, 6))
embedding = adata.obsm['X_umap']
x = embedding[:, 0]
y = embedding[:, 1]
plt.scatter(x, y, s=10, c=adata.obs['dpt_pseudotime'], cmap='viridis', alpha=0.9)
plt.colorbar(label='Latent time', shrink=0.5)
# plt.scatter(x[root_idx], y[root_idx], s=100, facecolors="none", edgecolors="black", alpha=1.0)
plt.axis('off')
plt.savefig(f"./figures/latent_time.svg", dpi=600)
