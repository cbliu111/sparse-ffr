import anndata
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import seaborn as sns
import h5py
import scipy
from scipy import cluster
from scipy.spatial.distance import squareform, pdist
import scienceplots

plt.style.use(['science', 'nature'])


def identify_medoid(indices, dist_mat):
    """
    Identify the medoid cell in cell_names.
    """
    xi = np.argmin(dist_mat[np.ix_(indices, indices)].sum(axis=0))
    return indices[xi]


adata = anndata.read_h5ad("./data/contours.h5ad")
modes = adata.obs['leiden'].to_numpy()
n_modes = np.unique(modes).shape[0]
states = adata.obs['state'].to_numpy()
color_list = sns.color_palette("deep")

with h5py.File("./data/gw_dist.h5", "r") as f:
    gw_dist_mat = f["/gw_dist"][...]
    iodms = f["/intra_dist"][...]
medoids = []
for m in np.unique(modes):
    indices = np.where(modes == m)[0]
    idx = identify_medoid(indices, gw_dist_mat)
    medoids.append(adata.obsm["X_umap"][idx])

inter_dist = pdist(medoids, 'euclidean')
Z = cluster.hierarchy.linkage(inter_dist, method='complete')
cluster.hierarchy.set_link_color_palette(['k'])
fig, ax = plt.subplots(figsize=(6, 2), linewidth=2.0, frameon=False)
plt.yticks([])
R = cluster.hierarchy.dendrogram(Z, p=0, truncate_mode='none', orientation='bottom', ax=None,
                                 above_threshold_color='k')
dendidx = np.array([int(s) for s in R['ivl']])
cluster.hierarchy.set_link_color_palette(None)
ax.set_xlabel('Shape mode', fontsize=15, fontweight='bold')
plt.axis('equal')
plt.axis('off')
fig.savefig(f"./figures/shape_dendrogram.svg", transparent=True, dpi=600)

text_pos = [
    [-0.5, -7],
    [-0.5, -3.5],
]

for s in np.unique(states):
    indices = np.where(states == s)[0]
    ms = modes[indices] + 1
    n, bins = np.histogram(ms, bins=range(n_modes+2)[1:])
    fig, ax = plt.subplots(figsize=(10, 5))
    n = n / np.sum(n)
    n *= 100
    n = np.around(n, 2)
    height = n
    shuffle_idx = np.array([int(s) for s in dendidx])
    heights = height[shuffle_idx]
    ax.bar(x=(np.delete(bins, 0) - 1) / 2, height=height, width=0.4, align='center', color=(0.2, 0.4, 0.6, 1),
             edgecolor='black', alpha=0.8)
    ax.set_ylabel(r'Abundance \%', fontsize=15, fontweight='bold')

    # only for paper
    ax.set_ylim([0, np.max(height) + 5])

    # ax.set_title('Shape mode distribution (N=' + str(len(IDX_dist)) + ')', fontsize=18, fontweight='bold')
    bartick = dendidx + 1
    ax.set_xticks((np.arange(np.max(modes) + 2) / 2)[1:])
    ax.set_xticklabels(tuple(bartick[:np.max(modes)+1]), fontsize=13, fontweight='bold')
    ax.yaxis.set_tick_params(labelsize=13)
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    for i, v in enumerate(height):
        ax.text((i - 0.25 + 1) / 2, v + 0.25, str(np.around(v, decimals=1)), color='black', fontweight='bold',
                  fontsize=13)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1)
    plt.text(text_pos[s-1][0], text_pos[s-1][1], f"Mode", fontweight='bold', fontsize=16, color='black')
    fig.savefig(f"./figures/shape_dist_{s}.svg", transparent=True, dpi=600)























