import anndata
import numpy as np
from utils import logger
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from pathlib import Path
import scienceplots
plt.style.use(['science', 'nature'])

adata = anndata.read_h5ad("./data/contours.h5ad")
logger.info(adata)

embedding = adata.obsm['X_umap']
time = adata.obs['time'].to_numpy()
sigma = 0.02

states = adata.obs['state'].to_numpy()
gmm_dicts = {}

for i, s in enumerate(np.unique(states)):
    index = np.where(states == s)[0]
    X = embedding[index]
    value = np.mean(adata.obsp['metric']) * sigma
    a = np.zeros((2, 2))
    np.fill_diagonal(a, value)
    covariances = np.repeat(a[None, ...], X.shape[0], axis=0)
    d = {'indices': index, 'means': X, 'covariances': covariances, 'weights': np.ones(X.shape[0]) * 1. / X.shape[0],
                               'precisions_cholesky': np.linalg.cholesky(np.linalg.inv(covariances))}
    gmm_dicts[f'state_dict_{i}'] = d
adata.uns['gmm_dicts'] = gmm_dicts
adata.write_h5ad(Path("./data/contours.h5ad"), compression='gzip')

plt.figure(figsize=(10, 6))
color_list = ['#4292C6', '#FD8D3C']
label_list = ['Day 1', 'Day 2-10']
expand = 2
xmin = embedding[:, 0].min() - expand
xmax = embedding[:, 0].max() + expand
ymin = embedding[:, 1].min() - expand
ymax = embedding[:, 1].max() + expand
xi, yi = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))

states = adata.obs['state']
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
    log_prob = gmm.score_samples(np.concatenate([xi.reshape(-1, 1), yi.reshape(-1, 1)], axis=1))
    prob = np.exp(log_prob)
    prob = prob.reshape(xi.shape[0], -1)
    plt.scatter(*means.T, s=10, c=color_list[i], alpha=0.3, label=label_list[i])
    plt.contour(xi, yi, prob, levels=12, colors=color_list[i], linestyles='-', linewidths=1)

plt.axis('off')
plt.legend(fontsize=20)
plt.savefig(f"./figures/gmm.svg", dpi=600)


