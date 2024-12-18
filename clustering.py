import numpy as np
from pathlib import Path
from utils.log import logger
import anndata
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
import seaborn as sns
plt.style.use(['science', 'nature'])

adata = anndata.read_h5ad("./data/contours.h5ad")
X = adata.obsp['metric']
time = adata.obs['time'].to_numpy()

# identify macrostates
first_state_index = np.where(time == 1.0)
second_state_index = np.where(time > 1.0)
state_label = np.zeros_like(time, dtype=int)
state_label[first_state_index] = 1
state_label[second_state_index] = 2
adata.obs['state'] = pd.Categorical(state_label)
adata.write_h5ad(Path("./data/contours.h5ad"), compression='gzip')

embedding = adata.obsm['X_umap']
states = adata.obs['state']

color_list = ['#4292C6', '#FD8D3C']
label_list = ['Day 1', 'Day 2-10']

plt.figure(figsize=(10, 6))
for i, s in enumerate(np.unique(states)):
    index = np.where(states == s)
    x = embedding[index, 0]
    y = embedding[index, 1]
    plt.scatter(x, y, s=10, c=color_list[i], alpha=0.3, label=label_list[i])
    sns.kdeplot(x=x.reshape(-1), y=y.reshape(-1), color=color_list[i])
plt.axis('off')
plt.legend(fontsize=20, loc='lower right')
plt.savefig(f"./figures/umap_cluster.svg", dpi=600)

