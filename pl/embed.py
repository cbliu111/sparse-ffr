import numpy as np
import matplotlib.pyplot as plt
from anndata import AnnData
from utils.log import logger


def scatter(
        adata: AnnData,
        basis='X_umap',
        c=None,
        cmap=None,
        alpha=0.5,
        save='',
        show=True,
):
    emb = adata.obsm[basis]
    lb = basis.split('_')[1].upper()
    point_size = 30000.0 / np.sqrt(adata.shape[0])
    if emb.shape[1] == 2:
        fig, axes = plt.subplots(figsize=(10, 6))
        if cmap is None or c is None:
            plt.scatter(emb[:, 0], emb[:, 1], marker='.', s=point_size, c='gray', alpha=alpha)
        else:
            color_id = adata.obs[c].to_numpy()
            max_c = color_id.max()
            min_c = color_id.min()
            plt.scatter(emb[:, 0], emb[:, 1], marker='.', s=point_size, c=color_id, cmap=cmap, alpha=alpha)
            cbar = plt.colorbar(ticks=np.linspace(min_c, max_c, 5),
                                ax=axes, shrink=0.5,
                                orientation='vertical')
            cbar.set_label(c)
    else:
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection="3d")
        ax.grid(b=False, color='gray',
                linestyle='-.', linewidth=0.3,
                alpha=0.8)
        if cmap is None or c is None:
            ax.scatter3D(emb[:, 0], emb[:, 1], emb[:, 2], marker='.', s=point_size, c='gray', alpha=alpha)
        else:
            color_id = adata.obs[c].to_numpy()
            max_c = color_id.max()
            plt_3d = ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], marker='.', s=point_size, c=color_id, cmap=cmap, alpha=alpha)
            cbar = plt.colorbar(plt_3d, ticks=np.arange(max_c + 1) + 1,
                                ax=ax, shrink=0.3,
                                orientation='vertical')
            cbar.set_label(c)
    if emb.shape[1] == 2:
        plt.axis('equal')
        plt.axis('off')
        plt.xlabel(f"{lb}1")
        plt.ylabel(f"{lb}2")
    else:
        ax.set_xlabel(f"{lb}1")
        ax.set_ylabel(f"{lb}2")
        ax.set_zlabel(f"{lb}3")
        #ax.set_xticks([])
        #ax.set_yticks([])
        #ax.set_zticks([])

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=600)
        logger.info(f"Save figure to {save}")
    if show:
        plt.show()
