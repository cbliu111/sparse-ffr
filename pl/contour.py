import matplotlib.pyplot as plt


def aligned_contours(adata, save: str="./figures/contours.pdf"):
    # plot mean contour and aligned contours
    m = adata.uns['mean_contour']
    data = adata.to_df().to_numpy()
    pts = int(adata.n_vars / 2)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    for i in range(100):
        axes[0].plot(data[i, :pts], data[i, pts:])
    for i in range(100):
        axes[1].plot(data[-i, :pts], data[-i, pts:])
    axes[0].plot(m[:, 1], m[:, 0], lw=3, color='k')
    axes[1].plot(m[:, 1], m[:, 0], lw=3, color='k')
    axes[0].set_title('First 100 contours')
    axes[1].set_title('Last 100 contours')
    plt.savefig(save, dpi=600)


