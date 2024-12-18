import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import anndata
from pathlib import Path
from scipy.ndimage import gaussian_filter
from utils import logger
import dynamo as dyn
from dynamo.tools.utils import nearest_neighbors
from scipy.sparse import csr_matrix
import pickle
import matplotlib.patches as patches
from skimage.feature import peak_local_max
import h5py
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform
import PIL.Image
import PIL.ImageDraw
import skimage.filters
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
from scipy import cluster
import seaborn as sns


def add_arrow(line, ax, position=None, direction='right', color=None, label=''):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    color:      if None, line color is taken.
    label:      label for arrow
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    dx = xdata[end_ind] - xdata[start_ind]
    dy = ydata[end_ind] - ydata[start_ind]
    size = 0.8
    x = xdata[start_ind]
    y = ydata[start_ind]

    arrow = patches.FancyArrow(x, y, dx, dy, color=color, width=0,
                               head_width=size, head_length=size,
                               label=label, length_includes_head=True,
                               overhang=0.3, zorder=10)
    ax.add_patch(arrow)



plt.style.use(['science', 'nature'])
adata = anndata.read_h5ad(Path("./data/contours.h5ad"))

X = adata.obsm['X_umap']
n_grid = 100
n_path = 600
n_cpu = 20
time_steps = int(5e5)
start_record_step = int(1e4)
expand = 1
eps = 1e-12
xmin, xmax = X[:, 0].min() - expand, X[:, 0].max() + expand
ymin, ymax = X[:, 1].min() - expand, X[:, 1].max() + expand
x_lim = (xmin, xmax)
y_lim = (ymin, ymax)

D = 0.003
sim_data = np.load(f"./data/sim_D{D}.npz")
num_tra = sim_data['num_traj']
total_Fx = sim_data['Fx']
total_Fy = sim_data['Fy']
path = sim_data['path']

p_tra = num_tra / (sum(sum(num_tra)))
pot_U = -np.log(p_tra + eps)  # -log P_ss
mean_Fx = total_Fx / (num_tra + eps)
mean_Fy = total_Fy / (num_tra + eps)
mean_Fx = mean_Fx.T
mean_Fy = mean_Fy.T
U = pot_U.T
P = p_tra.T

xlin = np.linspace(x_lim[0], x_lim[1], n_grid)
ylin = np.linspace(y_lim[0], y_lim[1], n_grid)
Xgrid, Ygrid = np.meshgrid(xlin, ylin)

###############################################################################################
""" identify basins """
###############################################################################################

coordinates = peak_local_max(-U)
plt.imshow(-U)
plt.scatter(coordinates[:, 1], coordinates[:, 0], color='orange')
plt.show()
print(coordinates)


fixed_points = np.array([
    [xlin[91], ylin[72]],
    [xlin[85], ylin[54]],
    [xlin[57], ylin[20]],
    [xlin[37], ylin[21]],
    [xlin[24], ylin[31]],
    [xlin[13], ylin[49]],
    [xlin[7], ylin[75]],
])

kmeans = KMeans(n_clusters=fixed_points.shape[0], init=fixed_points, n_init=1)
kmeans.fit(X)
cluster_centers = kmeans.cluster_centers_
distance_to_centers = cdist(X, fixed_points, metric='euclidean')
cluster_ids = np.argmin(distance_to_centers, axis=1)

color_list = sns.color_palette("deep")
plt.figure(figsize=(10, 6))
for m in np.unique(cluster_ids):
    index = np.where(cluster_ids == m)[0]
    m = int(m)
    x = X[index, 0]
    y = X[index, 1]
    plt.scatter(x, y, s=10, color=color_list[m], alpha=0.3)
    plt.text(x.mean(), y.mean(), f"{m + 1}", fontsize=25)
    sns.kdeplot(x=x.reshape(-1), y=y.reshape(-1), color=color_list[m], levels=1)
plt.axis('off')
plt.savefig(f"./figures/basin_modes_text.svg", dpi=600)
plt.close()

for m in np.unique(cluster_ids):
    plt.figure()
    index = np.where(cluster_ids == m)[0]
    m = int(m)
    contour = adata.obsm['realigned'][index]
    pts = int(adata.n_vars / 2)
    c = []
    for j in range(contour.shape[0]):
        c.append(np.array([contour[j][:pts], contour[j][pts:]]))
        plt.plot(contour[j][:pts], contour[j][pts:])
    plt.axis('off')
    plt.savefig(f"./figures/basin_mode_contour{m + 1}.svg", transparent=True, dpi=600)
    plt.close()

inter_dist = pdist(fixed_points, 'euclidean')
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
fig.savefig(f"./figures/basin_shape_dendrogram.svg", transparent=True, dpi=600)

text_pos = [
    [-0.5, -7],
    [-0.5, -3.5],
]

states = adata.obs['state'].to_numpy()
for s in np.unique(states):
    indices = np.where(states == s)[0]
    ms = cluster_ids[indices] + 1
    n, bins = np.histogram(ms, bins=range(fixed_points.shape[0]+2)[1:])
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
    ax.set_xticks((np.arange(np.max(cluster_ids) + 2) / 2)[1:])
    ax.set_xticklabels(tuple(bartick[:np.max(cluster_ids)+1]), fontsize=13, fontweight='bold')
    ax.yaxis.set_tick_params(labelsize=13)
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    for i, v in enumerate(height):
        ax.text((i - 0.25 + 1) / 2, v + 0.25, str(np.around(v, decimals=1)), color='black', fontweight='bold',
                  fontsize=13)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1)
    plt.text(text_pos[s-1][0], text_pos[s-1][1], f"Mode", fontweight='bold', fontsize=16, color='black')
    fig.savefig(f"./figures/basin_shape_dist_{s}.svg", transparent=True, dpi=600)

with h5py.File("./data/gw_dist.h5", "r") as f:
    gw_dist_mat = f["/gw_dist"][...]
    iodms = f["/intra_dist"][...]
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=3000, eps=1e-9)
for i in range(fixed_points.shape[0]):
    d = np.sum((X - fixed_points[i, :]) ** 2, axis=1)
    idx = np.argmin(d)
    distance_matrix = squareform(iodms[idx])
    coordinates = mds.fit_transform(distance_matrix)
    coordinates = coordinates - np.min(coordinates, axis=0)
    coordinates = np.ceil(coordinates).astype(int)
    coordinates = np.concatenate([coordinates, coordinates[0, :].reshape(1, -1)], axis=0)
    size = int(np.max(coordinates))
    size += int(0.5 * size)
    pad_size = int(0.25 * size)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    points = []
    for j in range(coordinates.shape[0]):
        points.append(tuple([coordinates[j, 0] + pad_size, coordinates[j, 1] + pad_size]))
    # plt.scatter(coordinates[:, 0], coordinates[:, 1], color="blue")
    draw.polygon(xy=points, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    kernel = skimage.morphology.ellipse(3, 3)
    out = skimage.morphology.dilation(mask, kernel)
    out = skimage.morphology.erosion(out, kernel)
    out = skimage.morphology.dilation(out, kernel)
    # mask = skimage.filters.gaussian(out, sigma=5)
    cs = skimage.measure.find_contours(mask, level=0.5)
    if len(cs) > 1:
        print("Multiple contours detected")
        # plt.scatter(coordinates[:, 0], coordinates[:, 1], color="blue")
        # for c in cs:
        #     plt.plot(c[:, 0], c[:, 1], color='black', alpha=0.8, linewidth=2)
        # plt.show()
    c = cs[0]
    plt.figure()
    plt.plot(c[:, 0], c[:, 1], color='black', alpha=0.8, linewidth=2)
    plt.axis('off')
    plt.savefig(f"./figures/basin_shape{i + 1}.svg", transparent=True, dpi=600)
    plt.close()


###############################################################################################
""" compute least action path """
###############################################################################################

cell_indices = []
for i in range(fixed_points.shape[0]):
    cell_indices.append(nearest_neighbors(fixed_points[i], adata.obsm["X_umap"]))
adata.uns['VecFld_umap'] = adata.uns['vf_dict']
adata.obsp['X_umap_distances'] = csr_matrix(adata.obsp['distances'])
adata.var['use_for_pca'] = False

lap_list = []
for i in range(fixed_points.shape[0] - 1):
    lap_t = True if i == 1 else False
    dyn.pd.least_action(
        adata,
        [adata.obs_names[cell_indices[i][0]][0]],
        [adata.obs_names[cell_indices[i + 1][0]][0]],
        vf_key='VecFld',
        adj_key='X_umap_distances',
        basis="umap",
        min_lap_t=lap_t,
        EM_steps=2,
        D=0.003,
    )
    lap_list.append(adata.uns["LAP_umap"])
pickle.dump(lap_list, open("least_action_path.pickle", "wb"))
lap_list = pickle.load(open("least_action_path.pickle", "rb"))

total_action = 0
plt.figure(figsize=(10, 6))
U = gaussian_filter(pot_U.T, sigma=1)
U1 = gaussian_filter(pot_U.T, sigma=3)
U[U1 > np.floor(np.max(U))] = np.nan
plt.contourf(Xgrid, Ygrid, U, cmap='jet', levels=100)
lap_coordinates = []
for lap in lap_list:
    prediction = lap['prediction']
    action = lap['action']
    line = plt.plot(*prediction[0].T, c="k", lw=3, alpha=0.5)[0]  # least action trajectory
    lap_coordinates.append(prediction[0])
    ax = plt.gca()
    add_arrow(line, ax)
    total_action += action[0][-1]

for i in range(fixed_points.shape[0]):
    plt.scatter(fixed_points[i, 0], fixed_points[i, 1], c="black", s=300)
    plt.text(fixed_points[i, 0]-0.25, fixed_points[i, 1]-0.25, str(i+1), fontsize=15, color='orange')
plt.axis("off")
plt.savefig(f"./figures/least_action_path.svg", dpi=600)

