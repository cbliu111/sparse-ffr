import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import anndata
from pathlib import Path
from scipy.ndimage import gaussian_filter
from matplotlib.colors import ListedColormap
from utils import logger
plt.style.use(['science', 'nature'])

adata = anndata.read_h5ad(Path("./data/contours.h5ad"))
X = adata.obsm['X_umap']
n_grid = 100
n_path = 600
n_cpu = 20
D = 0.003
time_steps = int(5e4)
start_record_step = int(1e4)
expand = 1
eps = 1e-12
xmin, xmax = X[:, 0].min() - expand, X[:, 0].max() + expand
ymin, ymax = X[:, 1].min() - expand, X[:, 1].max() + expand
x_lim = (xmin, xmax)
y_lim = (ymin, ymax)

sim_data = np.load(f"./data/sim_D{D}.npz")
num_tra = sim_data['num_traj']
total_Fx = sim_data['Fx']
total_Fy = sim_data['Fy']
path = sim_data['path']

p_tra = num_tra / (sum(sum(num_tra)))
print([sum(sum(num_tra)), n_path * (time_steps - start_record_step)])
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

GUx, GUy = np.gradient(U)
GPx, GPy = np.gradient(P)
Jx = mean_Fx * P - D * GPx
Jy = mean_Fy * P - D * GPy
E = Jy ** 2 + Jx ** 2
JJx = Jx / (np.sqrt(E) + eps)  # flux
JJy = Jy / (np.sqrt(E) + eps)
EE = mean_Fx ** 2 + mean_Fy ** 2
FFx = mean_Fx / (np.sqrt(EE) + eps)  # F
FFy = mean_Fy / (np.sqrt(EE) + eps)
Fgradx = -D * GUx
Fgrady = -D * GUy
EEE = Fgradx ** 2 + Fgrady ** 2
FFgradx = Fgradx / (np.sqrt(EEE) + eps)  # F gradient
FFgrady = Fgrady / (np.sqrt(EEE) + eps)

expand = 1

###############################################################################################
""" Landscape with observation data """
###############################################################################################

plt.figure(figsize=(10, 6))
plt.contourf(Xgrid, Ygrid, pot_U.T, cmap='jet', levels=100, alpha=0.8)
plt.colorbar(shrink=0.5, label=r"-$\log P_{ss}$")
plt.scatter(X[:, 0], X[:, 1], s=10, c='black', alpha=0.6)
plt.xlim([x_lim[0] - 1, x_lim[1] + 1])
plt.ylim([y_lim[0] - 1, y_lim[1] + 1])
plt.axis('off')
plt.savefig("./figures/landscape.svg", dpi=600)



###############################################################################################
""" Landscape with gradient and flux """
###############################################################################################

def default_quiver_args(arrow_size, arrow_len=None):
    if isinstance(arrow_size, (list, tuple)) and len(arrow_size) == 3:
        head_w, head_l, ax_l = arrow_size
    elif type(arrow_size) in [int, float]:
        head_w, head_l, ax_l = 10 * arrow_size, 12 * arrow_size, 8 * arrow_size
    else:
        head_w, head_l, ax_l = 10, 12, 8

    scale = 1 / arrow_len if arrow_len is not None else 1 / arrow_size

    return head_w, head_l, ax_l, scale


quiver_size = 2
head_w, head_l, ax_l, scale = default_quiver_args(quiver_size, 0.5)
quiver_kwargs = {
    "angles": "xy",
    "scale": scale,
    "scale_units": "xy",
    "width": 0.0003,
    "headwidth": head_w,
    "headlength": head_l,
    "headaxislength": ax_l,
    "minshaft": 1,
    "minlength": 1,
    "pivot": "mid",
    "linewidth": 0.1,
    "edgecolors": "black",
    "alpha": 1,
    "zorder": 10,
}

plt.figure(figsize=(10, 6))
U = gaussian_filter(pot_U.T, sigma=1)
U1 = gaussian_filter(pot_U.T, sigma=3)
U[U1 > np.floor(np.max(U))] = np.nan
plt.contourf(Xgrid, Ygrid, U,
             cmap='jet', levels=100, alpha=1)
plt.colorbar(shrink=0.5, label=r"-$\log P_{ss}$")
mg = np.arange(0, n_grid, 4)
plt.quiver(
    Xgrid[np.ix_(mg, mg)],
    Ygrid[np.ix_(mg, mg)],
    JJx[np.ix_(mg, mg)],
    JJy[np.ix_(mg, mg)],
    color='darkgreen',
    **quiver_kwargs
)
mg = np.arange(2, n_grid, 4)
plt.quiver(
    Xgrid[np.ix_(mg, mg)],
    Ygrid[np.ix_(mg, mg)],
    FFgradx[np.ix_(mg, mg)],
    FFgrady[np.ix_(mg, mg)],
    color='orange',
    **quiver_kwargs
)

plt.xlim([x_lim[0] - 1, x_lim[1] + 1])
plt.ylim([y_lim[0] - 1, y_lim[1] + 1])
plt.axis('off')
# plt.show()
plt.savefig("./figures/landscape_flux.svg", dpi=600)

###############################################################################################
""" Landscape with Forces """
###############################################################################################

quiver_kwargs = {
    "angles": "uv",
    "scale": scale,
    "scale_units": "xy",
    "width": 0.0003,
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
mg = np.arange(0, n_grid, 3)
plt.figure(figsize=(10, 6))
U = gaussian_filter(pot_U.T, sigma=1)
U1 = gaussian_filter(pot_U.T, sigma=3)
U[U1 > np.floor(np.max(U))] = np.nan
plt.contourf(Xgrid, Ygrid, U,
             cmap='jet', levels=100, alpha=1)
plt.colorbar(shrink=0.5, label=r"-$\log P_{ss}$")
plt.quiver(
    Xgrid[np.ix_(mg, mg)],
    Ygrid[np.ix_(mg, mg)],
    FFx[np.ix_(mg, mg)],
    FFy[np.ix_(mg, mg)],
    color='black',
    **quiver_kwargs
)

plt.xlim([x_lim[0] - 1, x_lim[1] + 1])
plt.ylim([y_lim[0] - 1, y_lim[1] + 1])
plt.axis('off')
# plt.show()
plt.savefig("./figures/landscape_force.svg", dpi=600)

###############################################################################################
""" Landscape for illustration figure """
###############################################################################################

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
xlin = np.linspace(x_lim[0], x_lim[1], n_grid + 10)
ylin = np.linspace(y_lim[0], y_lim[1], n_grid + 10)
Xgrid, Ygrid = np.meshgrid(xlin, ylin)
pot_U = np.pad(pot_U, ((5, 5), (5, 5)), mode='maximum')
surf = ax.plot_surface(Xgrid, Ygrid, gaussian_filter(pot_U.T, sigma=3), cmap='jet',
                       linewidth=0, antialiased=True)
#plt.xlim([x_lim[0], x_lim[1]])
#plt.ylim([y_lim[0], y_lim[1]])
ax.view_init(elev=80)
plt.axis('off')
plt.savefig("./figures/landscape_3d_demo.svg", dpi=600)

###############################################################################################
""" Landscape for showing results """
###############################################################################################

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 8))
Xgrid, Ygrid = np.meshgrid(xlin, ylin)
U = gaussian_filter(pot_U.T, sigma=1)
surf = ax.plot_surface(Xgrid, Ygrid, U, cmap='jet',
                       linewidth=0, antialiased=True)
ax.contour(Xgrid, Ygrid, U, levels=10, cmap="jet", linestyles="solid", offset=-1)
ax.set_xlim([x_lim[0]-3, x_lim[1]+3])
ax.set_ylim([y_lim[0]-3, y_lim[1]+3])
ax.view_init(elev=45, azim=-75)
# ax.set_xticks([])
# ax.set_yticks([])
ax.set_xlabel("UMAP1", fontdict={'fontsize': 20})
ax.set_ylabel("UMAP2", fontdict={'fontsize': 20})
ax.set_zlabel("Potential", fontdict={'fontsize': 20})
ax.zaxis.labelpad = 5
plt.savefig(f"./figures/landscape_3d_{D}.svg", dpi=600)
