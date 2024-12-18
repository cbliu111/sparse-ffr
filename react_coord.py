import numpy as np
import tl
import matplotlib.pyplot as plt
import scienceplots
import anndata
from pathlib import Path
import pickle
from scipy.interpolate import make_interp_spline, SmoothBivariateSpline
import warnings

warnings.simplefilter('ignore')
plt.style.use(['science', 'nature'])

###############################################################################################
""" obtain least action path and potential landscape functions """
###############################################################################################

lap_list = pickle.load(open("least_action_path.pickle", "rb"))
lap_coordinates = []
for lap in lap_list:
    prediction = lap['prediction']
    lap_coordinates.append(prediction[0])

lap_coordinates = np.concatenate(lap_coordinates, axis=0)
idx = np.argsort(lap_coordinates[:, 0])
lap_coordinates = lap_coordinates[idx]
eps = np.linspace(0, 1e-12, lap_coordinates.shape[0])  # avoid equality checking of x-axis
lap_func = make_interp_spline(lap_coordinates[:, 0] + eps, lap_coordinates[:, 1], k=3)
lap_x = np.linspace(lap_coordinates[1, 0], lap_coordinates[-1, 0], 100)
lap_path = lap_func(lap_x)
# plt.figure()
# plt.scatter(lap_coordinates[:, 0], lap_coordinates[:, 1])
# plt.plot(lap_x, lap_func(lap_x))
# plt.show()


adata = anndata.read_h5ad(Path("./data/contours.h5ad"))
X = adata.obsm['X_umap']
vf = tl.VectorField(adata, coord_basis='X_umap', velo_basis='vector_field', dims=2)
vf_func = lambda xx: vf.vector_field_function(xx, adata.uns['vf_dict'])
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

Ds = [
    # 0.,
    0.0001,
    0.0003,
    0.0008,
    0.0015,
    0.0024,
    0.003,
    0.005,
    0.008,
    0.01,
    0.015,
    0.024,
    0.03,
]
for i, D in enumerate(Ds):
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

    xx = np.ravel(Xgrid)
    yy = np.ravel(Ygrid)
    zz = np.ravel(U)
    U_func = SmoothBivariateSpline(xx, yy, zz)
    # plt.figure()
    # plt.imshow(U)
    # plt.show()
    # print(np.max(U), np.min(U))
    # plt.figure()
    # U1 = U_func(xlin, ylin).T
    # plt.imshow(U1)
    # print(np.max(U1), np.min(U1))
    # plt.show()

    plt.figure()
    react_coord_x = np.linspace(x_lim[0], x_lim[1], 100)
    lap_path = lap_func(react_coord_x)
    path_U = U_func(react_coord_x, lap_path, grid=False)
    plt.plot(react_coord_x, path_U)
    plt.xlabel("Reaction Coordinates", fontdict={'fontsize': 15})
    plt.ylabel("Potential", fontdict={'fontsize': 15})
    plt.ylim([3, 29])
    plt.title(f"D = {D}", fontdict={'fontsize': 15})
    plt.savefig(f"./figures/reaction_coord_D{D}.svg", transparent=True, dpi=600)
    plt.close()


