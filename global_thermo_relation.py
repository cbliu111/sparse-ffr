import numpy as np
import tl
import matplotlib.pyplot as plt
import scienceplots
import anndata
from pathlib import Path
import pickle
import matplotlib.patches as patches
from scipy.interpolate import make_interp_spline, SmoothBivariateSpline
from scipy.signal import find_peaks
import seaborn as sns
import pickle
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
adata = anndata.read_h5ad(Path("./data/contours.h5ad"))
X = adata.obsm['X_umap']
vf = tl.VectorField(adata, coord_basis='X_umap', velo_basis='vector_field', dims=2)
vf_func = lambda xx: vf.vector_field_function(xx, adata.uns['vf_dict'])
idx = np.argsort(lap_coordinates[:, 0])
lap_coordinates = lap_coordinates[idx]
eps = np.linspace(0, 1e-12, lap_coordinates.shape[0])  # avoid equality checking of x-axis
lap_func = make_interp_spline(lap_coordinates[:, 0] + eps, lap_coordinates[:, 1], k=3)
expand = 1
n_grid = 100
n_path = 600
xmin, xmax = X[:, 0].min() - expand, X[:, 0].max() + expand
ymin, ymax = X[:, 1].min() - expand, X[:, 1].max() + expand
x_lim = (xmin, xmax)
y_lim = (ymin, ymax)
xlin = np.linspace(x_lim[0], x_lim[1], n_grid)
ylin = np.linspace(y_lim[0], y_lim[1], n_grid)
Xgrid, Ygrid = np.meshgrid(xlin, ylin)

fixed_points = np.array([
    [xlin[91], ylin[72]],
    [xlin[85], ylin[54]],
    [xlin[57], ylin[20]],
    [xlin[37], ylin[21]],
    [xlin[24], ylin[31]],
    [xlin[13], ylin[49]],
    [xlin[7], ylin[75]],
])

Ds = [
    # 0.,
    # 0.0001,
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

forward_barrier_heights = []  # from state 1 to state 2
backward_barrier_heights = []  # from state 2 back to state 1
least_actions = []
mean_first_passage_times = []
log_integral_flux = []
epr = []

for i, D in enumerate(Ds):
    sim_data = np.load(f"./data/sim_D{D}.npz")
    num_tra = sim_data['num_traj']
    total_Fx = sim_data['Fx']
    total_Fy = sim_data['Fy']
    path = sim_data['path'].reshape(n_path, -1, 2)

    p_tra = num_tra / (sum(sum(num_tra)))
    pot_U = -np.log(p_tra + 1e-12)  # -log P_ss
    mean_Fx = total_Fx / (num_tra + 1e-12)
    mean_Fy = total_Fy / (num_tra + 1e-12)
    mean_Fx = mean_Fx.T
    mean_Fy = mean_Fy.T
    U = pot_U.T
    P = p_tra.T
    GUx, GUy = np.gradient(U)
    GPx, GPy = np.gradient(P)
    Jx = mean_Fx * P - D * GPx
    Jy = mean_Fy * P - D * GPy
    E = Jy ** 2 + Jx ** 2
    JJx = Jx / (np.sqrt(E) + 1e-12)  # flux
    JJy = Jy / (np.sqrt(E) + 1e-12)
    Fgradx = -D * GUx
    Fgrady = -D * GUy

    xx = np.ravel(Xgrid)
    yy = np.ravel(Ygrid)
    zz = np.ravel(U)
    U_func = SmoothBivariateSpline(xx, yy, zz)

    # compute the least action along the most probable path
    x0 = lap_coordinates  # maybe stop at the left basin loc?
    x = (x0[:-1] + x0[1:]) * 0.5
    dt = 1  # the value is unit-dependent, assign 1 without losing generality
    v = np.diff(x0, axis=0) / dt
    s = (v - vf_func(x)).flatten()
    s = 0.5 * s.dot(s) * dt / D  # the total action along the least action path
    least_actions.append(s)

    # compute forward and backward barrier height
    react_coord_x = np.linspace(x_lim[0], x_lim[1], 100)
    path_U = U_func(react_coord_x, lap_func(react_coord_x), grid=False)
    barrier_idx, _ = find_peaks(path_U, height=0)
    p_locs, p_values = react_coord_x[barrier_idx], path_U[barrier_idx]
    if i == 0:
        search_idx = np.where((p_locs < 11) & (p_locs > 8))
    else:
        search_idx = np.where((p_locs < 13) & (p_locs > 8))
    if search_idx is not int:
        search_idx = search_idx[0]
    state_barrier_idx = search_idx[np.argmax(p_values[search_idx])]
    basin_idx, _ = find_peaks(-path_U, height=np.min(-path_U)-1)
    b_locs, b_values = react_coord_x[basin_idx], -path_U[basin_idx]
    search_idx = np.where(b_locs < p_locs[state_barrier_idx])
    if search_idx is not int:
        search_idx = search_idx[0]
    left_basin_idx = search_idx[np.argmax(b_values[search_idx])]
    search_idx = np.where(b_locs > p_locs[state_barrier_idx])
    if search_idx is not int:
        search_idx = search_idx[0]
    right_basin_idx = search_idx[np.argmax(b_values[search_idx])]
    plt.figure()
    plt.plot(react_coord_x, path_U)
    plt.scatter(p_locs[state_barrier_idx], p_values[state_barrier_idx], c='red', s=15)
    plt.scatter(b_locs[left_basin_idx], -b_values[left_basin_idx], c='green', s=15)
    plt.scatter(b_locs[right_basin_idx], -b_values[right_basin_idx], c='green', s=15)
    plt.xlabel("Reaction Coordinates", fontdict={'fontsize': 15})
    plt.ylabel("Potential", fontdict={'fontsize': 15})
    plt.ylim([0, 29])
    plt.title(f"D = {D}", fontdict={'fontsize': 15})
    plt.savefig(f"./figures/reaction_coord_D{D}.svg", transparent=True, dpi=600)
    plt.close()
    forward_barrier_heights.append(p_values[state_barrier_idx] + b_values[right_basin_idx])
    backward_barrier_heights.append(p_values[state_barrier_idx] + b_values[left_basin_idx])

    # compute mean first passage time
    # sim_data = np.load(f"./data/sim_fix_init_D{D}.npz")
    # path = sim_data['path'].reshape(n_path, -1, 2)
    left_basin_loc = np.array([b_locs[left_basin_idx], -b_values[left_basin_idx]])
    barrier_loc = np.array([p_locs[state_barrier_idx], p_values[state_barrier_idx]])
    domain_range = 0.25 * np.sum((left_basin_loc - barrier_loc) ** 2)
    # domain_range = 10
    dist = np.sum((path - left_basin_loc) ** 2, axis=2)
    passage_times = []
    for j in range(n_path):
        hit_idx = np.where(dist[j] < domain_range)[0]
        if hit_idx.shape == (0,):
            continue
        else:
            passage_times.append(np.argmin(hit_idx) * dt)
    if len(passage_times) == 0:
        raise "No passage hit found!"
    mean_first_passage_times.append(np.mean(passage_times))

    # compute log_integral_flux
    xx = np.ravel(Xgrid)
    yy = np.ravel(Ygrid)
    zz = np.ravel(Jx)
    Jx_func = SmoothBivariateSpline(xx, yy, zz)
    xx = np.ravel(Xgrid)
    yy = np.ravel(Ygrid)
    zz = np.ravel(Jy)
    Jy_func = SmoothBivariateSpline(xx, yy, zz)
    lap_x = np.linspace(lap_coordinates[1, 0], lap_coordinates[-1, 0], 100)
    lap_y = lap_func(lap_x)
    dl = np.stack([np.diff(lap_x), np.diff(lap_y)], axis=1)
    x = (lap_x[:-1] + lap_x[1:]) * 0.5
    y = (lap_y[:-1] + lap_y[1:]) * 0.5
    path_Jx = Jx_func(x, y, grid=False)
    path_Jy = Jy_func(x, y, grid=False)
    dJ = np.stack([path_Jx, path_Jy], axis=1)
    c1 = np.abs(np.sum(dJ * dl))
    c2 = np.sum(np.sqrt(dl[:, 0]**2 + dl[:, 1]**2))
    log_integral_flux.append(np.log(c1/c2))

    # compute entropy production rate
    kb = 1.380649e-23
    T = 298.15
    a = GUx * Jx + GUy * Jy
    b = mean_Fx * Jx + mean_Fy * Jy
    epr.append(-np.sum((-kb * T * a - b)))

data_dict = {
    'Ds': np.array(Ds),
    'forward_barrier_heights': np.array(forward_barrier_heights),
    'backward_barrier_heights': np.array(backward_barrier_heights),
    'log_integral_flux': np.array(log_integral_flux),
    'mean_first_passage_times': np.array(mean_first_passage_times),
    'epr': np.array(epr),
    'least_actions': np.array(least_actions),
}
np.savez("./data/thermo_properties.npz", **data_dict)

data = np.load("./data/thermo_properties.npz")
Ds = data['Ds']
forward_barrier_heights = data['forward_barrier_heights']
backward_barrier_heights = data['backward_barrier_heights']
log_integral_flux = data['log_integral_flux']
mean_first_passage_times = data['mean_first_passage_times']
epr = data['epr']
least_actions = data['least_actions']

plt.figure()
plt.plot(Ds, forward_barrier_heights, color='b', marker='o', label='Forward barrier')
plt.plot(Ds, backward_barrier_heights, color='r', lw=2, marker='s', label='Backward barrier')
plt.legend(loc='best')
plt.xscale('log')
plt.xlabel("D", fontdict={'fontsize': 15})
plt.ylabel("Barrier Height", fontdict={'fontsize': 15})
plt.savefig(f"./figures/D-barrier_global.svg", transparent=True, dpi=600)

plt.figure()
plt.plot(Ds, log_integral_flux, color='k', marker='o')
plt.vlines(x=0.0024, ymin=-21, ymax=-11, colors='red', linestyles='dashed')
plt.xscale('log')
plt.xlabel("D", fontdict={'fontsize': 15})
plt.ylabel("Logarithm Flux", fontdict={'fontsize': 15})
plt.savefig(f"./figures/D-flux_global.svg", transparent=True, dpi=600)

plt.figure()
plt.plot(Ds, np.log(epr), color='k', marker='o')
plt.xscale('log')
plt.xlabel("D", fontdict={'fontsize': 15})
plt.ylabel("Logarithm EPR", fontdict={'fontsize': 15})
plt.savefig(f"./figures/D-epr_global.svg", transparent=True, dpi=600)

plt.figure()
plt.loglog(Ds, least_actions, color='k', marker='o')
plt.xlabel("D", fontdict={'fontsize': 15})
plt.ylabel("Least Action", fontdict={'fontsize': 15})
plt.savefig(f"./figures/D-action_global.svg", transparent=True, dpi=600)

plt.figure()
plt.plot(Ds, 1 / (np.exp(-1e-4 * least_actions)+1e-12), color='k', marker='o')
plt.xscale('log')
plt.xlabel("D", fontdict={'fontsize': 15})
plt.ylabel("Logarithm MFPT", fontdict={'fontsize': 15})
plt.savefig(f"./figures/D-mfpt_global.svg", transparent=True, dpi=600)

plt.figure()
plt.plot(least_actions, 1 / (np.exp(-1e-4 * least_actions)+1e-12), color='k', lw=2, marker='o')
plt.xlabel("Least Action", fontdict={'fontsize': 15})
plt.ylabel("Logarithm MFPT", fontdict={'fontsize': 15})
plt.savefig(f"./figures/action-mfpt_global.svg", transparent=True, dpi=600)

plt.figure()
plt.plot(log_integral_flux, least_actions, color='k', lw=2, marker='o')
plt.xlabel("Logarithm Flux", fontdict={'fontsize': 15})
plt.ylabel("Least Action", fontdict={'fontsize': 15})
plt.savefig(f"./figures/flux-action_global.svg", transparent=True, dpi=600)

plt.figure()
plt.plot(log_integral_flux, forward_barrier_heights, color='b', marker='o', label='Forward barrier')
plt.plot(log_integral_flux, backward_barrier_heights, color='r', lw=2, marker='s', label='Backward barrier')
plt.xlabel("Logarithm Flux", fontdict={'fontsize': 15})
plt.ylabel("Barrier Height", fontdict={'fontsize': 15})
plt.savefig(f"./figures/flux-barrier_global.svg", transparent=True, dpi=600)

plt.figure()
plt.plot(log_integral_flux, np.log(epr), color='k', lw=2, marker='o')
plt.xlabel("Logarithm Flux", fontdict={'fontsize': 15})
plt.ylabel("Logarithm EPR", fontdict={'fontsize': 15})
plt.savefig(f"./figures/flux-epr_global.svg", transparent=True, dpi=600)

plt.figure()
plt.plot(forward_barrier_heights, 1 / (np.exp(-1e-4 * least_actions)+1e-12), color='k', marker='o')
plt.xlabel("Barrier Height", fontdict={'fontsize': 15})
plt.ylabel("Logarithm MFPT", fontdict={'fontsize': 15})
plt.savefig(f"./figures/barrier-mfpt_global.svg", transparent=True, dpi=600)

plt.figure()
plt.plot(np.log(epr), 1 / (np.exp(-1e-4 * least_actions)+1e-12), color='k', marker='o')
plt.xlabel("Logarithm EPR", fontdict={'fontsize': 15})
plt.ylabel("Logarithm MFPT", fontdict={'fontsize': 15})
plt.savefig(f"./figures/epr-mfpt_global.svg", transparent=True, dpi=600)

plt.figure()
plt.plot(least_actions, forward_barrier_heights, color='b', marker='o', label='Forward barrier')
plt.plot(least_actions, backward_barrier_heights, color='r', lw=2, marker='s', label='Backward barrier')
plt.legend(loc='best')
plt.xlabel("Least Action", fontdict={'fontsize': 15})
plt.ylabel("Barrier Height", fontdict={'fontsize': 15})
plt.savefig(f"./figures/action-barrier_global.svg", transparent=True, dpi=600)

plt.figure()
plt.plot(least_actions, log_integral_flux, color='k', marker='o')
plt.xlabel("Least Action", fontdict={'fontsize': 15})
plt.ylabel("Logarithm Flux", fontdict={'fontsize': 15})
plt.savefig(f"./figures/action-flux_global.svg", transparent=True, dpi=600)


plt.figure()
total_flux = np.zeros(11)
epr = data_g['epr']
for i in range(fixed_points.shape[0]-1):
    data = np.load(f"./data/thermo_properties_{i}_{i+1}.npz")
    Ds = data['Ds']
    total_flux += np.exp(data['log_integral_flux'])
plt.plot(np.log(total_flux), np.log(epr), color='k', marker='o', lw=2)
plt.xlabel("Logarithm Flux", fontdict={'fontsize': 15})
plt.ylabel("Logarithm EPR", fontdict={'fontsize': 15})
plt.savefig(f"./figures/D-flux_global.svg", transparent=True, dpi=600)


plt.figure()
data = np.load(f"./data/thermo_properties_{0}_{1}.npz")
Ds = data['Ds']
epr = data['epr']
plt.plot(Ds, np.log(epr), color='k', lw=2, marker='o')
plt.xscale('log')
plt.xlabel("D", fontdict={'fontsize': 15})
plt.ylabel("Logarithm EPR", fontdict={'fontsize': 15})
plt.savefig(f"./figures/D-epr.svg", transparent=True, dpi=600)






plt.figure()
for i in range(fixed_points.shape[0]-1):
    data = np.load(f"./data/thermo_properties_{i}_{i+1}.npz")
    Ds = data['Ds']
    least_actions = data['least_actions']
    plt.loglog(Ds, least_actions, lw=2, marker='o', label=fr'{i}$\rightarrow${i+1}')
plt.legend(loc='best')
plt.xlabel("D", fontdict={'fontsize': 15})
plt.ylabel("Least Action", fontdict={'fontsize': 15})
plt.savefig(f"./figures/D-action.svg", transparent=True, dpi=600)









plt.figure()
for i in range(fixed_points.shape[0]-1):
    data = np.load(f"./data/thermo_properties_{i}_{i+1}.npz")
    forward_barrier_heights = data['forward_barrier_heights']
    least_actions = data['least_actions']
    plt.plot(least_actions, forward_barrier_heights, marker='o', lw=2, label=fr'{i}$\rightarrow${i+1}')
plt.legend(loc='best')
plt.xlabel("Least Action", fontdict={'fontsize': 15})
plt.ylabel("Barrier Height", fontdict={'fontsize': 15})
plt.savefig(f"./figures/action-barrier.svg", transparent=True, dpi=600)

plt.figure()
for i in range(fixed_points.shape[0]-1):
    data = np.load(f"./data/thermo_properties_{i}_{i+1}.npz")
    forward_barrier_heights = data['forward_barrier_heights']
    mean_first_passage_times = data['mean_first_passage_times']
    plt.plot(forward_barrier_heights, np.log(mean_first_passage_times), lw=2, marker='o', label=fr'{i}$\rightarrow${i+1}')
plt.legend(loc='best')
plt.xlabel("Barrier Height", fontdict={'fontsize': 15})
plt.ylabel("Logarithm MFPT", fontdict={'fontsize': 15})
plt.savefig(f"./figures/barrier-mfpt.svg", transparent=True, dpi=600)

plt.figure()
for i in range(fixed_points.shape[0]-1):
    data = np.load(f"./data/thermo_properties_{i}_{i+1}.npz")
    epr = data['epr']
    mean_first_passage_times = data['mean_first_passage_times']
    plt.plot(np.log(epr), np.log(mean_first_passage_times), lw=2, marker='o', label=fr'{i}$\rightarrow${i+1}')
plt.legend(loc='best')
plt.xlabel("Logarithm EPR", fontdict={'fontsize': 15})
plt.ylabel("Logarithm MFPT", fontdict={'fontsize': 15})
plt.savefig(f"./figures/epr-mfpt.svg", transparent=True, dpi=600)

plt.figure()
for i in range(fixed_points.shape[0]-1):
    data = np.load(f"./data/thermo_properties_{i}_{i+1}.npz")
    epr = data['epr']
    least_actions = data['least_actions']
    plt.plot(epr, least_actions, lw=2, marker='o', label=fr'{i}$\rightarrow${i+1}')
plt.legend(loc='best')
plt.xlabel("Logarithm EPR", fontdict={'fontsize': 15})
plt.ylabel("Least Action", fontdict={'fontsize': 15})
plt.savefig(f"./figures/epr-action.svg", transparent=True, dpi=600)

plt.figure()
for i in range(fixed_points.shape[0]-1):
    data = np.load(f"./data/thermo_properties_{i}_{i+1}.npz")
    epr = data['epr']
    log_integral_flux = data['log_integral_flux']
    plt.plot(log_integral_flux, np.log(epr), lw=2, marker='o', label=fr'{i}$\rightarrow${i+1}')
plt.legend(loc='best')
plt.xlabel("Logarithm Flux", fontdict={'fontsize': 15})
plt.ylabel("Logarithm EPR", fontdict={'fontsize': 15})
plt.savefig(f"./figures/flux-epr.svg", transparent=True, dpi=600)

plt.figure()
for i in range(fixed_points.shape[0]-1):
    data = np.load(f"./data/thermo_properties_{i}_{i+1}.npz")
    least_actions = data['least_actions']
    log_integral_flux = data['log_integral_flux']
    plt.plot(least_actions, log_integral_flux, lw=2, marker='o', label=fr'{i}$\rightarrow${i+1}')
plt.legend(loc='best')
plt.xlabel("Least Action", fontdict={'fontsize': 15})
plt.ylabel("Logarithm Flux", fontdict={'fontsize': 15})
plt.savefig(f"./figures/action-flux.svg", transparent=True, dpi=600)

plt.figure()
total_flux = np.zeros(11)
for i in range(fixed_points.shape[0]-1):
    data = np.load(f"./data/thermo_properties_{i}_{i+1}.npz")
    epr = data['epr']
    total_flux += np.exp(data['log_integral_flux'])
plt.plot(np.log(total_flux), np.log(epr), lw=2, marker='o', label=fr'{i}$\rightarrow${i+1}')
plt.legend(loc='best')
plt.xlabel("Logarithm Flux", fontdict={'fontsize': 15})
plt.ylabel("Logarithm EPR", fontdict={'fontsize': 15})
plt.savefig(f"./figures/flux-epr_global.svg", transparent=True, dpi=600)

plt.figure()
total_flux = np.zeros(11)
for i in range(fixed_points.shape[0]-1):
    data = np.load(f"./data/thermo_properties_{i}_{i+1}.npz")
    epr = data['epr']
    total_flux += np.exp(data['log_integral_flux'])
    mean_first_passage_times = data['mean_first_passage_times']
    plt.plot(np.log(total_flux), np.log(epr), lw=2, marker='o', label=fr'{i}$\rightarrow${i+1}')
plt.legend(loc='best')
plt.xlabel("Logarithm Flux", fontdict={'fontsize': 15})
plt.ylabel("Logarithm EPR", fontdict={'fontsize': 15})
plt.savefig(f"./figures/flux-epr_global.svg", transparent=True, dpi=600)

