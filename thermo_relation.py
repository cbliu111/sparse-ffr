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
from tl.mfpt import mfpt_parallel
import warnings

warnings.simplefilter('ignore')

plt.style.use(['science', 'nature'])

###############################################################################################
""" obtain least action path and potential landscape functions """
###############################################################################################

lap_list = pickle.load(open("least_action_path.pickle", "rb"))
adata = anndata.read_h5ad(Path("./data/contours.h5ad"))
X = adata.obsm['X_umap']
vf = tl.VectorField(adata, coord_basis='X_umap', velo_basis='vector_field', dims=2)
vf_func = lambda xx: vf.vector_field_function(xx, adata.uns['vf_dict'])
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
"""
for i in range(fixed_points.shape[0]-1):
    forward_barrier_heights = []  # from state 1 to state 2
    backward_barrier_heights = []  # from state 2 back to state 1
    least_actions = []
    mean_first_passage_times = []
    log_integral_flux = []
    epr = []
    lap = lap_list[i]
    start_point = fixed_points[i]
    end_point = fixed_points[i + 1]
    lap_coordinates = lap_list[i]['prediction'][0]
    idx = np.argsort(lap_coordinates[:, 0])
    lap_coordinates = lap_coordinates[idx]
    eps = np.linspace(0, 1e-12, lap_coordinates.shape[0])  # avoid equality checking of x-axis
    lap_func = make_interp_spline(lap_coordinates[:, 0] + eps, lap_coordinates[:, 1], k=3)

    for j, D in enumerate(Ds):
        sim_data = np.load(f"./data/sim_D{D}.npz")
        num_tra = sim_data['num_traj']
        total_Fx = sim_data['Fx']
        total_Fy = sim_data['Fy']

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
        x0 = lap_coordinates
        x = (x0[:-1] + x0[1:]) * 0.5
        dt = 1  # the value is unit-dependent, assign 1 without losing generality
        v = np.diff(x0, axis=0) / dt
        s = (v - vf_func(x)).flatten()
        s = 0.5 * s.dot(s) * dt / D  # the total action along the least action path
        least_actions.append(s)
        mfpt = 1 / (np.exp(-1e-4 * s)+1e-12)  # code adapted from dynamo, multiply by 1e-3 to avoid overflow, same as change time units
        mean_first_passage_times.append(mfpt)

        # compute forward and backward barrier height
        lap_x = np.linspace(lap_coordinates[:, 0].min(), lap_coordinates[:, 0].max(), 100)
        path_U = U_func(lap_x, lap_func(lap_x), grid=False)
        barrier_idx, height = find_peaks(path_U, height=0)
        U1 = U_func(start_point[0], start_point[1])[0, 0]
        U2 = U_func(end_point[0], end_point[1])[0, 0]
        if barrier_idx.size == 0:
            forward_barrier_heights.append(U2 - U1)
            backward_barrier_heights.append(U1 - U2)
        else:
            forward_barrier_heights.append(height['peak_heights'][0] - U1)
            backward_barrier_heights.append(height['peak_heights'][0] - U2)

        # mfpt = mfpt_parallel(
        #     vf_func,
        #     start_point,
        #     end_point,
        #     x_lim=x_lim,
        #     y_lim=y_lim,
        #     domain_range=None,
        #     max_time_steps=int(1e5),
        #     n_paths=100,
        #     diff_coeff=D,
        #     dt=1,
        #     cpus=20,
        # )
        # mean_first_passage_times.append(mfpt)

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
        c2 = np.sum(np.sqrt(dl[:, 0] ** 2 + dl[:, 1] ** 2))
        # log_integral_flux.append(np.log(c1 / c2))
        log_integral_flux.append(np.log(c1))

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
        'least_actions': np.array(least_actions),
        'epr': np.array(epr),
    }
    np.savez(f"./data/thermo_properties_{i}_{i+1}.npz", **data_dict)


data = np.load(f"./data/thermo_properties_{i}_{i+1}.npz")
Ds = data['Ds']
forward_barrier_heights = data['forward_barrier_heights']
backward_barrier_heights = data['backward_barrier_heights']
log_integral_flux = data['log_integral_flux']
mean_first_passage_times = data['mean_first_passage_times']
epr = data['epr']
least_actions = data['least_actions']

"""

plt.figure()
for i in range(fixed_points.shape[0]-1):
    data = np.load(f"./data/thermo_properties_{i}_{i+1}.npz")
    Ds = data['Ds']
    forward_barrier_heights = data['forward_barrier_heights']
    plt.plot(Ds, forward_barrier_heights, lw=2, marker='o', label=fr'{i}$\rightarrow${i+1}')
plt.legend(loc='best')
plt.xscale('log')
plt.xlabel("D", fontdict={'fontsize': 15})
plt.ylabel("Barrier Height", fontdict={'fontsize': 15})
plt.savefig(f"./figures/D-barrier.svg", transparent=True, dpi=600)

plt.figure()
for i in range(fixed_points.shape[0]-1):
    data = np.load(f"./data/thermo_properties_{i}_{i+1}.npz")
    Ds = data['Ds']
    mean_first_passage_times = data['mean_first_passage_times']
    plt.plot(Ds, np.log(mean_first_passage_times), lw=2, marker='o', label=fr'{i}$\rightarrow${i+1}')
plt.xscale('log')
plt.legend(loc='best')
plt.xlabel("D", fontdict={'fontsize': 15})
plt.ylabel("Logarithm MFPT", fontdict={'fontsize': 15})
plt.savefig(f"./figures/D-mfpt.svg", transparent=True, dpi=600)

plt.figure()
for i in range(fixed_points.shape[0]-1):
    data = np.load(f"./data/thermo_properties_{i}_{i+1}.npz")
    Ds = data['Ds']
    log_integral_flux = data['log_integral_flux']
    plt.plot(Ds, log_integral_flux, marker='o', lw=2, label=fr'{i}$\rightarrow${i+1}')
    # plt.vlines(x=0.008, ymin=-21, ymax=-11, colors='red', linestyles='dashed')
plt.xscale('log')
plt.legend(loc='best')
plt.xlabel("D", fontdict={'fontsize': 15})
plt.ylabel("Logarithm Flux", fontdict={'fontsize': 15})
plt.savefig(f"./figures/D-flux.svg", transparent=True, dpi=600)

data_g = np.load("./data/thermo_properties.npz")
Ds = data_g['Ds']
forward_barrier_heights = data_g['forward_barrier_heights']
backward_barrier_heights = data_g['backward_barrier_heights']
least_actions = data_g['least_actions']
mfpt = 1 / (np.exp(-1e-4 * least_actions)+1e-12)
epr = data_g['epr']
total_flux = np.zeros(11)
for i in range(fixed_points.shape[0]-1):
    data = np.load(f"./data/thermo_properties_{i}_{i+1}.npz")
    Ds = data['Ds']
    total_flux += np.exp(data['log_integral_flux'])

plt.figure()
plt.plot(Ds, forward_barrier_heights, color='b', marker='o', label='Forward barrier')
plt.plot(Ds, backward_barrier_heights, color='r', lw=2, marker='s', label='Backward barrier')
plt.legend(loc='best')
plt.xscale('log')
plt.xlabel("D", fontdict={'fontsize': 15})
plt.ylabel("Barrier Height", fontdict={'fontsize': 15})
plt.savefig(f"./figures/D-barrier_global.svg", transparent=True, dpi=600)
plt.figure()

plt.plot(least_actions, forward_barrier_heights, color='b', marker='o', label='Forward barrier')
plt.plot(least_actions, backward_barrier_heights, color='r', lw=2, marker='s', label='Backward barrier')
plt.legend(loc='best')
plt.xlabel("Least Action", fontdict={'fontsize': 15})
plt.ylabel("Barrier Height", fontdict={'fontsize': 15})
plt.savefig(f"./figures/action-barrier_global.svg", transparent=True, dpi=600)

plt.figure()
plt.plot(Ds, np.log(mfpt), color='k', marker='o')
plt.xscale('log')
plt.xlabel("D", fontdict={'fontsize': 15})
plt.ylabel("Logarithm MFPT", fontdict={'fontsize': 15})
plt.savefig(f"./figures/D-mfpt_global.svg", transparent=True, dpi=600)

plt.figure()
plt.plot(forward_barrier_heights, np.log(mfpt), color='k', marker='o')
plt.xlabel("Barrier Height", fontdict={'fontsize': 15})
plt.ylabel("Logarithm MFPT", fontdict={'fontsize': 15})
plt.savefig(f"./figures/barrier-mfpt_global.svg", transparent=True, dpi=600)

plt.figure()
plt.plot(Ds, np.log(total_flux), color='k', marker='o', lw=2)
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
plt.plot(np.log(total_flux), np.log(epr), color='k', marker='o', lw=2)
plt.xlabel("Logarithm Flux", fontdict={'fontsize': 15})
plt.ylabel("Logarithm EPR", fontdict={'fontsize': 15})
plt.savefig(f"./figures/flux-epr_global.svg", transparent=True, dpi=600)

plt.figure()
ax1 = plt.gca()
ax1.plot(forward_barrier_heights, np.log(total_flux), color='red', marker='o', lw=2)
ax1.set_xlabel("Barrier Height", fontdict={'fontsize': 15})
ax1.set_ylabel("Logarithm Flux", fontdict={'fontsize': 15}, color='red')
ax2 = ax1.twinx()
ax2.plot(forward_barrier_heights, np.log(mfpt), color='blue', marker='o', lw=2)
ax2.set_ylabel("Logarithm MFPT", fontdict={'fontsize': 15}, color='blue')
plt.savefig(f"./figures/barrier-flux-mfpt_global.svg", transparent=True, dpi=600)

plt.figure()
ax1 = plt.gca()
ax1.plot(np.log(mfpt), np.log(total_flux), color='red', marker='o', lw=2)
ax1.set_xlabel("Logarithm MFPT", fontdict={'fontsize': 15})
ax1.set_ylabel("Logarithm Flux", fontdict={'fontsize': 15}, color='red')
ax2 = ax1.twinx()
ax2.plot(np.log(mfpt), np.log(epr), color='blue', marker='o', lw=2)
ax2.set_ylabel("Logarithm EPR", fontdict={'fontsize': 15}, color='blue')
plt.savefig(f"./figures/mfpt-flux-epr_global.svg", transparent=True, dpi=600)



