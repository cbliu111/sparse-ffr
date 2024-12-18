from tl.simulate import sim_parallel
import anndata
from pathlib import Path
from utils import logger
import tl
from sklearn.mixture import GaussianMixture
import numpy as np

adata = anndata.read_h5ad(Path("./data/contours.h5ad"))
X = adata.obsm['X_umap']

############################################################################################################
""" Set parameters """
############################################################################################################

n_grid = 100
n_path = 600
n_cpu = 20
time_steps = int(5e5)
start_record_step = int(1e3)

############################################################################################################
""" Sample initial points """
############################################################################################################

gmm_dicts = adata.uns['gmm_dicts']
d = gmm_dicts['state_dict_0']
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
inits, _ = gmm.sample(n_samples=n_path)

# start_point = np.array([11.726526501202823, 2.7750460884787813])
# inits = np.repeat(start_point, n_path, axis=0)
# end_point = np.array([-9.074971218301792, 1.9145810098359082])

############################################################################################################
""" Simulate Langevin equation """
############################################################################################################

expand = 1
xmin, xmax = X[:, 0].min() - expand, X[:, 0].max() + expand
ymin, ymax = X[:, 1].min() - expand, X[:, 1].max() + expand
x_lim = (xmin, xmax)
y_lim = (ymin, ymax)

vf = tl.VectorField(adata, coord_basis='X_umap', velo_basis='vector_field', dims=2)


def func(x):
    return vf.vector_field_function(x, adata.uns['vf_dict'])


Ds = [
    # 0.,
    # 0.0001,
    0.0003,
    0.0008,
    0.0015,
    0.0024,
    # 0.003,
    # 0.005,
    # 0.008,
    # 0.01,
    0.015,
    0.024,
    0.03,
]
for D in Ds:
    print(f"Simulating {D}")
    results = sim_parallel(
        func,
        inits,
        x_lim,
        y_lim,
        diff_coeff=D,
        n_paths=n_path,
        n_time_steps=time_steps,
        start_time=start_record_step,
        Tra_grid=n_grid,
        cpus=n_cpu,
    )
    data_dict = {
        'num_traj': results[0],
        'Fx': results[1],
        'Fy': results[2],
        'path': results[3],
    }
    np.savez(f"./data/sim_D{D}.npz", **data_dict)
