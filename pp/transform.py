import scipy
import numpy as np
import copy
import sklearn
from pp.mtrack import generate_contours, align_contours, align_contour_to, decompose_homogenous_transform


def box_cox(adata, used_components=20):
    """
    using **scipy.stats.boxcox** to calculate box_cox_lambda and transform the pca vectors.
    """
    _m = adata.obsm['X_pca'][:, :used_components].copy()
    min_values = np.amin(_m, axis=0)
    box_cox_lambda = np.zeros(len(_m.T))
    shift_vector_mean = np.zeros(len(_m.T))
    shift_vector_std = np.zeros(len(_m.T))

    for k in range(len(_m.T)):
        shift_vector = _m.T[k]
        shift_vector = shift_vector - min_values[k] + 1
        shift_vector[shift_vector < 0] = 1e-12
        shift_vector, maxlog = scipy.stats.boxcox(shift_vector)
        shift_vector = np.asarray(shift_vector)
        box_cox_lambda[k] = maxlog
        shift_vector_mean[k] = np.mean(shift_vector)
        shift_vector_std[k] = np.std(shift_vector)
        _m.T[k] = (shift_vector - np.mean(shift_vector)) / np.std(shift_vector)

    # norm_vectors = sklearn.preprocessing.normalize(_m)
    norm_vectors = _m

    adata.uns['box_cox_transform'] = {
        'min_values': min_values,
        'box_cox_lambda': box_cox_lambda,
        'shift_vector_mean': shift_vector_mean,
        'shift_vector_std': shift_vector_std,
    }
    adata.obsm['X_pca_boxcox'] = norm_vectors


def procrustes_align(adata):
    _m = adata.X
    names = adata.obs_names
    pts = int(adata.n_vars / 2)
    contours_and_obj = []
    for i in range(adata.n_obs):
        c = np.array([_m[i, :pts], _m[i, pts:]]).T
        contours_and_obj.append((c, names))

    cell_contours, sort_obj_arr = generate_contours(contours_and_obj, closed_only=False)

    print(f"procrustes alignment 1/3: resampling...")
    for i in range(len(cell_contours)):
        cell_contours[i].resample(num_points=pts)
        cell_contours[i].axis_align()

    # randomly chose 500 contours for calculating the mean contour
    print(f"procrustes alignment 2/3: calculating reference contour...")
    num_sample_points = min(500, len(cell_contours))
    num_samples = max(len(cell_contours), num_sample_points)
    idx = np.random.choice(range(len(cell_contours)), num_samples, replace=False)
    sample_contours = [cell_contours[i] for i in idx]
    mean_contour, iters = align_contours(sample_contours, allow_reflection=True, allow_scaling=False, max_iters=20)

    print(f"procrustes alignment 3/3: realigning contours to reference contour...")
    for i in range(len(cell_contours)):
        align_contour_to(cell_contours[i], mean_contour, allow_reflection=True, allow_scaling=True)
        scale_back = decompose_homogenous_transform(cell_contours[i].to_world_transform)[1]
        cell_contours[i].scale(scale_back)

    points = [x.points for x in cell_contours]
    adata.obsm["procrustes_align"] = np.array(points).T












