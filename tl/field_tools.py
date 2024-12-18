import numpy as np
from scipy.spatial.distance import cdist, pdist
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union, TypedDict
from anndata import AnnData
from tqdm import tqdm
import multiprocessing as mp
import numdifftools as nd
from multiprocessing.dummy import Pool as ThreadPool
import itertools
from scipy.optimize import fsolve



class NormDict(TypedDict):
    xm: np.ndarray
    ym: np.ndarray
    xscale: float
    yscale: float
    fix_velocity: bool


class VecFldDict(TypedDict):
    X: np.ndarray
    valid_ind: float
    X_ctrl: np.ndarray
    ctrl_idx: float
    Y: np.ndarray
    beta: float
    V: np.ndarray
    C: np.ndarray
    P: np.ndarray
    VFCIndex: np.ndarray
    sigma2: float
    grid: np.ndarray
    grid_V: np.ndarray
    iteration: int
    tecr_traj: np.ndarray
    E_traj: np.ndarray
    norm_dict: NormDict


def con_K(
        x: np.ndarray,
        y: np.ndarray,
        beta: float = 0.1,
        method: str = "cdist",
        return_d: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    con_K constructs the kernel K, where K(i, j) = k(x, y) = exp(-beta * ||x - y||^2).

    Args:
        x:
            Original training data points.
        y:
            Control points used to build kernel basis functions.
        beta:
            parameter of Gaussian Kernel, k(x, y) = exp(-beta*||x-y||^2),
        method:
            metric method, "cdist" or "pdist".
        return_d:
            If True the intermediate 3D matrix x - y will be returned for analytical Jacobin.

    Returns:
        Tuple(K: the kernel to represent the vector field function, D:
    """
    D = []
    if method == "cdist" and not return_d:
        K = cdist(x, y, "sqeuclidean")
        if len(K) == 1:
            K = K.flatten()
    else:
        n = x.shape[0]
        m = y.shape[0]

        # https://stackoverflow.com/questions/1721802/what-is-the-equivalent-of-matlabs-repmat-in-numpy
        # https://stackoverflow.com/questions/12787475/matlabs-permute-in-python
        D = np.matlib.tile(x[:, :, None], [1, 1, m]) - np.transpose(np.matlib.tile(y[:, :, None], [1, 1, n]), [2, 1, 0])
        K = np.squeeze(np.sum(D ** 2, 1))
    K = -beta * K
    K = np.exp(K)

    if return_d:
        return K, D
    else:
        return K


def con_K_div_cur_free(
    x: np.ndarray, y: np.ndarray, sigma: int = 0.8, eta: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a convex combination of the divergence-free kernel T_df and curl-free kernel T_cf with a bandwidth sigma
    and a combination coefficient gamma.

    Args:
        x: Original training data points.
        y: Control points used to build kernel basis functions
        sigma: Bandwidth parameter.
        eta: Combination coefficient for the divergence-free or the curl-free kernels.

    Returns:
        A tuple of G (the combined kernel function), divergence-free kernel and curl-free kernel.

    See also: :func:`sparseVFC`.
    """
    m, d = x.shape
    n, d = y.shape
    sigma2 = sigma**2
    G_tmp = np.matlib.tile(x[:, :, None], [1, 1, n]) - np.transpose(np.matlib.tile(y[:, :, None], [1, 1, m]), [2, 1, 0])
    G_tmp = np.squeeze(np.sum(G_tmp**2, 1))
    G_tmp3 = -G_tmp / sigma2
    G_tmp = -G_tmp / (2 * sigma2)
    G_tmp = np.exp(G_tmp) / sigma2
    G_tmp = np.kron(G_tmp, np.ones((d, d)))

    x_tmp = np.matlib.tile(x, [n, 1])
    y_tmp = np.matlib.tile(y, [1, m]).T
    y_tmp = y_tmp.reshape((d, m * n), order="F").T
    xminusy = x_tmp - y_tmp
    G_tmp2 = np.zeros((d * m, d * n))

    tmp4_ = np.zeros((d, d))
    for i in tqdm(range(d), desc="Iterating each dimension in con_K_div_cur_free:"):
        for j in np.arange(i, d):
            tmp1 = xminusy[:, i].reshape((m, n), order="F")
            tmp2 = xminusy[:, j].reshape((m, n), order="F")
            tmp3 = tmp1 * tmp2
            tmp4 = tmp4_.copy()
            tmp4[i, j] = 1
            tmp4[j, i] = 1
            G_tmp2 = G_tmp2 + np.kron(tmp3, tmp4)

    G_tmp2 = G_tmp2 / sigma2
    G_tmp3 = np.kron((G_tmp3 + d - 1), np.eye(d))
    G_tmp4 = np.kron(np.ones((m, n)), np.eye(d)) - G_tmp2
    df_kernel, cf_kernel = (1 - eta) * G_tmp * (G_tmp2 + G_tmp3), eta * G_tmp * G_tmp4
    G = df_kernel + cf_kernel

    return G, df_kernel, cf_kernel



def unit_vector(vector):
    """Returns the unit vector of the vector."""
    vec_norm = np.linalg.norm(vector)
    if vec_norm == 0:
        return vec_norm, vector
    else:
        return vec_norm, vector / vec_norm


def angle(vector1, vector2):
    """Returns the angle in radians between given vectors"""
    # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python#answer-13849249
    v1_norm, v1_u = unit_vector(vector1)
    v2_norm, v2_u = unit_vector(vector2)

    if v1_norm == 0 or v2_norm == 0:
        return np.nan
    else:
        minor = np.linalg.det(np.stack((v1_u[-2:], v2_u[-2:])))
        if minor == 0:
            sign = 1
        else:
            sign = -np.sign(minor)
        dot_p = np.dot(v1_u, v2_u)
        dot_p = min(max(dot_p, -1.0), 1.0)
        return sign * np.arccos(dot_p)


def norm(
        X: np.ndarray, V: np.ndarray, T: Optional[np.ndarray] = None, fix_velocity: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Normalizes the X, Y (X + V) matrix to have zero means and unit covariance.
    We use the mean of X, Y's center (mean) and scale parameters (standard deviation) to normalize T.

    Args:
        X: Current state. This corresponds to, for example, the spliced transcriptomic state.
        V: Velocity estimates in delta t. This corresponds to, for example, the inferred spliced transcriptomic
            velocity estimated calculated by dynamo or velocyto, scvelo.
        T: Current state on a grid which is often used to visualize the vector field. This corresponds to, for example,
            the spliced transcriptomic state.
        fix_velocity: Whether to fix velocity and don't transform it. Default is True.

    Returns:
        A tuple of updated X, V, T and norm_dict which includes the mean and scale values for original X, V data used
        in normalization.
    """
    Y = X + V

    xm = np.mean(X, axis=0)
    ym = np.mean(Y, axis=0)

    x, y, t = (X - xm[None, :], Y - ym[None, :], T - (1 / 2 * (xm[None, :] + ym[None, :])) if T is not None else None)

    xscale, yscale = (np.sqrt(np.mean(x ** 2, axis=0))[None, :], np.sqrt(np.mean(y ** 2, axis=0))[None, :])

    X, Y, T = x / xscale, y / yscale, t / (1 / 2 * (xscale + yscale)) if T is not None else None

    X, V, T = X, V if fix_velocity else Y - X, T
    norm_dict = {"xm": xm, "ym": ym, "xscale": xscale, "yscale": yscale, "fix_velocity": fix_velocity}

    return X, V, T, norm_dict


def denorm(
        VecFld: Dict[str, Union[np.ndarray, None]],
        X_old: np.ndarray,
        V_old: np.ndarray,
        norm_dict: Dict[str, Union[np.ndarray, bool]],
) -> Dict[str, Union[np.ndarray, None]]:
    """Denormalize data back to the original scale.

    Args:
        VecFld: The dictionary that stores the information for the reconstructed vector field function.
        X_old: The original data for current state.
        V_old: The original velocity data.
        norm_dict: The norm_dict dictionary that includes the mean and scale values for X, Y used in normalizing the
            data.

    Returns:
        An updated VecFld dictionary that includes denormalized X, Y, X_ctrl, grid, grid_V, V, and the norm_dict key.
    """
    Y_old = X_old + V_old
    xm, ym = norm_dict["xm"], norm_dict["ym"]
    x_scale, y_scale = norm_dict["xscale"], norm_dict["yscale"]
    xy_m, xy_scale = (xm + ym) / 2, (x_scale + y_scale) / 2

    X = VecFld["X"]
    X_denorm = X_old
    Y = VecFld["Y"]
    Y_denorm = Y_old
    V = VecFld["V"]
    V_denorm = V_old if norm_dict["fix_velocity"] else (V + X) * y_scale + np.tile(ym, [V.shape[0], 1]) - X_denorm
    grid = VecFld["grid"]
    grid_denorm = grid * xy_scale + np.tile(xy_m, [grid.shape[0], 1]) if grid is not None else None
    grid_V = VecFld["grid_V"]
    grid_V_denorm = (
        (grid + grid_V) * xy_scale + np.tile(xy_m, [grid_V.shape[0], 1]) - grid if grid_V is not None else None
    )
    VecFld_denorm = {
        "X": X_denorm,
        "Y": Y_denorm,
        "V": V_denorm,
        "grid": grid_denorm,
        "grid_V": grid_V_denorm,
        "norm_dict": norm_dict,
    }

    return VecFld_denorm


def merge_dict(dict1: dict, dict2: dict, update=False) -> dict:
    """Merge two dictionaries.

    For overlapping keys, the values in dict 2 would replace values in dict 1.

    Args:
        dict1: The dict to be merged into and overwritten.
        dict2: The dict to be merged.

    Returns:
        The updated dict.
    """
    if update:
        dict1.update((k, dict2[k]) for k in dict1.keys() & dict2.keys())
    else:
        dict1.update((k, dict2[k]) for k in dict1.keys() | dict2.keys())

    return dict1


def vector_field_function_transformation(vf_func: Callable, Q: np.ndarray, func_inv_x: Callable) -> Callable:
    """Transform vector field function from PCA space to the original space.
    The formula used for transformation:
                                            :math:`\hat{f} = f Q^T`,
    where `Q, f, \hat{f}` are the PCA loading matrix, low dimensional vector field function and the
    transformed high dimensional vector field function.

    Args:
        vf_func: The vector field function.
        Q: PCA loading matrix with dimension d x k, where d is the dimension of the original space,
            and k the number of leading PCs.
        func_inv_x: The function that transform x back into the PCA space.

    Returns:
        The transformed vector field function.

    """
    return lambda x: vf_func(x @ Q.T) @ Q.T


def Jacobian_rkhs_gaussian(x: np.ndarray, vf_dict: VecFldDict, vectorize: bool = False) -> np.ndarray:
    """analytical Jacobian for RKHS vector field functions with Gaussian kernel.

    Args:
    x: Coordinates where the Jacobian is evaluated.
    vf_dict: A dictionary containing RKHS vector field control points, Gaussian bandwidth,
        and RKHS coefficients.
        Essential keys: 'X_ctrl', 'beta', 'C'

    Returns:
        Jacobian matrices stored as d-by-d-by-n numpy arrays evaluated at x.
            d is the number of dimensions and n the number of coordinates in x.
    """
    if x.ndim == 1:
        K, D = con_K(x[None, :], vf_dict["X_ctrl"], vf_dict["beta"], return_d=True)
        J = (vf_dict["C"].T * K) @ D[0].T
    elif not vectorize:
        n, d = x.shape
        J = np.zeros((d, d, n))
        for i, xi in enumerate(x):
            K, D = con_K(xi[None, :], vf_dict["X_ctrl"], vf_dict["beta"], return_d=True)
            J[:, :, i] = (vf_dict["C"].T * K) @ D[0].T
    else:
        K, D = con_K(x, vf_dict["X_ctrl"], vf_dict["beta"], return_d=True)
        if K.ndim == 1:
            K = K[None, :]
        J = np.einsum("nm, mi, njm -> ijn", K, vf_dict["C"], D)

    return -2 * vf_dict["beta"] * J


def Jacobian_rkhs_gaussian_parallel(x: np.ndarray, vf_dict: VecFldDict, cores: Optional[int] = None) -> np.ndarray:
    n = len(x)
    if cores is None:
        cores = mp.cpu_count()
    n_j_per_core = int(np.ceil(n / cores))
    xx = []
    for i in range(0, n, n_j_per_core):
        xx.append(x[i: i + n_j_per_core])
    # with mp.Pool(cores) as p:
    #    ret = p.starmap(Jacobian_rkhs_gaussian, zip(xx, itertools.repeat(vf_dict)))
    with ThreadPool(cores) as p:
        ret = p.starmap(Jacobian_rkhs_gaussian, zip(xx, itertools.repeat(vf_dict)))
    ret = [np.transpose(r, axes=(2, 0, 1)) for r in ret]
    ret = np.transpose(np.vstack(ret), axes=(1, 2, 0))
    return ret

def Jacobian_numerical(f: Callable, input_vector_convention: str = "row") -> Union[Callable, nd.Jacobian]:
    """
    Get the numerical Jacobian of the vector field function.
    If the input_vector_convention is 'row', it means that fjac takes row vectors
    as input, otherwise the input should be an array of column vectors. Note that
    the returned Jacobian would behave exactly the same if the input is an 1d array.

    The column vector convention is slightly faster than the row vector convention.
    So the matrix of row vector convention is converted into column vector convention
    under the hood.

    No matter the input vector convention, the returned Jacobian is of the following
    format:
            df_1/dx_1   df_1/dx_2   df_1/dx_3   ...
            df_2/dx_1   df_2/dx_2   df_2/dx_3   ...
            df_3/dx_1   df_3/dx_2   df_3/dx_3   ...
            ...         ...         ...         ...
    """
    fjac = nd.Jacobian(lambda x: f(x.T).T)
    if input_vector_convention == "row" or input_vector_convention == 0:

        def f_aux(x):
            x = x.T
            return fjac(x)

        return f_aux
    else:
        return fjac



def is_outside_domain(x: np.ndarray, domain: Tuple[float, float]) -> np.ndarray:
    x = x[None, :] if x.ndim == 1 else x
    return np.any(np.logical_or(x < domain[0], x > domain[1]), axis=1)


def grad(f: Callable, x: np.ndarray) -> nd.Gradient:
    """Gradient of scalar-valued function f evaluated at x"""
    return nd.Gradient(f)(x)


def laplacian(f: Callable, x: np.ndarray) -> float:
    """Laplacian of scalar field f evaluated at x"""
    hes = nd.Hessdiag(f)(x)
    return sum(hes)


def get_vf_dict(adata: AnnData, basis: str = "", vf_key: str = "VecFld") -> VecFldDict:
    """Get vector field dictionary from the `.uns` attribute of the AnnData object.

    Args:
        adata: `AnnData` object
        basis: string indicating the embedding data to use for calculating velocities. Defaults to "".
        vf_key: _description_. Defaults to "VecFld".

    Raises:
        ValueError: if vf_key or vfkey_basis is not included in the adata object.

    Returns:
        vector field dictionary
    """
    if basis is not None:
        if len(basis) > 0:
            vf_key = "%s_%s" % (vf_key, basis)

    if vf_key not in adata.uns.keys():
        raise ValueError(
            f"Vector field function {vf_key} is not included in the adata object! "
            f"Try firstly running dyn.vf.VectorField(adata, basis='{basis}')"
        )

    vf_dict = adata.uns[vf_key]
    return vf_dict


# ---------------------------------------------------------------------------------------------------
# jacobian



def elementwise_jacobian_transformation(Js: np.ndarray, qi: np.ndarray, qj: np.ndarray) -> np.ndarray:
    """Inverse transform low dimensional k x k Jacobian matrix (:math:`\partial F_i / \partial x_j`) back to the
    d-dimensional gene expression space. The formula used to inverse transform Jacobian matrix calculated from
    low dimension (PCs) is:
                                            :math:`Jac = Q J Q^T`,
    where `Q, J, Jac` are the PCA loading matrix, low dimensional Jacobian matrix and the inverse transformed high
    dimensional Jacobian matrix. This function takes only one row from Q to form qi or qj.

    Args:
        Js: k x k x n matrices of n k-by-k Jacobians.
        qi: The i-th row of the PC loading matrix Q with dimension d x k, corresponding to the effector gene i.
        qj: The j-th row of the PC loading matrix Q with dimension d x k, corresponding to the regulator gene j.

    Returns:
        The calculated Jacobian elements (:math:`\partial F_i / \partial x_j`) for each cell.
    """

    Js = np.atleast_3d(Js)
    n = Js.shape[2]
    ret = np.zeros(n)
    for i in tqdm(range(n), "calculating Jacobian for each cell"):
        ret[i] = qi @ Js[:, :, i] @ qj

    return ret


def Jacobian_kovf(
        x: np.ndarray, fjac_base: Callable, K: np.ndarray, Q: np.ndarray, exact: bool = False,
        mu: Optional[float] = None
) -> np.ndarray:
    """analytical Jacobian for RKHS vector field functions with Gaussian kernel.

    Args:
        x: Coordinates where the Jacobian is evaluated.
        vf_dict:
            A dictionary containing RKHS vector field control points, Gaussian bandwidth,
            and RKHS coefficients.
            Essential keys: 'X_ctrl', 'beta', 'C'

    Returns:
        Jacobian matrices stored as d-by-d-by-n numpy arrays evaluated at x.
            d is the number of dimensions and n the number of coordinates in x.
    """
    if K.ndim == 1:
        K = np.diag(K)

    if exact:
        if mu is None:
            raise Exception("For exact calculations of the Jacobian, the mean of the PCA transformation is needed.")

        s = np.sign(x @ Q.T + mu)
        if x.ndim > 1:
            G = np.zeros((Q.shape[1], Q.shape[1], x.shape[0]))
            KQ = K @ Q
            # KQ = (np.diag(K) * Q.T).T
            for i in range(x.shape[0]):
                G[:, :, i] = s[i] * Q.T @ KQ
        else:
            G = s * Q.T @ K @ Q
    else:
        G = Q.T @ K @ Q
        if x.ndim > 1:
            G = np.repeat(G[:, :, None], x.shape[0], axis=2)

    return fjac_base(x) - G


def subset_jacobian_transformation(Js: np.ndarray, Qi: np.ndarray, Qj: np.ndarray, cores: int = 1) -> np.ndarray:
    """Transform Jacobian matrix (:math:`\partial F_i / \partial x_j`) from PCA space to the original space.
    The formula used for transformation:
                                            :math:`\hat{J} = Q J Q^T`,
    where `Q, J, \hat{J}` are the PCA loading matrix, low dimensional Jacobian matrix and the inverse transformed high
    dimensional Jacobian matrix. This function takes multiple rows from Q to form Qi or Qj.

    Args:
        Js: Original (k x k) dimension Jacobian matrix
        Qi: PCA loading matrix with dimension n' x n_PCs of the effector genes, from which local dimension Jacobian matrix (k x k)
            will be inverse transformed back to high dimension.
        Qj: PCs loading matrix with dimension n' x n_PCs of the regulator genes, from which local dimension Jacobian matrix (k x k)
            will be inverse transformed back to high dimension.
        cores: Number of cores to calculate Jacobian. If cores is set to be > 1, multiprocessing will be used to
            parallel the Jacobian calculation.

    Returns:
        The calculated Jacobian matrix (n_gene x n_gene x n_obs) for each cell.
    """

    Js = np.atleast_3d(Js)
    Qi = np.atleast_2d(Qi)
    Qj = np.atleast_2d(Qj)
    d1, d2, n = Qi.shape[0], Qj.shape[0], Js.shape[2]

    ret = np.zeros((d1, d2, n))

    if cores == 1:
        ret = transform_jacobian(Js, Qi, Qj, pbar=True)
    else:
        if cores is None:
            cores = mp.cpu_count()
        n_j_per_core = int(np.ceil(n / cores))
        JJ = []
        for i in range(0, n, n_j_per_core):
            JJ.append(Js[:, :, i: i + n_j_per_core])
        with ThreadPool(cores) as p:
            ret = p.starmap(
                transform_jacobian,
                zip(JJ, itertools.repeat(Qi), itertools.repeat(Qj)),
            )
        ret = [np.transpose(r, axes=(2, 0, 1)) for r in ret]
        ret = np.transpose(np.vstack(ret), axes=(1, 2, 0))

    return ret


def transform_jacobian(Js: np.ndarray, Qi: np.ndarray, Qj: np.ndarray, pbar=False) -> np.ndarray:
    d1, d2, n = Qi.shape[0], Qj.shape[0], Js.shape[2]
    ret = np.zeros((d1, d2, n), dtype=np.float32)
    if pbar:
        iterj = tqdm(range(n), desc="Transforming subset Jacobian")
    else:
        iterj = range(n)
    for i in iterj:
        J = Js[:, :, i]
        ret[:, :, i] = Qi @ J @ Qj.T
    return ret


def average_jacobian_by_group(Js: np.ndarray, group_labels: List[str]) -> Dict[str, np.ndarray]:
    """
    Returns a dictionary of averaged jacobians with group names as the keys.
    No vectorized indexing was used due to its high memory cost.

    Args:
        Js: List of Jacobian matrices
        group_labels: list of group labels

    Returns:
        dictionary with group labels as keys and average Jacobians as values
    """
    groups = np.unique(group_labels)

    J_mean = {}
    N = {}
    for i, g in enumerate(group_labels):
        if g in J_mean.keys():
            J_mean[g] += Js[:, :, i]
            N[g] += 1
        else:
            J_mean[g] = Js[:, :, i]
            N[g] = 1
    for g in groups:
        J_mean[g] /= N[g]
    return J_mean


# ---------------------------------------------------------------------------------------------------
# Hessian


def Hessian_rkhs_gaussian(x: np.ndarray, vf_dict: VecFldDict) -> np.ndarray:
    """analytical Hessian for RKHS vector field functions with Gaussian kernel.

    Args:
        x: Coordinates where the Hessian is evaluated. Note that x has to be 1D.
        vf_dict: A dictionary containing RKHS vector field control points, Gaussian bandwidth,
            and RKHS coefficients.
            Essential keys: 'X_ctrl', 'beta', 'C'

    Returns:
        H: Hessian matrix stored as d-by-d-by-d numpy arrays evaluated at x.
            d is the number of dimensions.
    """
    x = np.atleast_2d(x)

    C = vf_dict["C"]
    beta = vf_dict["beta"]
    K, D = con_K(x, vf_dict["X_ctrl"], beta, return_d=True)

    K = K * C.T

    D = D.T
    D = np.eye(x.shape[1]) - 2 * beta * D @ np.transpose(D, axes=(0, 2, 1))

    H = -2 * beta * np.einsum("ij, jlm -> ilm", K, D)

    return H


def hessian_transformation(H: np.ndarray, qi: np.ndarray, Qj: np.ndarray, Qk: np.ndarray) -> np.ndarray:
    """Inverse transform low dimensional k x k x k Hessian matrix (:math:`\partial^2 F_i / \partial x_j \partial x_k`)
    back to the d-dimensional gene expression space. The formula used to inverse transform Hessian matrix calculated
    from low dimension (PCs) is:
                                            :math:`h = \sum_i\sum_j\sum_k q_i q_j q_k H_ijk`,
    where `q, H, h` are the PCA loading matrix, low dimensional Hessian matrix and the inverse transformed element from
    the high dimensional Hessian matrix.

    Args:
        H: k x k x k matrix of the Hessian.
        qi: The i-th row of the PC loading matrix Q with dimension d x k, corresponding to the effector i.
        Qj: The submatrix of the PC loading matrix Q with dimension d x k, corresponding to regulators j.
        Qk: The submatrix of the PC loading matrix Q with dimension d x k, corresponding to co-regulators k.

    Returns:
        h: The calculated Hessian matrix for the effector i w.r.t regulators j and co-regulators k.
    """

    h = np.einsum("ijk, di -> djk", H, qi)
    Qj, Qk = np.atleast_2d(Qj), np.atleast_2d(Qk)
    h = Qj @ h @ Qk.T

    return h


def elementwise_hessian_transformation(H: np.ndarray, qi: np.ndarray, qj: np.ndarray, qk: np.ndarray) -> np.ndarray:
    """Inverse transform low dimensional k x k x k Hessian matrix (:math:`\partial^2 F_i / \partial x_j \partial x_k`) back to the
    d-dimensional gene expression space. The formula used to inverse transform Hessian matrix calculated from
    low dimension (PCs) is:
                                            :math:`Jac = Q J Q^T`,
    where `Q, J, Jac` are the PCA loading matrix, low dimensional Jacobian matrix and the inverse transformed high
    dimensional Jacobian matrix. This function takes only one row from Q to form qi or qj.

    Args:
        H: k x k x k matrix of the Hessian.
        qi: The i-th row of the PC loading matrix Q with dimension d x k, corresponding to the effector i.
        qj: The j-th row of the PC loading matrix Q with dimension d x k, corresponding to the regulator j.
        qk: The k-th row of the PC loading matrix Q with dimension d x k, corresponding to the co-regulator k.

    Returns:
        h: The calculated Hessian elements for each cell.
    """

    h = np.einsum("ijk, i -> jk", H, qi)
    h = qj @ h @ qk

    return h


# ---------------------------------------------------------------------------------------------------
def Laplacian(H: np.ndarray) -> np.ndarray:
    """
    Computes the Laplacian of the Hessian matrix by summing the diagonal elements of the Hessian matrix (summing the unmixed second partial derivatives)
                                            :math: `\Delta f = \sum_{i=1}^{n} \frac{\partial^2 f}{\partial x_i^2}`
    Args:
        H: Hessian matrix
    """
    # when H has four dimensions, H is calculated across all cells
    if H.ndim == 4:
        L = np.zeros([H.shape[2], H.shape[3]])
        for sample_indx in range(H.shape[3]):
            for out_indx in range(L.shape[0]):
                L[out_indx, sample_indx] = np.diag(H[:, :, out_indx, sample_indx]).sum()
    else:
        # when H has three dimensions, H is calculated only on one single cell
        L = np.zeros([H.shape[2], 1])
        for out_indx in range(L.shape[0]):
            L[out_indx, 0] = np.diag(H[:, :, out_indx]).sum()

    return L


# ---------------------------------------------------------------------------------------------------
# dynamical properties
def _divergence(f: Callable, x: np.ndarray) -> float:
    """Divergence of the reconstructed vector field function f evaluated at x"""
    jac = nd.Jacobian(f)(x)
    return np.trace(jac)


def acceleration_(v: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Calculate acceleration by dotting the Jacobian and the velocity vector.

    Args:
        v: velocity vector
        J: Jacobian matrix

    Returns:
        Acceleration vector, with one element for the acceleration of each component
    """
    if v.ndim == 1:
        v = v[:, None]
    return J.dot(v)


def curvature_method1(a: np.array, v: np.array) -> float:
    """https://link.springer.com/article/10.1007/s12650-018-0474-6"""
    if v.ndim == 1:
        v = v[:, None]
    kappa = np.linalg.norm(np.outer(v, a)) / np.linalg.norm(v) ** 3

    return kappa


def curvature_method2(a: np.array, v: np.array) -> float:
    """https://dl.acm.org/doi/10.5555/319351.319441"""
    # if v.ndim == 1: v = v[:, None]
    kappa = (np.multiply(a, np.dot(v, v)) - np.multiply(v, np.dot(v, a))) / np.linalg.norm(v) ** 4

    return kappa


def torsion_(v, J, a):
    """only works in 3D"""
    if v.ndim == 1:
        v = v[:, None]
    tau = np.outer(v, a).dot(J.dot(a)) / np.linalg.norm(np.outer(v, a)) ** 2

    return tau


def compute_acceleration(vf, f_jac, X, Js=None, return_all=False):
    """Calculate acceleration for many samples via

    .. math::
    a = J \cdot v.

    """
    n = len(X)
    acce = np.zeros(n)
    acce_mat = np.zeros((n, X.shape[1]))

    v_ = vf(X)
    J_ = f_jac(X) if Js is None else Js
    for i in tqdm(range(n)):
        v = v_[i]
        J = J_[:, :, i]
        acce_mat[i] = acceleration_(v, J).flatten()
        acce[i] = np.linalg.norm(acce_mat[i])

    if return_all:
        return v_, J_, acce, acce_mat
    else:
        return acce, acce_mat


def compute_curvature(vf, f_jac, X, Js=None, formula=2):
    """Calculate curvature for many samples via

    Formula 1:
    .. math::
    \kappa = \frac{||\mathbf{v} \times \mathbf{a}||}{||\mathbf{V}||^3}

    Formula 2:
    .. math::
    \kappa = \frac{||\mathbf{Jv} (\mathbf{v} \cdot \mathbf{v}) -  ||\mathbf{v} (\mathbf{v} \cdot \mathbf{Jv})}{||\mathbf{V}||^4}
    """
    n = len(X)

    curv = np.zeros(n)
    v, _, _, a = compute_acceleration(vf, f_jac, X, Js=Js, return_all=True)
    cur_mat = np.zeros((n, X.shape[1])) if formula == 2 else None

    for i in tqdm(range(n)):
        if formula == 1:
            curv[i] = curvature_method1(a[i], v[i])
        elif formula == 2:
            cur_mat[i] = curvature_method2(a[i], v[i])
            curv[i] = np.linalg.norm(cur_mat[i])

    return curv, cur_mat


def compute_torsion(vf, f_jac, X):
    """Calculate torsion for many samples via

    .. math::
    \tau = \frac{(\mathbf{v} \times \mathbf{a}) \cdot (\mathbf{J} \cdot \mathbf{a})}{||\mathbf{V} \times \mathbf{a}||^2}
    """
    if X.shape[1] != 3:
        raise Exception(f"torsion is only defined in 3 dimension.")

    n = len(X)

    tor = np.zeros((n, X.shape[1], X.shape[1]))
    v, J, a_, a = compute_acceleration(vf, f_jac, X, return_all=True)

    for i in tqdm(range(n), desc="Calculating torsion"):
        tor[i] = torsion_(v[i], J[:, :, i], a[i])

    return tor


def compute_sensitivity(f_jac, X):
    """Calculate sensitivity for many samples via

    .. math::
    S = (I - J)^{-1} D(\frac{1}{{I-J}^{-1}})
    """
    J = f_jac(X)

    n_genes, n_genes_, n_cells = J.shape
    S = np.zeros_like(J)

    I = np.eye(n_genes)
    for i in tqdm(
            np.arange(n_cells),
            desc="Calculating sensitivity matrix with precomputed component-wise Jacobians",
    ):
        s = np.linalg.inv(I - J[:, :, i])  # np.transpose(J)
        S[:, :, i] = s.dot(np.diag(1 / np.diag(s)))
        # tmp = np.transpose(J[:, :, i])
        # s = np.linalg.inv(I - tmp)
        # S[:, :, i] = s * (1 / np.diag(s)[None, :])

    return S


def curl3d(f, x, method="analytical", VecFld=None, jac=None):
    """Curl of the reconstructed vector field f evaluated at x in 3D"""
    if jac is None:
        if method == "analytical" and VecFld is not None:
            jac = Jacobian_rkhs_gaussian(x, VecFld)
        else:
            jac = nd.Jacobian(f)(x)

    return np.array([jac[2, 1] - jac[1, 2], jac[0, 2] - jac[2, 0], jac[1, 0] - jac[0, 1]])


def curl2d(f, x, method="analytical", VecFld=None, jac=None):
    """Curl of the reconstructed vector field f evaluated at x in 2D"""
    if jac is None:
        if method == "analytical" and VecFld is not None:
            jac = Jacobian_rkhs_gaussian(x, VecFld)
        else:
            jac = nd.Jacobian(f)(x)

    curl = jac[1, 0] - jac[0, 1]

    return curl


def compute_curl(f_jac, X):
    """Calculate curl for many samples for 2/3 D systems."""
    if X.shape[1] > 3:
        raise Exception(f"curl is only defined in 2/3 dimension.")

    n = len(X)

    if X.shape[1] == 2:
        curl = np.zeros(n)
        f = curl2d
    else:
        curl = np.zeros((n, 3, 3))
        f = curl3d

    for i in tqdm(range(n), desc=f"Calculating {X.shape[1]}-D curl"):
        J = f_jac(X[i])
        curl[i] = f(None, None, method="analytical", VecFld=None, jac=J)

    return curl


def normalize_vectors(vectors, axis=1, **kwargs):
    """Returns the unit vectors of the vectors."""
    vec = np.array(vectors, copy=True)
    vec = np.atleast_2d(vec)
    vec_norm = np.linalg.norm(vec, axis=axis, **kwargs)

    vec_norm[vec_norm == 0] = 1
    vec = (vec.T / vec_norm).T
    return vec

def is_outside(X, domain):
    is_outside = np.zeros(X.shape[0], dtype=bool)
    for k in range(X.shape[1]):
        o = np.logical_or(X[:, k] < domain[k][0], X[:, k] > domain[k][1])
        is_outside = np.logical_or(is_outside, o)
    return is_outside


def index_condensed_matrix(n: int, i: int, j: int) -> int:
    """Return the index of an element in a condensed n-by-n square matrix by the row index i and column index j of the
    square form.

    Args:
        n: Size of the square form.
        i: Row index of the element in the square form.
        j: Column index of the element in the square form.

    Returns:
        The index of the element in the condensed matrix.
    """

    if i == j:
        warnings.warn("Diagonal elements (i=j) are not stored in condensed matrices.")
        return None
    elif i > j:
        i, j = j, i
    return int(i * (n - (i + 3) * 0.5) + j - 1)


def remove_redundant_points(X, tol=1e-4, output_discard=False):
    X = np.atleast_2d(X)
    discard = np.zeros(len(X), dtype=bool)
    if X.shape[0] > 1:
        dist = pdist(X)
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                if dist[index_condensed_matrix(len(X), i, j)] < tol:
                    discard[j] = True
        X = X[~discard]
    if output_discard:
        return X, discard
    else:
        return X


def form_triu_matrix(arr: np.ndarray) -> np.ndarray:
    """
    Construct upper triangle matrix from a 1d array.

    Args:
        arr: The array used to generate the upper triangle matrix.

    Returns:
        The generated upper triangle matrix.
    """
    n = int(np.ceil((np.sqrt(1 + 8 * len(arr)) - 1) * 0.5))
    M = np.zeros((n, n))
    c = 0
    for i in range(n):
        for j in range(n):
            if j >= i:
                if c < len(arr):
                    M[i, j] = arr[c]
                    c += 1
                else:
                    break
    return M


def find_fixed_points(
        x0_list: Union[list, np.ndarray],
        func_vf: Callable,
        domain: Optional[np.ndarray] = None,
        tol_redundant: float = 1e-4,
        return_all: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given sampling points, a function, and a domain, finds points for which func_vf(x) = 0.

    Args:
        x0_list: Array-like structure with sampling points
        func_vf: Function for which to find fixed points
        domain: Finds fixed points within the given domain of shape (n_dim, 2)
        tol_redundant: Margin outside of which points are considered distinct
        return_all: If set to true, always return a tuple of three arrays as output

    Returns:
        A tuple with the solutions, Jacobian matrix, and function values at the solutions.

    """

    def vf_aux(x):
        """auxillary function unifying dimensionality"""
        v = func_vf(x)
        if x.ndim == 1:
            v = v.flatten()
        return v

    X = []
    J = []
    fval = []
    for x0 in x0_list:
        x, info_dict, _, _ = fsolve(vf_aux, x0, full_output=True)

        outside = is_outside(x[None, :], domain)[0] if domain is not None else False
        if not outside:
            fval.append(info_dict["fvec"])
            # compute Jacobian
            Q = info_dict["fjac"]
            R = form_triu_matrix(info_dict["r"])
            J.append(Q.T @ R)
            X.append(x)
        elif return_all:
            X.append(np.zeros_like(x) * np.nan)
            J.append(np.zeros((len(x), len(x))) * np.nan)

    X = np.array(X)
    J = np.array(J)
    fval = np.array(fval)

    if return_all:
        return X, J, fval
    else:
        if X.size != 0:
            if tol_redundant is not None:
                X, discard = remove_redundant_points(X, tol_redundant, output_discard=True)
                J = J[~discard]
                fval = fval[~discard]

            return X, J, fval
        else:
            return None, None, None


