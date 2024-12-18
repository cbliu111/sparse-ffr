import numpy as np
from scipy.linalg.blas import dgemm
from scipy.linalg import lstsq
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union
# from pp.neighbor import k_nearest_neighbors
from scipy.spatial.distance import pdist, squareform
from pp.neighbor import fast_knn_from_precomputed
from anndata import AnnData
from tqdm import tqdm
from tl.field_tools import (
    angle,
    merge_dict,
    norm,
    denorm,
    VecFldDict,
    con_K,
    con_K_div_cur_free,
    Jacobian_rkhs_gaussian,
    Jacobian_rkhs_gaussian_parallel,
    Jacobian_numerical,
    Hessian_rkhs_gaussian,
    Laplacian,
    compute_curl,
    compute_torsion,
    compute_curvature,
    compute_sensitivity,
    compute_acceleration,
)


def sample_by_velocity(
        V: np.ndarray,
        n: int,
        seed: int = 0
) -> np.ndarray:
    """Sample method based on velocity.

    Args:
        V:
            Velocity associated with each element in the sample array.
        n:
            The number of samples.
        seed:
            The randomization seed. Defaults to 0.

    Returns:
        The sample data array.
    """
    np.random.seed(seed)
    tmp_V = np.linalg.norm(V, axis=1)
    p = tmp_V / np.sum(tmp_V)
    idx = np.random.choice(np.arange(len(V)), size=n, p=p, replace=False)
    return idx


def bandwidth_selector(X: np.ndarray) -> float:
    """
    This function computes an empirical bandwidth for a Gaussian kernel.
    """
    n, m = X.shape

    # _, distances = k_nearest_neighbors(
    #     X,
    #     k=max(2, int(0.2 * n)) - 1,
    #     exclude_self=False,
    #     pynn_rand_state=0,
    # )

    distances = squareform(pdist(X, 'euclidean'))

    n_neighbors = max(2, int(0.2 * n)) - 1
    knn_indices, distances = fast_knn_from_precomputed(distances, n_neighbors)

    d = np.mean(distances[:, 1:]) / 1.5
    return np.sqrt(2) * d


def get_P(
        Y: np.ndarray, V: np.ndarray, sigma2: float, gamma: float, a: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GET_P estimates the posterior probability and part of the energy.

    Args:
        Y: Velocities from the data.
        V: The estimated velocity: V=f(X), f being the vector field function.
        sigma2: sigma2 is defined as sum(sum((Y - V)**2)) / (N * D)
        gamma: Percentage of inliers in the samples. This is an inital value for EM iteration, and it is not important.
        a: Parameter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of
            outlier's variation space is a.

    Returns:
        Tuple of (posterior probability, energy) related to equations 27 and 26 in the SparseVFC paper.

    """

    D = Y.shape[1]
    temp1 = np.exp(-np.sum((Y - V) ** 2, 1) / (2 * sigma2))
    temp2 = (2 * np.pi * sigma2) ** (D / 2) * (1 - gamma) / (gamma * a)
    temp1[temp1 == 0] = np.min(temp1[temp1 != 0])
    P = temp1 / (temp1 + temp2)
    E = P.T.dot(np.sum((Y - V) ** 2, 1)) / (2 * sigma2) + np.sum(P) * np.log(sigma2) * D / 2

    return (P[:, None], E) if P.ndim == 1 else (P, E)


def linear_least_squares(
        a, b, residuals: bool = False
):
    """Return the least-squares solution to a linear matrix equation.

    Solves the equation `a x = b` by computing a vector `x` that minimizes the Euclidean 2-norm `|| b - a x ||^2`.
    The equation may be under-, well-, or over- determined (i.e., the number of linearly independent rows of `a` can be
    less than, equal to, or greater than its number of linearly independent columns).  If `a` is square and of full
    rank, then `x` (but for round-off error) is the "exact" solution of the equation.

    Args:
        a: The coefficient matrix.
        b: The ordinate or "dependent variable" values.
        residuals: Whether to compute the residuals associated with the least-squares solution. Defaults to False.

    Returns:
        The least-squares solution. If `residuals` is True, the sum of residuals (squared Euclidean 2-norm for each
        column in ``b - a*x``) would also be returned.
    """

    if type(a) != np.ndarray or not a.flags["C_CONTIGUOUS"]:
        warnings.warn(
            "Matrix a is not a C-contiguous numpy array. The solver will create a copy, which will result"
            + " in increased memory usage."
        )

    a = np.asarray(a, order="c")
    i = dgemm(alpha=1.0, a=a.T, b=a.T, trans_b=True)
    x = np.linalg.solve(i, dgemm(alpha=1.0, a=a.T, b=b))

    if residuals:
        return x, np.linalg.norm(np.dot(a, x) - b)
    else:
        return x


def lstsq_solver(lhs, rhs, method="drouin"):
    if method == "scipy":
        C = lstsq(lhs, rhs)[0]
    elif method == "drouin":
        C = linear_least_squares(lhs, rhs)
    else:
        warnings.warn("Invalid linear least squares solver. Use Drouin's method instead.")
        C = linear_least_squares(lhs, rhs)
    return C


def SparseVFC(
        X: np.ndarray,
        Y: np.ndarray,
        Grid: np.ndarray,
        M: int = 100,
        a: float = 5,
        beta: float = None,
        ecr: float = 1e-5,
        gamma: float = 0.9,
        lambda_: float = 3,
        minP: float = 1e-5,
        MaxIter: int = 500,
        theta: float = 0.75,
        velocity_based_sampling: bool = True,
        seed: int | np.ndarray = 0,
        lstsq_method: str = "drouin",
) -> VecFldDict:
    """Apply sparseVFC (vector field consensus) algorithm to learn a functional form of the vector field from random
    samples with outlier on the entire space robustly and efficiently. (Ma, Jiayi, etc. al, Pattern Recognition, 2013)

    Args:

        X: Current state. This corresponds to, for example, the spliced transcriptomic state.
        Y: Velocity estimates in delta t. This corresponds to, for example, the inferred spliced transcriptomic
            velocity or total RNA velocity based on metabolic labeling data estimated calculated by dynamo.
        Grid: Current state on a grid which is often used to visualize the vector field. This corresponds to, for example,
            the spliced transcriptomic state or total RNA state.
        M: The number of basis functions to approximate the vector field.
        a: Parameter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of
            outlier's variation space is `a`.
        beta: Parameter of Gaussian Kernel, k(x, y) = exp(-beta*||x-y||^2).
            If None, a rule-of-thumb bandwidth will be computed automatically.
        ecr: The minimum limitation of energy change rate in the iteration process.
        gamma: Percentage of inliers in the samples. This is an initial value for EM iteration, and it is not important.
        lambda_: Represents the trade-off between the goodness of data fit and regularization. Larger Lambda_ put more
            weights on regularization.
        minP: The posterior probability Matrix P may be singular for matrix inversion. We set the minimum value of P as
            minP.
        MaxIter: Maximum iteration times.
        theta: Define how could be an inlier. If the posterior probability of a sample is an inlier is larger than theta,
            then it is regarded as an inlier.
        seed: int or 1-d array_like, optional (default: `0`)
            Seed for RandomState. Must be convertible to 32 bit unsigned integers. Used in sampling control points.
            Default is to be 0 for ensure consistency between different runs.
        velocity_based_sampling:
        lstsq_method: The name of the linear least square solver, can be either 'scipy` or `douin`.

    Returns:
        A dictionary which contains:
            X: Current state.
            valid_ind: The indices of cells that have finite velocity values.
            X_ctrl: Sample control points of current state.
            ctrl_idx: Indices for the sampled control points.
            Y: Velocity estimates in delta t.
            beta: Parameter of the Gaussian Kernel for the kernel matrix (Gram matrix).
            V: Prediction of velocity of X.
            C: Finite set of the coefficients for the
            P: Posterior probability Matrix of inliers.
            VFCIndex: Indexes of inliers found by sparseVFC.
            sigma2: Energy change rate.
            grid: Grid of current state.
            grid_V: Prediction of velocity of the grid.
            iteration: Number of the last iteration.
            tecr_traj: Vector of relative energy changes rate comparing to previous step.
            E_traj: Vector of energy at each iteration,
        where V = f(X), P is the posterior probability and VFCIndex is the indexes of inliers found by sparseVFC.
        Note that V = `con_K(Grid, X_ctrl, beta).dot(C)` gives the prediction of velocity on Grid (but can also be any
        point in the gene expression state space).

    """
    print("[SparseVFC] begins...")

    X_ori, Y_ori = X.copy(), Y.copy()
    valid_ind = np.where(np.isfinite(Y.sum(1)))[0]
    X, Y = X[valid_ind], Y[valid_ind]
    N, D = Y.shape
    grid_U = None

    # Construct kernel matrix K
    tmp_X, uid = np.unique(X, axis=0, return_index=True)  # return unique rows
    M = min(M, tmp_X.shape[0])
    if velocity_based_sampling:
        print("Sampling control points based on data velocity magnitude...")
        idx = sample_by_velocity(Y[uid], M, seed=seed)
    else:
        idx = np.random.RandomState(seed=seed).permutation(tmp_X.shape[0])  # rand select some initial points
        idx = idx[range(M)]
    ctrl_pts = tmp_X[idx, :]

    if beta is None:
        h = bandwidth_selector(ctrl_pts)
        beta = 1 / h ** 2

    K = con_K(ctrl_pts, ctrl_pts, beta)
    U = con_K(X, ctrl_pts, beta)
    if Grid is not None:
        grid_U = con_K(Grid, ctrl_pts, beta)
    M = ctrl_pts.shape[0]

    # Initialization
    V = np.zeros((N, D))
    # Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
    # V = np.ones((N, D)) * np.mean(Y, axis=0)
    # V = Y + np.random.normal(0, 1, size=(N, D))
    C = np.zeros((M, D))
    # C = np.linalg.pinv(U).dot(V)
    i, tecr, E = 0, 1, 1
    # test this
    sigma2 = sum(sum((Y - V) ** 2)) / (N * D)
    sigma2 = 1e-7 if sigma2 < 1e-8 else sigma2
    tecr_vec = np.ones(MaxIter) * np.nan
    E_vec = np.ones(MaxIter) * np.nan
    P = None
    while i < MaxIter and tecr > ecr and sigma2 > 1e-8:
        # E_step
        E_old = E
        P, E = get_P(Y, V, sigma2, gamma, a)

        E = E + lambda_ / 2 * np.trace(C.T.dot(K).dot(C))
        E_vec[i] = E
        tecr = abs((E - E_old) / E)
        tecr_vec[i] = tecr

        # logger.report_progress(count=i, total=MaxIter, progress_name="E-step iteration")
        print(
            "iterate: %d, gamma: %.3f, energy: %.3f, energy change rate: %.3f, sigma2: %.3f"
            % (i, gamma, E, tecr, sigma2)
        )

        # M-step. Solve linear system for C.
        P = np.maximum(P, minP)
        # UP = U.T * numpy.matlib.repmat(P.T, M, 1)
        UP = U.T * np.tile(P.T, (M, 1))
        # lhs = UP.dot(U) + lambda_ * sigma2 * K  # in the original paper, there is no sigma2 term
        lhs = UP.dot(U) + lambda_ * K
        rhs = UP.dot(Y)

        C = lstsq_solver(lhs, rhs, method=lstsq_method)

        # Update V and sigma**2
        V = U.dot(C)
        Sp = sum(P)
        sigma2 = (sum(P.T.dot(np.sum((Y - V) ** 2, 1))) / np.dot(Sp, D))[0]

        # Update gamma
        numcorr = len(np.where(P > theta)[0])
        gamma = numcorr / X.shape[0]

        if gamma > 0.95:
            gamma = 0.95
        elif gamma < 0.05:
            gamma = 0.05

        i += 1
    if i == 0 and not (tecr > ecr and sigma2 > 1e-8):
        raise Exception(
            "please check your input parameters, "
            f"tecr: {tecr}, ecr {ecr} and sigma2 {sigma2},"
            f"tecr must larger than ecr and sigma2 must larger than 1e-8"
        )

    grid_V = None
    if Grid is not None:
        grid_V = np.dot(grid_U, C)

    VecFld = {
        "X": X_ori,
        "valid_ind": valid_ind,
        "X_ctrl": ctrl_pts,
        "ctrl_idx": idx,
        "Y": Y_ori,
        "beta": beta,
        "V": V,
        "C": C,
        "P": P,
        "VFCIndex": np.where(P > theta)[0],
        "sigma2": sigma2,
        "grid": Grid,
        "grid_V": grid_V,
        "iteration": i - 1,
        "tecr_traj": tecr_vec[:i],
        "E_traj": E_vec[:i],
    }

    print("SparseVFC finished.")
    return VecFld


class VectorField:
    def __init__(
            self,
            adata: AnnData,
            *,
            copy: bool = False,
            coord_basis: str = "X_umap",
            velo_basis: str = "X_velo",
            dims: int | list[int] | None = None,
            grid: bool = False,
            grid_num: int = 50,
            **kwargs,
    ):
        """
        VectorField class

            X: (dimension: n_obs x n_features), Original data.
            V: (dimension: n_obs x n_features), Velocities of cells in the same order and dimension of X.
            Grid: The function that returns diffusion matrix which can be dependent on the variables (for example, genes)
            M: `int` (default: None)
                The number of basis functions to approximate the vector field. By default it is calculated as
                `min(len(X), int(1500 * np.log(len(X)) / (np.log(len(X)) + np.log(100))))`. So that any datasets with less
                than  about 900 data points (cells) will use full data for vector field reconstruction while any dataset
                larger than that will at most use 1500 data points.
            a: `float` (default 5)
                Parameter of the model of outliers. We assume the outliers obey uniform distribution, and the volume of
                outlier's variation space is a.
            beta: `float` (default: None)
                Parameter of Gaussian Kernel, k(x, y) = exp(-beta*||x-y||^2).
                If None, a rule-of-thumb bandwidth will be computed automatically.
            ecr: `float` (default: 1e-5)
                The minimum limitation of energy change rate in the iteration process.
            gamma: `float` (default:  0.9)
                Percentage of inliers in the samples. This is an inital value for EM iteration, and it is not important.
                Default value is 0.9.
            lambda_: `float` (default: 3)
                Represents the trade-off between the goodness of data fit and regularization.
            minP: `float` (default: 1e-5)
                The posterior probability Matrix P may be singular for matrix inversion. We set the minimum value of P as
                minP.
            MaxIter: `int` (default: 500)
                Maximum iteration times.
            theta: `float` (default 0.75)
                Define how could be an inlier. If the posterior probability of a sample is an inlier is larger than theta,
                then it is regarded as an inlier.
            div_cur_free_kernels: `bool` (default: False)
                A logic flag to determine whether the divergence-free or curl-free kernels will be used for learning the
                vector field.
            sigma: `int`
                Bandwidth parameter.
            eta: `int`
                Combination coefficient for the divergence-free or the curl-free kernels.
            seed : int or 1-d array_like, optional (default: `0`)
                Seed for RandomState. Must be convertible to 32 bit unsigned integers. Used in sampling control points.
                Default is to be 0 for ensure consistency between different runs.
        """

        self.adata = adata.copy() if copy else adata
        X = adata.obsm[coord_basis]
        V = adata.obsm[velo_basis]

        if np.isscalar(dims):
            X, V = X[:, :dims], V[:, :dims]
        elif type(dims) is list:
            X, V = X[:, dims], V[:, dims]

        Grid = None
        if X.shape[1] < 4 or grid:
            # smart way for generating high dimensional grids and convert into a row matrix
            min_vec, max_vec = (
                X.min(0),
                X.max(0),
            )
            min_vec = min_vec - 0.01 * np.abs(max_vec - min_vec)
            max_vec = max_vec + 0.01 * np.abs(max_vec - min_vec)

            Grid_list = np.meshgrid(*[np.linspace(i, j, grid_num) for i, j in zip(min_vec, max_vec)])
            Grid = np.array([i.flatten() for i in Grid_list]).T

        self.data = {
            "X": X,
            "V": V,
            "Grid": Grid,
            "dims": dims
        }
        self.vf_dict = kwargs.pop("vf_dict", {})
        self.func = kwargs.pop("func", None)
        self.fixed_points = kwargs.pop("fixed_points", None)

        if X is not None and V is not None:
            self.parameters = kwargs
            self.parameters = merge_dict(
                self.parameters,
                {
                    "M": kwargs.pop("M", None) or # max(min([50, len(X)]), int(0.05 * len(X)) + 1),
                    min(len(X), int(1500 * np.log(len(X)) / (np.log(len(X)) + np.log(100)))),
                    "a": kwargs.pop("a", 5),
                    "beta": kwargs.pop("beta", None),
                    "ecr": kwargs.pop("ecr", 1e-5),
                    "gamma": kwargs.pop("gamma", 0.9),
                    "lambda_": kwargs.pop("lambda_", 3),
                    "minP": kwargs.pop("minP", 1e-5),
                    "MaxIter": kwargs.pop("MaxIter", 500),
                    "theta": kwargs.pop("theta", 0.75),
                    # "div_cur_free_kernels": kwargs.pop("div_cur_free_kernels", False),
                    "velocity_based_sampling": kwargs.pop("velocity_based_sampling", True),
                    # "sigma": kwargs.pop("sigma", 0.8),
                    # "eta": kwargs.pop("eta", 0.5),  # for divergence-free or the curl-free kernels
                    "seed": kwargs.pop("seed", 0),
                },
            )

        self.norm_dict = {}

    def one_epoch(self, normalize: bool = False, **kwargs):
        lstsq_method = kwargs.pop("lstsq_method", "drouin")  # scipy
        if normalize:
            X_norm, V_norm, T_norm, norm_dict = norm(self.data["X"], self.data["V"], self.data["Grid"])
            (self.data["X"], self.data["V"], self.data["Grid"], self.norm_dict,) = (
                X_norm,
                V_norm,
                T_norm,
                norm_dict,
            )
        else:
            X_norm, Y_norm, V_norm = None, None, None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            VecFld = SparseVFC(
                self.data["X"],
                self.data["V"],
                self.data["Grid"],
                **self.parameters,
                lstsq_method=lstsq_method,
            )
        if normalize:
            VecFld = denorm(VecFld, X_norm, V_norm, self.norm_dict)

        self.parameters = merge_dict(self.parameters, VecFld, update=True)

        self.func = lambda x: self.vector_field_function(x, VecFld)
        VecFld: dict
        VecFld["V"] = self.func(self.data["X"])
        VecFld["normalize"] = normalize

        return VecFld

    def train(
            self,
            normalize: bool = False,
            restart_num: int = 5,
            restart_seed: tuple[int] | None = (0, 100, 200, 300, 400),
            min_vel_corr: float = 0.6,
            **kwargs
    ):
        """
        Learn an function of vector field from sparse single cell samples in the entire space robustly.
        Reference: Regularized vector field learning with sparse approximation for mismatch removal, Ma, Jiayi, etc. al,
        Pattern Recognition

        Args:
            normalize: Logic flag to determine whether to normalize the data to have zero means and unit covariance. This is
                often required for raw dataset (for example, raw UMI counts and RNA velocity values in high dimension).
                But it is normally not required for low dimensional embeddings by PCA or other non-linear dimension
                reduction methods.
            restart_num: The number of retrials for vector field reconstructions.
            restart_seed: A list of seeds for each retrial. Must be the same length as `restart_num` or None.
            min_vel_corr: The minimal threshold for the cosine correlation between input velocities and learned velocities to consider as a successful
                vector field reconstruction procedure. If the cosine correlation is less than this threshold and restart_num > 1,
                `restart_num` trials will be attempted with different seeds to reconstruct the vector field function.
                This can avoid some reconstructions to be trapped in some local optimal.

        Returns:
            A dictionary which contains X, Y, beta, V, C, P, VFCIndex. Where V = f(X), P is the posterior
            probability and VFCIndex is the indexes of inliers which found by VFC.
        """

        self.parameters = merge_dict(self.parameters, kwargs, update=True)
        if restart_num > 0:
            if len(restart_seed) != restart_num:
                warnings.warn(
                    f"the length of {restart_seed} is different from {restart_num}, " f"using `np.range(restart_num) * 100"
                )
                restart_seed = np.arange(restart_num) * 100
            restart_counter, cur_vf_list, res_list = 0, [], []
            while True:
                kwargs.update({"seed": restart_seed[restart_counter]})
                cur_vf_dict = self.one_epoch(normalize, **kwargs)

                # consider refactor with .simulation.evaluation.py
                reference, prediction = (
                    cur_vf_dict["Y"][cur_vf_dict["valid_ind"]],
                    cur_vf_dict["V"][cur_vf_dict["valid_ind"]],
                )
                true_normalized = reference / (np.linalg.norm(reference, axis=1).reshape(-1, 1) + 1e-20)
                predict_normalized = prediction / (np.linalg.norm(prediction, axis=1).reshape(-1, 1) + 1e-20)
                res = np.mean(true_normalized * predict_normalized) * prediction.shape[1]

                cur_vf_list += [cur_vf_dict]
                res_list += [res]
                if res < min_vel_corr:
                    restart_counter += 1
                    warnings.warn(
                        f"current cosine correlation between input velocities and learned velocities is less than "
                        f"{min_vel_corr}. Make a {restart_counter}-th vector field reconstruction trial."
                    )
                else:
                    vf_dict = cur_vf_dict
                    break

                if restart_counter > restart_num - 1:
                    warnings.warn(
                        f"Cosine correlation between input velocities and learned velocities is less than"
                        f" {min_vel_corr} after {restart_num} trials of vector field reconstruction."
                    )
                    vf_dict = cur_vf_list[np.argmax(np.array(res_list))]
                    break
        else:
            vf_dict = self.one_epoch(normalize=normalize, **kwargs)

        vf_key = "vf_dict"
        vf_dict: dict
        vf_dict["method"] = 'SparseVFC'
        vf_dict["dims"] = self.data['dims']
        key = "velocity_umap_sparseVFC"
        X_copy_key = "X_umap_sparseVFC"

        self.adata.obsm[key] = vf_dict["V"]
        self.adata.obsm[X_copy_key] = vf_dict["X"]
        self.adata.uns[vf_key] = vf_dict

        control_point, inlier_prob, valid_ids = (
            "vf_control_point",
            "vf_inlier_prob",
            vf_dict["valid_ind"],
        )

        self.adata.obs[control_point], self.adata.obs[inlier_prob] = False, np.nan
        self.adata.obs.loc[self.adata.obs_names[vf_dict["ctrl_idx"]], control_point] = True
        self.adata.obs.loc[self.adata.obs_names[valid_ids], inlier_prob] = vf_dict["P"].flatten()

        # angles between observed velocity and that predicted by vector field across cells:
        cell_angles = np.zeros(self.adata.n_obs, dtype=float)
        for i, u, v in zip(valid_ids, self.data['V'][valid_ids], vf_dict["V"]):
            # fix the u, v norm == 0 in angle function
            cell_angles[i] = angle(u.astype("float64"), v.astype("float64"))

        self.adata.obs["obs_vf_angle"] = cell_angles

    @staticmethod
    def vector_field_function(
            x: np.ndarray,
            vf_dict: Dict,
            dim: Optional[Union[int, np.ndarray]] = None,
            kernel: str = "full",
            X_ctrl_ind: Optional[List] = None,
            **kernel_kwargs,
    ) -> np.ndarray:
        """vector field function constructed by sparseVFC.
        Reference: Regularized vector field learning with sparse approximation for mismatch removal, Ma, Jiayi, etc. al, Pattern Recognition

        Args:
            x: Set of cell expression state samples
            vf_dict: VecFldDict with stored parameters necessary for reconstruction
            dim: Index or indices of dimensions of the K gram matrix to return. Defaults to None.
            kernel: one of {"full", "df_kernel", "cf_kernel"}. Defaults to "full".
            X_ctrl_ind: Indices of control points at which kernels will be centered. Defaults to None.

        Raises:
            ValueError: If the kernel value specified is not one of "full", "df_kernel", or "cf_kernel"

        Returns:
            np.ndarray storing the `dim` dimensions of m x m gram matrix K storing the kernel evaluated at each pair of control points
        """
        if "div_cur_free_kernels" in vf_dict.keys():
            has_div_cur_free_kernels = True
        else:
            has_div_cur_free_kernels = False

        x = np.array(x)
        if x.ndim == 1:
            x = x[None, :]

        if has_div_cur_free_kernels:
            if kernel == "full":
                kernel_ind = 0
            elif kernel == "df_kernel":
                kernel_ind = 1
            elif kernel == "cf_kernel":
                kernel_ind = 2
            else:
                raise ValueError(f"the kernel can only be one of {'full', 'df_kernel', 'cf_kernel'}!")

            K = con_K_div_cur_free(
                x,
                vf_dict["X_ctrl"],
                vf_dict["sigma"],
                vf_dict["eta"],
            )[kernel_ind]
        else:
            Xc = vf_dict["X_ctrl"]
            K = con_K(x, Xc, vf_dict["beta"], **kernel_kwargs)

        if X_ctrl_ind is not None:
            C = np.zeros_like(vf_dict["C"])
            C[X_ctrl_ind, :] = vf_dict["C"][X_ctrl_ind, :]
        else:
            C = vf_dict["C"]

        K = K.dot(C)

        if dim is not None and not has_div_cur_free_kernels:
            if np.isscalar(dim):
                K = K[:, :dim]
            elif dim is not None:
                K = K[:, dim]

        return K

    def get_Jacobian(
            self,
            method: str = "analytical",
            input_vector_convention: str = "row",
            **kwargs: object
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get the Jacobian of the vector field function.
        If method is 'analytical':
        The analytical Jacobian will be returned and it always
        take row vectors as input no matter what input_vector_convention is.

        If method is 'numerical':
        If the input_vector_convention is 'row', it means that fjac takes row vectors
        as input, otherwise the input should be an array of column vectors. Note that
        the returned Jacobian would behave exactly the same if the input is an 1d array.

        The column vector convention is slightly faster than the row vector convention.
        So the matrix of row vector convention is converted into column vector convention
        under the hood.

        No matter the method and input vector convention, the returned Jacobian is of the
        following format:
                df_1/dx_1   df_1/dx_2   df_1/dx_3   ...
                df_2/dx_1   df_2/dx_2   df_2/dx_3   ...
                df_3/dx_1   df_3/dx_2   df_3/dx_3   ...
                ...         ...         ...         ...
        """
        if method == "numerical":
            return Jacobian_numerical(self.func, input_vector_convention)
        elif method == "parallel":
            return lambda x: Jacobian_rkhs_gaussian_parallel(x, self.vf_dict, **kwargs)
        elif method == "analytical":
            return lambda x: Jacobian_rkhs_gaussian(x, self.vf_dict, **kwargs)
        else:
            raise NotImplementedError(
                f"The method {method} is not implemented. Currently only "
                f"supports 'analytical', 'numerical', and 'parallel'."
            )

    def get_Hessian(self, method: str = "analytical") -> Callable:
        """
        Get the Hessian of the vector field function.
        If method is 'analytical':
        The analytical Hessian will be returned and it always
        take row vectors as input no matter what input_vector_convention is.

        No matter the method and input vector convention, the returned Hessian is of the
        following format:
                df^2/dx_1^2        df_1^2/(dx_1 dx_2)   df_1^2/(dx_1 dx_3)   ...
                df^2/(dx_2 dx_1)   df^2/dx_2^2          df^2/(dx_2 dx_3)     ...
                df^2/(dx_3 dx_1)   df^2/(dx_3 dx_2)     df^2/dx_3^2          ...
                ...                ...                  ...                  ...
        """
        if method == "analytical":
            return lambda x: Hessian_rkhs_gaussian(x, self.vf_dict)
        elif method == "numerical":
            if self.func is not None:
                raise Exception("numerical Hessian for vector field is not defined.")
            else:
                raise Exception("The perturbed vector field function has not been set up.")
        else:
            raise NotImplementedError(f"The method {method} is not implemented. Currently only supports 'analytical'.")

    def get_Laplacian(self, method: str = "analytical") -> Callable:
        """
        Get the Laplacian of the vector field. Laplacian is defined as the sum of the diagonal of the Hessian matrix.
        Because Hessian is originally defined for scalar function and here we extend it to vector functions. We will
        calculate the summation of the diagonal of each output (target) dimension.

        A Laplacian filter is an edge detector used to compute the second derivatives of an image, measuring the rate
        at which the first derivatives change (so it is the derivative of the Jacobian). This determines if a change in
        adjacent pixel values is from an edge or continuous progression.
        """
        if method == "analytical":
            return lambda x: Laplacian(H=x)
        elif method == "numerical":
            if self.func is not None:
                raise Exception("Numerical Laplacian for vector field is not defined.")
            else:
                raise Exception("The perturbed vector field function has not been set up.")
        else:
            raise NotImplementedError(f"The method {method} is not implemented. Currently only supports 'analytical'.")

    def evaluate(self, CorrectIndex: list, VFCIndex: list, siz: int) -> tuple[float, float, float]:
        """Evaluate the precision, recall, corrRate of the sparseVFC algorithm.

        Args:
            CorrectIndex: Ground truth indexes of the correct vector field samples.
            VFCIndex: Indexes of the correct vector field samples learned by VFC.
            siz: Number of initial matches.

        Returns:
            A tuple of precision, recall, corrRate, where Precision, recall, corrRate are Precision and recall of VFC, percentage of initial correct matches, respectively.

        See also:: :func:`sparseVFC`.
        """

        if len(VFCIndex) == 0:
            VFCIndex = range(siz)

        VFCCorrect = np.intersect1d(VFCIndex, CorrectIndex)
        NumCorrectIndex = len(CorrectIndex)
        NumVFCIndex = len(VFCIndex)
        NumVFCCorrect = len(VFCCorrect)

        corrRate = NumCorrectIndex / siz
        precision = NumVFCCorrect / NumVFCIndex
        recall = NumVFCCorrect / NumCorrectIndex

        print("correct correspondence rate in the original data: %d/%d = %f" % (NumCorrectIndex, siz, corrRate))
        print("precision rate: %d/%d = %f" % (NumVFCCorrect, NumVFCIndex, precision))
        print("recall rate: %d/%d = %f" % (NumVFCCorrect, NumCorrectIndex, recall))

        return corrRate, precision, recall

    def compute_divergence(
            self,
            X: np.ndarray = None,
            Js: np.ndarray = None,
            vectorize_size: int = 1000,
            method: str = "analytical",
            **kwargs
    ) -> np.ndarray:
        """Takes the trace of the jacobian matrix to calculate the divergence.
        Calculate divergence for many samples by taking the trace of a Jacobian matrix.

        vectorize_size is used to control the number of samples computed in each vectorized batch.
            If vectorize_size = 1, there's no vectorization whatsoever.
            If vectorize_size = None, all samples are vectorized.

        Args:
            X: Current state. Defaults to None, initialized from self.data
            method: Method for calculating the Jacobian, one of numerical, analytical, parallel

        Returns:
            The divergence of the Jacobian matrix as divergence np.ndarray across Jacobians for many samples
        """
        """

        Args:
            f_jac: function for calculating Jacobian from cell states
            X: cell states
            Js: Jacobian matrices for each sample, if X is not provided
            vectorize_size: number of Jacobian matrices to process at once in the vectorization

        Returns:

        """

        X = self.data["X"] if X is None else X

        f_jac = self.get_Jacobian(method=method)

        n = len(X)
        if vectorize_size is None:
            vectorize_size = n

        div = np.zeros(n)
        for i in tqdm(range(0, n, vectorize_size), desc="Calculating divergence"):
            J = f_jac(X[i: i + vectorize_size]) if Js is None else Js[:, :, i: i + vectorize_size]
            div[i: i + vectorize_size] = np.trace(J)

        return div

    def compute_curl(
            self,
            X: np.ndarray = None,
            method: str = "analytical",
            dim1: int = 0,
            dim2: int = 1,
            dim3: int = 2,
            **kwargs,
    ) -> np.ndarray:
        """Curl computation for many samples for 2/3 D systems.

        Args:
            X: Current state. Defaults to None, initialized from self.data
            method: Method for calculating the Jacobian, one of numerical, analytical, parallel
            dim1: index of first dimension
            dim2: index of second dimension
            dim3: index of third dimension

        Returns:
            np.ndarray storing curl
        """
        X = self.data["X"] if X is None else X
        if dim3 is None or X.shape[1] < 3:
            X = X[:, [dim1, dim2]]
        else:
            X = X[:, [dim1, dim2, dim3]]
        f_jac = self.get_Jacobian(method=method, **kwargs)
        return compute_curl(f_jac, X)

    def compute_acceleration(self, X: np.ndarray = None, method: str = "analytical", **kwargs) -> np.ndarray:
        """Calculate acceleration for many samples via

        .. math::
        a = || J \cdot v ||.

        Args:
            X: Current state. Defaults to None, initialized from self.data
            method: Method for calculating the Jacobian, one of numerical, analytical, parallel

        Returns:
            np.ndarray storing the vector norm of acceleration (across all genes) for each cell
        """
        X = self.data["X"] if X is None else X
        f_jac = self.get_Jacobian(method=method)
        return compute_acceleration(self.func, f_jac, X, **kwargs)

    def compute_curvature(
            self, X: np.ndarray = None, method: str = "analytical", formula: int = 2, **kwargs
    ) -> np.ndarray:
        """Calculate curvature for many samples via

        Formula 1:
        .. math::
        \kappa = \frac{||\mathbf{v} \times \mathbf{a}||}{||\mathbf{V}||^3}

        Formula 2:
        .. math::
        \kappa = \frac{||\mathbf{Jv} (\mathbf{v} \cdot \mathbf{v}) -  ||\mathbf{v} (\mathbf{v} \cdot \mathbf{Jv})}{||\mathbf{V}||^4}

        Args:
            X: Current state. Defaults to None, initialized from self.data
            method: Method for calculating the Jacobian, one of numerical, analytical, parallel
            formula: Choose between formulas 1 and 2 to compute the curvature. Defaults to 2.

        Returns:
            np.ndarray storing the vector norm of curvature (across all genes) for each cell
        """
        X = self.data["X"] if X is None else X
        f_jac = self.get_Jacobian(method=method)
        return compute_curvature(self.func, f_jac, X, formula=formula, **kwargs)

    def compute_torsion(self, X: np.ndarray = None, method: str = "analytical") -> np.ndarray:
        """Calculate torsion for many samples via

        .. math::
        \tau = \frac{(\mathbf{v} \times \mathbf{a}) \cdot (\mathbf{J} \cdot \mathbf{a})}{||\mathbf{V} \times \mathbf{a}||^2}

        Args:
            X: Current state. Defaults to None, initialized from self.data
            method: Method for calculating the Jacobian, one of numerical, analytical, parallel

        Returns:
            np.ndarray storing torsion for each sample
        """
        X = self.data["X"] if X is None else X
        f_jac = self.get_Jacobian(method=method)
        return compute_torsion(self.func, f_jac, X)

    def compute_sensitivity(self, X: np.ndarray = None, method: str = "analytical") -> np.ndarray:
        """Calculate sensitivity for many samples via

        .. math::
        S = (I - J)^{-1} D(\frac{1}{{I-J}^{-1}})

        Args:
            X: Current state. Defaults to None, initialized from self.data
            method: Method for calculating the Jacobian, one of numerical, analytical, parallel Defaults to "analytical".

        Returns:
            np.ndarray storing sensitivity matrix
        """
        X = self.data["X"] if X is None else X
        f_jac = self.get_Jacobian(method=method)
        return compute_sensitivity(f_jac, X)
