import os
import numpy as np
from scipy.sparse import issparse
import pandas as pd
import anndata
import warnings


def compute_step_transport_matrix(solver, **params):
    """
    Compute the optimal transport with stabilized numerics.
    Args:
    G: Growth (absolute)
    solver: transport_stablev2 or optimal_transport_duality_gap
    growth_iters:
  """

    import gc
    G = params['G']
    growth_iters = params['growth_iters']
    learned_growth = []
    tmap = None
    for i in range(growth_iters):
        if i == 0:
            row_sums = G
        else:
            row_sums = tmap.sum(axis=1)  # / tmap.shape[1]
        params['G'] = row_sums
        learned_growth.append(row_sums)
        tmap = solver(**params)
        gc.collect()

    if tmap is None:
        warnings.warn('No transport matrix was computed.')

    return tmap, learned_growth


# @ Lénaïc Chizat 2015 - optimal transport
def fdiv(l, x, p, dx):
    return l * np.sum(dx * (x * (np.log(x / p)) - x + p))


def fdivstar(l, u, p, dx):
    return l * np.sum((p * dx) * (np.exp(u / l) - 1))


def primal(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2):
    I = len(p)
    J = len(q)
    F1 = lambda x, y: fdiv(lambda1, x, p, y)
    F2 = lambda x, y: fdiv(lambda2, x, q, y)
    with (np.errstate(divide='ignore')):
        return F1(np.dot(R, dy), dx) + F2(np.dot(R.T, dx), dy) + (
                    epsilon * np.sum(R * np.nan_to_num(np.log(R)) - R + K) + np.sum(R * C)) / (I * J)


def dual(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2):
    I = len(p)
    J = len(q)
    F1c = lambda u, v: fdivstar(lambda1, u, p, v)
    F2c = lambda u, v: fdivstar(lambda2, u, q, v)
    return - F1c(- epsilon * np.log(a), dx) - F2c(- epsilon * np.log(b), dy) \
        - epsilon * np.sum(R - K) / (I * J)


# end @ Lénaïc Chizat

def optimal_transport_duality_gap(C, G, lambda1, lambda2, epsilon, batch_size, tolerance, tau, epsilon0, max_iter,
                                  **ignored):
    """
    Compute the optimal transport with stabilized numerics, with the guarantee that the duality gap is at most `tolerance`

    Parameters
    ----------
    C : 2-D ndarray
        The cost matrix. C[i][j] is the cost to transport cell i to cell j
    G : 1-D array_like
        Growth value for input cells.
    lambda1 : float, optional
        Regularization parameter for the marginal constraint on p
    lambda2 : float, optional
        Regularization parameter for the marginal constraint on q
    epsilon : float, optional
        Entropy regularization parameter.
    batch_size : int, optional
        Number of iterations to perform between each duality gap check
    tolerance : float, optional
        Upper bound on the duality gap that the resulting transport map must guarantee.
    tau : float, optional
        Threshold at which to perform numerical stabilization
    epsilon0 : float, optional
        Starting value for exponentially-decreasing epsilon
    max_iter : int, optional
        Maximum number of iterations. Print a warning and return if it is reached, even without convergence.

    Returns
    -------
    transport_map : 2-D ndarray
        The entropy-regularized unbalanced transport map
    """
    C = np.asarray(C, dtype=np.float64)
    epsilon_scalings = 5
    scale_factor = np.exp(- np.log(epsilon) / epsilon_scalings)

    I, J = C.shape
    dx, dy = np.ones(I) / I, np.ones(J) / J

    p = G
    q = np.ones(C.shape[1]) * np.average(G)

    u, v = np.zeros(I), np.zeros(J)
    a, b = np.ones(I), np.ones(J)

    epsilon_i = epsilon0 * scale_factor
    current_iter = 0

    duality_gap = np.nan
    R = None

    for e in range(epsilon_scalings + 1):
        duality_gap = np.inf
        u = u + epsilon_i * np.log(a)
        v = v + epsilon_i * np.log(b)  # absorb
        epsilon_i = epsilon_i / scale_factor
        _K = np.exp(-C / epsilon_i)
        alpha1 = lambda1 / (lambda1 + epsilon_i)
        alpha2 = lambda2 / (lambda2 + epsilon_i)
        K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
        a, b = np.ones(I), np.ones(J)
        old_a, old_b = a, b
        threshold = tolerance if e == epsilon_scalings else 1e-6

        while duality_gap > threshold:
            for i in range(batch_size if e == epsilon_scalings else 5):
                current_iter += 1
                old_a, old_b = a, b
                a = (p / (K.dot(np.multiply(b, dy)))) ** alpha1 * np.exp(-u / (lambda1 + epsilon_i))
                b = (q / (K.T.dot(np.multiply(a, dx)))) ** alpha2 * np.exp(-v / (lambda2 + epsilon_i))

                # stabilization
                if (max(max(abs(a)), max(abs(b))) > tau):
                    u = u + epsilon_i * np.log(a)
                    v = v + epsilon_i * np.log(b)  # absorb
                    K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
                    a, b = np.ones(I), np.ones(J)

                if current_iter >= max_iter:
                    warnings.warn("Reached max_iter with duality gap still above threshold. Returning")
                    return (K.T * a).T * b

            # The real dual variables. a and b are only the stabilized variables
            _a = a * np.exp(u / epsilon_i)
            _b = b * np.exp(v / epsilon_i)

            # Skip duality gap computation for the first epsilon scalings, use dual variables evolution instead
            if e == epsilon_scalings:
                R = (K.T * a).T * b
                pri = primal(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2)
                dua = dual(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2)
                duality_gap = (pri - dua) / abs(pri)
            else:
                duality_gap = max(
                    np.linalg.norm(_a - old_a * np.exp(u / epsilon_i)) / (1 + np.linalg.norm(_a)),
                    np.linalg.norm(_b - old_b * np.exp(v / epsilon_i)) / (1 + np.linalg.norm(_b)))

    if np.isnan(duality_gap):
        raise RuntimeError("Overflow encountered in duality gap computation, please report this incident")
    if R is None:
        warnings.warn("No transport map was computed.")
        return None

    return R / C.shape[1]


def transport_stablev2(C, lambda1, lambda2, epsilon, scaling_iter, G, tau, epsilon0, extra_iter, inner_iter_max,
                       **ignored):
    """
    Compute the optimal transport with stabilized numerics.
    Args:

        C: cost matrix to transport cell i to cell j
        lambda1: regularization parameter for marginal constraint for p.
        lambda2: regularization parameter for marginal constraint for q.
        epsilon: entropy parameter
        scaling_iter: number of scaling iterations
        G: growth value for input cells
    """

    warm_start = tau is not None
    epsilon_final = epsilon

    def get_reg(n):  # exponential decreasing
        return (epsilon0 - epsilon_final) * np.exp(-n) + epsilon_final

    epsilon_i = epsilon0 if warm_start else epsilon
    dx = np.ones(C.shape[0]) / C.shape[0]
    dy = np.ones(C.shape[1]) / C.shape[1]

    p = G
    q = np.ones(C.shape[1]) * np.average(G)

    u = np.zeros(len(p))
    v = np.zeros(len(q))
    b = np.ones(len(q))
    K = np.exp(-C / epsilon_i)

    alpha1 = lambda1 / (lambda1 + epsilon_i)
    alpha2 = lambda2 / (lambda2 + epsilon_i)
    epsilon_index = 0
    iterations_since_epsilon_adjusted = 0

    a = None

    for i in range(scaling_iter):
        # scaling iteration
        a = (p / (K.dot(np.multiply(b, dy)))) ** alpha1 * np.exp(-u / (lambda1 + epsilon_i))
        b = (q / (K.T.dot(np.multiply(a, dx)))) ** alpha2 * np.exp(-v / (lambda2 + epsilon_i))

        # stabilization
        iterations_since_epsilon_adjusted += 1
        if (max(max(abs(a)), max(abs(b))) > tau):
            u = u + epsilon_i * np.log(a)
            v = v + epsilon_i * np.log(b)  # absorb
            K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
            a = np.ones(len(p))
            b = np.ones(len(q))

        if (warm_start and iterations_since_epsilon_adjusted == inner_iter_max):
            epsilon_index += 1
            iterations_since_epsilon_adjusted = 0
            u = u + epsilon_i * np.log(a)
            v = v + epsilon_i * np.log(b)  # absorb
            epsilon_i = get_reg(epsilon_index)
            alpha1 = lambda1 / (lambda1 + epsilon_i)
            alpha2 = lambda2 / (lambda2 + epsilon_i)
            K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
            a = np.ones(len(p))
            b = np.ones(len(q))

    for i in range(extra_iter):
        a = (p / (K.dot(np.multiply(b, dy)))) ** alpha1 * np.exp(-u / (lambda1 + epsilon_i))
        b = (q / (K.T.dot(np.multiply(a, dx)))) ** alpha2 * np.exp(-v / (lambda2 + epsilon_i))

    if a is None:
        warnings.warn("No transport map was computed.")
        return None

    R = (K.T * a).T * b

    return R / C.shape[1]


def compute_transport_maps(
        adata,
        time_field: str = 'frame',
        metric_kw: str = 'distances',
        solver: str = 'duality_gap',
        growth_rate_field: np.ndarray = None,
        tmap_out='tmaps',
        overwrite=True,
        output_file_format='h5ad',
        **kwargs,
):
    tmap_dir, tmap_prefix = os.path.split(tmap_out) if tmap_out is not None else (None, None)
    tmap_prefix = tmap_prefix or "tmaps"
    tmap_dir = tmap_dir or '.'
    if not os.path.exists(tmap_dir):
        os.makedirs(tmap_dir)

    time_points = adata.obs[time_field].unique()  # sorted time sequence
    n_t = time_points.size
    t_pairs = [(time_points[i], time_points[i + 1]) for i in range(n_t - 1)]

    metric_matrix = adata.obsp[metric_kw]
    if issparse(metric_matrix):
        metric_matrix = metric_matrix.toarray()

    ot_config = {'local_pca': 30, 'growth_iters': 1, 'epsilon': 0.05, 'lambda1': 1, 'lambda2': 50,
                 'epsilon0': 1, 'tau': 10000, 'scaling_iter': 3000, 'inner_iter_max': 50, 'tolerance': 1e-8,
                 'max_iter': 1e7, 'batch_size': 5, 'extra_iter': 1000}

    for k in kwargs.keys():
        ot_config[k] = kwargs[k]

    if solver == 'fixed_iters':
        solver = transport_stablev2
    elif solver == 'duality_gap':
        solver = optimal_transport_duality_gap
    else:
        raise ValueError('Unknown solver')
    full_learned_growth_df = None
    save_learned_growth = ot_config.get('growth_iters', 1) > 1
    for t_pair in t_pairs:
        t0, t1 = t_pair

        path = tmap_prefix
        path += "_{}_{}".format(*t_pair)
        output_file = os.path.join(tmap_dir, path)
        if os.path.exists(output_file) and not overwrite:
            warnings.warn('Found existing tmap at ' + output_file + '. ')
            continue

        config = {**ot_config, 't0': t0, 't1': t1}

        # Computes the transport map from time t0 to time t1
        # t0: source time point
        # t1: destination time point

        p0_indices = adata.obs[time_field] == float(t0)
        p1_indices = adata.obs[time_field] == float(t1)
        i0 = np.where(p0_indices.to_numpy())[0]
        i1 = np.where(p1_indices.to_numpy())[0]
        name0 = adata[p0_indices, :].obs.index
        name1 = adata[p1_indices, :].obs.index

        # fetch the cost as the distance metric
        C = metric_matrix[i0, :][:, i1]
        config['C'] = C / (np.median(C)+1e-12)
        delta_t = t1 - t0

        if growth_rate_field is not None:
            config['G'] = np.power(growth_rate_field, delta_t)
        else:
            config['G'] = np.ones(C.shape[0])

        tmap, learned_growth = compute_step_transport_matrix(solver=solver, **config)
        learned_growth.append(tmap.sum(axis=1))
        obs_growth = {}
        for i in range(len(learned_growth)):
            g = learned_growth[i]
            g = np.power(g, 1.0 / delta_t)
            obs_growth['g' + str(i)] = g
        learned_growth_df = pd.DataFrame(index=name0, data=obs_growth)
        tmap = anndata.AnnData(tmap, obs=learned_growth_df, var=pd.DataFrame(index=name1))

        write_dataset(tmap, output_file, output_format=output_file_format)
        if save_learned_growth:
            full_learned_growth_df = learned_growth_df if full_learned_growth_df is None else pd.concat(
                (full_learned_growth_df, learned_growth_df), copy=False)
    if full_learned_growth_df is not None:
        full_learned_growth_df.to_csv(os.path.join(tmap_dir, tmap_prefix + '_g.txt'), sep='\t', index_label='id')


def write_dataset(adata, path, output_format='txt'):
    path = str(path)
    if not path.lower().endswith('.' + output_format):
        path += '.' + output_format
    if output_format == 'txt':
        x = adata.X.toarray() if issparse(adata.X) else adata.X
        pd.DataFrame(x, index=adata.obs.index, columns=adata.var.index).to_csv(path, index_label='id', sep='\t',
                                                                               doublequote=False)
    elif output_format == 'h5ad':
        adata.write(path)
    elif output_format == 'loom':
        adata.write_loom(adata, path)
    else:
        raise ValueError('Unknown file format')
