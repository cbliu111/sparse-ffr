import time
import numpy as np
from joblib import Parallel, delayed
from typing import Callable, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt


def mfpt_parallel(
        VecFnc: Callable,
        init_point: np.ndarray,
        fix_point: np.ndarray,
        domain_range: Tuple[float, float] = None,
        max_time_steps: int = int(1e6),
        x_lim: Tuple[float] = (-4, 15),
        y_lim: Tuple[float] = (-1, 12),
        n_paths: int = 400,
        diff_coeff: float = 0.00001,
        dt: float = 1,
        cpus: int = 20,
):
    """
    Compute mean first passage time between init point and
     fixed point of the Langevin dynamics using the provided vector field function: VecFnc
    Only for 2D field.

    Args:
        init_points: initial points
        x_lim:
        y_lim:
        dim:
        n_paths:
        n_time_steps:
        start_time: start time for recording the trajectory density
        Tra_grid:
        diff_coeff:
        dt:
        VecFnc:
        cpus:

    Returns:

    """
    if domain_range is None:
        basin_dist = np.sqrt(np.sum((fix_point - init_point) ** 2))
        domain_range = 0.5 * basin_dist
    def sim(i):
        D = diff_coeff
        init_xy = init_point
        x0 = init_xy[0]
        y0 = init_xy[1]

        # Initialize "path" variables
        x_p = x0
        y_p = y0
        # Evaluate potential (Integrate) over trajectory from init cond to  stable steady state
        for n_steps in np.arange(1, max_time_steps):
            # update dxdt, dydt
            dxdt, dydt = VecFnc([x_p, y_p])

            # update x, y
            dx = dxdt * dt + np.sqrt(2 * D) * np.sqrt(dt) * np.random.randn()
            dy = dydt * dt + np.sqrt(2 * D) * np.sqrt(dt) * np.random.randn()

            x_p = x_p + dx
            y_p = y_p + dy

            if x_p < x_lim[0]:
                x_p = 2 * x_lim[0] - x_p
            if y_p < x_lim[0]:
                y_p = 2 * y_lim[0] - y_p

            if x_p > x_lim[1]:
                x_p = 2 * x_lim[1] - x_p
            if y_p > y_lim[1]:
                y_p = 2 * y_lim[1] - y_p

            if np.sqrt(np.sum((fix_point - np.stack([x_p, y_p])) ** 2)) < domain_range:
                print(n_steps)
                return n_steps

        return None

    start = time.time()
    results = Parallel(n_jobs=cpus)(
        delayed(sim)(i)
        for i in tqdm(range(n_paths))
    )
    results = [r for r in results if r is not None]
    if len(results) == 0:
        raise "No passage found."

    print(f"simulation costs: {(time.time() - start) / 60} min")

    return np.mean(results)
