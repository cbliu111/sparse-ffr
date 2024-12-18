import time

import numpy as np
from joblib import Parallel, delayed
from typing import Callable, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt


def LHSample(D, bounds, N):
    """
    :param D: parameter number
    :param bounds:（list）
    :param N: LH
    :return: sample data
    """

    result = np.empty([N, D])
    temp = np.empty([N])
    d = 1.0 / N

    for i in range(D):
        for j in range(N):
            temp[j] = np.random.uniform(low=j * d, high=(j + 1) * d, size=1)[0]

        np.random.shuffle(temp)
        for j in range(N):
            result[j, i] = temp[j]

    b = np.array(bounds)
    lower_bounds = b[:, 0]
    upper_bounds = b[:, 1]
    if np.any(lower_bounds > upper_bounds):
        print('Error boundary.')
        return None

    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(result, (upper_bounds - lower_bounds), out=result), lower_bounds, out=result)
    return result


def sim_parallel(
        VecFnc: Callable,
        init_points: np.ndarray = None,
        x_lim: Tuple[float] = (-4, 15),
        y_lim: Tuple[float] = (-1, 12),
        dim: int = 2,
        n_paths: int = 400,
        n_time_steps: int = 5000000,
        start_time: float = 1000000,
        Tra_grid: int = 100,
        diff_coeff: float = 0.00001,
        dt: float = 5e-1,
        cpus: int = 20,
):
    """
    Simulate Langevin dynamics using the provided vector field function: VecFnc
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
    bounds = [x_lim, y_lim]
    if init_points is None:
        init_points = LHSample(dim, bounds, n_paths)

    def sim(i):
        D = diff_coeff

        x_path = []
        y_path = []
        num_tra = np.zeros((Tra_grid, Tra_grid))
        total_Fx = np.zeros((Tra_grid, Tra_grid))
        total_Fy = np.zeros((Tra_grid, Tra_grid))

        init_xy = init_points[i, :]
        x0 = init_xy[0]
        y0 = init_xy[1]

        # Initialize "path" variables
        x_p = x0
        y_p = y0
        dxdt, dydt = VecFnc([x_p, y_p])
        # Evaluate potential (Integrate) over trajectory from init cond to  stable steady state
        for n_steps in np.arange(1, n_time_steps):
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

            # update dxdt, dydt
            dxdt, dydt = VecFnc([x_p, y_p])

            if n_steps % 100 == 0:
                x_path.append(x_p)
                y_path.append(y_p)

            if n_steps > start_time:
                # find the location of the trajectory
                A = int((x_p - x_lim[0]) * Tra_grid / (x_lim[1] - x_lim[0]))
                B = int((y_p - y_lim[0]) * Tra_grid / (y_lim[1] - y_lim[0]))
                if A < Tra_grid and B < Tra_grid:
                    # if inside the range, record the point
                    num_tra[A, B] = num_tra[A, B] + 1
                    total_Fx[A, B] = total_Fx[A, B] + dxdt
                    total_Fy[A, B] = total_Fy[A, B] + dydt

        return num_tra, total_Fx, total_Fy, np.stack([x_path, y_path]).T

    start = time.time()
    results = Parallel(n_jobs=cpus)(
        delayed(sim)(i)
        for i in tqdm(range(n_paths))
    )

    num_tra = np.zeros((Tra_grid, Tra_grid))
    total_Fx = np.zeros((Tra_grid, Tra_grid))
    total_Fy = np.zeros((Tra_grid, Tra_grid))
    path = np.array([]).reshape(0, 2)

    for result in results:
        num_tra_i, total_Fx_i, total_Fy_i, path_i = result
        num_tra = num_tra + num_tra_i
        total_Fx = total_Fx + total_Fx_i
        total_Fy = total_Fy + total_Fy_i
        path = np.append(path, path_i, axis=0)

    print(f"simulation costs: {(time.time() - start) / 60} min")

    return num_tra, total_Fx, total_Fy, path
