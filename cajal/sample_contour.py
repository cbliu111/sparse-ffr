import os
import warnings
from typing import List, Iterator, Tuple
import numpy as np
import numpy.typing as npt
from skimage import measure
import tifffile
from scipy.spatial.distance import pdist
import itertools as it
from pathos.pools import ProcessPool
from .utilities import write_csv_block
import scipy

def resample_contour(contour, pts):
    bd = np.array(contour, dtype=np.float64)
    x = bd.T[0]
    y = bd.T[1]
    if x[-1] != x[0] and y[-1] != y[0]:
        x = np.append(x, x[0])
        y = np.append(y, y[0])
    sd = np.sqrt(np.power(x[1:] - x[0:-1], 2) + np.power(y[1:] - y[0:-1], 2))
    sd = np.append([1], sd)
    sid = np.cumsum(sd)
    ss = np.linspace(1, sid[-1], pts + 1)
    ss = ss[0:-1]
    splinerone = scipy.interpolate.splrep(sid, x, s=0)
    sx = scipy.interpolate.splev(ss, splinerone, der=0)
    splinertwo = scipy.interpolate.splrep(sid, y, s=0)
    sy = scipy.interpolate.splev(ss, splinertwo, der=0)
    contour_points = np.append([sy], [sx], axis=0)
    return contour_points.T

def cell_boundary(
    contour: npt.NDArray[np.float64],
    n_sample: int,
) -> List[Tuple[int, npt.NDArray[np.float64]]]:

    boundary_pts: npt.NDArray[np.float64]

    boundary_pts = resample_contour(contour, n_sample)
    return boundary_pts


def _compute_intracell_all(
    list_of_contours: list[npt.NDArray[np.float64]],
    labels: list[tuple],
    n_sample: int,
    pool: ProcessPool,
) -> Iterator[Tuple[str, npt.NDArray[np.float64]]]:

    cell_names = [f"frame {frame} fov {fov} label {label}" for frame, label, fov in labels]

    # compute_cell_boundaries: Callable[[str], List[Tuple[int, npt.NDArray[np.float64]]]]
    def compute_cell_boundaries(contour):
        return cell_boundary(
            contour,
            n_sample,
        )

    # cell_names_repeat: Iterator[Iterator[str]]
    # cell_names_repeat = map(it.repeat, cell_names)
    cell_bdary_lists: Iterator[
        Tuple[Iterator[str], Iterator[npt.NDArray[np.float64]]]
    ]
    cell_bdary_lists = zip(
        cell_names, pool.imap(compute_cell_boundaries, list_of_contours, chunksize=100)
    )
    # cell_bdary_list_iters: Iterator[
    #     Iterator[Tuple[str, npt.NDArray[np.float64]]]
    # ]
    # cell_bdary_list_iters = map(lambda tup: zip(tup[0], tup[1]), cell_bdary_lists)
    # cell_bdary_list_flattened: Iterator[Tuple[str, Tuple[int, npt.NDArray[np.float64]]]]
    # cell_bdary_list_flattened = it.chain.from_iterable(cell_bdary_list_iters)

    def restructure_and_get_pdist(
        tup: tuple[str, tuple[int, npt.NDArray[np.float64]]]
    ) -> tuple[str, npt.NDArray[np.float64]]:
        name = tup[0]
        pd = pdist(tup[1])
        return name, pd

    return pool.imap(
        restructure_and_get_pdist, cell_bdary_lists, chunksize=100
    )


def compute_icdm_all(
    list_of_contours: npt.NDArray[np.float64],
    list_of_labels: list[tuple],
    out_csv: str,
    n_sample: int,
    num_processes: int = 8,
) -> None:

    pool = ProcessPool(nodes=num_processes)
    name_dist_mat_pairs = _compute_intracell_all(
        list_of_contours, list_of_labels, n_sample, pool
    )
    batch_size: int = 100
    write_csv_block(out_csv, n_sample, name_dist_mat_pairs, batch_size)
    pool.close()
    pool.join()
    pool.clear()
    return None
