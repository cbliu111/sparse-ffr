import numpy as np
import csv
from scipy.spatial.distance import pdist
from scipy.sparse import csr_matrix, coo_matrix, coo_array
import multiprocessing
from scipy.spatial.distance import squareform
from cajal.qgw import quantized_icdm, quantized_gw_parallel_memory
import itertools as it
from utils.log import logger
from tqdm import tqdm
from multiprocessing import Pool
import gzip
from cajal.run_gw import (
    npt,
    DistanceMatrix,
    Matrix,
    Distribution,
    Optional,
    GW_cell,
    _init_gw_pool,
    Iterator,
    _gw_index,
)
import h5py


def uniform(n: int):
    """Compute the uniform distribution on n points, as a vector of floats."""
    return np.ones((n,), dtype=np.float64) / n


def get_dist(x):
    # the input shape is (2, pts), so transpose
    return pdist(x[0].T, metric=x[1])


def quantized_gw_parallel(
        dist_mat_dist: list[tuple[np.ndarray, np.ndarray]],
        num_processes: int,
        num_clusters: int,
        chunksize: int = 20,
):
    """
    Compute the quantized Gromov-Wasserstein distance in parallel between all cells in a family \
    of cells.

    Read icdms, quantize them, compute pairwise qGW distances between icdms.

    :param dist_mat_dist: distance matrix
    :param num_processes: number of Python processes to run in parallel
    :param num_clusters: Each cell will be partitioned into `num_clusters` many clusters.
    :param chunksize: How many q-GW distances should be computed at a time by each parallel process.
    """
    logger.info("Quantizing intracell distance matrices...")
    with Pool(processes=num_processes) as pool:
        args = [
            (cell_dm, cell_dist, num_clusters, None)
            for cell_dm, cell_dist in dist_mat_dist
        ]
        quantized_cells = list(
            tqdm(pool.imap(quantized_icdm.of_tuple, args), total=len(dist_mat_dist))
        )

    logger.info("Computing pairwise Gromov-Wasserstein distances...")
    gw_dists = quantized_gw_parallel_memory(quantized_cells, num_processes, chunksize)
    return gw_dists


def stringify_coupling_mat(A: npt.NDArray[np.float64]) -> list[str]:
    """Convert a coupling matrix into a string."""
    a = coo_matrix(A)
    return (
            [str(a.nnz)]
            + list(map(str, a.data))
            + list(map(str, a.row))
            + list(map(str, a.col))
    )


def csv_output_writer(
        indices: list[int],
        gw_coupling_csv: str,
        results_iterator: Iterator[tuple[int, int, Matrix, float]],
) -> Iterator[tuple[int, int, Matrix, float]]:
    file = open(gw_coupling_csv, "w", newline="")
    writer = csv.writer(file)
    writer.writerow(
        [
            "first_object",
            "first_object_sidelength",
            "second_object",
            "second_object_sidelength",
            "num_nonzero",
            "data",
            "row_indices",
            "col_indices",
        ]
    )
    for i, j, coupling_mat, gw_dist in results_iterator:
        writer.writerow(
            [indices[i], str(coupling_mat.shape[0]), indices[j], str(coupling_mat.shape[1])]
            + stringify_coupling_mat(coupling_mat)
        )
        yield i, j, coupling_mat, gw_dist
    file.close()


def gw_pairwise_parallel(
        dist_mat_dist: list[
            tuple[
                DistanceMatrix,  # Squareform distance matrix
                Distribution,  # Probability distribution on cells
            ]
        ],
        num_processes: int,
        gw_csv: str,
):
    """Compute the pairwise Gromov-Wasserstein distances between cells.

    Optionally one can also compute their coupling matrices.
    If appropriate file names are supplied, the output is also written to file.
    If computing a large number of coupling matrices, for reduced memory consumption it
    is suggested not to return the coupling matrices, and instead write them to file.

    :param dist_mat_dist: A list of pairs (A,a) where `A` is a squareform intracell
        distance matrix and `a` is a probability distribution on the points of
        `A`.
    :param num_processes: How many Python processes to run in parallel for the computation.

    :param gw_csv: Path to gw h5 file, saving couplings to `gw_h5file`.,
        and `couplings` is a h5py file, coupling_mat between cell i and cell j
         can be accessed using the address f"/coupling/{i}_{j}"
        and `coupling_mat` is a coupling matrix between the two cells.

    :return: returns `gw_dmat`,
        where gw_dmat is a square matrix whose (i,j) entry is the GW distance
        between two cells
    """
    GW_cells = []
    for A, a in dist_mat_dist:
        GW_cells.append(GW_cell(A, a))
    num_cells = len(dist_mat_dist)
    gw_dmat = np.zeros((num_cells, num_cells))
    NN = len(GW_cells)
    total_num_pairs = int((NN * (NN - 1)) / 2)
    ij = tqdm(it.combinations(range(num_cells), 2), total=total_num_pairs)
    with Pool(
            initializer=_init_gw_pool, initargs=(GW_cells,), processes=num_processes
    ) as pool:
        gw_data: Iterator[tuple[int, int, Matrix, float]]
        gw_data = pool.imap_unordered(_gw_index, ij, chunksize=20)
        gw_data = csv_output_writer(
            list(range(num_cells)),
            gw_csv,
            gw_data,
        )
        for i, j, coupling_mat, gw_dist in gw_data:
            gw_dmat[i, j] = gw_dist
            gw_dmat[j, i] = gw_dist
    return gw_dmat


def read_gw_couplings(
        gw_couplings_file: str, header: bool
) -> dict[tuple[int, int], coo_array]:
    """
    Read a list of Gromov-Wasserstein coupling matrices into memory.
    :param header: If True, the first line of the file will be ignored.
    :param gw_couplings_file: name of a file holding a list of GW coupling matrices in \
    COO form. The files should be in csv format. Each line should be of the form
    `cellA_name, cellA_sidelength, cellB_name, cellB_sidelength, num_nonzero, (data), (row), (col)`
    where `data` is a sequence of `num_nonzero` many floating point real numbers,
    `row` is a sequence of `num_nonzero` many integers (row indices), and
    `col` is a sequence of `num_nonzero` many integers (column indices).
    :return: A dictionary mapping pairs of names (firstcell, secondcell) to the GW \
    matrix of the coupling. `firstcell` and `secondcell` are in alphabetical order.
    """

    gw_coupling_mat_dict: dict[tuple[int, int], coo_array] = {}
    with open(gw_couplings_file, "r", newline="") as gw_file:
        csvreader = csv.reader(gw_file, delimiter=",")
        linenum = 1
        if header:
            _ = next(csvreader)
            linenum += 1
        for line in csvreader:
            cellA_name = line[0]
            cellA_sidelength = int(line[1])
            cellB_name = line[2]
            cellB_sidelength = int(line[3])
            num_non_zero = int(line[4])
            rest = line[5:]
            if 3 * num_non_zero != len(rest):
                raise Exception(
                    "On line " + str(linenum) + " data not in COO matrix form."
                )
            data = [float(x) for x in rest[:num_non_zero]]
            rows = [int(x) for x in rest[num_non_zero: (2 * num_non_zero)]]
            cols = [int(x) for x in rest[(2 * num_non_zero):]]
            coo = coo_array(
                (data, (rows, cols)), shape=(cellA_sidelength, cellB_sidelength)
            )
            linenum += 1
            if cellA_name < cellB_name:
                gw_coupling_mat_dict[(int(cellA_name), int(cellB_name))] = coo
            else:
                gw_coupling_mat_dict[(int(cellA_name), int(cellB_name))] = coo_array.transpose(
                    coo
                )
    return gw_coupling_mat_dict


def compute_gw_with_coupling(adata, gw_file, coupling_csv, cpus: int = 8, metric: str = 'euclidean'):
    _m = adata.to_df().to_numpy()
    pts = int(adata.n_vars / 2)
    _c = []
    for i in range(adata.n_obs):
        __c = np.array([_m[i][:pts], _m[i][pts:]])
        _c.append((__c, metric))

    with multiprocessing.Pool(processes=cpus) as pool:
        _dist = pool.map(get_dist, _c, chunksize=100)

    dist_mat_dist = [(c := squareform(d), uniform(c.shape[0])) for d in _dist]

    with h5py.File(gw_file, 'a') as f:
        if "/pdist" in f:
            del f["/pdist"]
        f.create_dataset("/pdist", data=_dist)
    del _dist
    gw_dist = gw_pairwise_parallel(dist_mat_dist, cpus, coupling_csv)

    with h5py.File(gw_file, 'a') as f:
        if "/gw_dist" in f:
            del f[f"/gw_dist"]
        f.create_dataset("/gw_dist", data=gw_dist)


def compute_gw_distances(adata, cpus: int = 8, method='gw', metric: str = 'euclidean'):
    logger.info("Computing intra-pair distances...")
    _m = adata.to_df().to_numpy()
    pts = int(adata.n_vars / 2)
    # _m = np.hstack([_m[:, :pts], _m[:, pts:]])
    # _m = _m.reshape(adata.n_obs, int(adata.n_vars / 2), 2)
    # _m = [(_m[i], metric) for i in range(adata.n_obs)]
    _c = []
    for i in range(adata.n_obs):
        __c = np.array([_m[i][:pts], _m[i][pts:]])
        _c.append((__c, metric))

    with multiprocessing.Pool(processes=cpus) as pool:
        _dist = pool.map(get_dist, _c, chunksize=100)

    logger.info("Computing gw distances...")
    dist_mat_dist = [(c := squareform(d), uniform(c.shape[0])) for d in _dist]
    gw_dist = []
    if method == 'gw':
        gw_coupling_mats: list[tuple[int, int, np.ndarray]]  # int, int, Matrix
        gw_dist, gw_coupling_mats = gw_pairwise_parallel(dist_mat_dist, num_processes=cpus)
        gw_dist = np.array(gw_dist)
    elif method == 'qgw':
        tup: list[tuple[int, int, float]]
        tup = quantized_gw_parallel(dist_mat_dist, num_processes=cpus, num_clusters=25)

        # extract cell names and pair-wise distances
        gw_dist_dict: dict[tuple[str, str], float] = {}
        for c in tup:
            first_cell, second_cell, gw_dist_str = c
            gw_dist = float(gw_dist_str)
            first_cell, second_cell = sorted([first_cell, second_cell])
            gw_dist_dict[(first_cell, second_cell)] = gw_dist
        all_cells_set = set()
        for cell_1, cell_2 in gw_dist_dict:
            all_cells_set.add(cell_1)
            all_cells_set.add(cell_2)
        all_cells = sorted(list(all_cells_set))
        if all_cells is None:
            names = set()
            for key in gw_dist_dict:
                names.add(key[0])
                names.add(key[1])
            all_cells = sorted(names)

        # transform to square matrix
        dist_list: list[float] = []
        for first_cell, second_cell in it.combinations(all_cells, 2):
            first_cell, second_cell = sorted([first_cell, second_cell])
            dist_list.append(gw_dist_dict[(first_cell, second_cell)])
        arr = np.array(dist_list, dtype=np.float_)
        gw_dist = squareform(arr, force="tomatrix")

    adata.obsp['metric'] = gw_dist
