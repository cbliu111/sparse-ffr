import anndata
import pickle
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import multiprocessing
from pp.metric import compute_gw_with_coupling, compute_gw_distances
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix, coo_matrix, coo_array
from utils.log import logger
from tqdm import tqdm
import h5py
import csv
from sklearn.manifold import MDS
import PIL.Image
import PIL.ImageDraw
import skimage.filters
from pp.metric import read_gw_couplings
from pathlib import Path
import scienceplots

plt.style.use(['science', 'nature'])


def identify_medoid(indices, dist_mat):
    """
    Identify the medoid cell in cell_names.
    """
    xi = np.argmin(dist_mat[np.ix_(indices, indices)].sum(axis=0))
    return indices[xi]


def cap(a, c):
    """
    Return a copy of `a` where values above `c` in `a` are replaced with `c`.
    """
    a1 = np.copy(a)
    a1[a1 >= c] = c
    return a1


def step_size(dist_mat) -> float:
    """
    Heuristic to estimate the step size a neuron was sampled at.
    """
    return np.min(dist_mat)


def generate_seek_pos(coupling_csv, header=True):
    seek_dict = {}
    file = open(coupling_csv, 'r')
    pos = 0
    n_line = 0
    if header:
        n_line += 1
        _ = next(file)
        pos += len(_) + 1
    for line in tqdm(file):
        n_line += 1
        a = line.split(',')
        seek_dict[(int(a[0]), int(a[2]))] = pos
        pos += len(line) + 1
    return seek_dict


def get_coupling_mat(coupling_csv, seek_dict, xi, xj):
    file = open(coupling_csv, 'r')
    if xj < xi:
        pos = seek_dict[(xj, xi)]
    else:
        pos = seek_dict[(xi, xj)]
    file.seek(pos)
    line = next(file).split(",")
    cellA_name = int(line[0])
    cellB_name = int(line[2])
    cellA_sidelength = int(line[1])
    cellB_sidelength = int(line[3])
    num_non_zero = int(line[4])
    rest = line[5:]
    if 3 * num_non_zero != len(rest):
        raise Exception(" data not in COO matrix form.")
    data = [float(x) for x in rest[:num_non_zero]]
    rows = [int(x) for x in rest[num_non_zero: (2 * num_non_zero)]]
    cols = [int(x) for x in rest[(2 * num_non_zero):]]
    coo = coo_array(
        (data, (rows, cols)), shape=(cellA_sidelength, cellB_sidelength)
    )
    if cellA_name < cellB_name:
        return coo
    else:
        return coo_array.transpose(coo)


def orient(
        medoid: int,
        obj_name: int,
        iodm,
        coupling_csv,
        seek_dict,
):
    """
    :param medoid: String naming the medoid object, its key in iodm
    :param obj_name: String naming the object to be compared to
    :param iodm: intra-object distance matrix given in square form
    :param gw_coupling_mat_dict: maps pairs (objA_name, objB_name) to scipy COO matrices
    :return: "oriented" squareform distance matrix
    """
    if obj_name < medoid:
        coupling_mat = get_coupling_mat(coupling_csv, seek_dict, obj_name, medoid)
    else:
        coupling_mat = coo_matrix.transpose(get_coupling_mat(coupling_csv, seek_dict, obj_name, medoid))

    i_reorder = np.argmax(coupling_mat.toarray(), axis=0).reshape(-1)
    return iodm[i_reorder][:, i_reorder]


def knn_graph(dmat, nn: int):
    """
    :param dmat: square form distance matrix
    :param nn: (nearest neighbors) - in the returned graph, nodes v and w will be \
    connected if v is one of the `nn` nearest neighbors of w, or conversely.
    :return: A (1,0)-valued adjacency matrix for a nearest neighbors graph, same shape as dmat.
    """
    a = np.argpartition(dmat, nn + 1, axis=0)
    sidelength = dmat.shape[0]
    graph = np.zeros((sidelength, sidelength), dtype=np.int_)
    for i in range(graph.shape[1]):
        graph[a[0: (nn + 1), i], i] = 1
    graph = np.maximum(graph, graph.T)
    np.fill_diagonal(graph, 0)
    return graph


def avg_shape(
        obj_names: list[int],
        gw_dist_mat: np.ndarray,
        iodms: dict[int, np.ndarray],
        coupling_csv,
        seek_dict,
):
    """
    Compute capped and uncapped average distance matrices. \
    In both cases the distance matrix is rescaled so that the minimal distance between two points \
    is 1. The "capped" distance matrix has a max distance of 2.

    :param obj_names: Keys for the gw_dist_dict and iodms.
    :param gw_dist_mat: distance matrix mapping ordered pairs (cellA_name, cellB_name) \
    to Gromov-Wasserstein distances.
    :param iodms: (intra-object distance matrices) - \
    Maps object names to intra-object distance matrices. Matrices are assumed to be given \
    in vector form rather than squareform.
    :param gw_coupling_mat_dict: Dictionary mapping ordered pairs (cellA_name, cellB_name) to \
    Gromov-Wasserstein coupling matrices from cellA to cellB.
    """
    num_objects = len(obj_names)
    medoid = identify_medoid(obj_names, gw_dist_mat)
    medoid_matrix = iodms[medoid]
    # Rescale to unit step size.
    ss = step_size(medoid_matrix)
    assert ss > 0
    medoid_matrix = medoid_matrix / step_size(medoid_matrix)
    dmat_accumulator_uncapped = np.copy(medoid_matrix)
    dmat_accumulator_capped = cap(medoid_matrix, 2.0)
    others = [obj for obj in obj_names if obj != medoid]
    for oi in tqdm(range(len(others))):
        obj_name = others[oi]
        iodm = iodms[obj_name]
        # Rescale to unit step size.
        iodm = iodm / step_size(iodms)
        reoriented_iodm = squareform(
            orient(
                medoid,
                obj_name,
                squareform(iodm, force="tomatrix"),
                coupling_csv,
                seek_dict,
            ),
            force="tovector",
        )
        # reoriented_iodm is not a distance matrix - it is a "pseudodistance matrix".
        # If X and Y are sets and Y is a metric space, and f : X -> Y, then \
        # d_X(x0, x1) := d_Y(f(x0),f(x1)) is a pseudometric on X.
        dmat_accumulator_uncapped += reoriented_iodm
        dmat_accumulator_capped += cap(reoriented_iodm, 2.0)
    # dmat_avg_uncapped can have any positive values, but none are zero,
    # because medoid_matrix is not zero anywhere.
    # dmat_avg_capped has values between 0 and 2, exclusive.
    return (
        dmat_accumulator_capped / num_objects,
        dmat_accumulator_uncapped / num_objects,
    )


def avg_shape_spt(
        obj_names: list[int],
        gw_dist_mat: np.ndarray,
        iodms: dict[int, np.ndarray],
        coupling_csv,
        seek_dict,
        k: int,
):
    """
    Given a set of cells together with their intracell distance matrices and
    the (precomputed) pairwise GW coupling matrices between cells, construct a
    morphological "average" of cells in the cluster. This function:

    * aligns all cells in the cluster with each other using the coupling matrices
    * takes a "local average" of all intracell distance matrices, forming a
      distance matrix which models the average local connectivity structure of the neurons
    * draws a minimum spanning tree through the intracell distance graph,
      allowing us to visualize this average morphology

    :param obj_names: Keys for the gw_dist_dict and iodms; unique identifiers for the cells.
    :param gw_dist_mat: distance matrix mapping ordered pairs (cellA_index, cellB_index) \
        to Gromov-Wasserstein distances between them.
    :param iodms: (intra-object distance matrices) - \
        Maps object names to intra-object distance matrices. Matrices are assumed to be given \
        in vector form rather than squareform.
    :gw_coupling_mat_dict: Dictionary mapping ordered pairs (cellA_name, cellB_name) to \
        Gromov-Wasserstein coupling matrices from cellA to cellB.
    :param k: how many neighbors in the nearest-neighbors graph.
    """
    dmat_avg_capped, dmat_avg_uncapped = avg_shape(
        obj_names, gw_dist_mat, iodms, coupling_csv, seek_dict
    )
    dmat_avg_uncapped = squareform(dmat_avg_uncapped)
    # So that 0s along diagonal don't get caught in min
    np.fill_diagonal(dmat_avg_uncapped, np.max(dmat_avg_uncapped))
    # When confidence at a node in the average graph is high, the node is not
    # very close to its nearest neighbor.  We can think of this as saying that
    # this node in the averaged graph is a kind of poorly amalgamated blend of
    # different features in different graphs.  Conversely, when confidence is
    # low, and the node is close to its nearest neighbor, we interpret this as
    # meaning that this node and its nearest neighbor appear together in many
    # of the graphs being averaged, so this is potentially a good
    # representation of some edge that really appears in many of the graphs.
    confidence = np.min(dmat_avg_uncapped, axis=0)
    d = squareform(dmat_avg_capped)
    G = knn_graph(d, k)
    d = np.multiply(d, G)
    # Get shortest path tree

    spt = dijkstra(d, directed=False, indices=0, return_predecessors=True)
    # Get graph representation by only keeping distances on edges from spt
    mask = np.array([True] * (d.shape[0] * d.shape[1])).reshape(d.shape)
    for i in range(1, len(spt[1])):
        if spt[1][i] == -9999:
            print("Disconnected", i)
            continue
        mask[i, spt[1][i]] = False
        mask[spt[1][i], i] = False
    retmat = squareform(dmat_avg_capped)
    retmat[mask] = 0
    return dmat_avg_uncapped, retmat, confidence


if __name__ == "__main__":

    adata = anndata.read_h5ad("./data/contours.h5ad")

    sc.tl.leiden(adata)
    replace_dict = {
        7: 0,
        2: 1,
        0: 2,
        10: 3,
        3: 4,
        6: 5,
        4: 6,
        9: 7,
        5: 8,
        1: 9,
        8: 10,
        11: 11,
    }
    modes = adata.obs['leiden'].to_numpy().astype(int)
    cp_modes = modes.copy()
    for key, value in replace_dict.items():
        indices = np.where(modes == key)[0]
        cp_modes[indices] = value
    adata.obs['leiden'] = cp_modes
    adata.write_h5ad(Path("./data/contours.h5ad"), compression='gzip')

    plt.figure(figsize=(10, 6))
    modes = adata.obs['leiden'].to_numpy()
    embedding = adata.obsm['X_umap']
    color_list = sns.color_palette("deep") + sns.color_palette("deep")[3:]

    for m in np.unique(modes):
        index = np.where(modes == m)[0]
        m = int(m)
        x = embedding[index, 0]
        y = embedding[index, 1]
        plt.scatter(x, y, s=10, color=color_list[m], alpha=0.3)
        plt.text(x.mean(), y.mean(), f"{m+1}", fontsize=25)
        sns.kdeplot(x=x.reshape(-1), y=y.reshape(-1), color=color_list[m], levels=1)
    plt.axis('off')
    plt.savefig(f"./figures/modes_leiden_text.svg", dpi=600)
    plt.close()

    for m in np.unique(modes):
        plt.figure()
        index = np.where(modes == m)[0]
        m = int(m)
        X = adata.obsm['realigned'][index]
        pts = int(adata.n_vars / 2)
        c = []
        for j in range(X.shape[0]):
            c.append(np.array([X[j][:pts], X[j][pts:]]))
            plt.plot(X[j][:pts], X[j][pts:])
        plt.axis('off')
        plt.savefig(f"./figures/mode_contour{m+1}.svg", dpi=600)
        plt.close()

    # lookup_dict = generate_seek_pos("./data/coupling.csv")
    # pickle.dump(lookup_dict, open("./data/coupling_mat_seek_dict.pkl", "wb"))

    lookup_dict = pickle.load(open("./data/coupling_mat_seek_dict.pkl", "rb"))
    coupling_csv_file = "./data/coupling.csv"
    with h5py.File("./data/gw_dist.h5", "r") as f:
        gw_dist_mat = f["/gw_dist"][...]
        iodms = f["/intra_dist"][...]
    modes = adata.obs['leiden']
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=3000, eps=1e-9)
    for m in np.unique(modes):
        plt.figure()
        index = np.where(modes == m)[0]
        m = int(m)
        # dmat_avg_uncapped, retmat, confidence = avg_shape_spt(index, gw_dist_mat, iodms, coupling_csv_file, lookup_dict, 15)
        dmat_avg_capped, dmat_avg_uncapped = avg_shape(index, gw_dist_mat, iodms, coupling_csv_file, lookup_dict)
        distance_matrix = squareform(dmat_avg_uncapped)
        # Initialize MDS
        coordinates = mds.fit_transform(distance_matrix)
        coordinates = coordinates - np.min(coordinates, axis=0)
        coordinates = np.ceil(coordinates).astype(int)
        coordinates = np.concatenate([coordinates, coordinates[0, :].reshape(1, -1)], axis=0)
        size = int(np.max(coordinates))
        size += int(0.1 * size)
        pad_size = int(0.05 * size)
        mask = np.zeros((size, size), dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        draw = PIL.ImageDraw.Draw(mask)
        points = []
        for i in range(coordinates.shape[0]):
            points.append(tuple([coordinates[i, 0]+pad_size, coordinates[i, 1]+pad_size]))
        # plt.scatter(coordinates[:, 0], coordinates[:, 1], color="blue")
        draw.polygon(xy=points, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        mask = skimage.filters.gaussian(mask, sigma=5)
        cs = skimage.measure.find_contours(mask, level=0.5)
        if len(cs) > 1:
            print("Multiple contours detected")
            plt.scatter(coordinates[:, 0], coordinates[:, 1], color="blue")
            c = cs[0]
            plt.plot(c[:, 0], c[:, 1], color='black', alpha=0.8, linewidth=2)
            plt.show()
            exit()
        c = cs[0]
        plt.plot(c[:, 0], c[:, 1], color='black', alpha=0.8, linewidth=2)
        plt.axis('off')
        plt.savefig(f"./figures/mode_average_contour{m+1}.svg", dpi=600)
        plt.close()

    with h5py.File("./data/gw_dist.h5", "r") as f:
        gw_dist_mat = f["/gw_dist"][...]
        iodms = f["/intra_dist"][...]
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=3000, eps=1e-9)
    for m in np.unique(modes):
        plt.figure()
        indices = np.where(modes == m)[0]
        m = int(m)
        medoid_ind = identify_medoid(indices, gw_dist_mat)
        distance_matrix = squareform(iodms[medoid_ind])
        coordinates = mds.fit_transform(distance_matrix)
        coordinates = coordinates - np.min(coordinates, axis=0)
        coordinates = np.ceil(coordinates).astype(int)
        coordinates = np.concatenate([coordinates, coordinates[0, :].reshape(1, -1)], axis=0)
        size = int(np.max(coordinates))
        size += int(0.5 * size)
        pad_size = int(0.25 * size)
        mask = np.zeros((size, size), dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        draw = PIL.ImageDraw.Draw(mask)
        points = []
        for i in range(coordinates.shape[0]):
            points.append(tuple([coordinates[i, 0] + pad_size, coordinates[i, 1] + pad_size]))
        # plt.scatter(coordinates[:, 0], coordinates[:, 1], color="blue")
        draw.polygon(xy=points, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        kernel = skimage.morphology.ellipse(3, 3)
        out = skimage.morphology.dilation(mask, kernel)
        out = skimage.morphology.erosion(out, kernel)
        out = skimage.morphology.dilation(out, kernel)
        # mask = skimage.filters.gaussian(out, sigma=5)
        cs = skimage.measure.find_contours(mask, level=0.5)
        if len(cs) > 1:
            print("Multiple contours detected")
            plt.scatter(coordinates[:, 0], coordinates[:, 1], color="blue")
            c = cs[0]
            plt.plot(c[:, 0], c[:, 1], color='black', alpha=0.8, linewidth=2)
            plt.show()
            exit()
        c = cs[0]
        plt.plot(c[:, 0], c[:, 1], color='black', alpha=0.8, linewidth=2)
        plt.axis('off')
        plt.savefig(f"./figures/mode_medoid_contour{m+1}.svg", dpi=600)
        plt.close()
