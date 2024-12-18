from typing import Literal
import scipy
import numpy as np
import numbers
from utils.log import logger
from scipy.sparse.csgraph import connected_components

def compute_eigen(
        transitions_sym: np.ndarray | None = None,
        connectivities: np.ndarray | None = None,
        n_comps: int = 15,
        sym: bool | None = None,
        sort: Literal["decrease", "increase"] = "decrease",
        random_state: float = 0,
):
    """\
    Compute eigen decomposition of transition matrix.

    Parameters
    ----------
    n_comps
        Number of eigenvalues/vectors to be computed, set `n_comps = 0` if
        you need all eigenvectors.
    sym
        Instead of computing the eigendecomposition of the assymetric
        transition matrix, computed the eigendecomposition of the symmetric
        Ktilde matrix.
    random_state
        A numpy random seed

    Returns
    -------
    Writes the following attributes.

    eigen_values : :class:`~numpy.ndarray`
        Eigenvalues of transition matrix.
    eigen_basis : :class:`~numpy.ndarray`
        Matrix of eigenvectors (stored in columns).  `.eigen_basis` is
        projection of data matrix on right eigenvectors, that is, the
        projection on the diffusion components.  these are simply the
        components of the right eigenvectors and can directly be used for
        plotting.
    """
    if transitions_sym is None:
        raise ValueError("Run `.compute_transitions` first.")
    matrix = transitions_sym
    # compute the spectrum
    if n_comps == 0:
        evals, evecs = scipy.linalg.eigh(matrix)
    else:
        n_comps = min(matrix.shape[0] - 1, n_comps)
        # ncv = max(2 * n_comps + 1, int(np.sqrt(matrix.shape[0])))
        ncv = None
        which = "LM" if sort == "decrease" else "SM"
        # it pays off to increase the stability with a bit more precision
        matrix = matrix.astype(np.float64)

        # Setting the random initial vector
        random_state = check_random_state(random_state)
        v0 = random_state.standard_normal(matrix.shape[0])
        evals, evecs = scipy.sparse.linalg.eigsh(
            matrix, k=n_comps, which=which, ncv=ncv, v0=v0
        )
        evals, evecs = evals.astype(np.float32), evecs.astype(np.float32)
    if sort == "decrease":
        evals = evals[::-1]
        evecs = evecs[:, ::-1]
    if connectivities is not None:
        cc = connected_components(connectivities)
        number_connected_components = cc[0]
        if number_connected_components > len(evals) / 2:
            logger.warning("Transition matrix has many disconnected components!")
    return evals, evecs


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.

    Examples
    --------
    >>> from sklearn.utils.validation import check_random_state
    >>> check_random_state(42)
    RandomState(MT19937) at 0x...
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )