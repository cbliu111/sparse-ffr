import anndata
from pp.metric import compute_gw_with_coupling
import time
from utils import logger


if __name__ == '__main__':
    adata = anndata.read_h5ad("./data/contours.h5ad")
    logger.info(adata)
    start = time.time()
    compute_gw_with_coupling(adata, gw_file="./data/gw.h5", coupling_csv="./data/coupling.csv", cpus=20)
    logger.info(f"gw with coupling used time: {(time.time() - start) / 60} min ")


