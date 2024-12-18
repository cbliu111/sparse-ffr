import numpy as np


def rescale_pseudo_time_to_real_time(
        adata,
        time_key: str = 'frame'
):
    dpt = adata.obs['dpt_pseudotime'].to_numpy().copy()
    rt = adata.obs[time_key].to_numpy()
    urt = np.unique(rt)
    if urt[0] == 0:
        ddt = urt[1] - urt[0]
        urt += ddt
        rt += ddt
    urt = np.append(np.array([0]), urt)
    dts = np.diff(urt)
    for i, dt in enumerate(dts):
        idx = rt == urt[i+1]
        idx_t = dpt[idx]
        sdt = np.sort(idx_t)
        if i > 0:
            mdt = sdt[1] - sdt[0]
        else:
            mdt = 0
        dpt[idx] = (idx_t - sdt[0]) / (sdt[-1] + mdt - sdt[0]) * dt + urt[i] + mdt

    adata.obs['latent_time'] = dpt


