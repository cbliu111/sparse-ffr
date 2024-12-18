import pandas as pd
import skimage
import numpy as np
import glob
import os
import pickle
import scipy
import matplotlib.pyplot as plt
from anndata import AnnData
import cv2
import multiprocessing
import copy
import math
from joblib import Parallel, delayed
from tqdm import tqdm
from os import PathLike
from utils.log import logger


def contour_area(cnt):
    cnt = np.array(cnt)
    area = cv2.contourArea(cnt)
    return area


def contour_aspect_ratio(cnt):
    cnt = np.array(cnt)
    # Orientation, Aspect_ratio
    (x, y), (MA, ma), orientation = cv2.fitEllipse(cnt)
    aspect_ratio = MA / ma
    return aspect_ratio


def contour_extent(cnt):
    cnt = np.array(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    rect_area = w * h
    extent = float(area) / rect_area
    return extent


def contour_equiv_diameter(cnt):
    cnt = np.array(cnt)
    area = cv2.contourArea(cnt)
    equi_diameter = np.sqrt(4 * area / np.pi)
    return equi_diameter


def contour_solidity(cnt):
    cnt = np.array(cnt)
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area
    return solidity


def reorient_contour(contour):
    xc = np.mean(contour[1])
    yc = np.mean(contour[0])
    bd0 = np.append([contour[0] - yc], [contour[1] - xc], axis=0)  # centralize
    xi = bd0[1]
    yi = bd0[0]
    s = np.sqrt((sum(np.power(xi, 2)) + sum(np.power(yi, 2))) / len(xi))
    xi = xi / s
    yi = yi / s  # rescale
    xiyi = np.append([xi], [yi], axis=0).transpose()
    u, S, rm = np.linalg.svd(xiyi, full_matrices=True)
    # this is a quick fix
    if np.isnan(rm).any():
        rm[rm != 1] = 1
    xynew = np.dot(xiyi, rm.transpose())
    xynew = xynew.transpose()  # align to eigenvector directions
    yc = xynew[1].mean()
    xc = xynew[0].mean()
    xon = xynew[0] - xc
    yon = xynew[1] - yc
    theta = np.empty(len(yon))
    theta[:] = np.nan  # arc tangent of each point
    for i in range(len(yon)):
        theta[i] = math.atan2(yon[i], xon[i])
    cc = np.argwhere(abs(theta) == min(abs(theta)))
    cc = cc[0]
    cc = cc[0]  # point on the direction of maximum eigenvalue
    ccid = np.append(range(cc, len(xon)), range(0, cc))
    ccid = ccid.astype(int)
    xon = xon[ccid]
    yon = yon[ccid]
    theta = theta[ccid]  # sort the points w.r.t. direction angles
    if theta[4] - theta[0] < 0:
        xon = np.append(xon[-1:0:-1], xon[0])
        yon = np.append(yon[-1:0:-1], yon[0])
    bds = np.append([yon], [xon], axis=0)
    return bds, s  # s: scale factor


def realign_contour(contour, reference):
    xc = np.sum(np.dot(contour[1], abs(contour[0]))) / np.sum(abs(contour[0]))
    yc = np.sum(np.dot(contour[0], abs(contour[1]))) / np.sum(abs(contour[1]))

    bd0 = np.append([contour[0] - yc], [contour[1] - xc], axis=0)
    bd = bd0
    bdr = reference

    xc = np.sum(np.dot(bdr[1], abs(bdr[0]))) / np.sum(abs(bdr[0]))
    yc = np.sum(np.dot(bdr[0], abs(bdr[1]))) / np.sum(abs(bdr[1]))

    bdr = np.append([bdr[0] - yc], [bdr[1] - xc], axis=0)
    temp = copy.deepcopy(bdr[1])
    bdr[1] = bdr[0]
    bdr[0] = temp
    temp = copy.deepcopy(bd[1])
    bd[1] = bd[0]
    bd[0] = temp
    N = len(bd[0])
    costold = np.mean(sum(sum(np.power((bdr - bd), 2))))
    bdout = copy.deepcopy(bd)
    # print('regbd3')
    for k in range(1, N + 1):
        idk = np.append(range(k, N + 1), range(1, k))
        bdt = np.empty([len(idk), 2])
        bdt[:] = np.nan
        for i in range(len(bd.transpose())):
            ind = int(idk[i] - 1)
            bdt[i] = bd.transpose()[ind]
        temp = np.dot(bdr, bdt)
        u, _, v = np.linalg.svd(temp)
        v = v.T
        q = np.dot(v, u.transpose())
        bdtemp = np.dot(bdt, q)
        costnew = np.mean(sum(sum(np.power((bdr.transpose() - bdtemp), 2))))
        if costnew < costold:
            bdout = copy.deepcopy(bdtemp)
            costold = copy.deepcopy(costnew)

    realigned_bd = copy.deepcopy(bdout.T)
    realigned_bd[:] = np.nan
    realigned_bd[0] = copy.deepcopy(bdout.T[1])
    realigned_bd[1] = copy.deepcopy(bdout.T[0])
    return realigned_bd


def get_realigned_contours(list_of_contours, pts: int = 150, num_cpus=20):
    contours = pd.Series(list_of_contours)
    kll = len(contours)
    features = np.zeros([kll, 2 * pts])
    scales = np.zeros([kll, 1])

    num_cores = multiprocessing.cpu_count()
    used_cores = min(num_cpus, num_cores)
    print(f"total cores: {num_cores}, used cores: {used_cores}")
    print(f"resampling...")
    for k in range(kll):
        c = resample_contour((contours.loc[k]), pts)
        contours.loc[k], scales[k] = reorient_contour(c.T)
        # scale back to original size
        features[k] = np.append([contours[k][1] * scales[k]], [contours[k][0] * scales[k]], axis=1)
    print(f"calculating reference contour...")
    c0 = [sum(x) / len(x) for x in zip(*features)]  # average over all contours, same as np.mean(c0, axis=0)
    mean_contour = np.append([c0[pts:]], [c0[0:pts]], axis=0)  # this is the mean contour
    # copy_mean_contour = copy.deepcopy(mean_contour)
    # copy_contour = copy.deepcopy(contours)
    print(f"realigning contours to reference contour...")
    # align all cells to the mean cell
    # print(contours[0].shape, scales[0].shape)
    rec = Parallel(n_jobs=num_cores)(
        delayed(realign_contour)(contours[k] * scales[k], mean_contour) for k in range(kll))
    realign_features = np.array([np.append(rec[i][1], rec[i][0]) for i in range(len(rec))])
    return features, mean_contour, realign_features


def fill_mask_holes(mask, area_th=10):
    for lv in np.unique(mask):
        if lv == 0:
            continue
        else:
            binary_mask = (mask == lv)
            binary_mask = skimage.morphology.remove_small_holes(binary_mask, area_threshold=area_th)
            mask[binary_mask] = lv
    return mask


def get_frame_fov_from_path(path):
    ids = os.path.splitext(os.path.basename(path))[0].split("_")
    file = int(ids[0][-1]) - 2
    frame = file * 4 + int(int(ids[1].replace("frame", "")) / 30)
    fov = int(ids[2].replace("fov", ""))
    return frame, fov


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


def read_masks(
        path_to_masks: PathLike,
        save: PathLike = '',
        pts: int = 150,
        cpus: int = 20,
):
    logger.info("Start reading contours...")
    contours = []
    masks_path = glob.glob(os.path.join(path_to_masks, '*.tif'))
    masks_path.sort(key=lambda p: get_frame_fov_from_path(p))
    frame_ids = []
    fov_ids = []
    labels = []
    n_obs = 0
    for path in tqdm(masks_path):
        frame, fov = get_frame_fov_from_path(path)
        mask = skimage.io.imread(path)
        fill_mask = fill_mask_holes(mask)  # fill holes in case of extra contours
        for label in np.unique(fill_mask):
            label = int(label)
            # print(f"Processing {path} with frame {frame} fov {fov} label {label}")
            if label < 1:
                continue
            label_mask = fill_mask == label
            cs = skimage.measure.find_contours(label_mask, level=0.5)
            if len(cs) > 1:
                close_mask = skimage.morphology.closing(label_mask, skimage.morphology.square(5))
                cs = skimage.measure.find_contours(close_mask, level=0.5)
            cs.sort(key=lambda x: x.shape[0], reverse=True)
            contours.append(cs[0])
            frame_ids.append(frame)
            fov_ids.append(fov)
            labels.append(label)
            n_obs += 1

    logger.info(f"Start realigning contours...")
    resample_contours, mean_contour, realign_contours = get_realigned_contours(contours, pts, cpus)

    pickle.dump(contours, open(os.path.join("./resample_contours.pkl"), "wb"))

    logger.info(f"Verify resample and realign contours...")
    idx = np.random.randint(len(contours), size=(25,))
    fig, axes = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            k = idx[i * 5 + j]
            xc = np.mean(contours[k][:, 0])
            yc = np.mean(contours[k][:, 1])
            ax.plot(contours[k][:, 0] - xc, contours[k][:, 1] - yc, linewidth=2, color='black')
            ax.plot(resample_contours[k, :pts], resample_contours[k, pts:], linewidth=2, color='red')
            ax.plot(realign_contours[k, :pts], realign_contours[k, pts:], linewidth=2, color='blue')
    plt.savefig("./figures/verify_contour_resample_align.pdf")
    logger.info("Figure saved in  ./figures/verify_contour_resample_align.pdf")

    adata = AnnData(
        resample_contours.reshape(n_obs, -1))  # restore the coordinate using np.array([f[:, :pts], f[:, pts:]])
    adata.obsm["realigned"] = realign_contours.reshape(n_obs, -1)
    adata.uns['mean_contour'] = mean_contour.T

    adata.obs_names = [f'contour_{i:d}' for i in range(n_obs)]

    vn = [f'contour_{i:d}_x' for i in range(pts)]
    vn += [f'contour_{i:d}_y' for i in range(pts)]
    adata.var_names = vn
    adata.obs['frame'] = pd.Categorical(frame_ids)
    adata.obs['fov'] = pd.Categorical(fov_ids)
    adata.obs['label'] = pd.Categorical(labels)

    contours = [c.astype(np.float32) for c in contours]
    # calculate some basic classical summary statistics
    adata.obs['area'] = pd.Categorical(map(contour_area, contours))
    adata.obs['aspect_ratio'] = pd.Categorical(map(contour_aspect_ratio, contours))
    adata.obs['extent'] = pd.Categorical(map(contour_extent, contours))
    adata.obs['equiv_diameter'] = pd.Categorical(map(contour_equiv_diameter, contours))
    adata.obs['solidity'] = pd.Categorical(map(contour_solidity, contours))

    if save:
        adata.write_h5ad(save, compression='gzip')
        logger.info(f"adata saved in  {save}")

    return adata
