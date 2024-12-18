from skimage.io import imsave
import h5py
import os
import numpy as np
import traceback
import logging


def obtain_img_mask(file, fov, frame):
    h5file = h5py.File(file, "r")
    channel = 0
    address = f"/images/channel_{channel}/fov_{fov}/frame_{frame}/image"
    image = h5file[address][:]
    mask_addr = f"/fov_{fov}/frame_{frame}"
    mask = h5file[mask_addr][:]
    h5file.close()
    return image, mask


def save_masks(save_path, file_list, frame_list, n_fov=30):
    if not os.path.exists(save_path + "/images"):
        os.makedirs(save_path + "/images")
    if not os.path.exists(save_path + "/masks"):
        os.makedirs(save_path + "/masks")
    for f, file in enumerate(file_list):
        for frame in frame_list[f]:
            for fov in range(n_fov):
                try:
                    image, mask = obtain_img_mask(file, fov, frame)
                    mask = mask.astype(np.uint8)
                    file_name = os.path.basename(file).split(".")[0]
                    name = f"{file_name}_frame{frame}_fov{fov}"
                    print(f"found mask : {name}")
                    imsave(save_path + "/images/" + name + ".tif", image)
                    imsave(save_path + "/masks/" + name + ".tif", mask)
                except Exception as e:
                    logging.error(traceback.format_exc())


if __name__ == "__main__":
    files = [f"F:/h 5 files/20240415-d1-10x00{i+2}.h5" for i in range(3)]
    frames = [
        [0, 30, 60, 90],
        [1, 31, 61, 91],
        [1, 31, 61, 91],
    ]
    save_masks("../..", files, frames)
