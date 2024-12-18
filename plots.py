import skimage
import matplotlib.pyplot as plt
import numpy as np


file = "./data/D1/15_RGB_0049.tif"
mask_file = "./data/D1_15_RGB_0049.npy"
img = skimage.io.imread(file)
masks = np.load(mask_file)
indices = np.random.choice(masks.shape[0], 10, replace=False)
for i in indices:
    plt.figure(figsize=(10, 10))
    contours = skimage.measure.find_contours(masks[i], 0.5)
    index = np.where(masks[i] > 0)
    single_cell_img = np.zeros_like(img)
    single_cell_img[index] = img[index]
    plt.imshow(single_cell_img)
    contours.sort(key=lambda a: a.shape[0], reverse=True)
    x = contours[0][:, 1]
    y = contours[0][:, 0]
    plt.plot(x, y, '--', color='orange', lw=0.5)
    plt.axis('off')
    plt.savefig(f"./figures/cell_seg1_{i}.png", dpi=600)

file = "./data/D3/0719-96-900-4.tif"
mask_file = "./data/D3_0719-96-900-4.npy"
img = skimage.io.imread(file)
masks = np.load(mask_file)
indices = np.random.choice(masks.shape[0], 10, replace=False)
for i in indices:
    plt.figure(figsize=(10, 10))
    contours = skimage.measure.find_contours(masks[i], 0.5)
    index = np.where(masks[i] > 0)
    single_cell_img = np.zeros_like(img)
    single_cell_img[index] = img[index]
    plt.imshow(single_cell_img)
    contours.sort(key=lambda a: a.shape[0], reverse=True)
    x = contours[0][:, 1]
    y = contours[0][:, 0]
    plt.plot(x, y, '--', color='orange', lw=0.5)
    plt.axis('off')
    plt.savefig(f"./figures/cell_seg2_{i}.png", dpi=600)



