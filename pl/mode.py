import matplotlib.pyplot as plt
import numpy as np
import scipy
import math


def plot_polar_frame(ax, n_cluster, max_r=1., fontsize=14):
    dx = max_r * 0.05
    dx1 = max_r * 0.15
    angles = get_polar_angles(n_cluster)
    n = len(angles)
    for i in range(n):
        a = angles[i]
        p = rotator(a, np.array([0, max_r - dx]))
        ax.arrow(0, 0, p[0], p[1], head_width=max_r * 0.03, color='k', alpha=0.3)
        p = rotator(a, np.array([0, max_r + dx1]))
        ax.text(p[0], p[1], 'mode %d' % (i + 1), ha='center', va='center', fontsize=fontsize)
    for i in range(5):
        r = max_r / 5 * (i + 1)
        c = plt.Circle((0, 0), r, color='k', alpha=0.3, fill=False)
        ax.add_patch(c)


def get_polar_angles(n_cluster):
    angles = []
    for i in range(n_cluster):
        a = round(i * 360 / n_cluster)
        angles.append(a)
    return angles


def plot_similar_umbrella(features, modes, color='cluster', save="./similar_umbrella.pdf"):
    dist = scipy.spatial.distance.cdist(features, modes, metric='euclidean')
    s = - dist ** 2
    similarity = scipy.special.softmax(s, axis=1)

    fig, ax = plt.figure(figsize=(10, 10))
    angles = get_polar_angles(len(modes))
    plot_polar_frame(ax, len(modes))
    arr = similarity.copy()
    axis_ids = np.argsort(arr, axis=1)[:, ::-1][:, :2]  # sort is ascending
    rotated_points = []
    for i, n in enumerate(axis_ids):
        i1, i2 = n[0], n[1]
        a1, a2 = angles[i1], angles[i2]
        x1, x2 = similarity[i, i1], similarity[i, i2]
        p1 = rotator(a1, np.array([0, x1]))
        p2 = rotator(a2, np.array([0, x2]))
        if abs(a1 - a2) == 180:
            rotated_points.append(p1)
        else:
            rotated_points.append(p1 + p2)
    rotated_points = np.array(rotated_points)
    if color == 'cluster':
        plt.scatter(rotated_points[:, 0], rotated_points[:, 1], marker='.', s=50, c=cluster_ids, cmap='RdBu')
    elif color == 'label':
        plt.scatter(rotated_points[:, 0], rotated_points[:, 1], marker='.', s=50, c=plot_labels, cmap='RdBu')
    else:
        raise ValueError("Color category not defined.")
    plt.axis('off')
    plt.savefig(save, dpi=600)


def plot_mode_distribution_umbrella(self):
    labels = np.unique(self.plot_labels)
    n_labels = len(labels)
    n_cluster = len(self.modes)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    self.plot_polar_frame(ax, max_r=0.3)
    angles = self.get_polar_angles()
    show_labels = [0, 6, 11]
    for i in range(n_labels):
        if i not in show_labels:
            continue
        l = labels[i]
        idx = self.plot_labels == l
        cid = self.cluster_ids[idx]
        hist, edges = np.histogram(cid, bins=np.arange(n_cluster + 1), density=True)
        cps = []
        for j in range(n_cluster):
            cp = self.rotator(angles[j], np.array([0, hist[j]]))
            cps.append(cp)
        cps.append(cps[0])
        cps = np.array(cps)
        ax.plot(cps[:, 0], cps[:, 1], lw=3, label=f"T{i + 1}")

        # ax.fill(corners_x, corners_y, 'orange', alpha=0.5)
    ax.legend()
    ax.axis('off')
    ax.axis('equal')
    file = self.figure_path + "mode_distribution_umbrella.svg"
    plt.savefig(file, dpi=600)


def rotator(a, point):
    a = math.radians(a)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    rps = r @ point.T
    return rps.T


def plot_mapping_umbrella(self, color='cluster'):
    fig, ax = plt.figure(figsize=(10, 10))
    angles = self.get_polar_angles()
    self.plot_polar_frame(ax)
    arr = self.map_weights.copy()
    axis_ids = np.argsort(arr, axis=1)[:, ::-1][:, :2]  # sort is ascending
    rotated_points = []
    for i, n in enumerate(axis_ids):
        i1, i2 = n[0], n[1]
        a1, a2 = angles[i1], angles[i2]
        x1, x2 = self.map_weights[i, i1], self.map_weights[i, i2]
        p1 = self.rotator(a1, np.array([0, x1]))
        p2 = self.rotator(a2, np.array([0, x2]))
        if abs(a1 - a2) == 180:
            rotated_points.append(p1)
        else:
            rotated_points.append(p1 + p2)
    rotated_points = np.array(rotated_points)
    if color == 'cluster':
        plt.scatter(rotated_points[:, 0], rotated_points[:, 1], marker='.', s=50, c=self.cluster_ids, cmap='RdBu')
    elif color == 'label':
        plt.scatter(rotated_points[:, 0], rotated_points[:, 1], marker='.', s=50, c=self.plot_labels, cmap='RdBu')
    else:
        raise ValueError("Color category not defined.")
    plt.axis('off')
    file = self.figure_path + "mapping_umbrella.svg"
    plt.savefig(file, dpi=600)


def plot_clustering_mapping_consistency(self):
    plt.figure()
    weight_ids = np.argmax(self.map_weights, axis=1)
    plt.hist(weight_ids - self.cluster_ids, bins=20)
    file = self.figure_path + "clustering_mapping_consistency.svg"
    plt.savefig(file, dpi=600)


def plot_shape_mode_dendrogram(self):
    pair_wise_dist = scipy.spatial.distance.pdist(self.modes, metric='euclidean')
    linkage = scipy.cluster.hierarchy.linkage(pair_wise_dist, method='complete')
    linkage[:, 2] = linkage[:, 2] * 5  # multiply distance manually 10times to plot better.
    scipy.cluster.hierarchy.set_link_color_palette(['k'])
    dendrogram = scipy.cluster.hierarchy.dendrogram(linkage, p=0, truncate_mode='mlab', orientation='bottom', ax=None,
                                                    above_threshold_color='k')

    plt.figure()
    plt.yticks([])
    dendidx = np.array(dendrogram['ivl'])
    scipy.cluster.hierarchy.set_link_color_palette(None)
    # mpl.rcParams['lines.linewidth'] = 1
    plt.axis('equal')
    plt.axis('off')
