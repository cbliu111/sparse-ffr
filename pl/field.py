from typing import Callable, List, Tuple, Union, Optional, Literal, Dict, Any
from matplotlib.axes import Axes
from anndata import AnnData
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from tl.field_tools import merge_dict


def flow_field(
    adata: AnnData,
    vecfld: Callable,
    n_grid: int = 100,
    start_points: Optional[np.ndarray] = None,
    integration_direction: Literal["forward", "backward", "both"] = "both",
    density: float = 2,
    linewidth: float = 2,
    streamline_color: Optional[str] = None,
    streamline_alpha: float = 0.6,
    color_start_points: Optional[float] = None,
    show=True,
    save: str = '',
    ax: Optional[Axes] = None,
    **streamline_kwargs,
):

    """Plots the flow field with line thickness proportional to speed.

    Code adapted from: http://be150.caltech.edu/2017/handouts/dynamical_systems_approaches.html

    Args:
        vecfld: an instance of the vector_field class.
        x_range: the range of values for x-axis.
        y_range: the range of values for y-axis.
        n_grid: the number of grid points to use in computing derivatives on phase portrait. Defaults to 100.
        start_points: the initial points from which the streamline will be drawn. Defaults to None.
        integration_direction: integrate the streamline in forward, backward or both directions. default is 'both'.
            Defaults to "both".
        background: the background color of the plot. Defaults to None.
        density: the density of the plt.streamplot function. Defaults to 1.
        linewidth: the multiplier of automatically calculated linewidth passed to the plt.streamplot function. Defaults
            to 1.
        streamline_color: the color of the vector field streamlines. Defaults to None.
        streamline_alpha: the alpha value applied to the vector field streamlines. Defaults to 0.6.
        color_start_points: the color of the starting point that will be used to predict cell fates. Defaults to None.
        save_show_or_return: whether to save, show or return the figure. Defaults to "return".
        save_kwargs: a dictionary that will be passed to the save_show_ret function. By default, it is an empty dictionary
            and the save_show_ret function will use the {"path": None, "prefix": 'plot_flow_field', "dpi": None,
            "ext": 'pdf', "transparent": True, "close": True, "verbose": True} as its parameters. Otherwise, you can
            provide a dictionary that properly modify those keys according to your needs. Defaults to {}.
        ax: the Axis on which to make the plot. Defaults to None.

    Returns:
        None would be returned by default. If `save_show_or_return` is set to be 'return', the Axes of the generated
        figure would be returned.
    """

    color, color_start_points = (
        "black" if streamline_color is None else streamline_color,
        "red" if color_start_points is None else color_start_points,
    )

    X_basis = adata.obsm["X_umap"][:, :2]
    min_, max_ = X_basis.min(0), X_basis.max(0)

    xlim = [
        min_[0] - (max_[0] - min_[0]) * 0.1,
        max_[0] + (max_[0] - min_[0]) * 0.1,
    ]
    ylim = [
        min_[1] - (max_[1] - min_[1]) * 0.1,
        max_[1] + (max_[1] - min_[1]) * 0.1,
    ]

    # Set up u,v space
    u = np.linspace(xlim[0], xlim[1], n_grid)
    v = np.linspace(ylim[0], ylim[1], n_grid)
    uu, vv = np.meshgrid(u, v)

    # Compute derivatives
    u_vel = np.empty_like(uu)
    v_vel = np.empty_like(vv)
    for i in range(uu.shape[0]):
        for j in range(uu.shape[1]):
            u_vel[i, j], v_vel[i, j] = vecfld(np.array([uu[i, j], vv[i, j]]))

    # Compute speed
    speed = np.sqrt(u_vel**2 + v_vel**2)

    # Make linewidths proportional to speed,
    # with minimal line width of 0.5 and max of 3
    # lw = lw_min + (lw_max - lw_min) * speed / speed.max()

    streamplot_kwargs = {
        "density": density * 2,
        "linewidth": None,
        "cmap": None,
        "norm": None,
        "arrowsize": 1,
        "arrowstyle": "fancy",
        "minlength": 0.1,
        "transform": None,
        "maxlength": 4.0,
        "zorder": 3,
    }
    linewidth *= 2 * speed / speed[~np.isnan(speed)].max()
    streamplot_kwargs.update({"linewidth": linewidth})

    streamplot_kwargs = merge_dict(streamplot_kwargs, streamline_kwargs, update=True)

    # Make stream plot
    if ax is None:
        ax = plt.gca()
    if start_points is None:
        s = ax.streamplot(
            uu,
            vv,
            u_vel,
            v_vel,
            color=color,
            **streamplot_kwargs,
        )
        set_arrow_alpha(ax, streamline_alpha)
        set_stream_line_alpha(s, streamline_alpha)
    else:
        if len(start_points.shape) == 1:
            start_points.reshape((1, 2))
        ax.scatter(*start_points, marker="*", zorder=4)

        s = ax.streamplot(
            uu,
            vv,
            u_vel,
            v_vel,
            start_points=start_points,
            integration_direction=integration_direction,
            color=color_start_points,
            **streamplot_kwargs,
        )
        set_arrow_alpha(ax, streamline_alpha)
        set_stream_line_alpha(s, streamline_alpha)
    fig = plt.gcf()
    if show:
        plt.show()
    if save:
        fig.savefig(save, dpi=600, bbox_inches="tight")


def set_arrow_alpha(ax=None, alpha=1.):
    ax = plt.gca() if ax is None else ax

    # iterate through the children of ax
    for art in ax.get_children():
        # we are only interested in FancyArrowPatches
        if not isinstance(art, patches.FancyArrowPatch):
            continue
        art.set_alpha(alpha)


def set_stream_line_alpha(s=None, alpha=1.):
    """s has to be a StreamplotSet"""
    s.lines.set_alpha(alpha)


