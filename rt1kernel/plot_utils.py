from fileinput import filename
import math
import os
import sys
from typing import Union,Tuple,List
from unicodedata import name

import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.figure 
import numpy as np
import pandas as pd
from numpy.core.fromnumeric import transpose
from PIL import Image
from scipy import misc, ndimage, optimize, signal
from scipy import special


params = {
        'font.family'      : 'Times New Roman', # font familyの設定
        'mathtext.fontset' : 'stix'           , # math fontの設定
        "font.size"        : 18               , # 全体のフォントサイズが変更されます。
        'xtick.labelsize'  : 15                , # 軸だけ変更されます。
        'ytick.labelsize'  : 15               , # 軸だけ変更されます
        'xtick.direction'  : 'in'             , # x axis in
        'ytick.direction'  : 'in'             , # y axis in 
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'axes.linewidth'   : 1.0              , # axis line width
        'axes.grid'        : True             , # make grid
        }       
        
plt.rcParams.update(**params)


def imshow_cbar(
    ax:plt.Axes,
    im0:np.ndarray,
    cbar_title: str|None=None,
    **kwargs
    ):

    im = ax.imshow(im0,**kwargs)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)    
    cax = divider.append_axes("right", size="5%", pad='3%')
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')
    if cbar_title is not None: cbar.set_title(cbar_title)
    ax.set_aspect('equal')



def contourf_cbar(
    ax:plt.Axes,
    im0:np.ndarray,
    cbar_title: str|None=None,
    **kwargs
    ):

    im = ax.imshow(im0,**kwargs)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)    
    cax = divider.append_axes("right", size="5%", pad='3%')
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')
    if cbar_title is not None: cbar.set_title(cbar_title)
    ax.set_aspect('equal')
    
def imshow_cbar_bottom(
    ax:plt.Axes,
    im0:np.ndarray,
    cbar_title=None,
    **kwargs
    ):
    im = ax.imshow(im0,**kwargs)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)    
    cax = divider.append_axes("bottom", size="5%", pad='3%')
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    if cbar_title is not None: cbar.set_title(cbar_title)
    ax.set_aspect('equal')

    
def scatter_cbar(
    ax:plt.Axes,
    x, y, c,
    cbar_title=None,
    **kwargs
    ):
    im = ax.scatter(x=x,y=y,c=c,**kwargs)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right' , size="5%", pad='3%')
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')
    if cbar_title is not None: cbar.set_label(cbar_title)
    ax.set_aspect('equal')

    
def cmap_line(
    ax:plt.Axes,
    x, y, C, 
    cmap='viridis',
    cbar_title=None,
    **kwargs
    ):
    norm = Normalize(vmin=y.min(), vmax=y.max())
    cmap = plt.get_cmap(cmap) # type: ignore

    for i,yi in enumerate(y):
        color = cmap(norm(yi))
        ax.plot(x, C[i,:], color=color,**kwargs)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad='3%')
    cbar = plt.colorbar(sm, cax=cax)
    if cbar_title is not None: cbar.set_label(cbar_title)


import matplotlib.figure 
from typing import Literal,cast
def plt_subplots(
    nrows: int = 1,
    ncols: int = 1,
    sharex: bool | Literal['none', 'all', 'row', 'col'] = False,
    sharey: bool | Literal['none', 'all', 'row', 'col'] = False,
    squeeze: bool = False,
    height_ratios=None,
    width_ratios=None,
    subplot_kw=None, 
    gridspec_kw=None,
    **fig_kw 
    )->tuple[ matplotlib.figure.Figure,list[list[plt.Axes]] ] :
    
    """
    Create a figure and a set of subplots.

    This utility wrapper makes it convenient to create common layouts of
    subplots, including the enclosing figure object, in a single call.

    Parameters
    ----------
    nrows, ncols : int, default: 1
        Number of rows/columns of the subplot grid.

    sharex, sharey : bool or {'none', 'all', 'row', 'col'}, default: False
        Controls sharing of properties among x (*sharex*) or y (*sharey*)
        axes:

        - True or 'all': x- or y-axis will be shared among all subplots.
        - False or 'none': each subplot x- or y-axis will be independent.
        - 'row': each subplot row will share an x- or y-axis.
        - 'col': each subplot column will share an x- or y-axis.

        When subplots have a shared x-axis along a column, only the x tick
        labels of the bottom subplot are created. Similarly, when subplots
        have a shared y-axis along a row, only the y tick labels of the first
        column subplot are created. To later turn other subplots' ticklabels
        on, use `~matplotlib.axes.Axes.tick_params`.

        When subplots have a shared axis that has units, calling
        `~matplotlib.axis.Axis.set_units` will update each axis with the
        new units.

    squeeze : bool, default: True
        - If True, extra dimensions are squeezed out from the returned
          array of `~matplotlib.axes.Axes`:

          - if only one subplot is constructed (nrows=ncols=1), the
            resulting single Axes object is returned as a scalar.
          - for Nx1 or 1xM subplots, the returned object is a 1D numpy
            object array of Axes objects.
          - for NxM, subplots with N>1 and M>1 are returned as a 2D array.

        - If False, no squeezing at all is done: the returned Axes object is
          always a 2D array containing Axes instances, even if it ends up
          being 1x1.

    width_ratios : array-like of length *ncols*, optional
        Defines the relative widths of the columns. Each column gets a
        relative width of ``width_ratios[i] / sum(width_ratios)``.
        If not given, all columns will have the same width.  Equivalent
        to ``gridspec_kw={'width_ratios': [...]}``.

    height_ratios : array-like of length *nrows*, optional
        Defines the relative heights of the rows. Each row gets a
        relative height of ``height_ratios[i] / sum(height_ratios)``.
        If not given, all rows will have the same height. Convenience
        for ``gridspec_kw={'height_ratios': [...]}``.

    subplot_kw : dict, optional
        Dict with keywords passed to the
        `~matplotlib.figure.Figure.add_subplot` call used to create each
        subplot.

    gridspec_kw : dict, optional
        Dict with keywords passed to the `~matplotlib.gridspec.GridSpec`
        constructor used to create the grid the subplots are placed on.

    **fig_kw
        All additional keyword arguments are passed to the
        `.pyplot.figure` call.

    Returns
    -------
    fig : `.Figure`

    ax : `~.axes.Axes` or list of Axes
        *ax* can be either a single `~.axes.Axes` object, or an array of Axes
        objects if more than one subplot was created.  The dimensions of the
        resulting array can be controlled with the squeeze keyword, see above.

        Typical idioms for handling the return value are::

            # using the variable ax for single a Axes
            fig, ax = plt.subplots()

            # using the variable axs for multiple Axes
            fig, axs = plt.subplots(2, 2)

            # using tuple unpacking for multiple Axes
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        The names ``ax`` and pluralized ``axs`` are preferred over ``axes``
        because for the latter it's not clear if it refers to a single
        `~.axes.Axes` instance or a collection of these.
    """

    fig,axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        subplot_kw=subplot_kw,# type: ignore
        gridspec_kw=gridspec_kw,# type: ignore
        **fig_kw)
    
    axs = np.array(axs)
    return fig, cast(list[list[plt.Axes]],axs.tolist())
    
    #if nrows == 1 and ncols ==1:
    #    return fig, cast(plt.Axes,axs[0])
    #elif nrows == 1 or ncols ==1:
    #    return fig, cast(list[plt.Axes],axs.tolist())
    #else :
    #    return fig, cast(list[list[plt.Axes]],axs.tolist())

    #return n0 *  (np.sqrt(((psi-psi_rmax)/psi0)**2+(1-1/b)**2)-0.5)*a / (a +   (np.sqrt(((psi-psi_rmax)/psi0)**2+(1-1/b)**2)-0.5)**2)*(1-np.exp(-100*(r-1)**2))
