import matplotlib.pyplot as plt 
from typing import Literal,cast
import numpy as np
import numpy.typing as npt
import rt1plotpy
import mpl_toolkits.axes_grid1
from matplotlib.colors import Normalize
import matplotlib.figure 
from typing import Literal,cast
from matplotlib.cm import ScalarMappable
import rt1kernel

__all__ = ['gaussian',
           'Length_scale_sq', 
           'Length_scale', 
           'rt1_ax_kwargs',
           'cycle',
           'imshow_cbar',
           'scatter_cbar',
           'func_ring',
           'imshow_cbar_bottom',
           'cmap_line',
           'plt_subplots',
           'plt_rt1_flux',]

params = {
        'font.family'      : 'Times New Roman', # font familyの設定
        'mathtext.fontset' : 'stix'           , # math fontの設定
        "font.size"        : 15               , # 全体のフォントサイズが変更されます。
        'xtick.labelsize'  : 12                , # 軸だけ変更されます。
        'ytick.labelsize'  : 12               , # 軸だけ変更されます
        'xtick.minor.visible' :True,
        'ytick.minor.visible' :True,
        'xtick.direction'  : 'in'             , # x axis in
        'ytick.direction'  : 'in'             , # y axis in 
        'axes.linewidth'   : 1.0              , # axis line width
        'axes.grid'        : True             , # make grid
        'figure.facecolor' : 'none',#透明にする.
        }

plt.rcParams.update(**params)

rt1_ax_kwargs = {'xlim'  :(0,1.1),
                 'ylim'  :(-0.7,0.7), 
                 'aspect': 'equal'
                }

cycle = plt.get_cmap("tab10")  # type: ignore

n0 = 2#25.99e16*0.8/2
a  = 1.348
b  = 0.5
rmax = 0.4577

def gaussian(r,z,n0=n0,a=a,b=b,rmax=rmax,separatrix=True):
    psi = rt1plotpy.mag.psi(r,z,separatrix=separatrix)
    br, bz = rt1plotpy.mag.bvec(r,z,separatrix=separatrix)
    b_abs = np.sqrt(br**2+bz**2)
    psi_rmax = rt1plotpy.mag.psi(rmax,0,separatrix=separatrix)
    psi0 = rt1plotpy.mag.psi(1,0,separatrix=separatrix)
    b0 = rt1plotpy.mag.b0(r,z,separatrix=separatrix)
    return n0 * np.exp(-a*(psi-psi_rmax)**2/psi0**2)*(b_abs/b0)**(-b) 

def Length_scale_sq(r,z):
    return 0.0001/(gaussian(r,z)+ 0.05)

def Length_scale(r,z):
    return np.sqrt( Length_scale_sq(r,z))

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
    x,y,C,
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


def func_ring(r,z,n0=1.0,a=1.0,b=1.0,r_center=0.58,radius=0.5,separatrix=False):
    """
    n0 :max_value,
    a  :width of layer,
    b  : shape,
    r_center :r center,
    radius :size of ring 
    """
    psi = rt1plotpy.mag.psi(r,z,separatrix)
    br, bz = rt1plotpy.mag.bvec(r,z,separatrix)
    b_abs = np.sqrt(br**2+bz**2)
    psi_rmax = rt1plotpy.mag.psi(r_center,0,separatrix)
    psi0 = rt1plotpy.mag.psi(1,0,separatrix)
    b0 = rt1plotpy.mag.b0(r,z,separatrix)
    bb = b_abs/b0/b
    return n0 * np.exp(- (np.sqrt(((psi-psi_rmax)/psi0)**2+(1-1/bb)**2)-radius)**2*100/a) /(1+np.exp(20000*(psi-psi0*1.05)))

def plt_rt1_flux(K:rt1kernel.Kernel2D_scatter,
    ax:plt.Axes,      
    separatrix:bool =True,
    is_inner:bool =False,
    append_frame :bool =True,
    **kwargs_contour,
    )->None:
    R,Z = np.meshgrid(K.r_plot,K.z_plot,indexing='xy')
    Psi = rt1plotpy.mag.psi(R,Z,separatrix=separatrix)
    extent = K.im_kwargs['extent']
    origin = K.im_kwargs['origin']
    kwargs = {'levels':20,'colors':'black','alpha':0.3}
    kwargs.update(kwargs_contour)
    if is_inner:
        Psi = Psi*K.mask
    else:
        mpsi_max = -rt1plotpy.mag.psi(0.3,0.,separatrix=separatrix)
        mpsi_min = -rt1plotpy.mag.psi(K.r_plot.min(),K.z_plot.max(),separatrix=separatrix)
        kwargs['levels'] = np.linspace(mpsi_min,mpsi_max,kwargs['levels'],endpoint=False)
    
    ax.contour(-Psi,extent=extent,origin=origin,**kwargs)
    if append_frame:
        K.append_frame(ax)


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
