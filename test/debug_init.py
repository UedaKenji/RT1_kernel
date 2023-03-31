import matplotlib.pyplot as plt 
import numpy as np
import rt1plotpy
import mpl_toolkits.axes_grid1
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

__all__ = ['gaussian',
           'Length_scale_sq', 
           'Length_scale', 
           'rt1_ax_kwargs',
           'cycle',
           'imshow_cbar',
           'scatter_cbar',
           'func_ring',
           'imshow_cbar_bottom',
           'cmap_line']

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
        }

plt.rcParams.update(**params)

rt1_ax_kwargs = {'xlim'  :(0,1.1),
                 'ylim'  :(-0.7,0.7), 
                 'aspect': 'equal'
                }

cycle = plt.get_cmap("tab10") 

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

def imshow_cbar(ax, im0,cbar_title=None,**kwargs):

    im = ax.imshow(im0,**kwargs)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)    
    cax = divider.append_axes("right", size="5%", pad='3%')
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')
    if cbar_title is not None: cbar.set_title(cbar_title)
    ax.set_aspect('equal')


def contourf_cbar(ax, im0,cbar_title=None,**kwargs):

    im = ax.imshow(im0,**kwargs)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)    
    cax = divider.append_axes("right", size="5%", pad='3%')
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')
    if cbar_title is not None: cbar.set_title(cbar_title)
    ax.set_aspect('equal')

    
def imshow_cbar_bottom(ax, im0,cbar_title=None,**kwargs):

    im = ax.imshow(im0,**kwargs)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)    
    cax = divider.append_axes("bottom", size="5%", pad='3%')
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    if cbar_title is not None: cbar.set_title(cbar_title)
    ax.set_aspect('equal')

    
def scatter_cbar(ax, x,y,c,cbar_title=None,**kwargs):
    im = ax.scatter(x=x,y=y,c=c,**kwargs)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right' , size="5%", pad='3%')
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')
    if cbar_title is not None: cbar.set_label(cbar_title)
    ax.set_aspect('equal')

    
def cmap_line(ax, x,y,C,cmap='viridis',cbar_title=None,**kwargs):
    
    norm = Normalize(vmin=y.min(), vmax=y.max())
    cmap = plt.get_cmap(cmap)

    for i,yi in enumerate(y):
        color = cmap(norm(yi))
        ax.plot(x, C[i,:], color=color,**kwargs)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad='3%')
    cbar = plt.colorbar(sm, cax=cax)
    if cbar_title is not None: cbar.set_label(cbar_title)





n0 = 1#25.99e16*0.8/2
a  = 1.348
b  = 0.5
rmax = 0.4577


def func_ring(r,z,n0=n0,a=a,b=b,rmax=rmax,separatrix=False):
    psi = rt1plotpy.mag.psi(r,z,separatrix)
    br, bz = rt1plotpy.mag.bvec(r,z,separatrix)
    b_abs = np.sqrt(br**2+bz**2)
    psi_rmax = rt1plotpy.mag.psi(rmax,0,separatrix)
    psi0 = rt1plotpy.mag.psi(1,0,separatrix)
    b0 = rt1plotpy.mag.b0(r,z,separatrix)
    b = b_abs/b0

    return n0 * np.exp(- (np.sqrt(((psi-psi_rmax)/psi0)**2+(1-1/b)**2)-0.5)**2*100)*(1-np.exp(-100*(r-1)**2))
    #return n0 *  (np.sqrt(((psi-psi_rmax)/psi0)**2+(1-1/b)**2)-0.5)*a / (a +   (np.sqrt(((psi-psi_rmax)/psi0)**2+(1-1/b)**2)-0.5)**2)*(1-np.exp(-100*(r-1)**2))
