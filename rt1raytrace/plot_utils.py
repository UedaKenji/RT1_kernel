from fileinput import filename
import math
import os
import sys
from typing import Union,Tuple,List
from unicodedata import name

import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import numpy as np
import pandas as pd
from numpy.core.arrayprint import IntegerFormat
from numpy.core.fromnumeric import transpose
from PIL import Image
from scipy import misc, ndimage, optimize, signal
from scipy import special


__all__ = ['imshow_cbar','contourf_cbar']


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
        'axes.grid'        : False             , # make grid
        }       
        
plt.rcParams.update(**params)


def imshow_cbar(f:plt.Figure, ax, im0, title:str=None,**kwargs):

    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    im = ax.imshow(im0,**kwargs)
    ax.set_title(title)
    ax.set_aspect('equal')
    f.colorbar(im,cax=cax)



def contourf_cbar(f:plt.Figure, ax, im0, title:str=None,**kwargs):

    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    im = ax.contourf(im0,**kwargs)
    ax.set_title(title)
    ax.set_aspect('equal')
    f.colorbar(im,cax=cax)
