from pkgutil import extend_path
import matplotlib.pyplot as plt
import numpy as np
from numpy import FPE_DIVIDEBYZERO, array, linalg, ndarray
import rt1plotpy
from typing import Optional, Union,Tuple,Callable,List
import time 
import math
from tqdm import tqdm
import scipy.linalg as linalg
from numba import jit
import warnings
from dataclasses import dataclass
import itertools
import scipy.sparse as sparse
import pandas as pd
import os
from .plot_utils import *  

__all__ = ['Kernel2D_scatter', 'Kernel2D_grid', 'Kernel1D']

class Kernel2D_scatter(rt1plotpy.frame.Frame):
    def __init__(self,
        dxf_file  :str,
        show_print:bool=False
        ) -> None:
        
        """
        import dxf file

        Parameters
        ----------
        dxf_file : str
            Path of the desired file.
        show_print : bool=True,
            print property of frames
        Note
        ----
        dxf_file is required to have units of (mm).
        """
        super().__init__(dxf_file,show_print)
        self.im_shape: Union[Tuple[int,int],None] = None
        print('you have to "create_induced_point()" or "set_induced_point()" next.')

    def create_induced_point(self,
        z_grid: np.ndarray,
        r_grid: np.ndarray,
        length_sq_fuction: Callable[[float,float],None],
        ) -> Tuple[np.ndarray,np.ndarray]:     
        """
        create induced point based on length scale function

        Parameters
        ----------
        z_grid: np.ndarray,
        r_grid: np.ndarray,
        length_sq_fuction: Callable[[float,float],None],

        Reuturns
        ----------
        zI: np.ndarray,
        rI: np.ndarray,  
        """
        
        if not 'r_bound'  in dir(self):
            print('set_bound() is to be done in advance!')
            return
        
        rr,zz = np.meshgrid(r_grid,z_grid)
        length_sq = length_sq_fuction(rr,zz)
        mask, _ = self.grid_input(R=r_grid, Z=z_grid)
        mask = (np.nan_to_num(mask) == 1)

        rI, zI = np.zeros(1),np.zeros(1)
        rI[0], zI[0] = r_grid[0],z_grid[0]
        is_short = True
        for i, zi in enumerate(tqdm(z_grid)):
            for j, ri in enumerate(r_grid):
                if mask[i,j]:
                    db_min = d2min(ri,zi,self.r_bound, self.z_bound)

                    if rI.size < 500:
                        d2_min = d2min(ri,zi,rI,zI)
                    else:
                        d2_min = d2min(ri,zi,rI[-500:],zI[-500:])

                    if length_sq[i,j] > min(db_min,d2_min):
                        is_short = True
                    elif is_short:
                        is_short = False
                        rI = np.append(rI,ri)
                        zI = np.append(zI,zi)                    

        rI,zI = rI[1:], zI[1:]

        self.zI, self.rI = zI, rI
        self.nI = rI.size
        self.length_scale_sq: Callable[[float,float],float] = length_sq_fuction 
        print('num of induced point is ',self.nI)
        return zI, rI

    def grid_input(self, 
        R: np.ndarray, 
        Z: np.ndarray, 
        fill_point: Tuple[float, float] = (0.5,0), 
        fill_point_2nd: Optional[Tuple[float, float]] = None, 
        isnt_print: bool = False
        ) -> Tuple[np.ndarray, dict]:
        mask,extent = self.__grid_input(R, Z, fill_point, fill_point_2nd, isnt_print)

        return mask, {"origin":"lower","extent":extent}

    def set_bound_grid(self,r,z):
        self.grid_input(r,z,isnt_print=True)
        r_grid,z_grid=np.meshgrid(r,z,indexing='xy')
        self.r_bound = r_grid[self.Is_bound]
        self.z_bound = z_grid[self.Is_bound]
        self.nb = self.z_bound.size

        rbrb = np.meshgrid(self.r_bound,self.r_bound)
        zbzb = np.meshgrid(self.z_bound,self.z_bound)
        self.rb_tau2 = (rbrb[0]-rbrb[1])**2
        self.zb_tau2 = (zbzb[0]-zbzb[1])**2
        print('num of bound point is ',self.nb)

    
    def save_inducing_point(self,
        name:str,
        is_plot:bool=False,
        figsize:tuple= (10,5)
        ):
        np.savez(file=name,
                 rI=self.rI,
                 zI=self.zI)

        if is_plot:
            fig,ax = plt.subplots(1,2,figsize=figsize)
            ax_kwargs = {'xlim'  :(0,1.1),
                        'ylim'  :(-0.7,0.7), 
                        'aspect': 'equal'
                            }
            ax[1].set(**ax_kwargs)
            ax[0].set(**ax_kwargs)
            self.append_frame(ax[0])
            self.append_frame(ax[1])
                        
            r_plot = np.linspace(0.05,1.05,500)
            z_plot = np.linspace(-0.7,0.7,500)
            R,Z = np.meshgrid(r_plot,z_plot)
            mask, im_kwargs = self.grid_input(R=r_plot, Z=z_plot)

            LS = self.length_scale(R,Z)

            contourf_cbar(fig,ax[0],LS*mask,cmap='turbo',**im_kwargs)

            self.set_bound_space(delta_l=20e-3)
            ax[0].set_title('Length scale distribution',size=15)
            ax[1].scatter(self.rI,self.zI,s=1,label='inducing_point')
                
            title = 'Inducing ponit: '+ str(self.nI)

            ax[1].set_title(title,size=15)
            ax[1].legend(fontsize=12)

            fig.suptitle(name)
            fig.savefig(name+'.png')

            
    def save_point(self,
        name:str,
        is_plot:bool=False,
        figsize:tuple= (10,5)
        ):
        np.savez(file=name,
                 zI=self.zI,
                 rI=self.rI,
                 rb=self.r_bound,
                 zb=self.z_bound)
        print('inducing points: '+str(self.nI)+' and boundary points: '+str(self.nb)+' are correctly saved at '+name)

        if is_plot:
            fig,ax = plt.subplots(1,2,figsize=figsize)
            ax_kwargs = {'xlim'  :(0,1.1),
                        'ylim'  :(-0.7,0.7), 
                        'aspect': 'equal'
                            }
            ax[1].set(**ax_kwargs)
            ax[0].set(**ax_kwargs)
            self.append_frame(ax[0])
            self.append_frame(ax[1])
                        
            r_plot = np.linspace(0.05,1.05,500)
            z_plot = np.linspace(-0.7,0.7,500)
            R,Z = np.meshgrid(r_plot,z_plot)
            mask, im_kwargs = self.grid_input(R=r_plot, Z=z_plot)

            LS = self.length_scale(R,Z)

            contourf_cbar(fig,ax[0],LS*mask,cmap='turbo',**im_kwargs)

            self.set_bound_space(delta_l=20e-3)

            ax[0].set_title('Length scale distribution',size=15)
                
            ax[1].scatter(self.rI,self.zI,s=1,label='inducing_point')
            title = 'Inducing ponit: '+ str(self.nI)
            if 'r_bound'  in dir(self):
                ax[1].scatter(self.r_bound, self.z_bound,s=1,label='boundary_point')
                title += '\nBoundary ponit: '+ str(self.nb)

            ax[1].set_title(title,size=15)
            ax[1].legend(fontsize=12)

            fig.suptitle(name)
            fig.savefig(name+'.png')

    def load_point(self,
        rI: np.ndarray,
        zI: np.ndarray,
        rb: np.ndarray,
        zb: np.ndarray,
        length_sq_fuction: Callable[[float,float],float],
        is_plot: bool = False,
        ) :  
        """
        set induced point by input existing data

        Parameters
        ----------
        zI: np.ndarray,
        rI: np.ndarray,
        length_sq_fuction: Callable[[float,float],None]
        """
        self.zI, self.rI = zI, rI
        self.z_bound, self.r_bound = zb, rb
        self.nI = rI.size
        self.nb = rb.size
        self.length_scale_sq: Callable[[float,float],float] = length_sq_fuction 
        if is_plot:
            fig,ax = plt.subplots(1,2,figsize=(10,5))
            ax_kwargs = {'xlim'  :(0,1.1),
                        'ylim'  :(-0.7,0.7), 
                        'aspect': 'equal'
                            }
            ax[1].set(**ax_kwargs)
            ax[0].set(**ax_kwargs)
            self.append_frame(ax[0])
            self.append_frame(ax[1])
                        
            r_plot = np.linspace(0.05,1.05,500)
            z_plot = np.linspace(-0.7,0.7,500)
            R,Z = np.meshgrid(r_plot,z_plot)
            mask, im_kwargs = self.grid_input(R=r_plot, Z=z_plot)

            LS = self.length_scale(R,Z)

            contourf_cbar(fig,ax[0],LS*mask,cmap='turbo',**im_kwargs)

            self.set_bound_space(delta_l=20e-3)

            ax[0].set_title('Length scale distribution',size=15)
                
            ax[1].scatter(self.rI,self.zI,s=1,label='inducing_point')
            title = 'Inducing ponit: '+ str(self.nI)
            if 'r_bound'  in dir(self):
                ax[1].scatter(self.r_bound, self.z_bound,s=1,label='boundary_point')
                title += '\nBoundary ponit: '+ str(self.nb)

            ax[1].set_title(title,size=15)
            ax[1].legend(fontsize=12)


    
    def set_bound_space(self,delta_l = 1e-2):
        """
        create induced point with equal space 

        Parameters
        ----------
        delta_l: space length [m] 

        Reuturns
        ----------
        """

        z_all, r_all = np.zeros(0),np.zeros(0)
        for entity in self.all_lines:
            r0,r1 = entity.start[0]/1000, entity.end[0]/1000 
            z0,z1 = entity.start[1]/1000, entity.end[1]/1000
            l = np.sqrt((z0-z1)**2 + (r0-r1)**2)
            n = int(l/delta_l) + 1 
            z = np.linspace(z0,z1,n)
            r = np.linspace(r0,r1,n)
            z_all = np.append(z_all,z)
            r_all = np.append(r_all,r)  

        for entity in self.all_arcs:
            angle = entity.end_angle- entity.start_angle
            angle = 360*( angle < 0 ) + angle 
            radius = entity.radius/1000 
            n = int(radius*angle/180*np.pi/delta_l) + 1
            #print(n,angle)
            theta = np.linspace(entity.start_angle,entity.start_angle+angle,n) / 180*np.pi
            r = entity.center[0]/1000 + radius*np.cos(theta)
            z = entity.center[1]/1000 + radius*np.sin(theta)
            z_all = np.append(z_all,z)
            r_all = np.append(r_all,r) 

        # 重複する点を除外する
        is_duplicate = np.zeros(z_all.size,dtype=np.bool8)
        for i in range(r_all.size-1):
            res = abs(z_all[i]-z_all[i+1:])+ abs(r_all[i]-r_all[i+1:])
            is_duplicate[i] = np.any(res < delta_l/100)

        r_all = r_all[~is_duplicate]
        z_all = z_all[~is_duplicate]


        self.r_bound = r_all
        self.z_bound = z_all 
        self.nb = self.z_bound.size

        
        print('num of bound point is ',self.nb)


    def set_induced_point(self,
        zI: np.ndarray,
        rI: np.ndarray,
        length_sq_fuction: Callable[[float,float],float],
        ) :     
        """
        set induced point by input existing data

        Parameters
        ----------
        zI: np.ndarray,
        rI: np.ndarray,
        length_sq_fuction: Callable[[float,float],None]
        """
        self.zI, self.rI = zI, rI
        self.nI = rI.size
        self.length_scale_sq: Callable[[float,float],float] = length_sq_fuction 
    
    def length_scale(self,r,z):
        return np.sqrt(self.length_scale_sq(r,z))
    
    def set_grid_interface(self,
        z_medium   : np.ndarray,
        r_medium   : np.ndarray,
        z_plot: np.ndarray=[None],
        r_plot: np.ndarray=[None],
        scale    : float = 1
        )  :
        
        if not 'rI'  in dir(self):
            print('set_induced_point() or create_induced_point() is to be done in advance')
            return
        
        s = scale
        lI = self.length_scale(self.rI,self.zI)
        KII = GibbsKer(x0=self.rI, x1=self.rI, y0=self.zI, y1=self.zI, lx0=lI*s, lx1=lI*s, isotropy=True)
        self.KII_inv = np.linalg.inv(KII+1e-5*np.eye(self.nI))
        
        self.mask_m,self.im_kwargs_m = self.grid_input(r_medium,z_medium,isnt_print=False)

        Z_medium,R_medium  = np.meshgrid(z_medium, r_medium, indexing='ij')

        lm = self.length_scale(R_medium.flatten(), Z_medium.flatten())
        lm = np.nan_to_num(lm,nan=1)
        self.Kps = GibbsKer(x0 = R_medium.flatten(),x1 = self.rI, y0 = Z_medium.flatten(), y1 =self.zI, lx0=lm*s, lx1=lI*s, isotropy=True)
        
        if z_plot[0] == None:
            return 
        else:
            dr, dz = r_medium[1]-r_medium[0],   z_medium[1]-z_medium[0]

            Kr1r1 = SEKer(x0=r_medium ,x1=r_medium, y0=0, y1=0, lx=dr, ly=1)
            Kz1z1 = SEKer(x0=z_medium ,x1=z_medium, y0=0, y1=0, lx=dz, ly=1)
            
            λ_r1, self.Q_r1 = np.linalg.eigh(Kr1r1)
            λ_z1, self.Q_z1 = np.linalg.eigh(Kz1z1)

            self.mask, self.im_kwargs = self.grid_input(r_plot,z_plot,isnt_print=False)

            self.KrHDr1 = SEKer(x0=r_plot,x1=r_medium, y0=0, y1=0, lx=dr, ly=1)
            self.KzHDz1 = SEKer(x0=z_plot,x1=z_medium, y0=0, y1=0, lx=dz, ly=1)

            self.Λ_z1r1_inv = 1 / np.einsum('i,j->ij',λ_z1,λ_r1)

            
    def convert_grid_media(self, fI:np.ndarray):
        f1 = self.Kps @ ( self.KII_inv @ fI)
        return f1.reshape(self.mask_m.shape), self.mask_m,self.im_kwargs_m

    
    def convert_grid(self, fI:np.ndarray) -> Tuple[np.ndarray,dict]:
        f1, _,_,  = self.convert_grid_media(fI)
        fHD = self.KzHDz1 @ (self.Q_z1 @ (self.Λ_z1r1_inv * (self.Q_z1.T @ f1 @ self.Q_r1)) @ self.Q_r1.T) @ self.KrHDr1.T
        return fHD, self.mask, self.im_kwargs
    
    def __grid_input(self, R: np.ndarray, Z: np.ndarray, fill_point: Tuple[float, float] = ..., fill_point_2nd: Optional[Tuple[float, float]] = None, isnt_print: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        return super().grid_input(R, Z, fill_point, fill_point_2nd, isnt_print)


@jit
def d2min(x,y,xs,ys):
    x_tau2 = (x- xs)**2
    y_tau2 = (y- ys)**2
    d2_min = np.min(x_tau2 + y_tau2)
    return d2_min

def SEKer(x0,x1,y0,y1,lx,ly):
    X = np.meshgrid(x0,x1,indexing='ij')
    Y = np.meshgrid(y0,y1,indexing='ij')
    return np.exp(- 0.5*( ((X[0]-X[1])/lx)**2 + ((Y[0]-Y[1])/ly)**2) )

def GibbsKer(
    x0 : np.ndarray,
    x1 : np.ndarray,
    y0 : np.ndarray,
    y1 : np.ndarray,
    lx0: np.ndarray,
    lx1: np.ndarray,
    ly0: Union[np.ndarray,bool]=None,
    ly1: Union[np.ndarray,bool]=None,
    isotropy: bool = False
    ) -> np.ndarray:  

    X  = np.meshgrid(x0,x1,indexing='ij')
    Y  = np.meshgrid(y0,y1,indexing='ij')
    Lx = np.meshgrid(lx0,lx1,indexing='ij')
    Lxsq = Lx[0]**2+Lx[1]**2 

    if isotropy:
        return 2*Lx[0]*Lx[1]/Lxsq *np.exp( -   ((X[0]-X[1])**2  +(Y[0]-Y[1])**2 )/ Lxsq )

    else:        
        Ly = np.meshgrid(ly0,ly1,indexing='ij')
        Lysq = Ly[0]**2+Ly[1]**2 
        return np.sqrt(2*Lx[0]*Lx[1]/Lxsq)*np.sqrt(2*Ly[0]*Ly[1]/Lysq)*np.exp( -   (X[0]-X[1])**2 / Lxsq  - (Y[0]-Y[1])**2 / Lysq )

@jit
def GibbsKer_fast(
    x0 : np.ndarray,
    x1 : np.ndarray,
    y0 : np.ndarray,
    y1 : np.ndarray,
    lx0: np.ndarray,
    lx1: np.ndarray,
    ) -> np.ndarray:  

    X  = np.meshgrid(x0,x1,indexing='ij')
    Y  = np.meshgrid(y0,y1,indexing='ij')
    Lx = np.meshgrid(lx0,lx1,indexing='ij')
    Lxsq = Lx[0]**2+Lx[1]**2 

    return 2*Lx[0]*Lx[1]/Lxsq *np.exp( -   ((X[0]-X[1])**2  +(Y[0]-Y[1])**2 )/ Lxsq )

class Kernel1D():
    pass 

class Kernel2D_grid():
    pass 