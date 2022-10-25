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
import os,sys
from .plot_utils import *  

sys.path.insert(0,os.pardir)
import rt1raytrace

__all__ = ['Kernel2D_scatter', 'Kernel2D_grid', 'Kernel1D','Observation_Matrix_integral']

@dataclass
class Observation_Matrix:
    H  : Union[np.ndarray,sparse.csr_matrix]
    ray: rt1raytrace.Ray
    
    def __post_init__(self):
        shape = self.H.shape
        self.H :sparse.csr_matrix = sparse.csr_matrix(self.H.reshape(shape[0]*shape[1],shape[2]))
        self.shape :tuple = shape
    
    def set_Direction(self,
        rI: np.ndarray
        ):
        Dcos = self.ray.Direction_Cos(R=rI)
        self.Exist: sparse.csr_matrix  = (self.H > 0)  
        self.Dcos = self.Exist.multiply(Dcos)
        pass 

    def projection_A(self,
        f :np.ndarray,
        t :np.ndarray,
        v :np.ndarray,
        reshape:bool=True
        ) -> np.ndarray:

        E :sparse.csr_matrix = (self.Exist@sparse.diags(np.log(f)-t)+1.j*self.Dcos@sparse.diags(v))
        E = E.expm1() + self.Exist
        A = np.array(self.H.multiply(E).sum(axis=1))

        if reshape:
            return A.reshape(self.shape[0:2])
        else:
            return A
    
    def Exp(self,
        a :np.ndarray,
        v :np.ndarray 
        ) -> sparse.csr_matrix:

        E :sparse.csr_matrix = (self.Exist@sparse.diags(a)+1.j*self.Dcos@sparse.diags(v))
        return  E.expm1() + self.Exist
        

    def projection(self,
        f :np.ndarray,
        reshape:bool=True
        ) -> np.ndarray: 

        g: np.ndarray = self.H @ f

        if reshape:
            return g.reshape(self.shape[0:2])
        else:
            return g

    def toarray(self):
        return self.H.toarray()

    def set_mask(self,mask:np.ndarray):
        if np.all(mask == True):
            return self.H
        H_d = self.toarray()
        H_d_masked = H_d[mask.flatten(),:]
        return sparse.csr_matrix(H_d_masked)


def cal_refractive_indices_metal2(
    cos_i: Union[np.ndarray,float], 
    n_R  : float = 1.7689, #for 400nm
    n_I  : float = 0.60521 #for 400nm
    ) -> Union[np.ndarray,float]:
    """""
    金属の反射率を計算する．
    :param cos_i: cos θ 
    :param n_R: 屈折率の実数部
    :param n_I: 屈折率の虚数部（消光係数）
    :return: s偏光の反射率（絶対値），p偏光の反射率（絶対値）
    """""
    sin_i = np.sqrt(1-cos_i**2)

    r_TE = (cos_i - np.sqrt((n_R**2 - n_I**2 - sin_i**2) + 2j*n_I*n_R))\
          /(cos_i + np.sqrt((n_R**2 - n_I**2 - sin_i**2) + 2j*n_I*n_R))
    r_TM = (-(n_R**2 - n_I**2 + 2j*n_R*n_I)*cos_i + np.sqrt((n_R**2 - n_I**2 - sin_i**2) + 2j*n_I*n_R))\
          /((n_R**2 - n_I**2 + 2j*n_R*n_I) *cos_i + np.sqrt((n_R**2 - n_I**2 - sin_i**2) + 2j*n_I*n_R))
    return (np.abs(r_TE)**2+ np.abs(r_TM)**2)/2
    
class Observation_Matrix_integral:

    indices = {'400nm':(1.7689 ,0.60521),
               '700nm':(0.70249,0.36890)}

    def load_model(path:str): 
        return pd.read_pickle(path)

    def save_model(self,path:str):
        self.path = path+'.pkl'
        self.abspath = os.path.abspath(self.path) 
        
        try:
            self.Hs_mask
            del self.Hs_mask
        except:
            pass 
        
        try:
            self.H_sum
            del self.H_sum
        except:
            pass 

        
        pd.to_pickle(self,path+'.pkl')
        

    def __init__(self,
        H_list: List[Observation_Matrix],
        ray0  : rt1raytrace.Ray,
        rI    : np.ndarray,
        zI    : np.ndarray) -> None:
        self.shape :tuple = H_list[0].shape 
        self.n  = len(H_list)
        self.mask = np.ones(self.shape[0:2],dtype=np.bool8)
        self.refs = [1.]*self.n
        self.ray_init = ray0
        self.Hs = H_list
        self.rI = rI 
        self.zI = zI 
        #self.is_mask = False
        pass

    def set_directon(self
        ):
        for H in self.Hs:
            H.set_Direction(rI=self.rI)
        pass 

    def set_mask(self,
        mask : Optional[np.ndarray] = None 
        ) -> None :
        #self.is_mask = True
        if not mask is None:
            self.mask = mask 
        
        self.Hs_mask: List[sparse.csr_matrix]  = [] 
        for H in self.Hs:
            self.Hs_mask.append(H.set_mask(self.mask))

    def set_uniform_ref(self,wave:str):
        n_R, n_I = self.indices[wave]
        for i in range(1,self.n):
            cos_factor = np.nan_to_num(self.Hs[i].ray.cos_factor)
            self.refs[i] = cal_refractive_indices_metal2(cos_factor,n_R,n_I)

    def set_fn_ref(self,fn: Callable[[np.ndarray,np.ndarray],Tuple[np.ndarray,np.ndarray]]):
        for i in range(1,self.n):
            Phi_ref = self.Hs[i].ray.Phi0
            Z_ref = self.Hs[i].ray.Z0
            n_R,n_I = fn(Z_ref,Phi_ref)
            cos_factor = np.nan_to_num(self.Hs[i].ray.cos_factor)
            self.refs[i] = cal_refractive_indices_metal2(cos_factor,n_R,n_I)
        pass
    
    def set_Hsum(self,
        mask: Optional[np.ndarray] = None 
        ) -> None :

        self.H_sum :sparse.csr_matrix = 0.

        if not mask is None:
            self.set_mask(mask)
        
        for i,H in enumerate(self.Hs_mask):
            ref = np.ones(H.shape[0])
            for j in range(i+1):
                ref_j = self.refs[j] * np.ones(self.shape[0:2])
                ref   = ref_j[self.mask] *ref
                
            ref = sparse.diags(ref)
            self.H_sum += ref @ H

    def projection(self,
        f :np.ndarray,
        reshape:bool=True
        ) -> np.ndarray: 

        if reshape:
            g = np.zeros(self.mask.shape)
            g[self.mask] = self.H_sum @ f
            return g
        else:        
            g = self.H_sum @ f
            return g


    def __call__(self):
        return self.H_sum
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
        boundary = 0, 
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


            ax[0].set_title('Length scale distribution',size=15)
                
            ax[1].scatter(self.rI,self.zI,s=1,label='inducing_point')
            title = 'Inducing ponit: '+ str(self.nI)
            if 'r_bound'  in dir(self):
                ax[1].scatter(self.r_bound, self.z_bound,s=1,label='boundary_point')
                title += '\nBoundary ponit: '+ str(self.nb)

            ax[1].set_title(title,size=15)
            ax[1].legend(fontsize=12)


    
    def set_bound_space(self,delta_l = 1e-2,is_change_local_variable=True):
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

        print('num of bound point is ',r_all.size)
        if is_change_local_variable:
            self.r_bound = r_all
            self.z_bound = z_all 
            self.nb = self.z_bound.size
        else:
            return r_all,z_all 

        


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
        scale    : float = 1,
        add_bound :bool=False,
        )  :
        
        if not 'rI'  in dir(self):
            print('set_induced_point() or create_induced_point() is to be done in advance')
            return
        
        s = scale
        Z_medium,R_medium  = np.meshgrid(z_medium, r_medium, indexing='ij')
        lm = self.length_scale(R_medium.flatten(), Z_medium.flatten())
        lm = np.nan_to_num(lm,nan=1)

        if add_bound:
            self.add_bound=True
            rIb = np.concatenate([self.rI,self.r_bound])
            zIb = np.concatenate([self.zI,self.z_bound])
            self.rIb,self.zIb=rIb,zIb
            lI = self.length_scale(rIb,zIb)
            KII = GibbsKer(x0=rIb, x1=rIb, y0=zIb, y1=zIb, lx0=lI*s, lx1=lI*s, isotropy=True)
            self.KII_inv = np.linalg.inv(KII+1e-5*np.eye(self.nI+self.nb))
            self.KpI = GibbsKer(x0 = R_medium.flatten(),x1 = rIb, y0 = Z_medium.flatten(), y1 =zIb, lx0=lm*s, lx1=lI*s, isotropy=True)        
        else:
            self.add_bound=False
            lI = self.length_scale(self.rI,self.zI)
            KII = GibbsKer(x0=self.rI, x1=self.rI, y0=self.zI, y1=self.zI, lx0=lI*s, lx1=lI*s, isotropy=True)
            self.KII_inv = np.linalg.inv(KII+1e-5*np.eye(self.nI))
            self.KpI = GibbsKer(x0 = R_medium.flatten(),x1 = self.rI, y0 = Z_medium.flatten(), y1 =self.zI, lx0=lm*s, lx1=lI*s, isotropy=True)
            
        
        self.mask_m,self.im_kwargs_m = self.grid_input(r_medium,z_medium,isnt_print=False)


        
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

            
    def convert_grid_media(self,
        fI:np.ndarray,
        boundary:float=0
        ):
        if self.add_bound:
            fI = np.concatenate([fI,boundary*np.ones(self.nb)])
        f1 = self.KpI @ ( self.KII_inv @ fI)
        return f1.reshape(self.mask_m.shape), self.mask_m,self.im_kwargs_m

    
    def convert_grid(self, 
        fI:np.ndarray,
        boundary:float=0,
        ) -> Tuple[np.ndarray,np.ndarray,dict]:
        f1, _,_,  = self.convert_grid_media(fI,boundary)
        fHD = self.KzHDz1 @ (self.Q_z1 @ (self.Λ_z1r1_inv * (self.Q_z1.T @ f1 @ self.Q_r1)) @ self.Q_r1.T) @ self.KrHDr1.T
        return fHD, self.mask, self.im_kwargs
    
    def __grid_input(self, R: np.ndarray, Z: np.ndarray, fill_point: Tuple[float, float] = ..., fill_point_2nd: Optional[Tuple[float, float]] = None, isnt_print: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        return super().grid_input(R, Z, fill_point, fill_point_2nd, isnt_print)

        
    def create_observation_matrix(self,
        ray  : rt1raytrace.Ray,
        Lnum : int=100
        ) -> np.ndarray:
        self.im_shape = ray.shape

        H = np.zeros((ray.shape[0], ray.shape[1], self.rI.size))

        Rray, Zray = ray.RZ_ray(Lnum=Lnum+1)
        dL = ray.Length / float(Lnum)
        
        Zray =0.5*(Zray[1:,:,:] + Zray[:-1,:,:])
        Rray =0.5*(Rray[1:,:,:] + Rray[:-1,:,:])

        lI = self.length_scale_sq(self.rI, self.zI)

        for i  in tqdm(range(ray.shape[0])):
            for j in range(ray.shape[1]):

                R    = Rray[:,i,j]
                Z    = Zray[:,i,j]
                dL2  = dL[i,j]
                l_ray = np.sqrt(self.length_scale_sq(R,Z))
                Krs =  GibbsKer(x0=R, x1=self.rI, y0=Z, y1=self.zI, lx0=l_ray*0.5, lx1=lI*0.5,isotropy=True)


                Krs_sum_inv   = 1/Krs.sum(axis=1)

                H[i,j,:] = np.einsum('i,ij->j', dL2*Krs_sum_inv, Krs ) 
        
        H[H < 1e-5] = 0

        return Observation_Matrix(H=H, ray=ray)

    def set_kernel(self,
        length_scale:float=1,
        is_bound :bool=True ,
        bound_value : float=0,
        bound_sig : float = 0.1,
        bound_space : float = 1e-2,
        ):
        ls = length_scale
        lI = ls*self.length_scale(self.rI,self.zI)
        rb,zb = self.set_bound_space(delta_l=bound_space,is_change_local_variable=False)
        lb = ls*self.length_scale(rb,zb)
        KII = GibbsKer(x0=self.rI     , x1=self.rI     , y0=self.zI     , y1=self.zI     , lx0=lI*ls, lx1=lI*ls, isotropy=True)
        if not is_bound: return KII 

        KIb = GibbsKer(x0=self.rI, x1=rb, y0=self.zI, y1=zb, lx0=lI*ls, lx1=lb*ls, isotropy=True)
        Kbb = GibbsKer(x0=rb     , x1=rb, y0=zb     , y1=zb, lx0=lb*ls, lx1=lb*ls, isotropy=True)
        Kbb+= bound_sig**2*np.eye(rb.size)

        Kb = KII - KIb @ np.linalg.inv(Kbb) @ KIb.T
        fpri  = KIb @ (np.linalg.inv(Kbb) @ (bound_value*np.ones(rb.size)))
        return Kb,fpri

        


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