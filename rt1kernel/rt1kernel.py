from pkgutil import extend_path
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numpy import FPE_DIVIDEBYZERO, array, linalg, ndarray
import rt1plotpy
from typing import Any, Optional, Union,Tuple,Callable, TypeVar,cast,List
import time 
import math
from tqdm import tqdm
import scipy.linalg as linalg
from numba import njit
import warnings
from dataclasses import dataclass
import itertools
import scipy.sparse as sparse
import pandas as pd
import os,sys
from .plot_utils import *  

try:
    from .. import rt1raytrace
except:
    sys.path.insert(0,os.path.join(os.path.dirname(__file__),os.pardir))
    import rt1raytrace
    sys.path.pop(0)
#sys.path.insert(0,os.pardir)



#sys.path.insert(0,os.path.join(os.path.dirname(__file__),os.pardir))
#import rt1raytrace

__all__ = ['Kernel2D_scatter',
           'Kernel1D',
           'Kernel2D_grid',
           'GibbsKer',
           'Observation_Matrix_integral',
           'Observation_Matrix',
           'Observation_Matrix_integral_load_model']

float_numpy = TypeVar(" float | NDArray[float64] ",float,npt.NDArray[np.float64])#type: ignore # 怒られているけど、実行する気エラーにならないので無視する。


def const_like(x:float, type_x:float_numpy)->float_numpy:
    return cast(float_numpy, x + 0*type_x)

def ones_like(type_x:float_numpy)->float_numpy:
    return cast(float_numpy, 1.0*type_x)

def zeros_like(type_x:float_numpy)->float_numpy:
    return cast(float_numpy, 0.0*type_x)


@dataclass
class Observation_Matrix:
    H  : sparse.csr_matrix|npt.NDArray[np.float64]
    ray: rt1raytrace.Ray
    
    def __post_init__(self):
        shape = self.H.shape
        self.H = sparse.csr_matrix(self.H.reshape(shape[0]*shape[1],shape[2]))
        self.shape :tuple = shape

    def __call__(self
        ) -> Any:
        return self.H
    
    def set_Direction(self,
            rI: npt.NDArray[np.float64],
            mask: npt.NDArray[np.bool_]|None=None,
        ):
        Dcos = self.ray.Direction_Cos(R=rI)
        if mask is not None:
            H = (self.H).toarray()  # type: ignore
            H = H[~mask.flatten(),:] 
            Dcos = Dcos[~mask.flatten(),:]
        else:
            H = self.H
            pass 
        self.Exist = sparse.csr_matrix(H > 0)  
        self.Dcos  :sparse.csr_matrix = self.Exist.multiply(Dcos)
        pass 

    def projection_A(self,
        f :npt.NDArray[np.float64],
        t :npt.NDArray[np.float64],
        v :npt.NDArray[np.float64],
        reshape:bool=True
        ) -> npt.NDArray[np.float64]:

        self.H = cast(sparse.csr_matrix,self.H)
        E :sparse.csr_matrix = (self.Exist@sparse.diags(np.log(f)-t)+1.j*self.Dcos@sparse.diags(v))
        E = E.expm1() + self.Exist
        A = np.array(self.H.multiply(E).sum(axis=1))

        if reshape:
            return A.reshape(self.shape[0:2])
        else:
            return A

    def projection_A2(self,
        a :npt.NDArray[np.float64],
        v :npt.NDArray[np.float64],
        reshape:bool=True
        ) -> npt.NDArray[np.float64]:

        self.H = cast(sparse.csr_matrix,self.H)
        E :sparse.csr_matrix = (self.Exist@sparse.diags(a)+1.j*self.Dcos@sparse.diags(v))
        E = E.expm1() + self.Exist

        A = np.array(self.H.multiply(E).sum(axis=1))

        if reshape:
            return A.reshape(self.shape[0:2])
        else:
            return A
    
    def Exp(self,
        a :npt.NDArray[np.float64],
        v :npt.NDArray[np.float64] 
        ) -> sparse.csr_matrix:

        E :sparse.csr_matrix = (self.Exist@sparse.diags(a)+1.j*self.Dcos@sparse.diags(v))
        return  E.expm1() + self.Exist

    def projection(self,
        f :npt.NDArray[np.float64],
        reshape:bool=True
        ) -> npt.NDArray[np.float64]: 

        g: npt.NDArray[np.float64] = self.H @ f

        if reshape:
            return g.reshape(self.shape[0:2])
        else:
            return g

    def toarray(self):
        self.H = cast(sparse.csr_matrix,self.H)
        return self.H.toarray()

    def set_mask(self,
        mask : npt.NDArray[np.bool_]
        )->sparse.csr_matrix:
        
        if np.all(mask == False):
            return sparse.csr_matrix(self.H)
        
        H_d = self.toarray()
        H_d_masked = H_d[~mask.flatten(),:]
        return sparse.csr_matrix(H_d_masked)
    
    def __matmul__(self,
        f: npt.NDArray[np.float64]
        )-> npt.NDArray[np.float64]:
        
        return (self.H @ f).reshape(*self.ray.shape)
            


def cal_refractive_indices_metal2(
    cos_i: float_numpy, 
    n_R  : float = 1.7689, #for 400nm
    n_I  : float = 0.60521 #for 400nm
    ) -> float_numpy:
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
    return (np.abs(r_TE)**2+ np.abs(r_TM)**2)/2 # type: ignore

from typing import TypeAlias


class Observation_Matrix_integral:

    indices = {'400nm':(1.7689 ,0.60521),
               '700nm':(0.70249,0.36890)}

    def load_model(path:str):  #type: ignore   #selfがないことを怒られた.
        
        return pd.read_pickle(path)

    def save_model(self,name:str,path:str=''):
        self.path = path+'.pkl'
        self.abspath = os.path.abspath(self.path) 
        class_name = type(self).__name__
        
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

        
        pd.to_pickle(self,path+class_name+'_'+name+'.pkl')
        

    def __init__(self,
        H_list: List[Observation_Matrix],
        ray0  : rt1raytrace.Ray,
        rI    : npt.NDArray[np.float64],
        zI    : npt.NDArray[np.float64]) -> None:
        self.shape :tuple = H_list[0].shape 
        self.n  = len(H_list)
        self.mask = np.zeros(self.shape[0:2],dtype=np.bool_)
        self.refs :list[npt.NDArray[np.float64]] = (self.n-1)*[np.empty(0)]
        self.ray_init = ray0
        self.Hs = H_list
        self.rI = rI 
        self.zI = zI 
        #self.is_mask = False
        pass

    def set_directon(self
        ):
        for H in self.Hs:
            H.set_Direction(rI=self.rI,mask=self.mask)
        pass 

    def set_mask(self,
        mask : Optional[npt.NDArray[np.bool_]] = None 
        ) -> None :
        #self.is_mask = True
        if not mask is None:
            self.mask = mask 
        
        self.Hs_mask: list[sparse.csr_matrix]  = [] 
        for H in self.Hs:
            self.Hs_mask.append(H.set_mask(self.mask))

    def set_uniform_ref(self,wave:str):
        n_R, n_I = self.indices[wave]
        for i in range(self.n-1):
            cos_factor = np.nan_to_num(self.Hs[i+1].ray.cos_factor)
            self.refs[i] = cal_refractive_indices_metal2(cos_factor,n_R,n_I)

    def set_fn_ref(self,fn: Callable):
        for i in range(self.n-1):
            Phi_ref = self.Hs[i+1].ray.Phi0
            Z_ref = self.Hs[i+1].ray.Z0
            n_R,n_I = fn(Z_ref,Phi_ref)
            cos_factor = np.nan_to_num(self.Hs[i+1].ray.cos_factor)
            self.refs[i] = cal_refractive_indices_metal2(cos_factor,n_R,n_I)
        pass
    
    def set_Hsum(self,
        mask: Optional[npt.NDArray[np.bool_]] = None 
        ) -> None :

        self.H_sum = 0

        if not mask is None:
            self.set_mask(mask)
        
        for i,H in enumerate(self.Hs_mask):
            ref = np.ones(H.shape[0])
            for j in range(i):
                ref_j = self.refs[j] * np.ones(self.shape[0:2])
                ref   = ref_j[~self.mask] *ref
                
            ref = sparse.diags(ref)
            self.H_sum += ref @ H
        
        self.H_sum  = sparse.csr_matrix(self.H_sum)

    def projection(self,
        f :npt.NDArray[np.float64],
        reshape:bool=True
        ) -> npt.NDArray[np.float64]: 

        if reshape:
            g = np.zeros(self.mask.shape)
            g[~self.mask] = self.H_sum @ f
            return g
        else:        
            g = self.H_sum @ f
            return g

    def __call__(self):
        return self.H_sum

def Observation_Matrix_integral_load_model(
    path:str
    ) -> Observation_Matrix_integral:
    return pd.read_pickle(path)

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
        #self.im_shape: Union[Tuple[int,int],None] = None
        self.V = None

        print('you have to "create_induced_point()" or "set_induced_point()" next.')

    def create_induced_point(self,
        z_grid: npt.NDArray[np.float64],
        r_grid: npt.NDArray[np.float64],
        length_sq_fuction: Callable[[npt.NDArray[np.float64],npt.NDArray[np.float64]],npt.NDArray[np.float64]],
        boundary = 0, 
        ) -> Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]] | None:     
        """
        create induced point based on length scale function

        Parameters
        ----------
        z_grid: npt.NDArray[np.float64],
        r_grid: npt.NDArray[np.float64],
        length_sq_fuction: Callable[[float,float],None],

        Reuturns
        ----------
        zI: npt.NDArray[np.float64],
        rI: npt.NDArray[np.float64],  
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

        rI,zI = cast(npt.NDArray[np.float64],rI[1:]), cast(npt.NDArray[np.float64],zI[1:])

        self.zI, self.rI = zI, rI
        self.nI = rI.size

        self.length_scale_sq: Callable[[npt.NDArray[np.float64],npt.NDArray[np.float64]],npt.NDArray[np.float64]]= length_sq_fuction 
        self.Lsq_I = self.length_scale_sq(self.rI,self.zI) 
        print('num of induced point is ',self.nI)
        return zI, rI

    def grid_input(self, 
        R: npt.NDArray[np.float64], 
        Z: npt.NDArray[np.float64], 
        fill_point: Tuple[float, float] = (0.5,0), 
        fill_point_2nd: Optional[Tuple[float, float]] = None, 
        isnt_print: bool = False
        ) -> Tuple[npt.NDArray[np.float64], dict]:
        mask,extent = self.__grid_input(R, Z, fill_point, fill_point_2nd, isnt_print)
        """
        this functions is to return 'mask' and 'imshow_kwargs' np.array for imshow plottting

        Parameters
        ----------
        R: npt.NDArray[np.float64],
            array of R axis with 1dim
        Z: npt.NDArray[np.float64],
            array of Z axis with 1dim

        fill_point: Tuple[float,float] = (0.5,0), optional,
        fill_point_2nd: Optional[Tuple[float,float]] = None, optional

        Reuturns
        ----------
        mask:
        imshow_kwargs:  {"origin":"lower","extent":extent}
        """
        return mask, {"origin":"lower","extent":extent}

    def set_bound_grid(self,r,z):
        #self.grid_input(r,z,isnt_print=True)
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
            fig = plt.figure(figsize=(10,5))
            axs = fig.subplots(1,2)
            axs:list[plt.Axes] = np.array(axs).tolist()
            ax_kwargs = {'xlim'  :(0,1.1),
                        'ylim'  :(-0.7,0.7), 
                        'aspect': 'equal'
                            }
            axs[1].set(**ax_kwargs)
            axs[0].set(**ax_kwargs)
            self.append_frame(axs[0])
            self.append_frame(axs[1])
                        
            r_plot = np.linspace(0.05,1.05,500)
            z_plot = np.linspace(-0.7,0.7,500)
            R,Z = np.meshgrid(r_plot,z_plot)
            mask, im_kwargs = self.grid_input(R=r_plot, Z=z_plot)

            LS = self.length_scale(R,Z)

            contourf_cbar(axs[0],LS*mask,cmap='turbo',**im_kwargs)

            axs[0].set_title('Length scale distribution',size=15)
            axs[1].scatter(self.rI,self.zI,s=1,label='inducing_point')
                
            title = 'Inducing ponit: '+ str(self.nI)

            axs[1].set_title(title,size=15)
            axs[1].legend(fontsize=12)

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
            fig = plt.figure(figsize=figsize)
            axs = fig.subplots(1,2)
            axs:list[plt.Axes] = np.array(axs).tolist()
            ax_kwargs = {'xlim'  :(0,1.1),
                        'ylim'  :(-0.7,0.7), 
                        'aspect': 'equal'
                            }
            axs[1].set(**ax_kwargs)
            axs[0].set(**ax_kwargs)
            self.append_frame(axs[0])
            self.append_frame(axs[1])
                        
            r_plot = np.linspace(0.05,1.05,500)
            z_plot = np.linspace(-0.7,0.7,500)
            R,Z = np.meshgrid(r_plot,z_plot)
            mask, im_kwargs = self.grid_input(R=r_plot, Z=z_plot)

            LS = self.length_scale(R,Z)


            contourf_cbar(axs[0],LS*mask,cmap='turbo',**im_kwargs)  

            axs[0].set_title('Length scale distribution',size=15)
                
            axs[1].scatter(self.rI,self.zI,s=1,label='inducing_point')
            title = 'Inducing ponit: '+ str(self.nI)
            if 'r_bound'  in dir(self):
                axs[1].scatter(self.r_bound, self.z_bound,s=1,label='boundary_point')
                title += '\nBoundary ponit: '+ str(self.nb)

            axs[1].set_title(title,size=15)
            axs[1].legend(fontsize=12)

            fig.suptitle(name)
            fig.savefig(name+'.png')

    def load_point(self,
            rI: npt.NDArray[np.float64],
            zI: npt.NDArray[np.float64],
            rb: npt.NDArray[np.float64],
            zb: npt.NDArray[np.float64],
            length_sq_fuction: Callable[[npt.NDArray[np.float64],npt.NDArray[np.float64]],npt.NDArray[np.float64]],
            is_plot: bool = False,
        ) :  
        """
        set induced point by input existing data

        Parameters
        ----------
        zI: npt.NDArray[np.float64],
        rI: npt.NDArray[np.float64],
        length_sq_fuction: Callable[[float,float],None]
        """
        self.zI, self.rI = zI, rI
        self.z_bound, self.r_bound = zb, rb
        self.nI = rI.size
        self.nb = rb.size
        self.length_scale_sq: Callable[[npt.NDArray[np.float64],npt.NDArray[np.float64]],npt.NDArray[np.float64]] = length_sq_fuction 
        self.Lsq_I = self.length_scale_sq(self.rI,self.zI) 
        if is_plot:
            fig = plt.figure(figsize=(10,5))
            axs = fig.subplots(1,2)
            axs:list[plt.Axes] = np.array(axs).tolist()

            ax_kwargs = {'xlim'  :(0,1.1),
                        'ylim'  :(-0.7,0.7), 
                        'aspect': 'equal'
                            }
            axs[1].set(**ax_kwargs)
            axs[0].set(**ax_kwargs)
            self.append_frame(axs[0])
            self.append_frame(axs[1])
                        
            r_plot = np.linspace(0.05,1.05,500)
            z_plot = np.linspace(-0.7,0.7,500)
            R,Z = np.meshgrid(r_plot,z_plot)
            mask, im_kwargs = self.grid_input(R=r_plot, Z=z_plot)

            LS = self.length_scale(R,Z)
            contourf_cbar(axs[0],LS*mask,cmap='turbo',**im_kwargs)


            axs[0].set_title('Length scale distribution',size=15)
                
            axs[1].scatter(self.rI,self.zI,s=1,label='inducing_point')
            title = 'Inducing ponit: '+ str(self.nI)
            if 'r_bound'  in dir(self):
                axs[1].scatter(self.r_bound, self.z_bound,s=1,label='boundary_point')
                title += '\nBoundary ponit: '+ str(self.nb)

            axs[1].set_title(title,size=15)
            axs[1].legend(fontsize=12)


    
    def set_bound_space(self,
        delta_l = 1e-2,
        is_change_local_variable
        :bool=True
        ) -> tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]] :
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
        is_duplicate = np.zeros(z_all.size,dtype=np.bool_)
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
            return r_all,z_all 

        else:
            return r_all,z_all 

        


    def set_induced_point(self,
            zI: npt.NDArray[np.float64],
            rI: npt.NDArray[np.float64],
            length_sq_fuction: Callable[[npt.NDArray[np.float64],npt.NDArray[np.float64]],npt.NDArray[np.float64]],
        ) :     
        """
        set induced point by input existing data

        Parameters
        ----------
        zI: npt.NDArray[np.float64],
        rI: npt.NDArray[np.float64],
        length_sq_fuction: Callable[[float,float],None]
        """
        self.zI, self.rI = zI, rI
        self.nI = rI.size
        self.length_scale_sq: Callable[[npt.NDArray[np.float64],npt.NDArray[np.float64]],npt.NDArray[np.float64]] = length_sq_fuction 
        self.Lsq_I = self.length_scale_sq(self.rI,self.zI) 
    
    def length_scale(self,r,z):
        return np.sqrt(self.length_scale_sq(r,z))
    
    def set_grid_interface(self,
            z_medium   : npt.NDArray[np.float64],
            r_medium   : npt.NDArray[np.float64],
            z_plot: npt.NDArray[np.float64] | None = None,
            r_plot: npt.NDArray[np.float64] | None = None,
            scale    : float = 1,
            add_bound :bool=False,
        ) :
        
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


        
        if z_plot is None or r_plot is None:
            return 
        else:
            self.r_plot,self.z_plot = r_plot,z_plot
            dr, dz = r_medium[1]-r_medium[0],   z_medium[1]-z_medium[0]

            Kr1r1 = SEKer(x0=r_medium ,x1=r_medium, y0=0., y1=0., lx=dr, ly=1)
            Kz1z1 = SEKer(x0=z_medium ,x1=z_medium, y0=0., y1=0., lx=dz, ly=1)
            
            λ_r1, self.Q_r1 = np.linalg.eigh(Kr1r1)
            λ_z1, self.Q_z1 = np.linalg.eigh(Kz1z1)

            self.mask, self.im_kwargs = self.grid_input(np.array(r_plot),np.array(z_plot),isnt_print=False)

            self.KrHDr1 = SEKer(x0=r_plot,x1=r_medium, y0=0, y1=0, lx=dr, ly=1)
            self.KzHDz1 = SEKer(x0=z_plot,x1=z_medium, y0=0, y1=0, lx=dz, ly=1)

            self.Λ_z1r1_inv = 1 / np.einsum('i,j->ij',λ_z1,λ_r1)

    def convert(self, 
        r:npt.NDArray[np.float64],
        z:npt.NDArray[np.float64],
        fI:npt.NDArray[np.float64],
        boundary:float=0,
        ) -> npt.NDArray[np.float64] : 
        
        if self.add_bound:
            fI = np.concatenate([fI,boundary*np.ones(self.nb)])
            
            rIb = np.concatenate([self.rI,self.r_bound])
            zIb = np.concatenate([self.zI,self.z_bound])
            lI = self.length_scale(rIb,zIb)

            self.rIb,self.zIb=rIb,zIb
            lI = self.length_scale(rIb,zIb)
            s = 1
            lm = self.length_scale(r.flatten(), z.flatten())
            phi = GibbsKer(x0 = r.flatten(),x1 = rIb, y0 = z.flatten(), y1 =zIb, lx0=lm*s, lx1=lI*s, isotropy=True)
    
        return phi@ (self.KII_inv @ fI), self.KII_inv @ fI
        
    def convert_grid_media(self,
        fI:npt.NDArray[np.float64],
        boundary:float=0
        ):
        if self.add_bound:
            fI = np.concatenate([fI,boundary*np.ones(self.nb)])
        f1 = self.KpI @ ( self.KII_inv @ fI)
        return f1.reshape(self.mask_m.shape), self.mask_m,self.im_kwargs_m

    
    def convert_grid(self, 
        fI:npt.NDArray[np.float64],
        boundary:float=0,
        ) -> Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64],dict]:
        f1, _,_,  = self.convert_grid_media(fI,boundary)
        fHD = self.KzHDz1 @ (self.Q_z1 @ (self.Λ_z1r1_inv * (self.Q_z1.T @ f1 @ self.Q_r1)) @ self.Q_r1.T) @ self.KrHDr1.T
        return fHD, self.mask, self.im_kwargs
    
    def __grid_input(self, R: npt.NDArray[np.float64], Z: npt.NDArray[np.float64], fill_point: Tuple[float, float] = ..., fill_point_2nd: Optional[Tuple[float, float]] = None, isnt_print: bool = False
        ) -> Tuple[npt.NDArray[np.float64], tuple]:
        return super().grid_input(R, Z, fill_point, fill_point_2nd, isnt_print)

        
    def create_observation_matrix(self,
        ray  : rt1raytrace.Ray,
        Lnum : int=100
        ) ->  Observation_Matrix:
        self.im_shape:tuple = ray.shape

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
    
    def create_obs_matrix_kernel(self,
            ray  : rt1raytrace.Ray,
            Lnum : int=100
        ):

        rIb,zIb = self.rIb,self.zIb
        lIb = self.length_scale(rIb,zIb)
        def phi(r:np.ndarray,z:np.ndarray):
            s = 1
            l = self.length_scale(r,z) 
            return GibbsKer(x0 = r.flatten(),x1 = rIb, y0 = z.flatten(), y1 =zIb, lx0=l*s, lx1=lIb*s, isotropy=True)    




        nIb = self.nI+self.nb
        im_shape = ray.R1.shape
        H = np.zeros((im_shape[0],im_shape[1],nIb))
        Rray,_,Zray,L = ray.ZΦRL_ray(Lnum=Lnum)

        for i in tqdm(range(im_shape[0])):
            for j in range(im_shape[1]):
                rray = Rray[:,i,j]
                zray = Zray[:,i,j]
                dL = L[1,i,j]-L[0,i,j]
                H[i,j] = np.sum( dL*(phi(rray,zray)@self.KII_inv),axis=0)

        H = H.reshape((im_shape[0]*im_shape[1],nIb))

        H = H[:,:self.nI]
        return H 

    def set_kernel(self,
                   
            length_scale:float=1,
            is_bound :bool=True ,
            bound_value : float=0,
            bound_sig : float = 0.1,
            bound_space : float = 1e-2,
            is_static_kernel:bool = True,  
            zero_value_index = None,
            mean: float= 0,

        )->Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:

        """

        Parameters
        ----------
        length_scale     :,
        is_bound         : Trueのとき境界条件が定められる。,
        bound_value      :,
        bound_sig        :,
        bound_space      :,
        is_static_kernel : TrueのときオブジェクトにKf_priとmuf_priが保存される。,

        Reuturns
        ----------
        K_ff_pri: hoge,
        mu_f_pri: hoge,

        """
        ls = length_scale
        lI = ls*self.length_scale(self.rI,self.zI)
        Kii = GibbsKer(x0=self.rI     , x1=self.rI     , y0=self.zI     , y1=self.zI     , lx0=lI*ls, lx1=lI*ls, isotropy=True)
        if not is_bound: 
            mu_f_pri = np.zeros_like(self.rI)
            Kf_pri = Kii 
        else:
            if zero_value_index is None:
                index = np.zeros(self.nI,dtype=bool)
            else:
                index = zero_value_index
                    
            rb,zb = self.set_bound_space(delta_l=bound_space,is_change_local_variable=False)
            zb,rb = np.concatenate([self.zI[index],zb]), np.concatenate([self.rI[index],rb])

            #rb,zb = self.set_bound_space(delta_l=bound_space,is_change_local_variable=False)
            lb = ls*self.length_scale(rb,zb)
            KIb = GibbsKer(x0=self.rI, x1=rb, y0=self.zI, y1=zb, lx0=lI*ls, lx1=lb*ls, isotropy=True)
            Kbb = GibbsKer(x0=rb     , x1=rb, y0=zb     , y1=zb, lx0=lb*ls, lx1=lb*ls, isotropy=True)
            Kbb+= bound_sig**2*np.eye(rb.size)

            Kb = Kii - KIb @ np.linalg.inv(Kbb) @ KIb.T
            Kf_pri  = Kb
            mu_f_pri  = mean + KIb @ (np.linalg.inv(Kbb) @ (bound_value*np.ones(rb.size)-mean))
            

        if is_static_kernel:
            self.Kf_pri = Kf_pri
            self.muf_pri = mu_f_pri 
            self.kernel_type = 'isotropic kernel'
                    
            self.Kf_pri_property = {
                #'kernel_type': self.kernel_type,
                'is_bound'   : is_bound ,
                #'mean_value' : mean_value,
                'bound_value': bound_value,
                'bound_sig'  : bound_sig,
                'bound_space': bound_space } 

        return Kf_pri,mu_f_pri
        
    
    def set_unifom_kernel(self,
                          
            length_scale:float=0.1,
            is_bound :bool=True ,
            mean_value : float=0.,
            bound_value : float=0,
            bound_sig : float = 0.1,
            bound_space : float = 1e-2,
            is_static_kernel:bool = False,  

        )->Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:

        """
        Parameters
        ----------
        length_scale     :
        is_bound         : Trueのとき境界条件が定められる。
        mean_value       : 
        bound_value      :
        bound_sig        : 
        bound_space      : 
        is_static_kernel : TrueのときオブジェクトにKf_priとmuf_priが保存される。

        Reuturns
        ----------
        K_ff_pri:
        mu_f_pri:
        """

        ls = length_scale
        Kii = SEKer(x0=self.rI, x1=self.rI, y0=self.zI, y1=self.zI, lx=ls, ly=ls)
        
        if not is_bound: 
            mu_f_pri = mean_value*np.ones_like(self.rI)
            Kf_pri = Kii 
        else:
            rb,zb = self.set_bound_space(delta_l=bound_space,is_change_local_variable=False)
            KIb = SEKer(x0=self.rI, x1=rb, y0=self.zI, y1=zb, lx=ls, ly=ls,)
            Kbb = SEKer(x0=rb     , x1=rb, y0=zb     , y1=zb, lx=ls, ly=ls,)
            Kbb+= bound_sig**2*np.eye(rb.size)

            
            mu_f_pri = mean_value*np.ones_like(self.rI)

            Kb = Kii - KIb @ np.linalg.inv(Kbb) @ KIb.T
            Kf_pri = Kb
            mu_f_pri  = mu_f_pri + KIb @ (np.linalg.inv(Kbb) @ (bound_value*np.ones(rb.size)-mean_value))

        if is_static_kernel:
            self.Kf_pri = Kf_pri
            self.muf_pri = mu_f_pri 
            self.kernel_type = 'uniform SE kernel'
                    
            self.Kf_pri_property = {
                #'kernel_type': self.kernel_type,
                'is_bound'   : is_bound ,
                'mean_value' : mean_value,
                'bound_value': bound_value,
                'bound_sig'  : bound_sig,
                'bound_space': bound_space } 


        return Kf_pri,mu_f_pri
    
    def set_laplace_kernel(self,     
        del_r :float = 1.,
        is_bound :bool=True,
        mean_value : float=0.,
        bound_value : float=0,
        bound_sig : float = 0.1,
        bound_space : float = 1e-2,
        is_static_kernel:bool = False,  

        )->Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:

        """
        Parameters
        ----------
        length_scale     :
        is_bound         : Trueのとき境界条件が定められる。
        mean_value       : 
        bound_value      :
        bound_sig        : 
        bound_space      : 
        is_static_kernel : TrueのときオブジェクトにKf_priとmuf_priが保存される。

        Reuturns
        ----------
        K_ff_pri:
        mu_f_pri:
        """
        def logKer(x0,x1,y0,y1,rsq0,rsq1):
            X0,X1 = np.meshgrid(x0,x1,indexing='ij')
            Y0,Y1 = np.meshgrid(y0,y1,indexing='ij')
            Rsq0,Rsq1 = np.meshgrid(rsq0,rsq1,indexing='ij')

            r = np.sqrt((X0-X1)**2+(Y0-Y1)**2)
            is_zero = r < 1e-5
            r[is_zero] = 1,
            Rsq = np.sqrt(Rsq0*Rsq1)
            K = -1/2/np.pi*np.log(r)*Rsq
            Rsq_0 = Rsq[is_zero]
            K[is_zero] = -1/4*Rsq_0**2*(2*np.log(Rsq_0)-1)

            return ((K-K.min())*del_r)**2
            
        Lsq_I = self.length_scale_sq(self.rI,self.zI)
        Kii = logKer(x0=self.rI, x1=self.rI, y0=self.zI, y1=self.zI,rsq0=Lsq_I,rsq1=Lsq_I)
        
        if not is_bound: 
            mu_f_pri = mean_value*np.ones_like(self.rI)
            Kf_pri = Kii 
        else:
            rb,zb = self.set_bound_space(delta_l=bound_space,is_change_local_variable=False)
            Lsq_b = bound_space**2*np.ones_like(rb)
            KIb = logKer(x0=self.rI, x1=rb, y0=self.zI, y1=zb,rsq0=Lsq_I,rsq1=Lsq_b)
            Kbb = logKer(x0=rb, x1=rb, y0=zb, y1=zb,rsq0=Lsq_b,rsq1=Lsq_b)
            Kbb+= bound_sig**2*np.eye(rb.size)

            
            mu_f_pri = mean_value*np.ones_like(self.rI)

            Kb = Kii - KIb @ np.linalg.inv(Kbb) @ KIb.T
            Kf_pri = Kb
            mu_f_pri  = mu_f_pri + KIb @ (np.linalg.inv(Kbb) @ (bound_value*np.ones(rb.size)-mean_value))

        if is_static_kernel:
            self.Kf_pri = Kf_pri
            self.muf_pri = mu_f_pri 
            self.kernel_type = 'uniform laplace kernel'
                    
            self.Kf_pri_property = {
                #'kernel_type': self.kernel_type,
                'is_bound'   : is_bound ,
                'mean_value' : mean_value,
                'bound_value': bound_value,
                'bound_sig'  : bound_sig,
                'bound_space': bound_space } 


        return Kf_pri,mu_f_pri
    
    
    def set_flux_kernel(self,
                        
        psi_scale        : float = 0.3,
        B_scale          : float = 0.3,
        is_bound         : bool  = True ,
        bound_value      : float = -2,
        bound_sig        : float = 0.05,
        bound_space      : float = 1e-2,
        mean             : float = 0,
        zero_value_index : npt.NDArray[np.bool_] |None = None, # requres b
        separatrix       : bool = False,
        is_static_kernel : bool = False,

        )->Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:

        rI,zI = self.rI,self.zI
        psi_i = rt1plotpy.mag.psi(rI,zI,separatrix=separatrix)
        br_i,bz_i = rt1plotpy.mag.bvec(rI,zI,separatrix=separatrix)
        babs_i = np.sqrt(br_i**2+bz_i**2)

                
        Psi_i = np.meshgrid(psi_i,psi_i,indexing='ij')
        logb = np.log(babs_i)
        logBabs_i = np.meshgrid(logb,logb,indexing='ij')
                
        psi_len  =psi_i.std()*psi_scale
        psi_psi = (Psi_i[0]-Psi_i[1])**2/ psi_len**2

        b_len = logb.std()*B_scale
        b_b   = (logBabs_i[0]-logBabs_i[1])**2/b_len**2
                
        Kflux_ii = np.exp(-0.5*(psi_psi+b_b))

        if not is_bound:
            mu_f_pri = mean*np.ones_like(self.rI)
            Kflux_pri = Kflux_ii
        
        else:
            if zero_value_index is None:
                index = np.zeros(self.nI,dtype=bool)
            else:
                index = zero_value_index
                    
            rb,zb = self.set_bound_space(delta_l=bound_space,is_change_local_variable=False)
            zo,ro = np.concatenate([zI[index],zb]), np.concatenate([rI[index],rb])
                    
            psi_o = rt1plotpy.mag.psi(ro,zo,separatrix=separatrix)
            br_o,bz_o = rt1plotpy.mag.bvec(ro,zo,separatrix=separatrix)
            babs_o = np.sqrt(br_o**2+bz_o**2)
                    

            Psi_o = np.meshgrid(psi_o,psi_o,indexing='ij') / psi_len
            lnBabs_o = np.meshgrid(np.log(babs_o),np.log(babs_o),indexing='ij') / b_len
            Psi_io = np.meshgrid(psi_i,psi_o,indexing='ij') / psi_len
            lnBabs_io = np.meshgrid(np.log(babs_i),np.log(babs_o),indexing='ij') / b_len

            
            Kb_oo =  np.exp(-0.5*((Psi_o[0]-Psi_o[1])**2  +(lnBabs_o[0]-lnBabs_o[1])**2))
            Kb_io =  np.exp(-0.5*((Psi_io[0]-Psi_io[1])**2+(lnBabs_io[0]-lnBabs_io[1])**2))
            
            out_sig = bound_sig
            out_value = bound_value
            Kb_oo_sig_inv = np.linalg.inv(Kb_oo+out_sig**2*np.eye(ro.size))
            Kflux_pri = Kflux_ii - Kb_io @ Kb_oo_sig_inv @ Kb_io.T 
            mu_f_pri  = mean+Kb_io @  (Kb_oo_sig_inv @ (out_value*np.ones(ro.size)-mean)  ) 

        if is_static_kernel:
            self.Kf_pri = Kflux_pri 
            self.muf_pri = mu_f_pri 
            self.kernel_type = 'flux kernel'
                
            self.Kf_pri_property = {
                #'kernel_type': self.kernel_type,
                'psi_scale'  : psi_scale,
                'B_scale'    : B_scale,
                'is_bound'   : is_bound ,
                'bound_value': bound_value,
                'bound_sig'  : bound_sig,
                'bound_space': bound_space } 

        return Kflux_pri,mu_f_pri
    
    def sampler(self,
        K   : Optional[npt.NDArray[np.float64]]=None,
        mu_f: npt.NDArray[np.float64] | float = 0.
        ) -> npt.NDArray[np.float64]:

        if K is None:
            K = self.Kf_pri
            mu_f = self.muf_pri

        K_hash = hash((K.sum(axis=1)).tobytes())  #type: ignore

        if self.V is None or (self.K_hash != K_hash):
            print('Eigenvalue decomposition is recalculated')
            lam,V = np.linalg.eigh(K) #type: ignore
            lam[lam<1e-5]= 1e-5
            self.V = V
            self.lam = lam
        else:
            self.V = self.V
            self.lam = self.lam
        
        self.K_hash = K_hash 
        
        noise = np.random.randn(self.nI)
        return  mu_f+ self.V @ (np.sqrt(self.lam) *  noise)  
    
    def plot_mosaic(self,
        ax:plt.Axes,      
        f:npt.NDArray[np.float64],
        size :float = 1.0, # type: ignore
        back_ground:float | None =None,
        cbar :bool=True,
        cbar_title: str|None = None,
        is_frame:bool=True,
        vmean: float|None=None,
        **kwargs_scatter,
        )->None:

        if 'vmax' in kwargs_scatter:
            vmax = kwargs_scatter['vmax']
        else:
            vmax = np.percentile(f,99)
        if 'vmin' in kwargs_scatter:
            vmin = kwargs_scatter['vmin']
        else:
            vmin = np.percentile(f,1)

    
        if vmean is not None:
            temp = ((vmax - vmean)  > (vmean-vmin))
            tempi = not ((vmax - vmean)  > (vmean-vmin))

            vmax = temp  *vmax + tempi *(2*vmean-vmin)
            vmin = tempi *vmin + temp  *(2*vmean-vmax)
        
        kwargs_scatter['vmax'] = vmax
        kwargs_scatter['vmin'] = vmin

        if back_ground is not None:
            cmap:str = 'viridis'
            alpha =1.0
            if 'cmap' in kwargs_scatter:
                cmap = str(kwargs_scatter['cmap'])
            if 'alpha' in kwargs_scatter:
                alpha = kwargs_scatter['alpha']
            
            ax.imshow(back_ground*self.mask,cmap=cmap,vmax=vmax,vmin=vmin,alpha=alpha,**self.im_kwargs) # type: ignore
        
        size:npt.NDArray[np.float64] = size**2*1e5 *self.Lsq_I
        im = ax.scatter(x=self.rI,y=self.zI,c=f,s=size,**kwargs_scatter)

        if cbar:
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right' , size="5%", pad='3%')
            cbar = plt.colorbar(im, cax=cax, orientation='vertical')
            if cbar_title is not None: cbar.set_label(cbar_title) # type: ignore
        
        #ax.set_aspect('equal')
        if is_frame: self.append_frame(ax=ax,add_coil=True)

    def  plt_rt1_flux(self,
        ax:plt.Axes,      
        separatrix:bool =True,
        is_inner:bool =False,
        append_frame :bool =True,
        **kwargs_contour,
        )->None:
        R,Z = np.meshgrid(self.r_plot,self.z_plot,indexing='xy')
        Psi = rt1plotpy.mag.psi(R,Z,separatrix=separatrix)
        extent = self.im_kwargs['extent']
        origin = self.im_kwargs['origin']
        kwargs = {'levels':20,'colors':'black','alpha':0.3}
        kwargs.update(kwargs_contour)
        if is_inner:
            Psi = Psi*self.mask
        else:
            mpsi_max = -rt1plotpy.mag.psi(0.3,0.,separatrix=separatrix)
            mpsi_min = -rt1plotpy.mag.psi(self.r_plot.min(),self.z_plot.max(),separatrix=separatrix)
            kwargs['levels'] = np.linspace(mpsi_min,mpsi_max,kwargs['levels'],endpoint=False)
        
        ax.contour(-Psi,extent=extent,origin=origin,**kwargs)
        if append_frame:
            self.append_frame(ax)

    

class Kernel2D_grid(rt1plotpy.frame.Frame):
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
        #self.im_shape: Union[Tuple[int,int],None] = None
        self.V = None

        print('you have to "create_induced_point()" or "set_induced_point()" next.')

    def set_grid(self,
        r_grid: npt.NDArray[np.float64],
        z_grid: npt.NDArray[np.float64],
        boundary = 0, 
        ) -> Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]] | None:     
        self.r_grid,self.z_grid = r_grid,z_grid
        self.nr,self.nz = r_grid.size, z_grid.size

        self.grid_shape=(self.nz,self.nr)

        self.Z_grid,self.R_grid = np.meshgrid(z_grid,r_grid,indexing='ij')

        self.Z_grid_fl = self.Z_grid.flatten() 
        self.R_grid_fl = self.R_grid.flatten()

        self.mask,self.im_kwargs = self.grid_input(r_grid,z_grid)

        self.set_bound()
        pass 

    def set_bound(self,
        does_contain_edge : bool = True,
        ):
        self.Is_out = self.Is_out + (self.fill==1)
        if does_contain_edge:
            self.Is_out[0,:] = True
            self.Is_out[-1,:] = True
            self.Is_out[:,0] = True
            self.Is_out[:,-1] = True
        pass 

    def create_observation_matrix(self,
        ray  : rt1raytrace.Ray,
        Lnum : int=100
        ) ->  sparse.csr_matrix:

        Rray,Zray = ray.RZ_ray(Lnum)
        dL = ray.Length/ (Lnum+1)
        H = create_grid_obs(Rray,Zray,dL,self.r_grid,self.z_grid)
        H = H.reshape(ray.shape[0]*ray.shape[1],self.nr*self.nz)
        return sparse.csr_matrix(H)
    
    def grid_input(self, 
        R: npt.NDArray[np.float64], 
        Z: npt.NDArray[np.float64], 
        fill_point: Tuple[float, float] = (0.5,0), 
        fill_point_2nd: Optional[Tuple[float, float]] = None, 
        isnt_print: bool = False
        ) -> Tuple[npt.NDArray[np.float64], dict]:
        mask,extent = self.__grid_input(R, Z, fill_point, fill_point_2nd, isnt_print)
        """
        this functions is to return 'mask' and 'imshow_kwargs' np.array for imshow plottting

        Parameters
        ----------
        R: npt.NDArray[np.float64],
            array of R axis with 1dim
        Z: npt.NDArray[np.float64],
            array of Z axis with 1dim

        fill_point: Tuple[float,float] = (0.5,0), optional,
        fill_point_2nd: Optional[Tuple[float,float]] = None, optional

        Reuturns
        ----------
        mask:
        imshow_kwargs:  {"origin":"lower","extent":extent}
        """
        return mask, {"origin":"lower","extent":extent}

    
    
    def __grid_input(self, R: npt.NDArray[np.float64], Z: npt.NDArray[np.float64], fill_point: Tuple[float, float] = ..., fill_point_2nd: Optional[Tuple[float, float]] = None, isnt_print: bool = False
        ) -> Tuple[npt.NDArray[np.float64], tuple]:
        return super().grid_input(R, Z, fill_point, fill_point_2nd, isnt_print)

        

    def set_kernel(self,
        is_bound :bool=True ,
        Length_sq :npt.NDArray[np.float64]|float = 1.,
        bound_value : float=0,
        bound_sig : float = 0.1,
        is_static_kernel:bool = True,
        mean :float = 0.,

        )->Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64]]:

        """

        Parameters
        ----------
        length_scale     :,
        is_bound         : Trueのとき境界条件が定められる。,
        bound_value      :,
        bound_sig        :,
        bound_space      :,
        is_static_kernel : TrueのときオブジェクトにKf_priとmuf_priが保存される。,

        Reuturns
        ----------
        K_ff_pri: hoge,
        mu_f_pri: hoge,

        """
        R,Z = self.R_grid_fl,self.Z_grid_fl
        if type(Length_sq) is float:
            Length = Length_sq*np.ones(R.size,dtype=np.float64)
            pass 
        else:
            Length = np.sqrt(Length_sq)

        Kgg = GibbsKer(x0=R, x1=R, y0=Z, y1=Z, lx0=Length, lx1=Length, isotropy=True)
        if not is_bound: 
            mu_f_pri = bound_value*np.ones_like(self.R_grid)
            Kf_pri = Kgg
        else:
            rb = self.R_grid[self.Is_out]
            zb = self.Z_grid[self.Is_out]
            lb = Length[self.Is_out]
            Kgb = GibbsKer(x0=R , x1=rb, y0=Z, y1=zb, lx0=Length, lx1=lb, isotropy=True)
            Kbb = GibbsKer(x0=rb, x1=rb, y0=zb,y1=zb, lx0=lb , lx1=lb, isotropy=True)
            Kbb+= bound_sig**2*np.eye(rb.size)
            Kbb_inv = np.linalg.inv(Kbb)
            Kb = Kgg - Kgb @ Kbb_inv @ Kgb.T
            Kf_pri  = Kb
            mu_f_pri  = mean + Kgb @ (Kbb_inv @ (bound_value*np.ones(rb.size)-mean))
            

        if is_static_kernel:
            self.Kf_pri = Kf_pri
            self.muf_pri = mu_f_pri 
            self.kernel_type = 'isotropic kernel'
                    
            self.Kf_pri_property = {
                #'kernel_type': self.kernel_type,
                'is_bound'   : is_bound ,
                #'mean_value' : mean_value,
                'bound_value': bound_value,
                'bound_sig'  : bound_sig } 

        return Kf_pri,mu_f_pri
    
    def create_laplacian_matrix(self
        )->sparse.csr_matrix:

        C = np.zeros((self.nz,self.nr,self.nz,self.nr))
        r_grid,z_grid = self.r_grid,self.z_grid
        dr = r_grid[1]-r_grid[0]
        dz = z_grid[1]-z_grid[0]

        ov_drsq  = 1/dr**2
        ov_dzsq  = 1/dz**2
        for i in range(120):
            for j in range(100):

                if i != 0:
                    C[i,j,i-1,j  ] = ov_dzsq 
                if i != 119:
                    C[i,j,i+1,j  ] = ov_dzsq 
                if j != 0:
                    C[i,j,i  ,j-1] = ov_drsq 
                if j != 99:
                    C[i,j,i  ,j+1] = ov_drsq 
                    

        C[:,:,self.Is_out] = 0
        C[self.Is_out,:,:] = 0

        for i in range(120):
            for j in range(100):
                C[i,j,i,j] = -2*ov_drsq  -2*ov_dzsq

        return sparse.csr_matrix(C.reshape(self.nz*self.nr, self.nz*self.nr))
    
    def create_deribative_matrix(self
        )->tuple[sparse.csr_matrix,sparse.csr_matrix]:
        
        Cr = np.zeros((self.nz,self.nr,self.nz,self.nr))
        Cz = np.zeros((self.nz,self.nr,self.nz,self.nr))
        r_grid,z_grid = self.r_grid,self.z_grid
        dr = r_grid[1]-r_grid[0]
        dz = z_grid[1]-z_grid[0]

        ov_dr  = 1/dr
        ov_dz  = 1/dz
        for i in range(120):
            for j in range(100):
                if j != 0:
                    pass
                Cr[i,j,i  ,j  ] = -ov_dr
                if j != 99:
                    Cr[i,j,i  ,j+1] = +ov_dr 
                if i != 0:
                    pass
                
                Cz[i,j,i,j    ] = -ov_dz 
                if i != 119:
                    Cz[i,j,i+1,j  ] = +ov_dz 
                
        Cr = sparse.csr_matrix(Cr.reshape(120*100,120*100))
        Cz = sparse.csr_matrix(Cz.reshape(120*100,120*100))
        return Cr,Cz

    
    def sampler(self,
        K   : Optional[npt.NDArray[np.float64]]=None,
        mu_f: npt.NDArray[np.float64] | float = 0.
        ) -> npt.NDArray[np.float64]:

        if K is None:
            K = self.Kf_pri
            mu_f = self.muf_pri

        K_hash = hash((K.sum(axis=1)).tobytes())  #type: ignore

        if self.V is None or (self.K_hash != K_hash):
            print('Eigenvalue decomposition is recalculated')
            lam,V = np.linalg.eigh(K) #type: ignore
            lam[lam<1e-5]= 1e-5
            self.V = V
            self.lam = lam
        else:
            self.V = self.V
            self.lam = self.lam
        
        self.K_hash = K_hash 
        
        noise = np.random.randn(self.R_grid.size)
        return  mu_f+ self.V @ (np.sqrt(self.lam) *  noise)  


@njit 
def create_grid_obs(
    Rray,Zray,dL,r_grid,z_grid):
    Lnum = Rray.shape[0]
    R,Z = r_grid,z_grid
    R_ext = np.empty(R.size+1,dtype=np.float64)  
    Z_ext = np.empty(Z.size+1,dtype=np.float64)
    R_ext[0] =  R[0]  - 0.5* (R[1]-R[0])
    R_ext[-1] = R[-1] + 0.5* (R[-1]-R[-2])
    R_ext[1:-1] = 0.5 * (R[:-1] + R[1:])
    Z_ext[0] =  Z[0]  - 0.5* (Z[1]-Z[0])
    Z_ext[-1] = Z[-1] + 0.5* (Z[-1]-Z[-2])
    Z_ext[1:-1] = 0.5 * (Z[:-1] + Z[1:])

    H = np.zeros((Rray.shape[1],Rray.shape[2],z_grid.size,r_grid.size))    
    
    for i in range(Rray.shape[1]):
        for j in range(Rray.shape[2]):
            ray_map =  np.zeros((z_grid.size,r_grid.size),dtype=np.int16)
            for k in range(Lnum):
                r,z = Rray[k,i,j], Zray[k,i,j]
                is_rin = (r>R_ext[0:-1]) * (r <=R_ext[1:])
                is_zin = (z>Z_ext[0:-1]) * (z <=Z_ext[1:])
                ir = np.arange(r_grid.size)[is_rin][0]
                iz = np.arange(z_grid.size)[is_zin][0]
                ray_map[iz,ir] += 1  
            H[i,j,:,:] = dL[i,j]*ray_map[:,:] 
        if i%10 == 0:
            print(i)
    
    return H 



@njit
def d2min(x,y,xs,ys):
    x_tau2 = (x- xs)**2
    y_tau2 = (y- ys)**2
    d2_min = np.min(x_tau2 + y_tau2)
    return d2_min

def SEKer(
    x0 : npt.NDArray[np.float64],
    x1 : npt.NDArray[np.float64],
    y0 : npt.NDArray[np.float64] |float,
    y1 : npt.NDArray[np.float64] |float,
    lx : float,
    ly : float,
    ) -> npt.NDArray[np.float64]:

    X = np.meshgrid(x0,x1,indexing='ij')
    Y = np.meshgrid(y0,y1,indexing='ij')
    return np.exp(- 0.5*( ((X[0]-X[1])/lx)**2 + ((Y[0]-Y[1])/ly)**2) )

def GibbsKer(
    x0 : npt.NDArray[np.float64],
    x1 : npt.NDArray[np.float64],
    y0 : npt.NDArray[np.float64],
    y1 : npt.NDArray[np.float64],
    lx0: npt.NDArray[np.float64],
    lx1: npt.NDArray[np.float64],
    ly0: npt.NDArray[np.float64] | bool  = False,
    ly1: npt.NDArray[np.float64] | bool  = False,
    isotropy: bool = False
    ) -> npt.NDArray[np.float64]:  

    X  = np.meshgrid(x0,x1,indexing='ij')
    Y  = np.meshgrid(y0,y1,indexing='ij')
    Lx = np.meshgrid(lx0,lx1,indexing='ij')
    Lxsq = Lx[0]**2+Lx[1]**2 

    if isotropy:
        return 2*Lx[0]*Lx[1]/Lxsq *np.exp( -   ((X[0]-X[1])**2  +(Y[0]-Y[1])**2 )/ Lxsq )

    else:
        Ly = np.meshgrid(ly0,ly1,indexing='ij')
        Lysq = Ly[0]**2+Ly[1]**2 
        return np.sqrt(2*Lx[0]*Lx[1]/Lxsq) *np.sqrt(2*Ly[0]*Ly[1]/Lysq) *np.exp( -(X[0]-X[1])**2 / Lxsq  - (Y[0]-Y[1])**2 / Lysq )# type: ignore

@njit
def GibbsKer_fast(
    x0 : npt.NDArray[np.float64],
    x1 : npt.NDArray[np.float64],
    y0 : npt.NDArray[np.float64],
    y1 : npt.NDArray[np.float64],
    lx0: npt.NDArray[np.float64],
    lx1: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:  

    X  = np.meshgrid(x0,x1,indexing='ij')
    Y  = np.meshgrid(y0,y1,indexing='ij')
    Lx = np.meshgrid(lx0,lx1,indexing='ij')
    Lxsq = Lx[0]**2+Lx[1]**2 

    return 2*Lx[0]*Lx[1]/Lxsq *np.exp( -   ((X[0]-X[1])**2  +(Y[0]-Y[1])**2 )/ Lxsq )

class Kernel1D():
    pass 
