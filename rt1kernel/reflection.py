from pkgutil import extend_path
import matplotlib.pyplot as plt
import numpy as np
import rt1plotpy
from typing import Optional, Union,Tuple,Callable,List,TypeVar,cast
import time 
import math
from tqdm import tqdm
from numba import jit
import warnings
from dataclasses import dataclass
import itertools
import scipy.linalg as linalg
import scipy.sparse as sparse
import scipy.optimize
import pandas as pd
import os,sys
from .plot_utils import *  

import torch
from torch.autograd.functional import hessian

from . import rt1kernel

try:
    from .. import rt1raytrace
except:
    sys.path.insert(0,os.path.join(os.path.dirname(__file__),os.pardir))
    import rt1raytrace
    sys.path.pop(0)

__all__ = ['Reflection_Kernel_grid',
           'sigmoid_inv',
           'sigmoid',
           'Reflection_tomography'
           ]


float_numpy = TypeVar(" float|np.ndarray ",float,np.ndarray) # type: ignore

def const_like(x:float, type_x:float_numpy)->float_numpy:
    return cast(float_numpy, x + 0*type_x)

def ones_like(type_x:float_numpy)->float_numpy:
    return cast(float_numpy, 1.0*type_x)

def zeros_like(type_x:float_numpy)->float_numpy:
    return cast(float_numpy, 0.0*type_x)


rt1_ax_kwargs = {'xlim'  :(0,1.1),
                 'ylim'  :(-0.7,0.7), 
                 'aspect': 'equal'
                }
class Reflection_Kernel_grid:

    def __init__(self,
        cI   : np.ndarray,
        hI   : np.ndarray,        
        c_len: float = 0.2,
        h_len: float = 0.15,
        mur_mean: float = -1,
        ) -> None:
        self.c_len, self.h_len = c_len,h_len
        self.cI, self.hI = cI,hI 
        self.CI, self.HI = np.meshgrid(cI,hI,indexing='ij')
        self.NI = self.HI.size
        self.mur_pri    = mur_mean*np.ones(self.CI.size)
        self.Kr_pri  = SEKer_cz(  self.CI     , self.CI     , self.HI     , self.HI     , self.c_len,self.h_len)

    def add_grad(self,
        C_inp: np.ndarray,
        H_inp: np.ndarray,
        Refgrad_inp: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
        Rg= Refgrad_inp
        Kr_dbI  = SEKer_dcz( C_inp, self.CI, H_inp, self.HI, self.c_len,self.h_len)
        Kr_dbdb = SEKer_ddcz(C_inp, C_inp,  H_inp, H_inp , self.c_len,self.h_len)
        

        Kr_dbdb_inv = np.linalg.inv(Kr_dbdb + 1e-3*np.eye(Rg.size)+10*np.diag(Rg.flatten()**2))
        self.mur_pri   = self.mur_pri+Kr_dbI.T @ (Kr_dbdb_inv @ Rg.flatten())

        self.Kr_pri = self.Kr_pri - Kr_dbI.T  @ Kr_dbdb_inv @ Kr_dbI
        return self.mur_pri,self.Kr_pri
    
    
    def add_value(self,
        C_inp: np.ndarray,
        H_inp: np.ndarray,
        Ref_inp: np.ndarray,
        sigma: Union[float,np.ndarray] = 1, 
        ):
        Rv = Ref_inp
        
        Kr_vI  = SEKer_cz( C_inp, self.CI, H_inp, self.HI, self.c_len,self.h_len)
        Kr_vv  = SEKer_cz(C_inp,  C_inp,   H_inp, H_inp,   self.c_len,self.h_len)

        

        Kr_vv_inv = np.linalg.inv(Kr_vv + sigma**2*np.eye(Rv.size))
        self.mur_pri   = self.mur_pri +Kr_vI.T @ (Kr_vv_inv @ Rv.flatten())

        self.Kr_pri = self.Kr_pri - Kr_vI.T  @ Kr_vv_inv @ Kr_vI
        return self.mur_pri,self.Kr_pri
    
    def make_sample(self,
        show: bool =True,
        ) -> np.ndarray:
        
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        
        f = np.random.multivariate_normal(self.mur_pri,self.Kr_pri)

        if show:              
            fig,axs=plt.subplots(1,2,figsize=(11,5))
            scatter_cbar(axs[0],x=self.CI.flatten(),y=self.HI.flatten(),c=sigmoid(f),marker='s',s=70,vmax=1,vmin=0)
            axs[0].set_ylabel(r'z[m]')
            axs[0].set_xlabel(r'$\cos\theta$')
            cmap_line(axs[1],
                      x=180/np.pi*np.arccos(self.cI),
                      y=self.hI,
                      C=sigmoid(f.reshape(self.CI.shape).T),
                      cmap='turbo',
                      cbar_title=r'z[m]',
                      alpha=0.5
                      )
            
            axs[1].plot(180/np.pi*np.arccos(self.cI),sigmoid(self.mur_pri.reshape(self.CI.shape)).mean(axis=1),'black')
            axs[1].set_xlim(0,90)
            axs[0].set_xlim(axs[0].get_xlim()[::-1])
            axs[1].set_xlabel(r'$\theta$[deg]')
            axs[1].set_ylim(0,1)
            axs[0].grid(False)

            plt.show()
        return f
    
    
    def make_plot(self,
        f :np.ndarray,
        show: bool =True,
        ) -> None:
        
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        
        if show:              
            fig,axs=plt_subplots(1,2,figsize=(11,5))
            scatter_cbar(axs[0],x=self.CI.flatten(),y=self.HI.flatten(),marker='s',c=sigmoid(f),s=75,vmax=1,vmin=0)
            axs[0][0].set_ylabel(r'$z$ [m]')
            axs[0][0].set_xlabel(r'$\cos\theta$')
            cmap_line(axs[1],
                      x=180/np.pi*np.arccos(self.cI),
                      y=self.hI,
                      C=sigmoid(f.reshape(self.CI.shape).T),
                      cmap='turbo',
                      cbar_title=r'$z$ [m]',
                      alpha=0.5
                      )
            
            axs[0][1].plot(180/np.pi*np.arccos(self.cI),sigmoid(self.mur_pri.reshape(self.CI.shape)).mean(axis=1),'black')
            axs[0][1].set_xlim(0,90)
            axs[0][0].set_xlim(axs[0].get_xlim()[::-1])
            axs[0][1].set_xlabel(r'$\theta$ [deg]')
            axs[0][1].set_ylim(0,1)
            #axs[0].set_ylim(-0.6,0.6)
            axs[0][0].grid(False)

        return 

    def set_ObsMarix_interface(self,
        Obs:rt1kernel.Observation_Matrix_integral,
        start_num:int=1,
        show:bool = True,
        ):
        h_max = self.hI.max()
        h_min = self.hI.min()
        Kr = SEKer_cz(  self.CI     , self.CI     , self.HI     , self.HI     , self.c_len,self.h_len)
        Kr_inv = np.linalg.inv(Kr+1e-5*np.eye(self.CI.size)) 
        self.T_I2w :List[np.ndarray] = []

        for n in range(start_num,len(Obs.Hs)):
            C_wall = np.nan_to_num(Obs.Hs[n].ray.cos_factor.copy())
            H_wall = Obs.Hs[n].ray.Z0.copy()
            H_wall[H_wall>h_max] = h_max
            H_wall[H_wall<h_min] = h_min            
            Kr_wI = SEKer_cz(  C_wall, self.CI, H_wall, self.HI, self.c_len, self.h_len)
            self.T_I2w.append(Kr_wI @ Kr_inv)



class Diffusion_kernel:
    '''
    This class imitates Lamber Reflection, and  requres inducing point of Z wall.
    '''

    def __init__(self,
        Obs: rt1kernel.Observation_Matrix_integral,
        zIb: np.ndarray,
        frame:rt1plotpy.frame.Frame,
        length_sacle_sq_I:np.ndarray,
        ) -> None:

        #########この部分は，RT-1の形に極めて依存########################
        is_wall_outer =  Obs.Hs[0].ray.R1 >0.55
        rb_set = Obs.Hs[0].ray.R1[is_wall_outer]  
        zb_set = Obs.Hs[0].ray.Z1[is_wall_outer]
        btype_set = Obs.Hs[0].ray.ref_type[is_wall_outer]
        bnum_set = Obs.Hs[0].ray.ref_num[is_wall_outer]
        #################################
        
        zIb = zIb
        NrI = np.zeros_like(zIb) 
        NzI = np.zeros_like(zIb) 
        rIb   = np.zeros_like(zIb) 
        for i,zIb_i in enumerate(zIb):
            index = np.argmin(abs(zb_set-zIb_i))
            rIb_i = rb_set[index]
            ref_num = bnum_set[index]
            ref_type = btype_set[index]
            nr,nz = frame.normal_vector(r=rIb_i,z=zIb_i,frame_type=ref_type,frame_num=ref_num)
            NrI[i] = nr
            NzI[i] = nz
            rIb[i] = rIb_i

                
        fig,ax =plt.subplots(figsize=(6,6))

        rt1_ax_kwargs = {'xlim'  :(0,1.1),
                    'ylim'  :(-0.7,0.7), 
                    'aspect': 'equal'
                }                
        ax.set(**rt1_ax_kwargs)
        frame.append_frame(ax,label=True)
        ax.scatter(x=rIb,
                    y=zIb,) 

        ax.quiver(rIb,zIb,NrI,NzI,)
        Ls_I = length_sacle_sq_I
        plt.scatter(Obs.rI,Obs.zI,s=Ls_I*100000,alpha=0.5)
        plt.show()
        
        ######################
        theta_max = np.pi*0.6
        n = 200
        ######################
        
        theta = np.linspace(0,theta_max,n,endpoint=False) 
        dtheta = theta[1]-theta[0]
        i,j = 0,0

        nI = Obs.shape[2]
        rI,zI = Obs.rI,Obs.zI
        H_diffusion = np.zeros((zIb.size,nI))

        for i in range(zIb.size):
            for j in range(nI):
                l_z = zIb[i] - zI[j]
                l_x = rIb[i]    - rI[j]*np.cos(theta)
                l_y = -rI[j] * np.sin(theta)

                l_sq = l_z**2 + l_x**2 + l_y**2

                drtheta = np.pi*rI[j]*dtheta

                cos = (NzI[i]*l_z+NrI[i]*l_x)/np.sqrt(l_sq)
                H_diffusion[i,j]=np.sum(abs(drtheta*Ls_I[j]*cos/l_sq))
                """
                if (i==10) *( j == 1000):
                    print(i,j)
                    plt.plot(cos)
                    plt.show()
                """

        Z_all_0 = np.array(Obs.Hs[1].ray.Z0).copy()

        Z_all_0[Z_all_0> zIb.max()] = zIb.max()
        Z_all_0[Z_all_0< zIb.min()] = zIb.min()
        Kzdif_II  = self.__SEKer_z(z1=zIb,     z2=zIb, z_len=0.1)
        Kzdif_z0I = self.__SEKer_z(z1=Z_all_0, z2=zIb, z_len=0.1)

        Kzdif_II_inv =np.linalg.inv(Kzdif_II+1e-5*np.eye(zIb.size)) 
        self.Interp       =Kzdif_z0I @Kzdif_II_inv
        self.H_diff_I =H_diffusion

    def __matmul__(self,
        f:np.ndarray)->np.ndarray:
        return self.Interp @ (self.H_diff_I @ f)

    def __SEKer_z(self,z1,z2,z_len):
        z1 = z1.flatten()
        z2 = z2.flatten()
        Z  = np.meshgrid(z1,z2,indexing='ij') 
        return np.exp(-0.5*(Z[0]-Z[1])**2/z_len**2)




class Reflection_tomography:

    def __init__(self,
        Ref   :Reflection_Kernel_grid,
        Obs   :rt1kernel.Observation_Matrix_integral,  
        Kernel:rt1kernel.Kernel2D_scatter,
        H_diff:Optional[Diffusion_kernel]=None,
        ):
        self.Ref = Ref
        self.Obs = Obs
        self.Kernel = Kernel
        self.Kf_pri = self.Kernel.Kf_pri.copy()
        self.muf_pri = self.Kernel.muf_pri.copy()
        self.im_shape = self.Obs.shape[:2]
        self.H_diff = H_diff
        self.K_pri_inv = np.linalg.inv(self.Kf_pri+1e-6*np.eye(self.muf_pri.size))

    
    def set_obs(self,
        g_obs_list    :List[np.ndarray],
        ref_true      :None,
        name_list     :Optional[List[str]] = None,
        f_true_list   :Optional[List[np.ndarray]]=None,
        sig_im        :Optional[np.ndarray]=None,
        ref_mask      :Optional[np.ndarray]=None,
        n_reflection  :int=1,
        alpha_d_fix   :Optional[float] = None,
        ) :
        self.num_im = len(g_obs_list)
        self.f_true_list = f_true_list 
        self.ref_true = ref_true
        self.n_frame = len(g_obs_list)
        self.n_reflection = n_reflection
        
        self.g_obs_list = g_obs_list
        g_size = g_obs_list[0].size
        
        # sig_imageは共通だとしている．
        if sig_im is None:
            sig_inv_spa = sparse.diags(np.ones(g_size)) 
        else:
            sig_inv_spa = sparse.diags(1/sig_im.flatten())
            
        if ref_mask is None:
            self.ref_mask = np.zeros(g_size,dtype=np.bool_)
        else:
            self.ref_mask = (ref_mask.flatten() == True)

        if name_list is None:
            self.name_list = []
            for i in range(self.n_frame):
                if i == 0:
                    self.name_list.append('1st')
                elif i == 1:
                    self.name_list.append('2nd')
                elif i == 2:
                    self.name_list.append('3rd')
                else:
                    self.name_list.append(str(i+1)+'th')
            

        self.sigi_obs_list     = [sig_inv_spa @ g_obs.flatten() for g_obs    in self.g_obs_list   ]
        self.sigi_obs_tor_list = [torch.FloatTensor(sigi_obs)   for sigi_obs in self.sigi_obs_list]
        self.sigiH0 = sig_inv_spa@self.Obs.Hs[0].H
        self.sigiH1 = sig_inv_spa@self.Obs.Hs[1].H
        self.sigiH2 = sig_inv_spa@self.Obs.Hs[2].H
        self.H0sigi2H0 = self.sigiH0.T @ self.sigiH0 

                
        self.Kr_II_pri_tor = torch.FloatTensor(self.Ref.Kr_pri)
        self.Kr_II_pri_inv = np.linalg.inv(self.Ref.Kr_pri+1e-3*np.eye(self.Ref.NI))
        self.Kr_II_pri_inv_tor = torch.FloatTensor(self.Kr_II_pri_inv)

        self.T_I20_tor = torch.FloatTensor(self.Ref.T_I2w[0])
        self.T_I21_tor = torch.FloatTensor(self.Ref.T_I2w[1])
        self.mask_w_tor = torch.FloatTensor(~self.ref_mask)
        self.mur_pri_tor = torch.FloatTensor(self.Ref.mur_pri)
        self.iter:int=0

        self.T0_tor    = torch.FloatTensor(np.einsum('i,ij->ij',~self.ref_mask,self.Ref.T_I2w[0])) 
        self.T1_tor    = torch.FloatTensor(np.einsum('i,ij->ij',~self.ref_mask,self.Ref.T_I2w[1])) 

        self.enable_diffusion = False
        if self.H_diff is not None: 
            self.enable_diffusion = True
            self.msigi_Interp_diff =  np.einsum('i,ij->ij',~self.ref_mask,sig_inv_spa @ self.H_diff.Interp)
            self.H_diff_I = self.H_diff.H_diff_I
            self.Hdsigi2Hd = self.H_diff.H_diff_I.T @ (self.msigi_Interp_diff.T@self.msigi_Interp_diff) @ self.H_diff_I

    def calc_f(self,
        f_list   :List[np.ndarray],
        refI_now :np.ndarray,
        Kr_now   :np.ndarray,
        sig_scale:float = 0.01,
        is_plot:bool=True,
        alpha_d :Optional[float]=None,
        )-> Tuple[List[np.ndarray],List[np.ndarray]]:
        
        ## 本来はsimoidは最後にするのが正しいがそこまで影響は少ないと思われる

        lam,V = np.linalg.eigh(Kr_now+1e-1*np.eye(self.Ref.NI))
        exist = lam>1e-5
        lam =lam[exist]
        V = V[:,exist]
        L = np.diag(np.sqrt(lam)) @ V.T
        sgmI_sample= (L.T @ np.random.randn(lam.size,490)).T +np.broadcast_to(refI_now,(490,refI_now.size))
        sgmI_sample = sigmoid(sgmI_sample)
        #sgmI_sample = sigmoid(np.random.multivariate_normal(refI_now,Kr_now+1e-1*np.eye(self.Ref.NI),490))
        self.sgmI_sample = sgmI_sample
        ref0_sample = self.Ref.T_I2w[0] @sgmI_sample.T
        ref1_sample = self.Ref.T_I2w[1] @sgmI_sample.T
        ##
        ref0_mean_sp    = sparse.diags(~self.ref_mask   *(ref0_sample   ).mean(axis=1))
        ref0_sq_mean_sp = sparse.diags(~self.ref_mask   *(ref0_sample**2).mean(axis=1))
        del sgmI_sample 


        if self.n_reflection == 1:     
            Mat0 =  self.sigiH0.T @ (ref0_mean_sp @ self.sigiH1) #  sparse.diags @ A is 4 times faster than A @ sparse.diags 
            Hsig2iH_mean= (self.H0sigi2H0
                        + self.sigiH1.T @(ref0_sq_mean_sp @self.sigiH1)
                        + Mat0  
                        + Mat0.T
                        ).toarray()
            
            del Mat0
            sigiH_mean = self.sigiH0 + ref0_mean_sp @self.sigiH1 
            if alpha_d is not None:
                Mat0 = alpha_d**2 *self.Hdsigi2Hd

                Mat1 = (alpha_d  *(self.sigiH0.T @self.msigi_Interp_diff) 
                       +alpha_d  *(self.sigiH1.T @(ref0_mean_sp @self.msigi_Interp_diff))
                       ) @ self.H_diff.H_diff_I 
                
                Hsig2iH_mean += Mat0 +Mat1 +Mat1.T

                #print((Mat0 +Mat1 +Mat1.T).shape)
                sigiH_mean += alpha_d *( self.msigi_Interp_diff @self.H_diff.H_diff_I )
                sigiH_mean = np.array(sigiH_mean)

            #sigiH_mean = np.array(sigiH_mean)

            

        elif self.n_reflection == 2:

            ref01_mean_sp     = sparse.diags(~self.ref_mask *(ref0_sample    *ref1_sample   ).mean(axis=1))
            ref0sq1_mean_sp   = sparse.diags(~self.ref_mask *(ref0_sample**2 *ref1_sample   ).mean(axis=1))
            ref0sq1sq_mean_sp = sparse.diags(~self.ref_mask *(ref0_sample**2 *ref1_sample**2).mean(axis=1))
            Mat0 = self.sigiH1.T @( ref0_mean_sp    @self.sigiH0 )
            Mat1 = self.sigiH0.T @( ref01_mean_sp   @self.sigiH2 )
            Mat2 = self.sigiH1.T @( ref0sq1_mean_sp @self.sigiH2 )
            Hsig2iH_mean= (self.H0sigi2H0
                        +  self.sigiH1.T @( ref0_sq_mean_sp   @self.sigiH1 )
                        +  self.sigiH2.T @( ref0sq1sq_mean_sp @self.sigiH2 )
                        +  Mat0 +Mat0.T +Mat1 +Mat1.T +Mat2 +Mat2.T
                        ).toarray()
            del Mat0,Mat1,Mat2
            sigiH_mean = self.sigiH0 + ref0_mean_sp @self.sigiH1 + ref01_mean_sp @self.sigiH2


            if alpha_d is not None:
                Mat0 = (alpha_d**2) *self.Hdsigi2Hd

                Mat1 = (alpha_d  *(self.sigiH0.T @self.msigi_Interp_diff) 
                       +alpha_d  *(self.sigiH1.T @(ref0_mean_sp  @self.msigi_Interp_diff))
                       +alpha_d  *(self.sigiH2.T @(ref01_mean_sp @self.msigi_Interp_diff))
                       ) @ self.H_diff.H_diff_I 
                
                Hsig2iH_mean += Mat0 +Mat1 +Mat1.T


                sigiH_mean   += alpha_d *( self.msigi_Interp_diff @self.H_diff_I )
            
            if type(sigiH_mean) is not np.ndarray:
                sigiH_mean = np.array(sigiH_mean)

        else: return print('n_reflenction is wrong number')
        
        rI,zI = self.Obs.rI,self.Obs.zI 
            
        def calc_core_fast(
            f:np.ndarray,
            i:int=0, ##list_index,
            ):
            r_f = f - self.muf_pri
            exp_f = np.exp(f)
            fxf = np.einsum('i,j->ij',exp_f,exp_f)

            c1 = 1/sig_scale**2 *(Hsig2iH_mean  @ exp_f-sigiH_mean.T @ self.sigi_obs_list[i]) * exp_f 
            C1 = 1/sig_scale**2 *Hsig2iH_mean * fxf 

            Psi_df   = -c1 - self.K_pri_inv @ r_f 

            Psi_dfdf = -C1 - np.diag(c1) - self.K_pri_inv

            DPsi = Psi_dfdf
            NPsi = Psi_df

            delta_f = - np.linalg.solve(DPsi,NPsi)

            delta_f[delta_f<-3] = -3
            delta_f[delta_f>+3] = +3

            return delta_f,DPsi
        
        Kf_pos_list = []
        f_rmse      = []
        for i in range(self.n_frame):

            f = f_list[i].copy()
                
            for j in range(30):
                delta_f,DPsi = calc_core_fast(f=f,i=i)
                if (res := (delta_f.std())) >= 1e-8:
                    pass
                else:
                    break
                f += delta_f*1
                if j%5 == 0:
                    #check_diff(f) 
                    pass  
            
            print(i,j,res)
            muf_pos = f 
            f_list[i] = muf_pos
            Kf_pos_list.append(np.linalg.inv(-DPsi))

            if self.ref_true is not None: 
                f_rmse.append((np.exp(muf_pos)-self.f_true_list[i]).std()/self.f_true_list[i].mean())

            
            if self.f_true_list is not None and is_plot:    
                f_true = self.f_true_list[i]
                Kf_pos = Kf_pos_list[i]
                vmax = f_true.max()*1.1
                fig,axs = plt.subplots(1,3,figsize=(15,5),sharey=True)
                for ax in axs:
                    ax.set(**rt1_ax_kwargs)
                    self.Kernel.append_frame(ax)

                rI,zI = self.Obs.rI,self.Obs.zI 
                scatter_cbar(axs[0],x=rI,y=zI,c=np.exp(muf_pos),s=0.7*1e5*self.Kernel.Lsq_I,vmin=0,vmax=vmax)
                scatter_cbar(axs[1],x=rI,y=zI,c=(np.exp(muf_pos)-f_true),s=0.7*1e5*self.Kernel.Lsq_I,vmax=0.3,vmin=-0.3,cmap='RdBu_r')
                axs[0].set_title('f_'+self.name_list[i]+', Iteration: '+str(self.iter))
                axs[1].set_title('nrmse = '+str(f_rmse[i])[:8])

                scatter_cbar(axs[2],x=rI,y=zI,c=np.exp(muf_pos)*np.sqrt(np.diag(Kf_pos)),s=0.7*1e5*self.Kernel.Lsq_I,vmin=0,vmax=0.03,cmap='BuPu')
                plt.show()
            
            
        self.iter +=1


        return f_list,Kf_pos_list,f_rmse
    
    def calc_r_torch(self,
        refI       :np.ndarray,
        f_list     :List[np.ndarray],
        Kf_now_list:List[np.ndarray],
        sig_scale  :float =0.01,
        w          :float=1.0,
        alpha_d    :Optional[float]=None,
        is_hessian :bool=True,
        is_plot    :bool = True,
        )->Tuple[np.ndarray,np.ndarray]:
        
        N = len(f_list)

        expf_list = [np.exp(f_list[i]+0.5*np.diag(Kf_now_list[i])) for i in range(N)]

        if alpha_d is None:
            sigiH0f_tor_list = [torch.FloatTensor((self.sigiH0 @expf)) for expf in expf_list]
        else:
            sigiHsum = self.sigiH0+alpha_d*(self.msigi_Interp_diff@self.H_diff_I)
            sigiH0f_tor_list = [torch.FloatTensor((sigiHsum @expf)) for expf in expf_list]

        sigiH1f_tor_list = [torch.FloatTensor((self.sigiH1@expf)) for expf in expf_list]
        sigiH2f_tor_list = [torch.FloatTensor((self.sigiH2@expf)) for expf in expf_list]
        
        sigiH0f_sigiH1f_sum = sum([sigiH0f_tor_list[i]*sigiH1f_tor_list[i] for i in range(N) ])
        sigiH1f_sq_sum      = sum([sigiH1f_tor_list[i]*sigiH1f_tor_list[i] for i in range(N) ])
        sigiobs_sigiH1f_sum = sum([self.sigi_obs_tor_list[i]*sigiH1f_tor_list[i] for i in range(N) ])

        if self.n_reflection == 2:    
            sigiH0f_sigiH2f_sum = sum([sigiH0f_tor_list[i]*sigiH2f_tor_list[i] for i in range(N) ])
            sigiH1f_sigiH2f_sum = sum([sigiH1f_tor_list[i]*sigiH2f_tor_list[i] for i in range(N) ])
            sigiH2f_sq_sum      = sum([sigiH2f_tor_list[i]*sigiH2f_tor_list[i] for i in range(N) ])
            sigiobs_sigiH2f_sum = sum([self.sigi_obs_tor_list[i]*sigiH2f_tor_list[i] for i in range(N) ])

        """ ##リスト化に対応してないLL関数
        def LL_for_r_torch(rI:torch.Tensor):
            Ref0 = self.mask_w_tor*1/(1+torch.exp(-self.T_I20_tor@rI))
            term1 = -1/2*torch.sum(2*Ref0*sigiH1f_tor*sigiH0f_tor+Ref0**2*sigiH1f_tor**2)*1.0
            term2 = torch.sum(Ref0*sigi_obs_tor *sigiH1f_tor)
            rI= rI-self.mur_pri_tor
            term3 = -1/2*torch.sum(rI@(self.Kr_II_pri_inv_tor@rI))
            #return term3
            return (1/sig_scale**2*(term1+term2)+w**2*term3)
        """
        def LL_torch_ref1(rI:torch.Tensor):
            Ref0 = self.T0_tor @ (1/(1+torch.exp(-rI)))
            Term_1and2 = -1/2*torch.sum(2*Ref0*(sigiH0f_sigiH1f_sum-sigiobs_sigiH1f_sum)+Ref0**2*sigiH1f_sq_sum )

            rI= rI-self.mur_pri_tor
            Term3 = -1/2*torch.sum(rI@(self.Kr_II_pri_inv_tor@rI))
            #return term3
            return (1/sig_scale**2*(Term_1and2)+w**2*Term3)
        
        def  LL_torch_ref2(rI:torch.Tensor):
            Ref0 = self.T0_tor @ (1/(1+torch.exp(-rI)))
            Ref1 = self.T1_tor @ (1/(1+torch.exp(-rI)))
            Term_1and2 = -1/2*torch.sum( 
                 2*Ref0            *(sigiH0f_sigiH1f_sum-sigiobs_sigiH1f_sum)
                +2*Ref0    *Ref1   *(sigiH0f_sigiH2f_sum-sigiobs_sigiH2f_sum) 
                +2*Ref0**2 *Ref1   * sigiH1f_sigiH2f_sum
                +  Ref0**2          *sigiH1f_sq_sum
                +  Ref0**2 *Ref1**2 *sigiH2f_sq_sum )

            rI= rI-self.mur_pri_tor
            Term3 = -1/2*torch.sum(rI@(self.Kr_II_pri_inv_tor@rI))
            #return term3
            return (1/sig_scale**2*(Term_1and2)+w**2*Term3)
                
        if   self.n_reflection == 1:
            LL_torch:Callable[[torch.Tensor],torch.Tensor] = LL_torch_ref1
        elif self.n_reflection ==2 : 
            LL_torch:Callable[[torch.Tensor],torch.Tensor] = LL_torch_ref2
        else:
            print('something err')
            return 
        Phi0 = LL_torch(torch.zeros(refI.size)).numpy().astype(np.float32)

        def Phi(x):
            x = torch.FloatTensor(x).requires_grad_(False)
            y = LL_torch(x)
            return -y.numpy().astype(np.float64)/abs(Phi0)


            
        def Phi_grad(x: torch.Tensor):
            x = torch.FloatTensor(x).requires_grad_(True)
            y = LL_torch(x)
            y.backward()
            return -x.grad.numpy().astype(np.float32)/abs(Phi0)
        
        def Phi_hessian(x: torch.Tensor):
            x = torch.FloatTensor(x).requires_grad_(True)
            print('calc hessian')
            return hessian( LL_torch,x).numpy().astype(np.float32)
        
        x = refI
        time0 = time.time()
        res = scipy.optimize.fmin_cg(f=Phi,x0=x.astype(np.float32),fprime=Phi_grad,full_output=True)
        time1 = time.time()
        print('takes '+str(time1-time0)+' sec.')
        print(Phi_grad(res[0]).std())
        refI = res[0]
        Phi_i= res[1]

        if not is_hessian:
            return refI,Phi_i

        
        time0 = time.time()
        Kr_pos_inv = -Phi_hessian(refI)
        time1 = time.time()
        print('takes '+str(time1-time0)+' sec.')
        lam,V = np.linalg.eigh(Kr_pos_inv)
        lam[lam<1e-5]= 1e-5
        Kr_pos = V @ np.diag(1/lam) @ V.T
        if self.ref_true is not None or is_plot:    
            ref_true = self.ref_true
            fig,axs=plt.subplots(1,3,figsize=(15,5),sharey=True)
            CI,HI = self.Ref.CI,self.Ref.HI
            scatter_cbar(axs[0],x=CI,y=HI,c=sigmoid(refI),s=200,vmax=1,vmin=0)
            
            scatter_cbar(axs[1],x=CI,y=HI,c=sigmoid(refI)-sigmoid(ref_true),s=200,vmin=-0.5,vmax=0.5,cmap='RdBu_r')
            
            r_RMSE = (sigmoid(refI)-sigmoid(self.ref_true)).std()
            axs[1].set_title('nrmse = '+str(r_RMSE)[:8])


            scatter_cbar(axs[2],x=CI,y=HI,c=0.5*(sigmoid(refI+np.sqrt(np.diag(Kr_pos)))-sigmoid(refI-np.sqrt(np.diag(Kr_pos)))),s=200,cmap='BuPu',vmin=0)
            for ax in axs:
                ax.set_xlim(ax.get_xlim()[::-1])
                ax.set_ylim(-0.6,0.6)
            plt.show()
        else :
            r_RMSE = None


        return refI,Kr_pos,Phi_i,r_RMSE
    
    def calc_diffusion(self,
        f_list      :List[np.ndarray],
        refI    :np.ndarray,
        para :float = 1,
        alpha_bk :float = None 
        ): 
        sgmI = sigmoid(refI)
        ref0 = self.Ref.T_I2w[0] @sgmI
        ref1 = self.Ref.T_I2w[1] @sgmI

        temp1,temp2 = 0,0 
        for i,f in enumerate(f_list):
            expf = np.exp(f)
            sigiHd_exp = self.msigi_Interp_diff @ (self.H_diff.H_diff_I @expf)
            prj = self.sigiH0 @ expf + ref0 *(self.sigiH1 @expf) + ref0*ref1*(self.sigiH2 @expf) 
            #cm = plt.imshow(prj.reshape(200,200))
            #plt.colorbar(cm)
            #plt.show()
            temp1 += np.sum((self.sigi_obs_list[i]-prj)*sigiHd_exp)
            temp2 += np.sum(sigiHd_exp**2)
            #print(i,temp1,temp2)

        alpha_d = temp1/temp2  

        print('coef of diffusion is '+str(alpha_d))

        if alpha_bk is not None:
            d_alpha = alpha_d -alpha_bk
            alpha_d = para*d_alpha + alpha_bk

        if alpha_d < 0 : alpha_d=0


        return alpha_d 
    
    def calc_diffusion2(self,
        f_list      :List[np.ndarray],
        Kf_now_list:List[np.ndarray],
        refI_now    :np.ndarray,
        Kr_now  :np.ndarray,

        ): 
        N = len(f_list)
        sgmI_sample=self.sgmI_sample

        #sgmI_sample = sigmoid(np.random.multivariate_normal(refI,Kr_now+1e-1*np.eye(self.Ref.NI),490))
        ref0_sample = self.Ref.T_I2w[0] @sgmI_sample.T
        ref1_sample = self.Ref.T_I2w[1] @sgmI_sample.T
        ##
        ref0_mean_sp  = sparse.diags(~self.ref_mask   *(ref0_sample   ).mean(axis=1))
        ref01_mean_sp = sparse.diags(~self.ref_mask   *(ref0_sample*ref1_sample).mean(axis=1))
        expf_list = [np.exp(f_list[i]+0.5*np.diag(Kf_now_list[i])) for i in range(N)]
        temp1,temp2 = 0,0 
        for i in range(N):
            expf = expf_list[i]
            sigiHd_exp = self.msigi_Interp_diff @ (self.H_diff.H_diff_I @expf)
            prj = self.sigiH0 @ expf + ref0_mean_sp @ (self.sigiH1 @expf) + ref01_mean_sp@(self.sigiH2 @expf) 
            #cm = plt.imshow(prj.reshape(200,200))
            #plt.colorbar(cm)
            #plt.show()
            temp1 += np.sum((self.sigi_obs_list[i]-prj)*sigiHd_exp)
            temp2 += np.sum(sigiHd_exp**2)
            #print(i,temp1,temp2)

        alpha_d = temp1/temp2  

        print('coef of diffusion is '+str(alpha_d))

        if alpha_d < 0 : alpha_d=0

        return alpha_d 
    
    def projection(self,
        f   :np.ndarray,
        refI:np.ndarray,
        n_relection:int=1,
        ref_mask :Optional[np.ndarray]=None,
        diffusion_coef:float = 0,):

        if ref_mask is None:                  
            #ref_maskが無指定ならば, set_obsで代入したref_maskを使用する.
            ref_mask = self.ref_mask
        elif type(ref_mask) is not np.ndarray:
            #ref_mask がnumpy じゃなければ，強制的にmaskなし
            ref_mask = np.array([False])
        elif  type(ref_mask.flatten()[0]) is not np.bool_:
            #numpyのtypeがboolじゃなければ，boolに治す.
            ref_mask = (ref_mask >= 1).flatten()
        else:
            #以上の条件を満たせば，そのまま代入可能
            ref_mask = ref_mask.flatten()

        g:np.ndarray = 0
        if n_relection == 0:
            g = self.Obs.Hs[0].projection(f) 
            
        elif n_relection == 1:
            Ref0 = ~ref_mask *sigmoid(self.Ref.T_I2w[0] @ refI)
            Ref0 = Ref0.reshape(*self.im_shape)
            g = self.Obs.Hs[0].projection(f) + Ref0 *self.Obs.Hs[1].projection(f)

        elif n_relection == 2:
            Ref0 = ~ref_mask *sigmoid(self.Ref.T_I2w[0] @ refI)
            Ref0 = Ref0.reshape(*self.im_shape)
            
            Ref1 = sigmoid(self.Ref.T_I2w[1] @ refI)
            Ref1 = Ref1.reshape(*self.im_shape)
            g = self.Obs.Hs[0].projection(f) + Ref0 *self.Obs.Hs[1].projection(f) +  Ref0*Ref1 *self.Obs.Hs[2].projection(f)
        else:
            return print('err in projection')
        
        if self.H_diff is not None:
            g = g+ ~ref_mask.reshape(*self.im_shape) *diffusion_coef*(self.H_diff@f).reshape(*self.im_shape)
        
        return g




def SEKer_z(z1,z2,z_len):
    z1 = z1.flatten()
    z2 = z2.flatten()
    Z  = np.meshgrid(z1,z2,indexing='ij') 
    return np.exp(-0.5*(Z[0]-Z[1])**2/z_len**2)



def SEKer_cz(c1,c2,z1,z2,c_len,z_len):
    c1 = c1.flatten()
    c2 = c2.flatten()
    z1 = z1.flatten()
    z2 = z2.flatten()
    C  = np.meshgrid(c1,c2,indexing='ij') 
    Z  = np.meshgrid(z1,z2,indexing='ij') 
    return np.exp(-0.5*(Z[0]-Z[1])**2/z_len**2)*np.exp(-0.5*(C[0]-C[1])**2/c_len**2)
def SEKer_dcz(c1,c2,z1,z2,c_len,z_len):
    c1 = c1.flatten()
    c2 = c2.flatten()
    z1 = z1.flatten()
    z2 = z2.flatten()
    C  = np.meshgrid(c1,c2,indexing='ij') 
    Z  = np.meshgrid(z1,z2,indexing='ij') 
    return  np.exp(-0.5*(Z[0]-Z[1])**2/z_len**2)*-(C[0]-C[1])/c_len**2*np.exp(-0.5*(C[0]-C[1])**2/c_len**2)
    
def SEKer_ddcz(c1,c2,z1,z2,c_len,z_len):
    c1 = c1.flatten()
    c2 = c2.flatten()
    z1 = z1.flatten()
    z2 = z2.flatten()
    C  = np.meshgrid(c1,c2,indexing='ij') 
    Z  = np.meshgrid(z1,z2,indexing='ij') 
    return np.exp(-0.5*(Z[0]-Z[1])**2/z_len**2)*(1-(C[0]-C[1])**2/c_len**2)/c_len**2*np.exp(-0.5*(C[0]-C[1])**2/c_len**2)


def sigmoid(f):
    return 1/(1+np.exp(-f))

def sigmoid_inv(f):
    return np.log(f/(1-f))

