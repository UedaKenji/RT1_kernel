from pkgutil import extend_path
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numpy import FPE_DIVIDEBYZERO, array, linalg, ndarray
import rt1plotpy
from typing import Optional, Union,Tuple,Callable,List,TypeAlias
import time 
import math
from tqdm import tqdm
import scipy.linalg as linalg
from numba import jit
import warnings
from dataclasses import dataclass
import itertools
import scipy.sparse as sps
import pandas as pd
import os,sys
from .plot_utils import *  

import rt1kernel

sys.path.insert(0,os.pardir)
import rt1raytrace
import sparse_dot_mkl



__all__ = ['GPT_lin', 
           'GPT_av',  
           'GPT_cis', 
           'GPT_log', 
           'GPT_log_grid']

csr  = sps.csr_matrix


def csr_cos(A ,Exist)->csr:
    """
 
    """
    data = np.array( np.cos(A[Exist==True]) ).flatten() # type: ignore
    return sps.csr_array( (data, Exist.indices, Exist.indptr),shape=Exist.shape)

__all__ = []

"""
class GPT_av_old:
    def __init__(self,
        Obs: rt1kernel.Observation_Matrix_integral,
        Kernel: rt1kernel.Kernel2D_scatter,
        ) -> None:
        self.Obs = Obs
        self.rI = Obs.rI 
        self.zI = Obs.zI 
        self.nI = Obs.zI.size  
        self.Kernel = Kernel
        pass

    def set_kernel(self,
        K_a :np.ndarray,
        K_v :np.ndarray,
        a_pri:np.ndarray |float = 0,
        v_pri:np.ndarray |float = 0,
        regularization:float = 1e-5,
        ):
        K_a += regularization*np.eye(self.nI)
        K_v += regularization*np.eye(self.nI)

        self.K_a = 0.5*(K_a + K_a.T)
        self.K_v = 0.5*(K_v + K_v.T)
        K_a_inv = np.linalg.inv(self.K_a)
        K_v_inv = np.linalg.inv(self.K_v)
        self.K_a_inv = 0.5*(K_a_inv+K_a_inv.T)
        self.K_v_inv = 0.5*(K_v_inv+K_v_inv.T)
        self.a_pri = a_pri
        self.v_pri = v_pri

        pass 

    def set_sig(self,
        sigma:np.ndarray,
        A_cos:np.ndarray,
        A_sin:np.ndarray,
        num:int=0,
        ):
        sigma = sigma.flatten()
        A_cos = A_cos.flatten()
        A_sin = A_sin.flatten()
        self.sig_inv = 1/sigma
        self.sig2_inv = 1/sigma**2
        Dcos = self.Obs.Hs[num].Dcos
        H    = self.Obs.Hs[num].H

        self.sigiH   :sps.csr_matrix =  sps.diags(self.sig_inv) @ H  
        self.sigiHT  :sps.csr_matrix = self.sigiH.multiply(Dcos)  # type: ignore
        self.sigiHT2 :sps.csr_matrix = self.sigiHT.multiply(Dcos)
        self.SiA = self.sig_inv*(A_cos + A_sin*1.j)
        
    def set_sig2(self,
        sigma:np.ndarray,
        A_cos:np.ndarray,
        A_sin:np.ndarray,
        num:int=0,
        ):
        sigma = sigma.flatten()
        A_cos = A_cos.flatten()
        A_sin = A_sin.flatten()
        self.sig_inv = 1/sigma
        self.sig2_inv = 1/sigma**2
        Dcos  = 1.j*self.Obs.Hs[num].Dcos
        #Dcos  = 1.j*self.Obs.Hs[num].Exist
        H    = self.Obs.Hs[num].H

        self.sigiH =  sps.csr_matrix( sps.diags(self.sig_inv) @ H )
        self.sigiHT  :sps.csr_matrix = self.sigiH.multiply(Dcos) 
        self.sigiHT2 :sps.csr_matrix = self.sigiHT.multiply(Dcos)
        self.SiA = self.sig_inv*(A_cos + A_sin*1.j)

    def calc_core2(self,
        a:np.ndarray,
        v:np.ndarray,
        num:int=0
        ):
        r_a = a - self.a_pri
        r_v = v - self.v_pri
        Exp = self.Obs.Hs[num].Exp(a,v)

        SiHE  :sps.csr_matrix = self.sigiH.multiply(Exp) # m*n 
        SiHTE :sps.csr_matrix = self.sigiHT.multiply(Exp)
        SiHT2E:sps.csr_matrix = self.sigiHT2.multiply(Exp)

        SiHE_conj = SiHE.conjugate()
        SiHTE_conj= SiHTE.conjugate()
        SiR = np.asarray(SiHE.sum(axis=1)).flatten()- self.SiA 
        
        #fig,ax=plt.subplots(1,2,figsize=(10,5))
        #imshow_cbar(fig,ax[0],(np.sum(SiHE,axis=1).real).reshape(200,200))
        #imshow_cbar(fig,ax[1],(np.sum(SiHE,axis=1).imag).reshape(200,200))
        #plt.show()
        
        #fig,ax=plt.subplots(1,2,figsize=(10,5))
        #imshow_cbar(fig,ax[0],(SiR.real).reshape(200,200))
        #imshow_cbar(fig,ax[1],(SiR.imag).reshape(200,200))
        #plt.show()
        SiR_conj = np.conj(SiR)
        c1 = (SiHE.T   @ SiR_conj).real 
        c2 = (SiHTE.T  @ SiR_conj).real
        c3 = (SiHT2E.T @ SiR_conj).real

        C1 = ((SiHE_conj.T  @ SiHE ).real).toarray()

        C2 = ((SiHTE_conj.T @ SiHTE).real).toarray()

        C3 = ((SiHTE_conj.T @ SiHE ).real).toarray()


        Psi_da   = -c1 - self.K_a_inv @ r_a 
        Psi_dv   = -c2 - self.K_v_inv @ r_v
        Psi_dada = -C1 - np.diag(c1)*1 - self.K_a_inv
        Psi_dvdv = -C2*1 + np.diag(c3)*1- self.K_v_inv
        Psi_dadv = (-C3*1 + np.diag(c2)*0).T
        Psi_dvda =  Psi_dadv.T


        nI = self.nI
        
        NPsi      = np.zeros((2*nI))
        NPsi[:nI] = Psi_da[:]
        NPsi[nI:] = Psi_dv[:]
        #NPsi = np.concatenate([Psi_da,Psi_dv])

        DPsi = np.zeros((2*nI,2*nI))

        #plt.hist(((SiHE.T   @ SiR_conj).real).flatten(),bins=50)
        #plt.hist(((SiHE.T   @ SiR_conj).imag).flatten(),bins=50)
        #plt.hist(((SiHTE.T   @ SiR_conj).real).flatten(),bins=50)
        #plt.hist(((SiHTE.T   @ SiR_conj).imag).flatten(),bins=50)
        #plt.hist(Psi_da,bins=50)
        #plt.show()

        DPsi[:nI,:nI] = Psi_dada[:,:]
        DPsi[nI:,nI:] = Psi_dvdv[:,:]*1
        DPsi[:nI,nI:] = Psi_dvda[:,:]*1
        DPsi[nI:,:nI] = Psi_dadv[:,:]*1

        #plt.figure(figsize=(30,30))
        #c = plt.imshow(DPsi,cmap='turbo',vmax=abs(DPsi).max(),vmin=-abs(DPsi).max())
        #plt.colorbar(c)
        #plt.show()

        del Psi_dada,Psi_dvdv,Psi_dvda,Psi_dadv

        delta_av = - np.linalg.solve(DPsi+0*np.eye(self.nI*2),NPsi)
        
        delta_av[delta_av<-3] = -3
        delta_av[delta_av>+3] = +3
        delta_a = delta_av[:nI]
        delta_v = delta_av[nI:]

        return delta_a,delta_v

    def calc_core(self,
        a:np.ndarray,
        v:np.ndarray,
        num:int=0
        ):
        r_a = a - self.a_pri
        r_v = v - self.v_pri
        Exp = self.Obs.Hs[num].Exp(a,v)

        SiHE  :sps.csr_matrix = self.sigiH.multiply(Exp) # m*n 
        SiHTE :sps.csr_matrix = self.sigiHT.multiply(Exp)
        SiHT2E:sps.csr_matrix = self.sigiHT2.multiply(Exp)

        SiHE_conj = SiHE.conjugate()
        SiHTE_conj = SiHTE.conjugate()
        SiR = np.asarray(SiHE.sum(axis=1)).flatten()- self.SiA 
        SiR_conj = np.conj(SiR)
        c1 = (SiHE.T @ SiR_conj).real 
        c2 = (1.j*SiHTE.T @ SiR_conj).real
        c3 = (SiHT2E.T @ SiR_conj).real

        C1 = ((SiHE_conj.T @ SiHE).real).toarray()

        C1_2 = ((SiHE_conj.T @ SiHE).real).toarray()
        C2 = ((SiHTE_conj.T @ SiHTE).real).toarray()
        C3 = ((1.j*SiHTE_conj.T @ SiHE).real).toarray()

        Psi_da   = -c1 - self.K_a_inv @ r_a 
        Psi_dv   = -c2 - self.K_v_inv @ r_v
        Psi_dada = -C1 - np.diag(c1)*1 - self.K_a_inv
        Psi_dvdv = +C2*0 - np.diag(c3)*1- self.K_v_inv
        Psi_dadv = -C3*0 - np.diag(c2)*0 
        Psi_dvda = Psi_dadv.T 


        nI = self.nI
        DPsi = np.zeros((2*nI,2*nI))
        NPsi = np.concatenate([Psi_da,Psi_dv])

        DPsi[:nI,:nI] = Psi_dada[:,:]
        DPsi[nI:,nI:] = Psi_dvdv[:,:]
        DPsi[:nI,nI:] = Psi_dvda[:,:]*0
        DPsi[nI:,:nI] = Psi_dadv[:,:]*0

        print(abs((DPsi-DPsi.T)).max())
        #plt.figure(figsize=(30,30))
        #c = plt.imshow(DPsi,cmap='turbo',vmax=abs(DPsi).max(),vmin=-abs(DPsi).max())
        #plt.colorbar(c)
        #plt.show()

        del Psi_dada,Psi_dvdv,Psi_dvda,Psi_dadv

        delta_av = - np.linalg.solve(DPsi,NPsi)
        
        delta_av[delta_av<-3] = -3
        delta_av[delta_av>+3] = +3
        delta_a = delta_av[:nI]
        delta_v = delta_av[nI:]

        return delta_a,delta_v
        
    def check_diff(self,
        a:np.ndarray,
        v:np.ndarray):
            
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        A = self.Obs.Hs[0].projection_A2(a,v)
        imshow_cbar(fig,ax[0],A.real)
        imshow_cbar(fig,ax[1],A.imag)
        
        plt.show()
"""

class GPT_lin:
    def __init__(self,
        Obs: rt1kernel.Observation_Matrix_integral,
        Kernel: rt1kernel.Kernel2D_scatter,
        ) -> None:
        self.Obs = Obs
        self.rI = Obs.rI 
        self.zI = Obs.zI 
        self.nI = Obs.zI.size  
        self.Kernel = Kernel
        pass

    def set_prior(self,
        K :np.ndarray,
        f_pri :np.ndarray | float = 0,
        regularization:float = 1e-6,
        ):
        K += regularization*np.eye(self.nI)

        self.K_inv = np.linalg.inv(K)
        self.f_pri = f_pri

    def calc_core(self):
        self.K_pos = np.linalg.inv( self.Hsig2iH+self.K_inv )
        self.mu_f_pos = self.f_pri + self.K_pos @ (self.sigiH.T @ (self.Sigi_obs-self.sigiH @self.f_pri))

        return self.mu_f_pos,self.K_pos



    def set_sig(self,
        sig_array:np.ndarray,
        g_obs:np.ndarray,
        sig_scale:float=1.0,
        num:int=0,
        ):
        self.g_obs=g_obs.reshape(self.Obs.shape[:2])
        self.sig_scale = sig_scale
        sig_array = sig_array.flatten()
        g_obs = g_obs.flatten()
        self.sig_inv = 1/sig_array
        #self.sig2_inv = 1/sig_array**2
        H    = self.Obs.Hs[num].H

        self.Sigi_obs = self.sig_inv*(g_obs)
        self.sigiH = sps.csr_matrix(sps.diags(self.sig_inv) @ H )
        sigiH_t = sps.csr_matrix( self.sigiH.T )

        #self.Hsig2iH  = (self.sigiH.T @ self.sigiH).toarray() 
        # self.Hsig2iH = sparse_dot_mkl.gram_matrix_mkl(sigiH_t,transpose=True,dense=True)　#2時間溶かした戦犯
        self.Hsig2iH = sparse_dot_mkl.dot_product_mkl(sigiH_t,self.sigiH ,dense=True)

    
    def check_diff(self,
        f:np.ndarray):
            
        fig,ax = plt_subplots(1,3,figsize=(12,3.))
        ax = ax[0][:]
        g = self.Obs.Hs[0].projection(f)
        imshow_cbar(ax[0],g,origin='lower')
        ax[0].set_title('Hf')
        vmax = (abs(g-self.g_obs)).max()
        imshow_cbar(ax= ax[1],im0 = g-self.g_obs,vmin=-vmax,vmax=vmax,cmap='RdBu_r',origin='lower')
        ax[1].set_title('diff_im')
        
        ax[2].hist((g-self.g_obs).flatten(),bins=50)

        ax[2].tick_params( labelleft=False)
        plt.show()


class GPT_av:
    def __init__(self,
        Obs: rt1kernel.Observation_Matrix_integral,
        Kernel: rt1kernel.Kernel2D_scatter,
        ) -> None:
        self.Obs = Obs
        self.rI = Obs.rI 
        self.zI = Obs.zI 
        self.nI = Obs.zI.size  
        self.Kernel = Kernel
        self.H   = Obs.Hs[0].H
        self.Dec = Obs.Hs[0].Dcos
        self.Exist = Obs.Hs[0].Exist
        pass

    def set_kernel(self,
        K_a :np.ndarray,
        K_v :np.ndarray,
        a_pri:np.ndarray | float = 0,
        v_pri:np.ndarray | float = 0,
        regularization:float = 1e-5,
        ):
        K_a += regularization*np.eye(self.nI)
        K_v += regularization*np.eye(self.nI)

        self.K_a = 0.5*(K_a + K_a.T)
        self.K_v = 0.5*(K_v + K_v.T)
        K_a_inv = np.linalg.inv(self.K_a)
        K_v_inv = np.linalg.inv(self.K_v)
        self.K_a_inv = 0.5*(K_a_inv+K_a_inv.T)
        self.K_v_inv = 0.5*(K_v_inv+K_v_inv.T)
        self.a_pri = a_pri
        self.v_pri = v_pri

        self.K_f_inv = np.zeros((2*self.nI,2*self.nI))
        self.K_f_inv[:self.nI ,:self.nI ] = self.K_a_inv[:,:]
        self.K_f_inv[ self.nI:, self.nI:] = self.K_v_inv[:,:]

        pass 

        
    def set_sig(self,
        sigma:np.ndarray,
        A_cos:np.ndarray,
        A_sin:np.ndarray,
        num:int=0,
        #sig_scale:float = 1.0,
        ):
        sigma = sigma.flatten()
        A_cos = A_cos.flatten()
        A_sin = A_sin.flatten()
        self.sig_inv = 1/sigma
        self.sig2_inv = 1/sigma**2
        #Dcos  = 1.j*self.Obs.Hs[num].Dcos
        H    = self.Obs.Hs[num].H

        self.sigiH   : sps.csr_matrix = sps.diags(self.sig_inv) @ H  
        self.sigiA = np.hstack((self.sig_inv*A_cos, self.sig_inv*A_sin))
    

    def calc_core(self,
        a:np.ndarray,
        v:np.ndarray,
        num:int=0,
        sig_scale:float = 1.0
        ):
        r_a = a - self.a_pri
        r_v = v - self.v_pri
        
        sig_scale_inv = 1/sig_scale
        DecV = sps.csr_matrix(self.Dec @ sps.diags(v))
        Hc = self.sigiH.multiply(csr_cos(DecV,self.Exist))
        Hs = self.sigiH.multiply(DecV.sin())
        Exp_a = sps.diags(np.exp(a))
        self.Rc = sps.csr_matrix(Hc @ Exp_a )
        self.Rs = sps.csr_matrix(Hs @ Exp_a )
        Ac = np.asarray(self.Rc.sum(axis=1)).flatten()
        As = np.asarray(self.Rs.sum(axis=1)).flatten()
        sigi_g  = np.hstack((Ac,As))
        self.resA = sig_scale_inv *(sigi_g-self.sigiA)

        self.Rc_Dec = self.Rc.multiply(self.Dec)
        self.Rs_Dec = self.Rs.multiply(self.Dec)

        Jac = sps.vstack(
                (sps.hstack((self.Rc, -self.Rs_Dec)),
                 sps.hstack((self.Rs,  self.Rc_Dec)))
                ) *sig_scale_inv
        
        Jac_t = sps.csr_matrix(Jac.T)
        
        nabla_Phi = -np.array(Jac_t @ self.resA) - np.hstack((self.K_a_inv @r_a, self.K_v_inv @r_v)) #type: ignore

        #W1 = sparse_dot_mkl.gram_matrix_mkl( sps.csr_matrix(Jac.T),dense=True,transpose=True)
        #W1 = W1 + W1.T - np.diag(W1.diagonal())

        W1 = sparse_dot_mkl.dot_product_mkl( Jac_t ,Jac ,dense=True)
        loss = abs(nabla_Phi).mean()
        # W2 = self._W2()

        laplace_Phi = - W1  - self.K_f_inv # -W2*1
        self.laplace_Phi = laplace_Phi
        delta_f = - np.linalg.solve(laplace_Phi, nabla_Phi)
        delta_f[delta_f<-5] = -5
        delta_f[delta_f>+5] = +5 

        delta_a = delta_f[:self.nI]
        delta_v = delta_f[self.nI:]
        return delta_a, delta_v,loss
    
    def K_pos(self,
        consider_w2 :bool = True,
        ):
        if consider_w2:
            K_inv = -self.laplace_Phi+self._W2()
        else:
            K_inv = -self.laplace_Phi
        return np.linalg.inv(K_inv)

    def _W2(self,
        ):

        Rc_Dec2 =  self.Rc_Dec.multiply(self.Dec)
        Rs_Dec2 =  self.Rs_Dec.multiply(self.Dec)

        d_aa = sps.hstack((  self.Rc.T, self.Rs.T )) @ self.resA
        d_vv = sps.hstack(( -Rc_Dec2.T, -Rs_Dec2.T )) @ self.resA
        d_av = sps.hstack((-self.Rs_Dec.T,  self.Rc_Dec.T))  @ self.resA
        
        W2 = np.zeros((2*self.nI, 2*self.nI))
        W2[:self.nI,  :self.nI ] = np.diag(d_aa)[:,:]
        W2[ self.nI:, :self.nI ] = np.diag(d_av)[:,:]
        W2[:self.nI ,  self.nI:] = np.diag(d_av)[:,:]
        W2[ self.nI:,  self.nI:] = np.diag(d_vv)[:,:]

        return W2



    def check_diff(self,
        a:np.ndarray,
        v:np.ndarray):
        m = self.H.shape[0]
        fig,axs = plt_subplots(2,2,figsize=(8,8),sharex=True,sharey=True)
        A = self.Obs.Hs[0].projection_A2(a,v)
        g_cos,g_sin = A.real,A.imag
        y = self.sigiA
        y_cos = (1/self.sig_inv*y[:m]).reshape(*self.Obs.shape[:2])
        y_sin = (1/self.sig_inv*y[m:]).reshape(*self.Obs.shape[:2])
        imshow_cbar(axs[0][0],g_cos)
        imshow_cbar(axs[1][0],g_sin)
        diff1 = g_cos-y_cos
        diff2 = g_sin-y_sin
        vmax = max(np.percentile(diff1,95),-np.percentile(diff1,5),np.percentile(diff2,95),-np.percentile(diff2,5)) #type: ignore
        imshow_cbar(axs[0][1],g_cos-y_cos,cmap='RdBu_r',vmax=vmax,vmin=-vmax)
        imshow_cbar(axs[1][1],g_sin-y_sin,cmap='RdBu_r',vmax=vmax,vmin=-vmax)
        
        axs[0][0].set_title(r'$g^{\cos}$')
        axs[1][0].set_title(r'$g^{\sin}$')
        axs[0][1].set_title(r'diff: $g^{\cos}-y^{\cos}$')
        axs[1][1].set_title(r'diff: $g^{\sin}-y^{\sin}$')
        plt.show()

GPT_cis = GPT_av


class GPT_log:
    def __init__(self,
        Obs: rt1kernel.Observation_Matrix_integral,
        Kernel: rt1kernel.Kernel2D_scatter,
        ) -> None:
        self.Obs = Obs
        self.rI = Obs.rI 
        self.zI = Obs.zI 
        self.nI = Obs.zI.size  
        self.Kernel = Kernel
        pass

    def set_kernel(self,
        K :np.ndarray,
        f_pri :np.ndarray | float = 0,
        regularization:float = 1e-6,
        ):
        K += regularization*np.eye(self.nI)

        self.K_inv = np.linalg.inv(K)
        self.f_pri = f_pri

    def set_sig(self,
        sig_array:np.ndarray,
        g_obs:np.ndarray,
        sig_scale:float=1.0,
        num:int=0,
        ):
        self.g_obs=g_obs.reshape(self.Obs.shape[:2])
        self.sig_scale = sig_scale
        sig_array = sig_array.flatten()
        g_obs = g_obs.flatten()
        self.sig_inv = 1/sig_array
        #self.sig2_inv = 1/sig_array**2
        H    = self.Obs.Hs[num].H

        self.Sigi_obs = self.sig_inv*(g_obs)
        self.sigiH = sps.csr_matrix(sps.diags(self.sig_inv) @ H )
        sigiH_t = sps.csr_matrix( self.sigiH.T )

        #self.Hsig2iH  = (self.sigiH.T @ self.sigiH).toarray() 
        # self.Hsig2iH = sparse_dot_mkl.gram_matrix_mkl(sigiH_t,transpose=True,dense=True)　#2時間溶かした戦犯
        self.Hsig2iH = sparse_dot_mkl.dot_product_mkl(sigiH_t,self.sigiH ,dense=True)

    
    def check_diff(self,
        f:np.ndarray):
            
        fig,ax = plt.subplots(1,3,figsize=(10,4))
        g = self.Obs.Hs[0].projection(np.exp(f))
        imshow_cbar(ax[0],g,origin='lower')
        ax[0].set_title('Hf')
        vmax = (abs(g-self.g_obs)).max()
        imshow_cbar(ax= ax[1],im0 = g-self.g_obs,vmin=-vmax,vmax=vmax,cmap='RdBu_r',origin='lower')
        ax[1].set_title('diff_im')
        
        ax[2].hist((g-self.g_obs).flatten(),bins=50)
        plt.show()

    
    def calc_core_fast(self,
        f:np.ndarray,
        num:int=0,
        ):
        r_f = f - self.f_pri
        exp_f = np.exp(f)
        fxf = np.einsum('i,j->ij',exp_f,exp_f)
        
        SiR = self.sigiH @ exp_f - self.Sigi_obs

        c1 = 1/self.sig_scale**2 *(self.sigiH.T @ SiR) * exp_f #self.sigiH.T @ SiR =  self.Hsig2iH  @ exp_f -  self.sigiH.T@ self.Sigi_obs
        C1 = 1/self.sig_scale**2 *self.Hsig2iH * fxf 

        Psi_df   = -c1 - self.K_inv @ r_f 

        Psi_dfdf = -C1 - np.diag(c1) - self.K_inv

        DPsi = Psi_dfdf
        NPsi = Psi_df
        loss = abs(NPsi).mean()

        delta_f = - np.linalg.solve(DPsi,NPsi)

        delta_f[delta_f<-3] = -3
        delta_f[delta_f>+3] = +3

        return delta_f,loss
    
    def set_postprocess(self,
        f:npt.NDArray[np.float64],
        ):
        exp_f = np.exp(f)
        fxf = np.einsum('i,j->ij',exp_f,exp_f)
        
        SiR = self.sigiH @ exp_f - self.Sigi_obs

        c1 = 1/self.sig_scale**2 *(self.sigiH.T @ SiR) * exp_f #self.sigiH.T @ SiR =  self.Hsig2iH  @ exp_f -  self.sigiH.T@ self.Sigi_obs
        C1 = 1/self.sig_scale**2 *self.Hsig2iH * fxf 

        Psi_dfdf = -C1 - np.diag(c1) - self.K_inv

        DPsi = Psi_dfdf
        self.Kf_pos_inv = -DPsi
        self.Kf_pos     = np.linalg.inv(self.Kf_pos_inv)
        self.sigf_pos = np.sqrt(np.diag(self.Kf_pos))

        pass
    

class GPT_log_grid:
    def __init__(self,
        H: sps.csr_matrix,
        ray:rt1raytrace.Ray,
        Kernel: rt1kernel.Kernel2D_grid,
        ) -> None:
        self.H = H  
        self.Kernel = Kernel
        self.ng = Kernel.R_grid.size
        self.im_shape = ray.shape
        pass

    def set_priori(self,
        K :np.ndarray,
        f_pri :np.ndarray | float = 0,
        regularization:float = 1e-6,
        ):
        K += regularization*np.eye(self.ng)

        self.K_inv = np.linalg.inv(K)
        self.f_pri = f_pri

    def set_sig(self,
        sig_array:np.ndarray,
        g_obs:np.ndarray,
        sig_scale:float=1.0,
        num:int=0,
        ):
        self.g_obs=g_obs.reshape(*self.im_shape)
        self.sig_scale = sig_scale
        sig_array = sig_array.flatten()
        g_obs = g_obs.flatten()
        self.sig_inv = 1/sig_array
        #self.sig2_inv = 1/sig_array**2
        H    = self.H

        self.Sigi_obs = self.sig_inv*(g_obs)
        self.sigiH = sps.csr_matrix(sps.diags(self.sig_inv) @ H )
        sigiH_t = sps.csr_matrix( self.sigiH.T )

        #self.Hsig2iH  = (self.sigiH.T @ self.sigiH).toarray() 
        # self.Hsig2iH = sparse_dot_mkl.gram_matrix_mkl(sigiH_t,transpose=True,dense=True)　#2時間溶かした戦犯
        self.Hsig2iH = sparse_dot_mkl.dot_product_mkl(sigiH_t,self.sigiH ,dense=True)

    
    def check_diff(self,
        f:np.ndarray):
            
        fig,ax = plt_subplots(1,3,figsize=(10,4))
        ax = ax[0][:]
        g = self.H @ np.exp(f.flatten())

        g = g.reshape(*self.im_shape)
        imshow_cbar(ax[0],g,origin='lower')
        ax[0].set_title('Hf')
        vmax = (abs(g-self.g_obs)).max()
        imshow_cbar(ax= ax[1],im0 = g-self.g_obs,vmin=-vmax,vmax=vmax,cmap='RdBu_r',origin='lower')
        ax[1].set_title('diff_im')
        
        ax[2].hist((g-self.g_obs).flatten(),bins=50)
        ax[2].tick_params( labelleft=False)
        plt.show()

    
    def calc_core_fast(self,
        f:np.ndarray,
        num:int=0,
        ):
        r_f = f - self.f_pri
        exp_f = np.exp(f)
        fxf = np.einsum('i,j->ij',exp_f,exp_f)
        
        SiR = self.sigiH @ exp_f - self.Sigi_obs

        c1 = 1/self.sig_scale**2 *(self.sigiH.T @ SiR) * exp_f #self.sigiH.T @ SiR =  self.Hsig2iH  @ exp_f -  self.sigiH.T@ self.Sigi_obs
        C1 = 1/self.sig_scale**2 *self.Hsig2iH * fxf 

        Psi_df   = -c1 - self.K_inv @ r_f 

        Psi_dfdf = -C1 - np.diag(c1) - self.K_inv

        DPsi = Psi_dfdf
        NPsi = Psi_df
        loss = abs(NPsi).mean()

        delta_f = - np.linalg.solve(DPsi,NPsi)

        delta_f[delta_f<-3] = -3
        delta_f[delta_f>+3] = +3

        return delta_f,loss
    
    def set_postprocess(self,
        f:npt.NDArray[np.float64],
        ):
        exp_f = np.exp(f)
        fxf = np.einsum('i,j->ij',exp_f,exp_f)
        
        SiR = self.sigiH @ exp_f - self.Sigi_obs

        c1 = 1/self.sig_scale**2 *(self.sigiH.T @ SiR) * exp_f #self.sigiH.T @ SiR =  self.Hsig2iH  @ exp_f -  self.sigiH.T@ self.Sigi_obs
        C1 = 1/self.sig_scale**2 *self.Hsig2iH * fxf 

        Psi_dfdf = -C1 - np.diag(c1) - self.K_inv

        DPsi = Psi_dfdf
        self.Kf_pos_inv = -DPsi
        self.Kf_pos     = np.linalg.inv(self.Kf_pos_inv)
        self.sigf_pos = np.sqrt(np.diag(self.Kf_pos))

        pass
"""

class GPT_log_torch:
    import torch
    def __init__(self,
        Obs: rt1kernel.Observation_Matrix_integral,
        Kernel: rt1kernel.Kernel2D_scatter,
        ) -> None:
        self.Obs = Obs
        self.rI = Obs.rI 
        self.zI = Obs.zI 
        self.nI = Obs.zI.size  
        self.Kernel = Kernel
        pass

    def set_kernel(self,
        K :np.ndarray,
        f_pri :np.ndarray = 0,
        regularization:float = 1e-6,
        ):
        K += regularization*np.eye(self.nI)

        self.K_inv = np.linalg.inv(K)
        self.f_pri = f_pri

    def set_sig(self,
        sigma:np.ndarray,
        g_obs:np.ndarray,
        num:int=0,
        ):
        self.g_obs=g_obs
        sigma = sigma.flatten()
        g_obs = g_obs.flatten()
        self.sig_inv = 1/sigma
        self.sig2_inv = 1/sigma**2
        H    = self.Obs.Hs[num].H

        self.Sigi_obs = self.sig_inv*(g_obs)
        self.sigiH = sps.csr_matrix(sps.diags(self.sig_inv) @ H )

        self.Hsig2iH = (self.sigiH.T @ self.sigiH).toarray() 
        

    def calc_core(self,
        f:np.ndarray,
        num:int=0
        ):
        Exist = self.Obs.Hs[num].Exist
        E = sps.csr_matrix(Exist@sps.diags(f))
        Exp =  E.expm1() + Exist
        
        SiHE  :sps.csr_matrix = self.sigiH.multiply(Exp)
        SiR = np.array(SiHE.sum(axis=1)).flatten() - self.Sigi_obs
        r_f = f - self.f_pri

        c1 = (SiHE.T @ SiR) 

        C1 = (SiHE.T @ SiHE)

        Psi_df   = -c1 - self.K_inv @ r_f 

        Psi_dfdf = -C1 - np.diag(c1) - self.K_inv

        DPsi = Psi_dfdf
        NPsi = Psi_df

        delta_f = - np.linalg.solve(DPsi,NPsi)

        delta_f[delta_f<-3] = -3
        delta_f[delta_f>+3] = +3

        return delta_f
    
    def check_diff(self,
        f:np.ndarray):
            
        fig,ax = plt.subplots(1,3,figsize=(10,4))
        g = self.Obs.Hs[0].projection(np.exp(f))
        imshow_cbar(fig,ax[0],g)
        ax[0].set_title('Hf')
        vmax = (abs(g-self.g_obs)).max()
        imshow_cbar(fig,ax[1],g-self.g_obs,vmin=-vmax,vmax=vmax,cmap='turbo')
        ax[1].set_title('diff_im')
        
        ax[2].hist((g-self.g_obs).flatten(),bins=50)
        plt.show()

    
    def calc_core_fast(self,
        f:np.ndarray,
        num:int=0
        ):
        r_f = f - self.f_pri
        exp_f = np.exp(f)
        fxf = np.einsum('i,j->ij',exp_f,exp_f)
        
        SiR = self.sigiH @ exp_f - self.Sigi_obs

        c1 = (self.sigiH.T @ SiR) * exp_f
        C1 = self.Hsig2iH * fxf 

        Psi_df   = -c1 - self.K_inv @ r_f 

        Psi_dfdf = -C1 - np.diag(c1) - self.K_inv

        DPsi = Psi_dfdf
        NPsi = Psi_df

        delta_f = - np.linalg.solve(DPsi,NPsi)

        delta_f[delta_f<-3] = -3
        delta_f[delta_f>+3] = +3

        return delta_f

"""
        