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

import rt1kernel

sys.path.insert(0,os.pardir)
import rt1raytrace

__all__ = []

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
        pass

    def set_kernel(self,
        K_a :np.ndarray,
        K_v :np.ndarray,
        a_pri:np.ndarray = 0,
        v_pri:np.ndarray = 0,
        regularization:float = 1e-6,
        ):
        K_a += regularization*np.eye(self.nI)
        K_v += regularization*np.eye(self.nI)

        self.K_a_inv = np.linalg.inv(K_a)
        self.K_v_inv = np.linalg.inv(K_v)
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

        self.sigiH   : sparse.csr_matrix =  sparse.diags(self.sig_inv) @ H  
        self.sigiHT  :sparse.csr_matrix = self.sigiH.multiply(Dcos) 
        self.sigiHT2 :sparse.csr_matrix = self.sigiHT.multiply(Dcos)
        self.SiA = self.sig_inv*(A_cos + A_sin*1.j)


    def calc_core(self,
        a:np.ndarray,
        v:np.ndarray,
        num:int=0
        ):
        r_a = a - self.a_pri
        r_v = v - self.v_pri
        Exp = self.Obs.Hs[num].Exp(a,v)

        SiHE  :sparse.csr_matrix = self.sigiH.multiply(Exp)
        SiHTE :sparse.csr_matrix = self.sigiHT.multiply(Exp)
        SiHT2E:sparse.csr_matrix = self.sigiHT2.multiply(Exp)

        SiHE_conj = np.conjugate(SiHE)
        SiHTE_conj = np.conjugate(SiHTE)
        SiR = np.asarray(np.sum(SiHE,axis=1)).flatten()- self.SiA 
        SiR_conj = np.conj(SiR)
        c1 = (SiHE.T @ SiR_conj).real 
        c2 = (1.j*SiHTE.T @ SiR_conj).real
        c3 = (SiHT2E.T @ SiR_conj).real

        C1 = ((SiHE_conj.T @ SiHE).real).toarray()
        C2 = ((SiHTE_conj.T @ SiHTE).real).toarray()
        C3 = ((1.j*SiHTE_conj.T @ SiHE).real).toarray()

        Psi_da   = -c1 - self.K_a_inv @ r_a 
        Psi_dv   = -c2 - self.K_v_inv @ r_v
        Psi_dada = -C1 - np.diag(c1)*0 - self.K_a_inv
        Psi_dvdv = +C2 + np.diag(c3)*0 - self.K_v_inv
        Psi_dadv = -C3 - np.diag(c2)*0 
        Psi_dvda = Psi_dadv.T 

        nI = self.nI
        DPsi = np.empty((2*nI,2*nI))
        NPsi = np.concatenate([Psi_da,Psi_dv])

        DPsi[:nI,:nI] = Psi_dada[:,:]
        DPsi[nI:,nI:] = Psi_dvdv[:,:]
        DPsi[:nI,nI:] = Psi_dvda[:,:]
        DPsi[nI:,:nI] = Psi_dadv[:,:]

        del Psi_dada,Psi_dvdv,Psi_dvda,Psi_dadv

        delta_av = - np.linalg.solve(DPsi,NPsi)
        
        delta_av[delta_av<-3] = -3
        delta_av[delta_av>+3] = +3
        delta_a = delta_av[:nI]
        delta_v = delta_av[nI:]

        return delta_a,delta_v

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
        sigma = sigma.flatten()
        g_obs = g_obs.flatten()
        self.sig_inv = 1/sigma
        self.sig2_inv = 1/sigma**2
        H    = self.Obs.Hs[num].H

        self.Sig_obs = self.sig_inv*(g_obs)
        self.sigiH   : sparse.csr_matrix =  sparse.diags(self.sig_inv) @ H 

        self.Hsig2iH  = (self.sigiH.T @ self.sigiH).toarray() 
        

    def calc_core(self,
        f:np.ndarray,
        num:int=0
        ):
        Exist = self.Obs.Hs[num].Exist
        E :sparse.csr_matrix = (Exist@sparse.diags(f))
        Exp =  E.expm1() + Exist
        
        SiHE  :sparse.csr_matrix = self.sigiH.multiply(Exp)
        SiR = np.array(SiHE.sum(axis=1)).flatten() - self.Sig_obs
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

    
    def calc_core_fast(self,
        f:np.ndarray,
        num:int=0
        ):
        r_f = f - self.f_pri
        exp_f = np.exp(f)
        fxf = np.einsum('i,j->ij',exp_f,exp_f)
        
        SiR = self.sigiH @ exp_f - self.Sig_obs

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