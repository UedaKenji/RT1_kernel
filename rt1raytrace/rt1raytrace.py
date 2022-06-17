import matplotlib.pyplot as plt
import numpy as np
from numpy import FPE_DIVIDEBYZERO, array, linalg, ndarray
from rt1plotpy import frame
from typing import Optional, Union,Tuple,Callable,List
import time 
import math
from sympy import Q
from tqdm import tqdm
import scipy.linalg as linalg
from numba import jit
import warnings
from dataclasses import dataclass
import itertools
import scipy.sparse as sparse
import pandas as pd
import os

from  .plot_utils import *

__all__ = ['Raytrace','Ray']


@dataclass
class Ray:
    Ve_theta0: np.ndarray
    Ho_theta0: np.ndarray
    R0  : Union[np.ndarray, float]
    Phi0: Union[np.ndarray, float] = 0.
    Z0  : Union[np.ndarray, float] = 0.
    cos_factor: Union[np.ndarray,float] = 1

    raytraced: bool = False

    def __post_init__(self):
        self.shape = (self.Ve_theta0.shape)
        self.cos_factor = np.abs(self.cos_factor)

    def set_raytraced(self, Length, ref_type, ref_num):
        self.reytraced = True        
        self.Length   : np.ndarray = Length      
        self.ref_type : np.ndarray = ref_type
        self.ref_num  : np.ndarray = ref_num

        self.R1, self.Phi1, self.Z1,_ = self.ZΦRL_ray()
        
    def correct_no_intersection(self):
        self.Length[self.ref_num==-1] = 0.      
        

    def ZΦRL_ray(self,
        Lmax: Union[np.ndarray,float,None] = None,
        Lnum: int=1,
        Lmin: float = 0., 
        ) ->  Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if Lmax is None:
            Lmax = self.Length

        if Lnum == 1:
            L = Lmax*np.ones(self.shape)
        else:
            L = np.linspace(Lmin,Lmax*np.ones(self.shape), Lnum)
        
        CosVe, SinVe = np.cos(self.Ve_theta0)     , np.sin(self.Ve_theta0)
        CosHo, SinHo = np.cos(self.Ho_theta0-self.Phi0), np.sin(self.Ho_theta0-self.Phi0)
        X0,Y0 = self.R0*np.cos(self.Phi0), self.R0*np.sin(self.Phi0)
        X = X0-L*CosVe*CosHo
        Y = Y0+L*CosVe*SinHo
        Z = self.Z0+L*SinVe
        R = np.sqrt(X**2+Y**2)
        Phi = np.arctan2(Y,X)
        return R,Phi,Z,L

    def RZ_ray(self,
        Lnum: int=1,
        Lmax: Union[np.ndarray,float,None] = None,
        Lmin: float = 0., 
        ) ->  Tuple[np.ndarray, np.ndarray]:
        R,_,Z,_ = self.ZΦRL_ray(Lmax=Lmax,Lnum=Lnum,Lmin=Lmin)
        return R,Z

        
    def XYZ_ray(self,
        Lnum: int=1,
        Lmax: Union[np.ndarray,float,None] = None,
        Lmin: float = 0., 
        ) ->  Tuple[np.ndarray, np.ndarray]:
        R,Phi,Z,_ = self.ZΦRL_ray(Lmax=Lmax,Lnum=Lnum,Lmin=Lmin)
        X = R*np.cos(Phi)
        Y = R*np.sin(Phi)
        return X,Y,Z

    def Direction_Cos(self,
        R: np.ndarray,
        flatten:bool =True,
        ):
        R1 = (self.R0 * np.sin(self.Ho_theta0)).flatten()
        Tan = np.sqrt(1+np.tan(self.Ve_theta0)**2).flatten() 
        Cos = np.einsum('i,j->ij',R1,1/R)
        Cos[Cos>1] = 1.0
        Direction_cos = np.einsum('ij,i->ij',Cos,1/Tan)

        if flatten == False:
            Direction_cos = Direction_cos.reshape(*self.shape,R.size)
            print(Direction_cos.shape)
        
        return Direction_cos


class Raytrace(frame.Frame):
    def load_model(
        path:str, 
        is_plot:bool=False
        ) :
        model:Raytrace = pd.read_pickle(path) 
        if is_plot: model.show_ims()
        return model

    def save_model(self,
        name:str,
        path:str=''
        ):
        N = len(self.rays)
        class_name = type(self).__name__

        fig,ax = plt.subplots(1,N,figsize=(N*8,10))
        for i in range(N):
            imshow_cbar(fig,ax[i],self.rays[i].Length,origin='lower',cmap='turbo')

        fig.suptitle('flocal length: '+str(self.focal_length)+'\n'
                    +'image_size: '+str(self.image_size)+'\n'
                    +'center_angles: '+str(self.center_angles)+'\n'
                    +'location: '+str(self.location)+'\n'
                    +'rotation: '+str(self.rotation),fontsize=25)
        fig.tight_layout()
        fig.savefig(path+class_name+'_'+name+'.png',tight_layout=True,facecolor="white") 
        pd.to_pickle(self,path+class_name+'_'+name+'.pkl')

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
        print('you have to "set_camera()" or "set_angles()" next.')
    
    def set_camera(self,
        focal_length: float,
        image_size  : Tuple[float,float],
        image_shape : Tuple[int,int],
        location    : Tuple[float,float],
        center_angles: Tuple[float,float],
        rotation    : float=0,
        ) -> None:
        
        """
        set camera infomation

        Parameters
        ----------
        focal_length: float,
            focal_length [m]
        image_length: Tuple[float,float],
            like as ( Height [m], Width [m] )
        image_shape  : Tuple[int,int],
            this param is the shape of array associated with image, like as ( num of H, num of W )
        location: Tuple[float,float],
            equal to ( Z_cam, R_cam ) 
        center_angles: Tuple[float,float],
            equal to (h_angle[deg], w_angle)
        rotation: float=0,
            the angle of camera rotation [rad]
        Reuturns
        ----------
        None
        """
        h_length, w_length = image_size
        h_num, w_num = image_shape
        h_ang0, w_ang0 = center_angles
        
        self.focal_length = focal_length
        self.image_size = image_size 
        self.center_angles = center_angles
        self.location = location 
        self.rotation = rotation 

        self.im_shape = image_shape
        self.Z_cam, self.R_cam = location
        cos,sin =  np.cos(rotation*np.pi/180), np.sin(rotation*np.pi /180)

        h = np.linspace( 0.5*h_length *( -1 + 1/h_num), 0.5 *h_length *( 1 - 1/h_num), h_num )
        w  = np.linspace( 0.5*w_length *( -1 + 1/w_num), 0.5 *w_length *( 1 - 1/w_num), w_num )

        h_ang = np.arctan(h/focal_length) #+ np.pi * h_ang0 / 180
        w_ang = np.arctan(w/focal_length) #+ np.pi * w_ang0 / 180

        Ve_cam,Ho_cam = np.meshgrid(h_ang,w_ang,indexing='ij')
        self.Ho_cam, self.Ve_cam= cos *Ho_cam - sin*Ve_cam+ np.pi * w_ang0 / 180 , sin * Ho_cam + cos*Ve_cam+ np.pi * h_ang0 / 180

        self.ray_1st = Ray( Ve_theta0 = self.Ve_cam,
                            Ho_theta0 = self.Ho_cam,
                            R0        = self.R_cam,
                            Z0        = self.Z_cam,
                            Phi0      = 1,
                            )
 
    
    def set_angles(self,
        location: Tuple[float,float],
        H_angles: np.ndarray         ,
        W_angles: np.ndarray         ,
        ):
        
        """
        set ray angles of camera insted of set_camera() 

        Parameters
        ----------
        location: Tupple[float,float],
            equal to ( Z_cam, R_cam ) 
        H_angles: np.ndarray         ,
            2 dim numpy array of h angles [rad]
        W_angles: np.ndarray         ,
            2 dim numpy array of h angles [rad]

        Reuturns
        ----------
        None
        """
        self.im_shape = H_angles.shape
        self.Z_cam,   self.R_cam = location
        self.Ve_cam, self.Ho_cam = H_angles, W_angles
        self.ray_1st = Ray( Ve_theta0 = self.Ve_cam,
                            Ho_theta0 = self.Ho_cam,
                            R0        = self.R_cam ,
                            Z0        = self.Z_cam ,)

        
    def main(self,
        N:int=1,
        Lmax:float=3.0,
        Lnum:int=100):

        if not 'Ve_cam'  in dir(self):
            print('set_camera() or set_angle() is to be done in advance!')
            return

        self.rays :list[Ray] = []
        self.rays.append(self.ray_1st)
        self.print_raytrace(1)
        self.raytrace(self.rays[0], Lmax=Lmax, Lnum=Lnum, ignore_1st_intersection=True)
        
        for i in range(1,N):
            self.print_raytrace(i+1)
            Ve_ref,Ho_ref,cos_factor =  self.reflection(self.rays[i-1])

            ray = Ray( Ve_theta0 = Ve_ref,
                       Ho_theta0 = Ho_ref,
                       R0        = self.rays[i-1].R1 ,
                       Phi0      = self.rays[i-1].Phi1,
                       Z0        = self.rays[i-1].Z1  ,
                       cos_factor= cos_factor)
            self.rays.append(ray)

            self.raytrace(self.rays[i], Lmax, Lnum=Lnum, ignore_1st_intersection=False)

            
    def main2(self,
        N:int=1,
        Lmax:float=3.0,
        Lnum: Union[int, List[int]]=100):

        if not 'Ve_cam'  in dir(self):
            print('set_camera() or set_angle() is to be done in advance!')
            return

        if type(Lnum) is int:
            Lnum = [Lnum]*N

        

        self.rays :list[Ray] = []
        self.rays.append(self.ray_1st)
        self.print_raytrace(1)
        self.raytrace2(self.rays[0], Lmax=Lmax, Lnum=Lnum[0], ignore_1st_intersection=False)
        
        for i in range(1,N):
            self.print_raytrace(i+1)
            if i == 1:
                Ve_ref = self.rays[0].Ve_theta0
                Ho_ref = self.rays[0].Ho_theta0 + self.rays[0].Phi1- self.rays[0].Phi0
                cos_factor = 1.  
            else: 
                Ve_ref,Ho_ref,cos_factor =  self.reflection(self.rays[i-1])

            ray = Ray( Ve_theta0 = Ve_ref,
                       Ho_theta0 = Ho_ref,
                       R0        = self.rays[i-1].R1 ,
                       Phi0      = self.rays[i-1].Phi1,
                       Z0        = self.rays[i-1].Z1  ,
                       cos_factor= cos_factor)
            self.rays.append(ray)

            self.raytrace2(self.rays[i], Lmax, Lnum=Lnum[i], ignore_1st_intersection=False)


    def print_raytrace(self,N:int=1):
        if N == 1:
            return print("\n### start 1st raytrace ###")
        elif N == 2:
            return print("\n### start 2nd raytrace ###")
        elif N == 3:
            return print("\n### start 3rd raytrace ###")
            
        else:
            return print("\n### start "+str(N)+"th raytrace ###")



    def normal_wrapper(self,ray:Ray):
        return self.normal_vector2(R=ray.R1, Z=ray.Z1, frame_type=ray.ref_type, frame_num=ray.ref_num)

    def reflection(self,
        ray:Ray):

        if ray.Length is None:
            print("Error! raytrace have not been carried out!")

        dPhi=ray.Phi1- ray.Phi0
        Ve_theta0  = ray.Ve_theta0
        Ho_theta0  = ray.Ho_theta0

        Ho_theta_p = Ho_theta0+dPhi
        Ve_theta_p = Ve_theta0
        CosH,SinH = np.cos(Ho_theta_p), np.sin(Ho_theta_p)
        CosV,SinV = np.cos(Ve_theta_p), np.sin(Ve_theta_p)
        
        Pr, Pp, Pz = -CosV*CosH, CosV*SinH, SinV
        Nr,Nz = self.normal_wrapper(ray)
        N_dot_P = Pr*Nr + Pz*Nz
        Rr, Rp, Rz = Pr - 2*N_dot_P*Nr,  Pp, Pz - 2*N_dot_P*Nz
        
        Ve_theta1 = np.arctan(Rz/ np.sqrt(Rr**2+Rp**2))
        Ho_theta1 = np.arctan2(Rp, -Rr) 

        return Ve_theta1, Ho_theta1, N_dot_P 

    def show_rays(self,

        ):
        '''
        失敗した例のみ表示
        '''
        fig,ax = plt.subplots(figsize=(20,20))

        hoge = self.rays[-1].ref_type==-1
        N = len(self.rays)
        self.append_frame(ax)
        ax.set(**rt1_ax_kwargs)
        for n in range(N):
            R,Z=self.rays[n].RZ_ray(Lnum=50)
            R = R[:,hoge]
            Z = Z[:,hoge]
            for i in range(R.shape[1]):
                ax.plot(R[:,i],Z[:,i],color=cycle(int(i%10)))
        


    def show_ims(self,

        ):
        '''
        失敗した例のみ表示
        '''
        N = len(self.rays)
        fig,ax = plt.subplots(1,N,figsize=(N*8,7))
        for i in range(N):
            imshow_cbar(fig,ax[i],self.rays[i].Length,origin='lower',cmap='turbo')

        fig.suptitle('flocal length: '+str(self.focal_length)+', '
                    +'image_size: '+str(self.image_size)+', '
                    +'center_angles: '+str(self.center_angles)+', '
                    +'location: '+str(self.location)+', '
                    +'rotation: '+str(self.rotation),fontsize=25)
        
    
    def raytrace(self,
        ray: Ray,
        Lmax:     Union[np.ndarray,float] = 3.0,
        Lnum:     int=500,
        ignore_1st_intersection:bool = False,
        ):
        

        self.n_ve, self.n_ho = self.im_shape[0], self.im_shape[1]
        R_ray, _ , Z_ray, LL = ray.ZΦRL_ray(Lnum=Lnum, Lmax=Lmax,Lmin=1e-3)
        time.sleep(0.2)
        #Time =  ElapsedTime()
        
        R_dray_sta = R_ray[:-1,:,:]
        R_dray_end = R_ray[1:,:,:]
        Z_dray_sta = Z_ray[:-1,:,:]
        Z_dray_end = Z_ray[1:,:,:]
        del Z_ray, R_ray

        line_index_list =[]
        line_Rsol_list =[]
        line_Zsol_list =[]
        arc_index_list =[]
        arc_Rsol_list =[]
        arc_Zsol_list =[]
    
        R1, Z1 = R_dray_sta, Z_dray_sta
        R2, Z2 = R_dray_end, Z_dray_end
        # 交点が存在するかの判定(直線)#############
        for i in tqdm(range(len(self.all_lines)), desc='calculating Lines'):
            R4, Z4 = self.all_lines[i].start[0]/1000, self.all_lines[i].start[1]/1000
            R3, Z3 = self.all_lines[i].end[0]/1000, self.all_lines[i].end[1]/1000
        
            D = (R4-R3) * (Z2-Z1) - (R2-R1) * (Z4-Z3)
            W1, W2 = Z3*R4-Z4*R3, Z1*R2 - Z2*R1
        
            R_inter = ( (R2-R1) * W1 - (R4-R3) * W2 ) / D
            Z_inter = ( (Z2-Z1) * W1 - (Z4-Z3) * W2 ) / D
            del W1,W2,D
            
            is_in_Rray_range = (R2 - R_inter) * (R1 - R_inter) < 0 
            is_in_Zray_range = (Z2 - Z_inter) * (Z1 - Z_inter) < 0 
            is_in_Rfra_range = (R4 - R_inter) * (R3 - R_inter) < 0 
            is_in_Zfra_range = (Z4 - Z_inter) * (Z3 - Z_inter) < 0 
            is_in_range = np.logical_or(is_in_Rfra_range,is_in_Zfra_range) * np.logical_or(is_in_Rray_range,is_in_Zray_range) 
            # 水平や垂直  な線に対応するため
            index = np.nonzero(is_in_range)
            line_index_list.append(index)
            line_Rsol_list.append(R_inter[index])
            line_Zsol_list.append(Z_inter[index])
            #Time.printtime('Line:'+str(i+1)+'/'+str(len(self.all_lines)))

            del R_inter,Z_inter

        lR = R2-R1
        lZ = Z2-Z1
        S  = R2*Z1 - R1*Z2        
        # 交点が存在するかの判定(弧) #############
        for i in tqdm(range(len(self.all_arcs)), desc='calculating Arcs '):
            Rc, Zc =(self.all_arcs[i].center[0]/1000, self.all_arcs[i].center[1]/1000)
            radius = self.all_arcs[i].radius/1000
            sta_angle, end_angle = self.all_arcs[i].start_angle ,self.all_arcs[i].end_angle 

            D = (lR**2+lZ**2)*radius**2 + 2*lR*lZ*Rc*Zc - 2*(lZ*Rc-lR*Zc)*S - lR**2 *Zc**2 -lZ**2*Rc**2-S**2 #判別式
            exist = D > 0
            index = np.nonzero(D > 0 ) # 判別式が正、すなわち交点が存在する条件のインデックス

            #lR_i  = lR[index]
            #lZ_i  = lZ[index]
            #S_i   = S[index]
            #D_i   = D[index]
            Ri1 = (lR**2 *Rc + lR*lZ *Zc - lZ *S + lR * np.sqrt(D*exist) ) / (lR**2 + lZ**2)  * exist# １つ目の交点のR座標
            Zi1 = (lZ**2 *Zc + lR*lZ *Rc + lR *S + lZ * np.sqrt(D*exist) ) / (lR**2 + lZ**2)  * exist# １つ目の交点のZ座標
            Ri2 = (lR**2 *Rc + lR*lZ *Zc - lZ *S - lR * np.sqrt(D*exist) ) / (lR**2 + lZ**2)  * exist# 2つ目の交点のR座標
            Zi2 = (lZ**2 *Zc + lR*lZ *Rc + lR *S - lZ * np.sqrt(D*exist) ) / (lR**2 + lZ**2)  * exist# 2つ目の交点のZ座標
            del D, exist

            is_in_ray_range1  = np.logical_and((R2 - Ri1) * (R1 - Ri1) < 0 ,(Z2 - Zi1) * (Z1- Zi1) < 0) # 交点1が線分内にあるか判定
            is_in_ray_range2  = np.logical_and((R2 - Ri2) * (R1 - Ri2) < 0 ,(Z2 - Zi2) * (Z1- Zi2) < 0) # 交点2が線分内にあるか判定

            cos1 = (Ri1-Rc) / radius
            sin1 = (Zi1-Zc) / radius
            atan = np.arctan2(sin1,cos1)
            theta1 = np.where(atan > 0, 180/np.pi*atan, 360+180/np.pi*atan)
            cos2 = (Ri2-Rc) / radius    
            sin2 = (Zi2-Zc) / radius 
            atan = np.arctan2(sin2,cos2)
            theta2 = np.where(atan > 0, 180/np.pi*atan, 360+180/np.pi*atan)

            del cos1,sin1,atan,cos2,sin2

            is_in_arc1 =  (end_angle - theta1) * (sta_angle - theta1) * (end_angle-sta_angle) <= 0 # 交点1が弧の範囲内あるか判定
            is_in_arc2 =  (end_angle - theta2) * (sta_angle - theta2) * (end_angle-sta_angle) <= 0 # 交点1が弧の範囲内あるか判定

            is_real_intercept1 = is_in_ray_range1 * is_in_arc1
            is_real_intercept2 = is_in_ray_range2 * is_in_arc2
            is_real_intercept  = is_real_intercept1 + is_real_intercept2

            Rsol = Ri1 * is_real_intercept1 + Ri2 * is_real_intercept2
            del Ri1,Ri2
            Zsol = Zi1 * is_real_intercept1 + Zi2 * is_real_intercept2
            del Zi1,Zi2


            index = np.nonzero(is_real_intercept)
            arc_index_list.append(index)
            arc_Rsol_list.append(Rsol[index])
            arc_Zsol_list.append(Zsol[index])
            del Rsol,Zsol
        
        # 一番初めに交わる交点とその種類を整理する #############
        self.intersect_info = {}
        self.intersect_info['frame_type'] =    np.zeros(self.im_shape,dtype='int8')
        self.intersect_info['frame_num']  =    np.zeros(self.im_shape,dtype='int8')
        self.intersect_info['frame_type2'] =   np.zeros(self.im_shape,dtype='int8')
        self.intersect_info['frame_num2']  =   np.zeros(self.im_shape,dtype='int8')
        self.intersect_info['index_1st']   =-1*np.ones(self.im_shape,dtype='int16')
        self.intersect_info['index_2nd']   =-1*np.ones(self.im_shape,dtype='int16')
        self.intersect_info['R_1st']       =   np.zeros(self.im_shape)
        self.intersect_info['Z_1st']      =   np.zeros(self.im_shape)
        self.intersect_info['R_2nd']      =   np.zeros(self.im_shape)
        self.intersect_info['Z_2nd']      =   np.zeros(self.im_shape)

        # 一番はじめに通過する壁の番号を判定している
        """
        i_size = 0
        i_mainline = -1
        for i in range(len(self.all_lines)):
            if len(line_Rsol_list[i]) > i_size:
                i_size = len(line_Rsol_list[i])
                i_mainline = i 
                
        print(i_mainline)
        """

        self.test = line_index_list


        for i in range(len(self.all_arcs)):
            for j in range(arc_index_list[i][0].size):
                Li   = int(arc_index_list[i][0][j])
                vi    = int(arc_index_list[i][1][j])
                hi    = int(arc_index_list[i][2][j])
                target_Li_1st = self.intersect_info['index_1st'][vi,hi]
                target_Li_2nd = self.intersect_info['index_2nd'][vi,hi]
                if (Li < target_Li_1st or target_Li_1st == -1):
                    self.intersect_info['frame_type2'][vi,hi] = self.intersect_info['frame_type'][vi,hi] 
                    self.intersect_info['frame_num2'][vi,hi]  = self.intersect_info['frame_num'][vi,hi]
                    self.intersect_info['index_2nd'][vi,hi]   = self.intersect_info['index_1st'][vi,hi] 

                    self.intersect_info['frame_type'][vi,hi] = 1  # 0 means 'line'
                    self.intersect_info['frame_num'][vi,hi] = i  # 0 means 'line'
                    self.intersect_info['index_1st'][vi,hi] = Li  
                elif (Li < target_Li_2nd or target_Li_2nd == -1):
                    self.intersect_info['frame_type2'][vi,hi] = 1  # 0 means 'line'
                    self.intersect_info['frame_num2'][vi,hi] = i  # 0 means 'line'
                    self.intersect_info['index_2nd'][vi,hi] = Li  


        for i in range(len(self.all_lines)):
            for j in range(line_index_list[i][0].size):
                Li = int(line_index_list[i][0][j])
                vi    = int(line_index_list[i][1][j])
                hi    = int(line_index_list[i][2][j])
                target_Li_1st = self.intersect_info['index_1st'][vi,hi]
                target_Li_2nd = self.intersect_info['index_2nd'][vi,hi]
                if (Li < target_Li_1st or target_Li_1st == -1):
                    
                    self.intersect_info['frame_type2'][vi,hi] = self.intersect_info['frame_type'][vi,hi] 
                    self.intersect_info['frame_num2'][vi,hi]  = self.intersect_info['frame_num'][vi,hi]
                    self.intersect_info['index_2nd'][vi,hi]   = self.intersect_info['index_1st'][vi,hi] 

                    self.intersect_info['frame_type'][vi,hi] = 0  # 0 means 'line'
                    self.intersect_info['frame_num'][vi,hi] = i  # 0 means 'line'
                    self.intersect_info['index_1st'][vi,hi] = Li  
                    
                elif ( Li < target_Li_2nd or target_Li_2nd == -1):
                    self.intersect_info['frame_type2'][vi,hi] = 0  # 0 means 'line'
                    self.intersect_info['frame_num2'][vi,hi] = i  # 0 means 'line'
                    self.intersect_info['index_2nd'][vi,hi] = Li  
 
        for i in range(self.im_shape[0]):
            for j in range(self.n_ho):
                first_index = self.intersect_info['index_1st'][i,j]
                secod_index = self.intersect_info['index_2nd'][i,j]
                self.intersect_info['R_1st'][i,j] = R_dray_sta[first_index,i,j]
                self.intersect_info['Z_1st'][i,j] = Z_dray_sta[first_index,i,j]
                self.intersect_info['R_2nd'][i,j] = R_dray_sta[secod_index,i,j]
                self.intersect_info['Z_2nd'][i,j] = Z_dray_sta[secod_index,i,j]
        
        del R_dray_end,R_dray_sta,Z_dray_sta,Z_dray_end

        if ignore_1st_intersection:
            self.intersect_info['R_1st']      =   self.intersect_info['R_2nd']
            self.intersect_info['Z_1st']      =   self.intersect_info['Z_2nd']
            self.intersect_info['frame_type'] =   self.intersect_info['frame_type2']
            self.intersect_info['frame_num']  =   self.intersect_info['frame_num2']
            self.intersect_info['index_1st']  =   self.intersect_info['index_2nd']

        no_interception =  self.intersect_info['index_1st']== -1

        if no_interception.any() :
            time.sleep(0.2)
            print('!WARNING!, there are rays without intersection.')
            time.sleep(0.2)

        
        ####----------------- strictly calc ------------------------####

        Is_Line = self.intersect_info['frame_type'] == 0
        Is_Arc = self.intersect_info['frame_type'] == 1

        # 宣言部
        R3,R4,Z3,Z4      = np.zeros(self.im_shape), np.zeros(self.im_shape), np.zeros(self.im_shape), np.zeros(self.im_shape)
        Rc, Zc, radius   = np.zeros(self.im_shape), np.zeros(self.im_shape), np.zeros(self.im_shape)
        sta_angle, end_angle = np.zeros(self.im_shape), np.zeros(self.im_shape)
        L_1st, L_2nd = np.zeros(self.im_shape), np.zeros(self.im_shape)
        R_1st,Z_1st  = np.zeros(self.im_shape), np.zeros(self.im_shape)

        # init 
        for i in range(self.im_shape[0]):
            for j in range(self.im_shape[1]):
                n = self.intersect_info['frame_num'][i,j]

                if(self.intersect_info['frame_type'][i,j] == 0 ): # ラインですか？
                    R3[i,j], Z3[i,j] = self.all_lines[n].start[0]/1000, self.all_lines[n].start[1]/1000
                    R4[i,j], Z4[i,j] = self.all_lines[n].end[0]/1000,   self.all_lines[n].end[1]/1000
                if(self.intersect_info['frame_type'][i,j] == 1 ): # 弧ですかぁ？
                    Rc[i,j], Zc[i,j] = self.all_arcs[n].center[0]/1000, self.all_arcs[n].center[1]/1000
                    radius[i,j]      = self.all_arcs[n].radius/1000
                    sta_angle[i,j]   = self.all_arcs[n].start_angle 
                    end_angle[i,j]   = self.all_arcs[n].end_angle

                index = self.intersect_info['index_1st'][i,j]
                L_1st[i,j] = LL[ index ,i,j]
                L_2nd[i,j] = LL[ index+1,i,j]
                #R_1st[i,j]        = R_dray_sta[index,i,j]
                R_1st[i,j]        = self.intersect_info['R_1st'][i,j]
                #Z_1st[i,j]        = Z_dray_sta[index,i,j]
                Z_1st[i,j]        = self.intersect_info['Z_1st'][i,j]

        frame_type = self.intersect_info['frame_type']
        frame_num  = self.intersect_info['frame_num']

        frame_num[no_interception] = -1

        self.L_1st = L_1st
        self.L_2nd = L_2nd

        del self.intersect_info

        N = 100

        for i in tqdm(range(N),desc='detailed calculation'):
            R1,Z1  = ray.RZ_ray(Lmax=L_1st)
            R2,Z2  = ray.RZ_ray(Lmax=L_2nd)
            
            #Z1,R1 = self.__ZR(L_1st, self.H_theta, self.W_theta,self.Z0,self.R0)
            #Z2,R2 = self.__ZR(L_2nd, self.H_theta, self.W_theta,self.Z0,self.R0)

            Is_finish = np.abs(L_2nd- L_1st) < 1.e-10

            D = (R4-R3) * (Z2-Z1) - (R2-R1) * (Z4-Z3) 
            
            D_is_0 = (D <= 1e-15)
            D += D_is_0 * 1e-15
            W1, W2 = Z3*R4-Z4*R3, Z1*R2 - Z2*R1

            R_inter = ( (R2-R1) * W1 - (R4-R3) * W2 ) / D 
            Z_inter = ( (Z2-Z1) *W1 - (Z4-Z3) * W2 ) / D  

            #R_inter = R_inter * np.logical_not(Is_iter_finish) #+ R_inter

            lR = R2-R1
            lZ = Z2-Z1
            S  = R2*Z1 - R1*Z2
            D = (lR**2+lZ**2)*radius**2 + 2*lR*lZ*Rc*Zc - 2*(lZ*Rc-lR*Zc)*S - lR**2 *Zc**2 -lZ**2*Rc**2-S**2 #判別式
            D = Is_Arc * D
            Ri1 = (lR**2 *Rc + lR*lZ *Zc - lZ *S + lR * np.sqrt(D) ) / (lR**2 + lZ**2)  # １つ目の交点のR座標
            Zi1 = (lZ**2 *Zc + lR*lZ *Rc + lR *S + lZ * np.sqrt(D) ) / (lR**2 + lZ**2)  # １つ目の交点のZ座標
            Ri2 = (lR**2 *Rc + lR*lZ *Zc - lZ *S - lR * np.sqrt(D) ) / (lR**2 + lZ**2)  # 2つ目の交点のR座標
            Zi2 = (lZ**2 *Zc + lR*lZ *Rc + lR *S - lZ * np.sqrt(D) ) / (lR**2 + lZ**2)  # 2つ目の交点のZ座標

            cos1 = (Ri1-Rc) / radius
            sin1 = (Zi1-Zc) / radius
            atan = np.arctan2(sin1,cos1)
            theta1 = np.where(atan > 0, 180/np.pi*atan, 360+180/np.pi*atan)
            cos2 = (Ri2-Rc) / radius    
            sin2 = (Zi2-Zc) / radius 
            atan = np.arctan2(sin2,cos2)
            theta2 = np.where(atan > 0, 180/np.pi*atan, 360+180/np.pi*atan)

            Is_Arc1 =  (end_angle - theta1) * (sta_angle - theta1) * (end_angle-sta_angle) <= 0 # 交点1が弧の範囲内あるか判定
            Is_Arc2 =  (end_angle - theta2) * (sta_angle - theta2) * (end_angle-sta_angle) <= 0 # 交点1が弧の範囲内あるか判定
            Is_Arc1 = Is_Arc1 * Is_Arc
            Is_Arc2 = Is_Arc2 * Is_Arc

            Is_Arc_and = Is_Arc1*Is_Arc2
            Is_Arc1_priori  = (Ri2 - R_1st)**2 + (Zi2 - Z_1st)**2  > (Ri1 - R_1st)**2 + (Zi1 - Z_1st)**2 

            Is_Arc1 = Is_Arc1 * np.logical_not(Is_Arc2) + Is_Arc_and *Is_Arc1_priori
            Is_Arc2 = Is_Arc2 * np.logical_not(Is_Arc1) + Is_Arc_and *np.logical_not(Is_Arc1_priori)

            R_sol=  Ri1 *Is_Arc1 + Ri2 *Is_Arc2 + np.nan_to_num(R_inter) *Is_Line
            Z_sol=  Zi1 *Is_Arc1 + Zi2 *Is_Arc2 + np.nan_to_num(Z_inter) *Is_Line

            Li = (L_1st* (R2-R_sol+Z2-Z_sol) + L_2nd* (R_sol-R1+Z_sol-Z1) ) / (R2-R1+Z2-Z1)
            Li = ~no_interception*Li + no_interception*Lmax

            Is_finish += np.isnan(Li)

            L_sol = ~Is_finish*np.nan_to_num(Li)   + Is_finish * L_1st

            #R_sol, phi_sol, Z_sol   = self.RφZ(L_sol, Ve_theta, Ho_theta, Z0, R0, Phi0)
            R_sol, Z_sol= ray.RZ_ray(Lmax=L_sol)

            L_2nd = L_1st
            L_1st = L_sol

            #Time.printtime(str(i+1)+'/'+str(N))

            ray.set_raytraced(Length=L_sol, ref_type=frame_type, ref_num=frame_num)

        return #R_sol, phi_sol, Z_sol, L_sol, frame_type, frame_num



    def raytrace2(self,
        ray: Ray,
        Lmax:     Union[np.ndarray,float] = 3.0,
        Lnum:     int=500,
        ignore_1st_intersection:bool = False,
        ):
        

        self.n_ve, self.n_ho = self.im_shape[0], self.im_shape[1]

        Lmax,Lmin = Lmax,1e-3
        L_sol  = 0

        for i in range(3):
            R_ray, _ , Z_ray, LL = ray.ZΦRL_ray(Lnum=Lnum, Lmax=Lmax,Lmin=Lmin)
            time.sleep(0.2)
            #Time =  ElapsedTime()
            
            R_dray_sta = R_ray[:-1,:,:]
            R_dray_end = R_ray[1:,:,:]
            Z_dray_sta = Z_ray[:-1,:,:]
            Z_dray_end = Z_ray[1:,:,:]
            del Z_ray, R_ray

            line_index_list =[]
            line_Rsol_list =[]
            line_Zsol_list =[]
            arc_index_list =[]
            arc_Rsol_list =[]
            arc_Zsol_list =[]
        
            R1, Z1 = R_dray_sta, Z_dray_sta
            R2, Z2 = R_dray_end, Z_dray_end
            # 交点が存在するかの判定(直線)#############
            for i in tqdm(range(len(self.all_lines)), desc='calculating Lines'):
                R4, Z4 = self.all_lines[i].start[0]/1000, self.all_lines[i].start[1]/1000
                R3, Z3 = self.all_lines[i].end[0]/1000, self.all_lines[i].end[1]/1000
            
                D = (R4-R3) * (Z2-Z1) - (R2-R1) * (Z4-Z3)
                W1, W2 = Z3*R4-Z4*R3, Z1*R2 - Z2*R1
            
                R_inter = ( (R2-R1) * W1 - (R4-R3) * W2 ) / D
                Z_inter = ( (Z2-Z1) * W1 - (Z4-Z3) * W2 ) / D
                del W1,W2,D
                
                is_in_Rray_range = (R2 - R_inter) * (R1 - R_inter) < 0 
                is_in_Zray_range = (Z2 - Z_inter) * (Z1 - Z_inter) < 0 
                is_in_Rfra_range = (R4 - R_inter) * (R3 - R_inter) < 0 
                is_in_Zfra_range = (Z4 - Z_inter) * (Z3 - Z_inter) < 0 
                is_in_range = np.logical_or(is_in_Rfra_range,is_in_Zfra_range) * np.logical_or(is_in_Rray_range,is_in_Zray_range) 
                # 水平や垂直  な線に対応するため
                index = np.nonzero(is_in_range)
                line_index_list.append(index)
                line_Rsol_list.append(R_inter[index])
                line_Zsol_list.append(Z_inter[index])
                #Time.printtime('Line:'+str(i+1)+'/'+str(len(self.all_lines)))

                del R_inter,Z_inter

            lR = R2-R1
            lZ = Z2-Z1
            S  = R2*Z1 - R1*Z2        
            # 交点が存在するかの判定(弧) #############
            for i in tqdm(range(len(self.all_arcs)), desc='calculating Arcs '):
                Rc, Zc =(self.all_arcs[i].center[0]/1000, self.all_arcs[i].center[1]/1000)
                radius = self.all_arcs[i].radius/1000
                sta_angle, end_angle = self.all_arcs[i].start_angle ,self.all_arcs[i].end_angle 

                D = (lR**2+lZ**2)*radius**2 + 2*lR*lZ*Rc*Zc - 2*(lZ*Rc-lR*Zc)*S - lR**2 *Zc**2 -lZ**2*Rc**2-S**2 #判別式
                exist = D > 0
                index = np.nonzero(D > 0 ) # 判別式が正、すなわち交点が存在する条件のインデックス

                #lR_i  = lR[index]
                #lZ_i  = lZ[index]
                #S_i   = S[index]
                #D_i   = D[index]
                Ri1 = (lR**2 *Rc + lR*lZ *Zc - lZ *S + lR * np.sqrt(D*exist) ) / (lR**2 + lZ**2)  * exist# １つ目の交点のR座標
                Zi1 = (lZ**2 *Zc + lR*lZ *Rc + lR *S + lZ * np.sqrt(D*exist) ) / (lR**2 + lZ**2)  * exist# １つ目の交点のZ座標
                Ri2 = (lR**2 *Rc + lR*lZ *Zc - lZ *S - lR * np.sqrt(D*exist) ) / (lR**2 + lZ**2)  * exist# 2つ目の交点のR座標
                Zi2 = (lZ**2 *Zc + lR*lZ *Rc + lR *S - lZ * np.sqrt(D*exist) ) / (lR**2 + lZ**2)  * exist# 2つ目の交点のZ座標
                del D, exist

                is_in_ray_range1  = np.logical_and((R2 - Ri1) * (R1 - Ri1) < 0 ,(Z2 - Zi1) * (Z1- Zi1) < 0) # 交点1が線分内にあるか判定
                is_in_ray_range2  = np.logical_and((R2 - Ri2) * (R1 - Ri2) < 0 ,(Z2 - Zi2) * (Z1- Zi2) < 0) # 交点2が線分内にあるか判定

                cos1 = (Ri1-Rc) / radius
                sin1 = (Zi1-Zc) / radius
                atan = np.arctan2(sin1,cos1)
                theta1 = np.where(atan > 0, 180/np.pi*atan, 360+180/np.pi*atan)
                cos2 = (Ri2-Rc) / radius    
                sin2 = (Zi2-Zc) / radius 
                atan = np.arctan2(sin2,cos2)
                theta2 = np.where(atan > 0, 180/np.pi*atan, 360+180/np.pi*atan)

                del cos1,sin1,atan,cos2,sin2

                is_in_arc1 =  (end_angle - theta1) * (sta_angle - theta1) * (end_angle-sta_angle) <= 0 # 交点1が弧の範囲内あるか判定
                is_in_arc2 =  (end_angle - theta2) * (sta_angle - theta2) * (end_angle-sta_angle) <= 0 # 交点1が弧の範囲内あるか判定

                is_real_intercept1 = is_in_ray_range1 * is_in_arc1
                is_real_intercept2 = is_in_ray_range2 * is_in_arc2
                is_real_intercept  = is_real_intercept1 + is_real_intercept2

                Rsol = Ri1 * is_real_intercept1 + Ri2 * is_real_intercept2
                del Ri1,Ri2
                Zsol = Zi1 * is_real_intercept1 + Zi2 * is_real_intercept2
                del Zi1,Zi2


                index = np.nonzero(is_real_intercept)
                arc_index_list.append(index)
                arc_Rsol_list.append(Rsol[index])
                arc_Zsol_list.append(Zsol[index])
                del Rsol,Zsol
            
            # 一番初めに交わる交点とその種類を整理する #############
            self.intersect_info = {}
            self.intersect_info['frame_type'] =  -1*np.ones(self.im_shape,dtype='int8')
            self.intersect_info['frame_num']  =  -1*np.ones(self.im_shape,dtype='int8')
            self.intersect_info['frame_type2'] = -1*np.ones(self.im_shape,dtype='int8')
            self.intersect_info['frame_num2']  = -1*np.ones(self.im_shape,dtype='int8')
            self.intersect_info['index_1st']   =-1*np.ones(self.im_shape,dtype='int16')
            self.intersect_info['index_2nd']   =-1*np.ones(self.im_shape,dtype='int16')
            self.intersect_info['R_1st']       =   np.zeros(self.im_shape)
            self.intersect_info['Z_1st']      =   np.zeros(self.im_shape)
            self.intersect_info['R_2nd']      =   np.zeros(self.im_shape)
            self.intersect_info['Z_2nd']      =   np.zeros(self.im_shape)

            # 一番はじめに通過する壁の番号を判定している
            """
            i_size = 0
            i_mainline = -1
            for i in range(len(self.all_lines)):
                if len(line_Rsol_list[i]) > i_size:
                    i_size = len(line_Rsol_list[i])
                    i_mainline = i 
                    
            print(i_mainline)
            """

            self.test = line_index_list


            for i in range(len(self.all_arcs)):
                for j in range(arc_index_list[i][0].size):
                    Li   = int(arc_index_list[i][0][j])
                    vi    = int(arc_index_list[i][1][j])
                    hi    = int(arc_index_list[i][2][j])
                    target_Li_1st = self.intersect_info['index_1st'][vi,hi]
                    target_Li_2nd = self.intersect_info['index_2nd'][vi,hi]
                    if (Li < target_Li_1st or target_Li_1st == -1):
                        self.intersect_info['frame_type2'][vi,hi] = self.intersect_info['frame_type'][vi,hi] 
                        self.intersect_info['frame_num2'][vi,hi]  = self.intersect_info['frame_num'][vi,hi]
                        self.intersect_info['index_2nd'][vi,hi]   = self.intersect_info['index_1st'][vi,hi] 

                        self.intersect_info['frame_type'][vi,hi] = 1  # 0 means 'line'
                        self.intersect_info['frame_num'][vi,hi] = i  # 0 means 'line'
                        self.intersect_info['index_1st'][vi,hi] = Li  
                    elif (Li < target_Li_2nd or target_Li_2nd == -1):
                        self.intersect_info['frame_type2'][vi,hi] = 1  # 0 means 'line'
                        self.intersect_info['frame_num2'][vi,hi] = i  # 0 means 'line'
                        self.intersect_info['index_2nd'][vi,hi] = Li  


            for i in range(len(self.all_lines)):
                for j in range(line_index_list[i][0].size):
                    Li = int(line_index_list[i][0][j])
                    vi    = int(line_index_list[i][1][j])
                    hi    = int(line_index_list[i][2][j])
                    target_Li_1st = self.intersect_info['index_1st'][vi,hi]
                    target_Li_2nd = self.intersect_info['index_2nd'][vi,hi]
                    if (Li < target_Li_1st or target_Li_1st == -1):
                        
                        self.intersect_info['frame_type2'][vi,hi] = self.intersect_info['frame_type'][vi,hi] 
                        self.intersect_info['frame_num2'][vi,hi]  = self.intersect_info['frame_num'][vi,hi]
                        self.intersect_info['index_2nd'][vi,hi]   = self.intersect_info['index_1st'][vi,hi] 

                        self.intersect_info['frame_type'][vi,hi] = 0  # 0 means 'line'
                        self.intersect_info['frame_num'][vi,hi] = i  # 0 means 'line'
                        self.intersect_info['index_1st'][vi,hi] = Li  
                        
                    elif ( Li < target_Li_2nd or target_Li_2nd == -1):
                        self.intersect_info['frame_type2'][vi,hi] = 0  # 0 means 'line'
                        self.intersect_info['frame_num2'][vi,hi] = i  # 0 means 'line'
                        self.intersect_info['index_2nd'][vi,hi] = Li  
    
            for i in range(self.im_shape[0]):
                for j in range(self.n_ho):
                    first_index = self.intersect_info['index_1st'][i,j]
                    secod_index = self.intersect_info['index_2nd'][i,j]
                    self.intersect_info['R_1st'][i,j] = R_dray_sta[first_index,i,j]
                    self.intersect_info['Z_1st'][i,j] = Z_dray_sta[first_index,i,j]
                    self.intersect_info['R_2nd'][i,j] = R_dray_sta[secod_index,i,j]
                    self.intersect_info['Z_2nd'][i,j] = Z_dray_sta[secod_index,i,j]
            
            del R_dray_end,R_dray_sta,Z_dray_sta,Z_dray_end

            if ignore_1st_intersection:
                self.intersect_info['R_1st']      =   self.intersect_info['R_2nd']
                self.intersect_info['Z_1st']      =   self.intersect_info['Z_2nd']
                self.intersect_info['frame_type'] =   self.intersect_info['frame_type2']
                self.intersect_info['frame_num']  =   self.intersect_info['frame_num2']
                self.intersect_info['index_1st']  =   self.intersect_info['index_2nd']

            no_interception =  self.intersect_info['index_1st']== -1

            if no_interception.any() :
                time.sleep(0.2)
                print('!WARNING!, there are rays without intersection.')
                time.sleep(0.2)
                #self.intersect_info['frame_type'] = self.frame_type_bk
                #self.intersect_info['frame_num']  = self.frame_num_bk
                #self.intersect_info['index_1st']  = self.index_1st_bk


            
           
            L_1st, L_2nd = np.zeros(self.im_shape), np.zeros(self.im_shape)

            for i in range(self.im_shape[0]):
                for j in range(self.im_shape[1]):
                    index = self.intersect_info['index_1st'][i,j]
                    L_1st[i,j] = LL[ index ,i,j]
                    L_2nd[i,j] = LL[ index+1,i,j]

            frame_type = self.intersect_info['frame_type']
            frame_num  = self.intersect_info['frame_num']


            Lmin = L_1st
            Lmax = L_2nd
            '''
            fig,ax = plt.subplots(figsize=(20,20))
            rt1_ax_kwargs = {'xlim'  :(0,1.1),
                            'ylim'  :(-0.7,0.7), 
                            'aspect': 'equal'
                            }
            self.append_frame(ax)
            ax.set(**rt1_ax_kwargs)
            R_1st,Z_1st = ray.RZ_ray(Lmax=L_1st)
            R_2nd,Z_2nd = ray.RZ_ray(Lmax=L_2nd)

            ax.scatter(R_1st.flatten(),Z_1st.flatten(),s=1)
            ax.scatter(R_2nd.flatten(),Z_2nd.flatten(),s=1)
            plt.show()
            '''
            L_sol_bk = L_sol 
            
            L_sol = (Lmin+Lmax) /2
            #self.frame_type_bk = self.intersect_info['frame_type']
            #self.frame_num_bk  = self.intersect_info['frame_num']  
            #self.index_1st_bk  = self.intersect_info['index_1st']  

     
            print((L_sol_bk-L_sol).max(),(L_sol_bk-L_sol).std()) 

        del self.intersect_info

        L_sol = (Lmin+Lmax) /2 
        ray.set_raytraced(Length=L_sol, ref_type=frame_type, ref_num=frame_num)



        return #R_sol, phi_sol, Z_sol, L_sol, frame_type, frame_num

                
    def RφZ(self,L,Ve_theta, Ho_theta,Z0,R0,Phi0=0):
        #print(Phi.shape,V_theta.shape,H_theta.shape)
        CosVe, SinVe = np.cos(Ve_theta)     , np.sin(Ve_theta)
        CosHo, SinHo = np.cos(Ho_theta-Phi0), np.sin(Ho_theta-Phi0)
        X0,Y0 = R0*np.cos(Phi0), R0*np.sin(Phi0)
        X = X0-L*CosVe*CosHo
        Y = Y0+L*CosVe*SinHo
        Z = Z0+L*SinVe
        R = np.sqrt(X**2+Y**2)
        Phi = np.arctan2(Y,X)       
        
        
        return R,Phi,Z
        

    def creating_ZR_ray(self,phi_num):
        Phi = np.linspace(0,self.phi_sol,phi_num)
        #Phi = np.linspace(0,self.phi_1st,phi_num)
        Z_i,R_i = self.__ZR(Phi,self.H_theta, self.W_theta,self.Z0,self.R0)
        print(Z_i.shape)
        
        dR = abs(R_i[:-1,:,:]-R_i[1:,:,:])
        R = 0.5*(R_i[1:,:,: ]+R_i[1:,:,:])
        dZ = abs(Z_i[:-1,:,:]-Z_i[1:,:,:])
        dPhi = abs(Phi[:-1,:,:]-Phi[1:,:,:])
        dL = np.sqrt(R*dPhi**2 + dR**2 + dZ**2)

        dL2 = np.zeros_like(R_i)
        dL2[:-1,:,:] += 0.5*dL
        dL2[1:,:,:] += 0.5*dL

        self.Zray,self.Rray = Z_i, R_i
        self.dL = dL
        self.dL2 = dL2
        self.Phi = Phi 
        self.L = np.sum(dL, axis=0)
        return  self.Zray, self.Rray
        
    def __ZR(self,Phi,H_theta,W_theta,Z0,R0):
        #print(Phi.shape,V_theta.shape,H_theta.shape)
        func = np.sin(Phi)/np.sin(Phi+W_theta)
        Z_phi = Z0 + R0 * np.tan(H_theta) * func
        R_phi = R0 * np.sqrt( func**2 - 2*np.cos(W_theta)*func + 1)
        return Z_phi,R_phi
        
    def ZR(self,Phi,H_theta,W_theta,Z0,R0):
        #print(Phi.shape,V_theta.shape,H_theta.shape)
        func = np.sin(Phi)/np.sin(Phi+W_theta)
        Z_phi = Z0 + R0 * np.tan(H_theta) * func
        R_phi = R0 * np.sqrt( func**2 - 2*np.cos(W_theta)*func + 1)
        return Z_phi,R_phi

    def create_projection_matrix(self,Z_inp,R_inp,phi_num:int=1000):
        self.creating_ZR_ray(phi_num)
        self.Z_inp = Z_inp 
        self.R_inp = R_inp
        self.nZi = Z_inp.size
        self.nRi = R_inp.size

        nL = phi_num
        
        nZ = self.Z_inp.size
        nR = self.R_inp.size

        dR = self.R_inp[1:] -self.R_inp[:-1] 
        dZ = self.Z_inp[1:] -self.Z_inp[:-1]
        dS = np.kron(dZ,dR).reshape(nZ-1,nR-1)
        dS = np.broadcast_to(dS,(nL,nZ-1,nR-1))
        dR = self.R_inp[1:] -self.R_inp[:-1] 
        dZ = self.Z_inp[1:] -self.Z_inp[:-1]

        R_all2 = np.broadcast_to(self.R_inp, (nL,nR))
        self.Z_inp2 = np.broadcast_to(self.Z_inp, (nL,nZ))
        dW = np.zeros((nL,nZ,nR))


        #Time = ElapsedTime()
        Tensor = np.zeros((self.h_ang.size,self.w_ang.size, nZ, nR))
        for i in tqdm(range(self.h_ang.size)):
            for j in range(self.w_ang.size):
                R_target = self.Rray[:,i,j]
                Z_target = self.Zray[:,i,j]

                dL3 = self.dL2[:,i,j]
                dL3 = np.broadcast_to(dL3,(nR-1,nZ-1,nL)).T

                Z_target2 =  np.broadcast_to(Z_target,(nZ-1,nL)).T
                R_target2 =  np.broadcast_to(R_target,(nR-1,nL)).T
                R_is_inrange = (R_all2[:,:-1] <= R_target2) * (R_target2 < R_all2[:,1:])
                Z_is_inrange = (self.Z_inp2[:,:-1] <= Z_target2) * (Z_target2 < self.Z_inp2[:,1:])


                dR1 = (R_target2  - R_all2[:,:-1]) * R_is_inrange
                dR2 = (R_all2[:,1:] - R_target2  ) * R_is_inrange
                dZ1 = (Z_target2  - self.Z_inp2[:,:-1]) * Z_is_inrange
                dZ2 = (self.Z_inp2[:,1:] - Z_target2  ) * Z_is_inrange

                W =  dL3 / dS 

                dW[:,:,:] = 0.
                dW[:,:-1,:-1] += np.einsum('ki,kj -> kij',dZ2,dR2)   * W
                dW[:,+1:,:-1] += np.einsum('ki,kj -> kij',dZ1,dR2)   * W
                dW[:,:-1,+1:] += np.einsum('ki,kj -> kij',dZ2,dR1)   * W
                dW[:,+1:,+1:] += np.einsum('ki,kj -> kij',dZ1,dR1)   * W

                W = np.sum(dW,axis=0)
                Tensor[i,j,:,:] = W

            #Time.printtime(i)
        self.Obs_tensor = Tensor
        self.Obs_matrix = Tensor.reshape(self.nx*self.ny, self.nRi*self.nZi)        


    def save_phis(self,name:str,path:str=''):
        np.savez(file=path+name+'_phis',
                 phi_sol =self.phi_sol,
                 v_ang=self.h_ang,
                 h_ang=self.w_ang,
                 ZR0=np.array([self.Z0,self.R0,])) 

    def load_phis(self,name:str,path:str=''):
        load = np.load(file=path+name+'_phis.npz')
        self.phi_sol = load['phi_sol']
        self.h_ang = load['v_ang']
        self.w_ang = load['h_ang']
        self.Z0 = load['ZR0'][0]
        self.R0 = load['ZR0'][1]

        self.nx = self.w_ang.size
        self.ny = self.h_ang.size
        self.H_theta, self.W_theta = np.meshgrid(self.h_ang,self.w_ang,indexing='ij')


    def save_tensors(self,name:str,path:str=''):
        np.savez(file=path+name+'_tensors',
                 Obs_tensor =self.Obs_tensor,
                 y_out=self.h_ang,
                 x_out=self.w_ang,
                 Z_inp=self.Z_inp,
                 R_inp=self.R_inp,
                 ZR0=np.array([self.Z0,self.R0,])) 
                 
        fig,ax = plt.subplots(1,2,figsize=(12,6))
        ax[0].set_aspect('equal')
        ax[0].set_xlim(0,1.1)
        ax[0].set_ylim(-0.7,0.7)
        ZZ_out,RR_out = np.meshgrid(self.Z_inp,self.R_inp,indexing='ij')
        ax[0].scatter(RR_out,ZZ_out,color='black',s=1)
        ax[0].set_ylabel('Z[m]')
        ax[0].set_xlabel('R[m]')
        self.append_frame(ax[0],label=True)

        YY_out,XX_out = np.meshgrid(self.h_ang,self.w_ang,indexing='ij')
        ax[1].set_aspect('equal')
        ax[1].pcolormesh(XX_out,YY_out,self.Obs_tensor.sum(axis=(2,3)))
        
        ax[1].set_ylabel('Y[rad]')
        ax[1].set_xlabel('X[rad]')
        ax[1].set_title('R0 ='+str(self.R0)+', Z0='+str(self.Z0))
        fig.savefig(path+name+'.png')

        self.append_frame(plt.axes())

    def load_tensors(self,name:str,path:str=''):
        load = np.load(file=path+name+'_tensors.npz')
        self.Obs_tensor = load['Obs_tensor']
        self.h_ang = load['y_out']
        self.w_ang = load['x_out']
        self.Z_inp = load['Z_inp']
        self.R_inp = load['R_inp']
        self.Z0 = load['ZR0'][0]
        self.R0 = load['ZR0'][1]

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
