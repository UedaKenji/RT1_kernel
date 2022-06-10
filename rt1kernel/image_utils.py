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

def remove_impulse_noises(img,threshold=1000,s=(5,5)):
    img_median = ndimage.median_filter(img, size=s)
    return np.where(abs(img-img_median) < threshold,img,img_median)


    
def load_tif_pile(
    filename_list:List[str],
    filename_headder:str,
    title    :str='',
    path     :str='',
    ):
    n = len(filename_list)
    img_frame = []
    for i, filename in enumerate(filename_list):
        img_pil = Image.open(path+filename_headder+filename+'.tif')
        n_frame = img_pil.n_frames

        for j in range(n_frame):
            img_pil.seek(j)
            img = np.asarray(img_pil,dtype=np.float64)

    pass

def load_tif_file(file_name,path='',BackGround=0.,remove=False,s=None,ignore_first=False):
    image_dict = {}

    img_pil = Image.open(path+'\\'+file_name+'.tif')
    #print(path+'\\'+file_name+'.tif is successfully imported.')
    frame = img_pil.n_frames
    #fig,ax = plt.subplots(frame,1,figsize=(20,50))

    for i in range(frame):
        img_pil.seek(i)
        img = np.asarray(img_pil,dtype=np.float64)
        if remove: img = remove_impulse_noises(img,remove,s)
        img -= BackGround  # substruct Back ground 
        image_dict[file_name+'_'+str(i+1)] = img

    #print(image_dict.keys())
    return image_dict

def simple_noise_replace(img,threshold=1500,Back=0.,result_plot=False,size=(10,10)):
    img_median = ndimage.median_filter(img, size=size)
    miss = abs(img - img_median) > threshold
    img_cr = np.logical_not(miss) * img + miss * img_median -Back
    if result_plot == True:
        fig,ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(15,6))
        imshow_cbar(fig,ax[0],img)
        imshow_cbar(fig,ax[1],miss*1.0)
        imshow_cbar(fig,ax[2],img_cr)
        ax[0].set_title('Origin')
        ax[1].set_title('Noise detect')
        ax[2].set_title('Corrected')
    
    return img_cr,miss

def detect_missing_area(img,threshold=2000):
    img_median = ndimage.median_filter(img, size=(15,5))
    return abs(img - img_median) > threshold

def plot_summery(GP,calib_phase = 0, save=False,file=None): # GPはGP_SM.kernel_Img クラスから生成されたインスタンス
    import math

    #plt.imshow(Shot.Im0,cmap='plasma')
    #Mu_Phase[file_name] = test.mu_Phase
    m_factor = np.logical_not(GP.miss_Im) 


    fig,ax = plt.subplots(3,2,figsize=(20,25))
    imshow_cbar(fig,ax[0,0],GP.mu_f    ,title = "Noise removed" ,save=False)
    imshow_cbar(fig,ax[0,1],GP.mu_intensity  ,title = "Bright"   ,cmap='jet',save=False)
    imshow_cbar(fig,ax[1,0],GP.mu_fringe   ,title = "Fringe" ,cmap='seismic',save=False)
    imshow_cbar(fig,ax[1,1],GP.mu_Amp     ,title = "Amplitude" ,cmap='jet',save=False)
    imshow_cbar(fig,ax[2,0],GP.mu_Gamma ,title = "Gamma" ,cmap='jet',save=False)#,vmax=2,vmin=0)
    imshow_cbar(fig,ax[2,1],GP.mu_Phase-calib_phase,
                    title = "Phase[rad],: $ \mu $ ="+str(GP.W_para['freqs'][1])+"[/pix]" ,
                    cmap='jet',save=False,vmax=0.5,vmin=-0.5)
                    #vmin=-math.pi,vmax=math.pi,cmap='hsv',save=False)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save:
        fig.suptitle(file)
        fig.savefig(file+"_analyed.png")


class Intensity_calib_bg: 

    def load_model(path:str):
        return pd.read_pickle(path)

    def __init__(self,
        im_all  :np.ndarray,
        t       :np.ndarray,
        n_order :str=4,
        lenx    :float=50,
        leny    :float=50,
        sig     :float=0.1,
        hw_window  :Tuple[slice,slice] = (slice(None),slice(None)),
        ):
        h_slice, w_slice = hw_window

        self.original_im_shape = (im_all.shape[1], im_all.shape[2])
        im_all = im_all[:,h_slice,w_slice]
        nt,ny,nx = im_all.shape 

        self.h_slice, self.w_slice = h_slice, w_slice
        self.n_order = n_order
        self.ny,self.nx = ny,nx

        x = np.linspace(0,nx-1,nx)
        y = np.linspace(0,ny-1,ny)

        self.lenx,self.leny = lenx, leny 
        Kx0x0 = self.Kse(x,x,len=lenx)
        Ky0y0 = self.Kse(y,y,len=leny)
        Kt0t0 = self.Kpol(t,t,n=n_order)
        #Kt0t0 = self.Kpol_exp(t,t,n=n_order)

        λ_x0x0, Q_x0x0 = np.linalg.eigh(Kx0x0)
        Q_x0x0 = Q_x0x0[:,λ_x0x0 > 1e-6]
        λ_x0x0 = λ_x0x0[λ_x0x0 > 1e-6]

        λ_y0y0, Q_y0y0 = np.linalg.eigh(Ky0y0)
        Q_y0y0 = Q_y0y0[:,λ_y0y0 > 1e-6]
        λ_y0y0 = λ_y0y0[λ_y0y0 > 1e-6]

        λ_t0t0, Q_t0t0 = np.linalg.eigh(Kt0t0)
        Q_t0t0 = Q_t0t0[:,λ_t0t0 > 1e-6]
        λ_t0t0 = λ_t0t0[λ_t0t0 > 1e-6]

        
        Λ_yx_sig =  np.einsum('i,j->ij',λ_y0y0,λ_x0x0) +  sig**2*np.ones((λ_y0y0.size,λ_x0x0.size))
        Λ_tyx_inv = np.einsum('i,jk->ijk',1./λ_t0t0, 1./Λ_yx_sig)



        temp = np.einsum('li,ijk->ljk',Q_t0t0.T,im_all)
        temp = np.einsum('lj,ijk->ilk',Q_y0y0.T,temp)
        temp = np.einsum('lk,ijk->ijl',Q_x0x0.T,temp)

        temp = Λ_tyx_inv*temp

        temp = np.einsum('li,ijk->ljk',Q_t0t0,temp)
        temp = np.einsum('lj,ijk->ilk',Q_y0y0,temp)
        temp = np.einsum('lk,ijk->ijl',Q_x0x0,temp)

        alpha = temp

        x1 = np.linspace(0,nx-1,nx)
        y1 = np.linspace(0,ny-1,ny)


        K_x1x0 = self.Kse(x1,x,len=lenx)
        K_y1y0 = self.Kse(y1,y,len=leny)
        K_t1t0_2 = self.Kpol_t0(t,n=n_order)
        #K_t1t0_2 = self.Kpol_exp_t0(t,n=n_order)


        temp = np.einsum('li,ijk->ljk',K_t1t0_2,alpha)
        temp = np.einsum('lj,ijk->ilk',K_y1y0,temp)
        temp = np.einsum('lk,ijk->ijl',K_x1x0,temp)

        print(temp.shape)

        self.im1 = temp

        self.linearized_intensity = self.im1[0].mean()
        self.t = t

    def save_model(self,
        path:str,
        print_image:bool=True):

        pd.to_pickle(self,path+'.pkl')
        if print_image:
            fig,ax = plt.subplots(1,self.n_order+1,sharex=True,sharey=True, figsize=((self.n_order+1)*4,5),tight_layout=True)
            fig.suptitle('mean of 0-order im: '+str(self.im1[0].mean())[:6]+', t_max: '+str(self.t.max())[:8]+'s'+', leny: '+str(self.leny)
                        +', lenx: '+str(self.lenx))
            for i in range(self.n_order+1):
                imshow_cbar(fig,ax[i],self.im1[i])
                ax[i].set_title( str(i)+'-order')

            fig.patch.set_alpha(1) 
            fig.savefig(path+'.png', facecolor='white')
            

    def plot_coef_im(self):

        fig,ax = plt.subplots(1,self.n_order+1, sharex=True,sharey=True,figsize=((self.n_order+1)*6,5))
        fig.suptitle('normalized coef ims')
        for i in range(self.n_order+1):
            imshow_cbar(fig,ax[i],self.im1[i])
            ax[i].set_title( str(i)+' order')

    def plot_check(self,
        im_all  :np.ndarray,
        t_all   :np.ndarray,
        point_list:List[tuple] =[(250,250),(10,10),(10,-10),(-10,-10),(-10,10)],
        t_max = None, 
        ax = None,
        ) -> None:

        if t_max is None:
            t_max = t_all.max()

        t = np.linspace(0,t_max,100)
        lines = [np.zeros_like(t) for i in point_list]


        for i,ti in enumerate(t):
            f = self.func_imt(ti)
            for j, point in enumerate(point_list):
                lines[j][i] = f[point]

        if ax is None:
            fig,ax = plt.subplots(figsize=(5,6))

        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for j, point in enumerate(point_list):
            ax.plot(t_all,im_all[:,point[0],point[1]],'o',color=cycle[j])
            ax.plot(t, lines[j],label=str(point),color=cycle[j])
        ax.legend()



    def func_imt(self,t, is_original_size=False):
        im = 0
        for i in range(self.n_order+1):
            im += self.im1[i]*t**int(i)

        if is_original_size:
            im_out = np.ones(self.original_im_shape)*im.mean()
            im_out[self.h_slice,self.w_slice] = im
            im = im_out

        return  im 
        
    def Kse(self,x0,x1,len):
        XX = np.meshgrid(x0,x1,indexing='ij')
        return np.exp(-0.5*(XX[0]-XX[1])**2/len**2) 


    def Kpol(self,t0,t1,n=1):
        TT = np.meshgrid(t0,t1,indexing='ij')
        TT_sum = np.zeros(TT[0].shape)
        for i in range(n+1):
            TT_sum += (TT[0]**int(i)) *(TT[1]**int(i))
        return TT_sum

    def Kpol_exp(self,t0,t1,n=1):
        TT = np.meshgrid(t0,t1,indexing='ij')
        TT_sum = np.zeros(TT[0].shape)
        for i in range(n+1):
            TT_sum += (TT[0]**i) *(TT[1]**i)* 0.8**int(2*i)
        return TT_sum

    def check_hist(self,im_all,t):
        im_all = im_all[:,self.h_slice,self.w_slice]

        fig,ax = plt.subplots(t.size,figsize=(5,4*t.size))
        for i in range(t.size):
            dif = (im_all[i]-self.func_imt(t[i]))#/self.func_imt(t[i])

            dif = dif.flatten()
            bins=200
            ax[i].hist(dif,bins=bins,density=True)
            ax[i].set_title(str(i)+', t:'+str(t[i])[:6]+', mean:' + str(im_all[i].mean())[:6]+', dif_mean:'+str(dif.mean())[:8]+', dif_std:'+str(dif.std())[:6])

            xmin,xmax =     -dif.std()*5,+dif.std()*5

            x = np.linspace(xmin,xmax,100)

            f = np.exp(-0.5*(x-dif.mean())**2/ dif.std()**2)/ (np.sqrt(2*np.pi)*dif.std())

            ax[i].plot(x,f,'--',alpha=0.8)

            ax[i].set_xlim(xmin,xmax)

    def Kpol_exp_t0(self,t0,n=1):
        K_sum = np.zeros((n+1,t0.size))
        for i in range(n+1):
            K_sum[i,:] = 0.8**int(i) * t0**i
        return K_sum


    def Kpol_t0(self,t0,n=1):
        K_sum = np.zeros((n+1,t0.size))
        for i in range(n+1):
            K_sum[i,:] = 1* t0**(i)
        return K_sum

        
class Intensity_calib: 
    def load_model(path:str):
        return pd.read_pickle(path)

    def __init__(self,
        im_all  :np.ndarray,
        t       :np.ndarray,
        n_order :str=4,
        lenx    :float=0.1,
        leny    :float=0.1,
        sig     :float=0.1,
        hw_window  :Tuple[slice,slice] = (slice(None),slice(None)),
        background :Union[float,Intensity_calib_bg]=None
        ):
        h_slice, w_slice = hw_window
        if background is None:
            bg = 0
        elif type(background) == float or type(background) == int:
            bg = background
        else:
            bg = background.func_imt(t=0)[hw_window]
        
        self.original_im_shape = (im_all.shape[1], im_all.shape[2])
        im_all = im_all[:,h_slice,w_slice] -bg
        self.bg = bg
        nt,ny,nx = im_all.shape 

        self.t_max = t.max()
        t          = t / self.t_max
        self.t     = t 

        self.h_slice, self.w_slice = h_slice, w_slice
        self.n_order = n_order
        self.ny,self.nx = ny,nx
        x = np.linspace(0,nx-1,nx)
        y = np.linspace(0,ny-1,ny)

        print(lenx)

        Kx0x0 = self.Kse(x,x,len=lenx)
        Ky0y0 = self.Kse(y,y,len=leny)
        Kt0t0 = self.Kpol(t,t,n=n_order)

        λ_x0x0, Q_x0x0 = np.linalg.eigh(Kx0x0)
        Q_x0x0 = Q_x0x0[:,λ_x0x0 > 1e-5]
        λ_x0x0 = λ_x0x0[λ_x0x0 > 1e-5]

        λ_y0y0, Q_y0y0 = np.linalg.eigh(Ky0y0)
        Q_y0y0 = Q_y0y0[:,λ_y0y0 > 1e-5]
        λ_y0y0 = λ_y0y0[λ_y0y0 > 1e-5]

        λ_t0t0, Q_t0t0 = np.linalg.eigh(Kt0t0)
        Q_t0t0 = Q_t0t0[:,λ_t0t0 > 1e-5]
        λ_t0t0 = λ_t0t0[λ_t0t0 > 1e-5]

        
        Λ_yx_sig =  np.einsum('i,j->ij',λ_y0y0,λ_x0x0) +  sig**2*np.ones((λ_y0y0.size,λ_x0x0.size))
        Λ_tyx_inv = np.einsum('i,jk->ijk',1./λ_t0t0, 1./Λ_yx_sig)



        temp = np.einsum('li,ijk->ljk',Q_t0t0.T,im_all)
        temp = np.einsum('lj,ijk->ilk',Q_y0y0.T,temp)
        temp = np.einsum('lk,ijk->ijl',Q_x0x0.T,temp)

        temp = Λ_tyx_inv*temp

        temp = np.einsum('li,ijk->ljk',Q_t0t0,temp)
        temp = np.einsum('lj,ijk->ilk',Q_y0y0,temp)
        temp = np.einsum('lk,ijk->ijl',Q_x0x0,temp)

        alpha = temp

        x1 = np.linspace(0,nx-1,nx)
        y1 = np.linspace(0,ny-1,ny)


        K_x1x0 = self.Kse(x1,x,len=lenx)
        K_y1y0 = self.Kse(y1,y,len=leny)
        K_t1t0_2 = self.Kpol_t0(t,n=n_order)


        temp = np.einsum('li,ijk->ljk',K_t1t0_2,alpha)
        temp = np.einsum('lj,ijk->ilk',K_y1y0,temp)
        temp = np.einsum('lk,ijk->ijl',K_x1x0,temp)

        print(temp.shape)

        self.im1 = temp
        self.coef = np.ones((self.n_order+1,ny,nx))

        self.linear_sensitivity = self.im1[1]/self.t_max
        self.f_max = im_all.max()

    def plot_check(self,
        im_all  :np.ndarray,
        t_all   :np.ndarray,
        point_list:List[tuple] =[(250,250),(100,100),(100,-100),(-100,-100),(-100,100)],
        t_max = None, 
        ax = None,
        ) -> None:

        if t_max is None:
            t_max = t_all.max()
        
        t = np.linspace(0,t_max,100)
        lines = [np.zeros_like(t) for i in point_list]


        for i,ti in enumerate(t):
            f = self.func_imt(ti)
            for j, point in enumerate(point_list):
                lines[j][i] = f[point]

        if ax is None:
            fig,ax = plt.subplots(figsize=(5,6))

        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for j, point in enumerate(point_list):
            ax.plot(t_all,im_all[:,point[0],point[1]],'o',color=cycle[j])
            ax.plot(t, lines[j],label=str(point),color=cycle[j])
        ax.legend()

    def save_model(self,
        path:str,
        print_image:bool=True):

        pd.to_pickle(self,path+'.pkl')
        if print_image:
            fig,ax = plt.subplots(1,self.n_order+3,figsize=((self.n_order+3)*5,5),tight_layout=True)
            fig.suptitle('mean linear sensitivity: '+str(self.linear_sensitivity.mean())[:8]+', t_max: '+str(self.t.max())[:8]+'s'+', f_max: '+str(self.f_max)[:8]
            +'background: '+str(self.bg.mean())[:6]+'\n h_slice: '+str(self.h_slice)+', w_slice: '+str(self.w_slice))

            imshow_cbar(fig,ax[1],self.bg,title='background')
            for i in range(self.n_order+1):
                imshow_cbar(fig,ax[i+2],self.im1[i])
                ax[i+2].set_title('coef of '+str(i)+'-order')

            
            t = np.linspace(0,self.t.max()*self.t_max+0.1,100)
            point_list = [(int(self.ny/2),int(self.nx/2)),(100,100),(100,-100),(-100,-100),(-100,100)]
            
            lines = [np.zeros_like(t) for i in point_list]


            for i,ti in enumerate(t):
                f = self.func_imt(ti)
                for j, point in enumerate(point_list):
                    lines[j][i] = f[point]

            cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for j, point in enumerate(point_list):
                ax[0].plot(t, lines[j],label=str(point),color=cycle[j])
            ax[0].set_xlabel('exposure time [s]')
            ax[0].set_ylabel('count [a. u.]')
            ax[0].legend()

            fig.patch.set_alpha(1) 
            fig.savefig(path+'.png', facecolor='white')
            

    def plot_coef_im2(self):

        fig,ax = plt.subplots(1,self.n_order+1,sharex=True,sharey=True, figsize=((self.n_order+1)*5,5))
        fig.suptitle('normalized coef ims')
        for i in range(self.n_order+1):
            imshow_cbar(fig,ax[i],self.im1[i])
            ax[i].set_title('coef of '+str(i)+' order')


    def plot_coef_im(self):
        fig,ax = plt.subplots(1,self.n_order-1,sharex=True,sharey=True, figsize=((self.n_order-1)*5,5))
        fig.suptitle('normalized coef ims')
        for i in range(self.n_order-1):
            imshow_cbar(fig,ax[i],self.coef[i+1]*float(2**16)**(i+1))
            ax[i].set_title('coef of '+str(i+1)+' order')

    def func_imt(self,t, is_original_size=False,is_bg=True):
        im = 0
        t  = t/self.t_max
        for i in range(0,self.n_order+1):
            im += self.im1[i]*t**int(i)

        im = im +is_bg*self.bg

        if is_original_size:
            im_out = np.ones(self.original_im_shape)*im.mean()
            im_out[self.h_slice,self.w_slice] = im
            im = im_out

        return  im

    def func_imt_diff(self,t, is_original_size=False,is_bg=True):
        im = 0
        t  = t/self.t_max
        for i in range(1, self.n_order+1):
            im += i*t**int(i-1)*self.im1[i]

        if is_original_size:
            im_out = np.ones(self.original_im_shape)*im.mean()
            im_out[self.h_slice,self.w_slice] = im
            im = im_out
        return  im /self.t_max

    def im_inverse_t(self,im,is_print=False,is_bg=True, is_original_size=False,n=10):
        f = (im-self.bg-self.im1[0])/self.im1[1]*self.t_max

        if is_original_size:
            im = im[self.h_slice,self.w_slice]

        for i in range(n):
            f = f - (self.func_imt(f,is_bg=is_bg)-im)/self.func_imt_diff(f)
            if is_print: print((self.func_imt(f,is_bg=is_bg)-im).mean())
        
        print((self.func_imt(f,is_bg=is_bg)-im).std())

        if is_original_size:
            im_out = np.zeros(self.original_im_shape)
            im_out[self.h_slice,self.w_slice] = f
            f = im_out

        return  f
        
    def Kse(self,x0,x1,len):
        XX = np.meshgrid(x0,x1,indexing='ij')
        return np.exp(-0.5*(XX[0]-XX[1])**2/len**2) 


    def Kpol(self,t0,t1,n=1):
        TT = np.meshgrid(t0,t1,indexing='ij')
        TT_sum = np.zeros(TT[0].shape)
        for i in range(n+1):
        
            TT_sum += (TT[0]**i) *(TT[1]**i)
        #for i in range(n):
        #    TT_sum += (TT[0]**(i+1))*(TT[1]**(i+1))
        return TT_sum 
        
    def check_hist(self,im_all,t):
        im_all = im_all[:,self.h_slice,self.w_slice]

        fig,ax = plt.subplots(t.size,figsize=(5,4*t.size))
        for i in range(t.size):
            dif = (im_all[i]-self.func_imt(t[i]))/(self.func_imt(t[i])-self.bg)
            dif = dif.flatten()
            bins=200
            ax[i].hist(dif,bins=bins,density=True)
            ax[i].set_title(str(i)+', t:'+str(t[i])[:6]+', mean:' + str(im_all[i].mean())[:6]+', dif_mean:'+str(dif.mean())[:8]+', dif_std:'+str(dif.std())[:6])

            xmin,xmax =     -dif.std()*5,+dif.std()*5

            x = np.linspace(xmin,xmax,100)

            f = np.exp(-0.5*(x-dif.mean())**2/ dif.std()**2)/ (np.sqrt(2*np.pi)*dif.std())

            ax[i].plot(x,f,'--',alpha=0.8)

            ax[i].set_xlim(xmin,xmax)
            

    def Kpol_t0(self,t0,n=1):
        K_sum = np.zeros((n+1,t0.size))
        for i in range(n+1):
            K_sum[i,:] = 1* t0**(i)
        #for i in range(n):
        #    K_sum[i,:] = 1* t0**(i+1)
        return K_sum

class image_set:
    def __init__(self,
        filename_list   :List[str],
        filename_headder:str='',
        title           :str='',
        path            :str='',
        n_frame       :int=10,
        ):
        self.names = filename_list
        self.headder = filename_headder
        self.title = title
        self.n_frame = n_frame
        self.path_list:List[str] = []
        self.imgs_raw :dict[str,List[np.ndarray]] = {} 
        
        for filename in self.names:
            name = path+filename_headder+filename+'.tif'
            self.path_list.append(name)
            img_pil = Image.open(name)

            img_list :List[np.ndarray] = []

            for j in range(n_frame):
                img_pil.seek(j)
                img = np.asarray(img_pil,dtype=np.float64)
                img_list.append(img)
            
            self.imgs_raw[filename] = img_list
    
    def pile(self,
        is_plot   :bool = True,
        is_remove :bool = True,
        threshold :float= 1000,
        size      :tuple= (5,5) 
        ):
        
        self.im_miss = {name: list([]) for name in self.names}
        self.im_cr_ave :List[np.ndarray] = []
        for i in range(self.n_frame):
            
            im_cr_ave = 0.

            for name in self.names:
                im = self.imgs_raw[name][i]
                if is_remove:
                    im_cr, miss =  simple_noise_replace(img=im,threshold=threshold,result_plot=is_plot, size=size)
                    plt.show()
                else: 
                    im_cr, miss = im, False 
                print(i,name,im_cr.mean(),im_cr.std())
                self.im_miss[name].append(miss)

                im_cr_ave += im_cr 
            
            im_cr_ave /= len(self.names) 
            self.im_cr_ave.append(im_cr_ave)

    def set_time(self,
        exposure_time : float ,      #[s]
        initial_time  : float = 1.0, #[s]
        readout_time  : float = 0.,  #[s]
        is_ms_print   : bool  = False,
        ):
        self.exposure_time = exposure_time
        self.time_sta   :List[str] = []
        self.time_end   :List[str] = []
        self.time_names :List[str] = []
        
        for i in range(self.n_frame):
            if i == 0:
                time_start = initial_time
            else :
                time_start = time_end + readout_time 
            time_end   = time_start+self.exposure_time

            self.time_sta.append(time_start)
            self.time_end.append(time_start)

            if is_ms_print:
                time_name = str(time_start*1000)[:6]+'-'+str(time_end*1000)[:6] + 'ms'
            else:
                time_name = str(time_start)[:6]+'-'+str(time_end)[:6] +'s' 
            self.time_names.append(time_name)
    
    def plot(self,
        fig      :plt.Figure=None,
        n_rows   :int= 1,
        i_start  :int =0,
        is_save  :bool = False, 
        save_path:str = ''
        ):
        self.i_start, self.n_rows = i_start, n_rows
        n = self.n_frame - i_start
        n_cols = int((n-1)/n_rows) + 1

        if fig is not  None:
            axes = fig.subplots(n_rows,n_cols,sharex = True,sharey=True)
        else:
            fig,axes = plt.subplots(n_rows,n_cols,sharex = True,sharey=True, figsize=(4*n_cols,4*n_rows+1), tight_layout=True)

        for i in range(i_start, self.n_frame):
            if n_rows == 1:
                j = i - i_start
                ax = axes[j]    
            else:
                j1 = int((i-i_start)/n_cols) 
                j2 = (i-i_start) - j1*n_cols
                ax = axes[j1,j2]
            imshow_cbar(fig,ax,self.im_cr_ave[i])
            ax.set_title(str(i)+': '+self.time_names[i])

            ax.set_xticks(np.linspace(0,self.im_cr_ave[i].shape[1],5))
            ax.set_yticks(np.linspace(0,self.im_cr_ave[i].shape[0],5))
            ax.grid()

        
        fig.suptitle(self.title+'#'+self.headder+str(self.names))

        if is_save:
            fig.savefig(save_path+self.title+'.svg')
            fig.set_facecolor('white')
            fig.savefig(save_path+self.title+'.png',transparent=False)
            return 
        else: 
            return  fig

    def calib_intensity(self,
        calib            :Intensity_calib,
        wavelength_factor:float= 1.0,
        is_plot          :bool = False,
        is_save          :bool = False,
        save_path        :str  = "",
        ):
        if not (type(calib) == Intensity_calib): return print('calib class not match!!!')
        
        self.intensity :List[np.ndarray] = []
        for im in self.im_cr_ave:
            t = calib.im_inverse_t(im)
            intensity = wavelength_factor * calib.linear_sensitivity.mean() * t / self.exposure_time 
            self.intensity.append(intensity)

        if is_plot:  
            im_list = self.intensity
            i_start, n_rows = self.i_start,self.n_rows
            n = self.n_frame - i_start
            n_cols = int((n-1)/n_rows) + 1

            fig,axes = plt.subplots(n_rows,n_cols,sharex = True,sharey=True, figsize=(4*n_cols,4*n_rows+1), tight_layout=True)

            vmax = max([im.max() for im in im_list])
            for i in range(i_start, self.n_frame):
                if n_rows == 1:
                    j = i - i_start
                    ax = axes[j]    
                else:
                    j1 = int((i-i_start)/n_cols) 
                    j2 = (i-i_start) - j1*n_cols
                    ax = axes[j1,j2]
                imshow_cbar(fig,ax,im_list[i],vmin=0,vmax=vmax)
                ax.set_title(str(i)+': '+self.time_names[i]+' [mW]')

                ax.set_xticks(np.linspace(0,im_list[i].shape[1],5))
                ax.set_yticks(np.linspace(0,im_list[i].shape[0],5))
                ax.grid()
            
            fig.suptitle(self.title+'_intensity#'+self.headder+str(self.names))
            
                    
            if is_save:
                fig.savefig(save_path+self.title+'_intensity.svg')
                fig.set_facecolor('white')
                fig.savefig(save_path+self.title+'_intensity.png',transparent=False)
                return 
            else: 
                return  fig

        



            

if __name__ == '__main__':
    print(__name__)
    print(__file__)
