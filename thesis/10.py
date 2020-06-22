# -*- coding: utf-8 -*-

"""
Created on Tue Feb  4 17:03:07 2020

@author: agoetz
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
from os.path import isfile, join
from multiprocessing.pool import ThreadPool
import gc
from helper import radial_profile, focused_otf
from scipy.optimize import curve_fit
import pickle as pk
path='/home/alex/Desktop/'


def exp(x,a,b,c):
    temp=np.exp(-x/a)*b+c
    return(temp)

extx=7.2e-3
pxx=1236#1626
exty=5.4e-3
pxy=1236
zoom=23.65

#%% magnification

def fsin(x,w,a,b,c):
    out=np.sin(x*w+c)*a+b
    return(out)
    
x=np.linspace(0,extx,pxx)   

paths=[]

est_zoom=21
paths.append([path+'data/mtf/far/20x_mag/6/',60])
paths.append([path+'data/mtf/far/20x_mag/7/',70])
paths.append([path+'data/mtf/far/20x_mag/10/',100])

# est_zoom=21
# paths.append([path+'data/mtf/near/20x_mag/9/',90])
# paths.append([path+'data/mtf/near/20x_mag/10/',100])


# est_zoom=11
# paths.append(['/home/alex/Desktop/Lab/10x_mag/9/',90])
# paths.append(['/home/alex/Desktop/Lab/10x_mag/10/',100])

# est_zoom=11
# paths.append([path+'data/mtf/near/10x_mag/6/',60])
# paths.append([path+'data/mtf/near/10x_mag/7/',70])


zoom_list=[]
for i in range(len(paths)):
    files = [f for f in listdir(paths[i][0]) if isfile(join(paths[i][0], f))]
    
    img=np.zeros((len(files),pxy,pxx))
    for j in range(len(files)):
        if(files[i][-4:]=="tiff"):
            im = np.array(Image.open(paths[i][0]+files[j]))
            im=im[:pxx,150:pxx+150]
            img[j]=im
            # print("lp/mm:",paths[i][1]," number:",j)   
            
            cross=im[0,:]
            lps=extx*paths[i][1]*1e3/est_zoom
            
            est_freq=2*np.pi*lps/extx  
            est_mean=np.mean(cross)
            
           
            p0=[est_freq,20,est_mean,5e3]
            par=curve_fit(fsin,x,cross.astype(np.float),p0=p0)
            # plt.plot(x,fsin(x,*par[0]))
            # plt.plot(x,fsin(x,*p0))
            # plt.plot(x,cross)
            
            freq=par[0][0]
            
            linepair_object=2*np.pi/freq
            linepair_image=1/paths[i][1]*1e-3
            zoom=linepair_object/linepair_image
            zoom_list.append(zoom)

zoom=np.mean(zoom_list)    
print("magnification: ",zoom)


#%% line pairs
samples=3

pxx=1626
pxy=1236

dat=np.empty((21,samples,pxy,pxx))
fdat=np.empty((21,samples,pxy,pxx))

lp_path=path+'data/mtf/far/20x_lp/'

for i in range(1,21):
    for j in range(samples):
        im = np.array(Image.open(lp_path+str(i*10)+"_"+str(j)+".tif"))
        dat[i,j]=im
        
lp_y=np.empty((21))

for i in range(1,21):
    temp=0
    for j in range(samples):
        mean=np.std(dat[i,j])
        temp+=mean
    temp=temp/3
    lp_y[i]=temp
lp_y=lp_y[1:]

lp_x=np.empty((21))
for i in range(21):
    lp_x[i]=i*10*1e3
lp_x=lp_x[1:]

lp_y=lp_y/np.max(lp_y)

pk.dump([lp_x,lp_y],open(path+'data/mtf/far/pks/20xlp.pk', "wb"))
    
#%% slant edge

sl_path=path+'data/mtf/far/20x_slant/'

files = [f for f in listdir(sl_path) if isfile(join(sl_path, f))]
syl=[]

for j in range(len(files)):
    if(files[j][-4:]=="tiff"):
        print("slant edge ",j)
        im = np.array(Image.open(sl_path+files[j]))
        im=im[:pxx,150:pxx+150]
        
        # plt.figure(str(files[j]))
        # plt.imshow(im)
           
        img=im/np.max(im)
        
        img2=np.where(img < np.mean(img), 0, 1)       
        
        if(np.mean(img2[:,0])>0.5):
            # print("flipped ",j)
            img=np.flip(img)
            img2=np.flip(img2)
        
        
        y_axis=np.arange(pxy)
        
        clist=np.empty((pxy))
        for i in range(pxy):
            clist[i]=img2[i].tolist().index(1)        
        z = np.polyfit(y_axis, clist, 1)
        
        
        #############################
        p=np.poly1d(z)      
        fit=p(y_axis)
        

        cut=0.2
        ausr=np.where(fit-np.std(clist)*cut<clist)

        # plt.figure(str(j)+"1")        
        # plt.plot(y_axis,fit,label="fit")
        # plt.plot(y_axis,clist,label="clist")
        # plt.plot(y_axis,fit-np.std(clist)*cut,label="fit - std")
        # plt.plot(y_axis[ausr],clist[ausr],label="not to be ignored")  
        # plt.legend()
        
        y_axis2=y_axis[ausr]
        clist2=clist[ausr]
        z = np.polyfit(y_axis2, clist2, 1)
        p=np.poly1d(z)      
        fit=p(y_axis2)
        
        cut=0.2
        ausr=np.where(fit-np.std(clist2)*cut<clist2)

        # plt.figure(str(j)+"2")           
        # plt.plot(y_axis2,fit,label="fit")
        # plt.plot(y_axis2,clist2,label="clist")
        # plt.plot(y_axis2,clist2-np.std(clist2)*cut,label="clist - std")
        # plt.plot(y_axis2[ausr],clist2[ausr],label="not to be ignored")  
        # plt.legend()   
        
        y_axis3=y_axis2[ausr]
        clist3=clist2[ausr]
        
        z = np.polyfit(y_axis3, clist3, 1) 
        
        #################################
        
        alpha=np.arctan(z[0])
        print(alpha)
        
        pr=np.empty((2,pxx*pxy))
        k=0
        for i in range(pxx):
            for m in range(pxy):
                l=np.sqrt((i+1)**2+(m+1)**2)
                beta=np.arctan((m+1)/(i+1))
                x=np.cos(alpha+beta)*l
                pr[0,k]=x
                pr[1,k]=img[m,i]
                k+=1
        
        center=clist[0]
        
        indices=np.argsort(pr[0])
        x=pr[0][indices]
        y=pr[1][indices]
        
        xstart=np.argmin(np.abs(x-center+100))
        xfin=np.argmin(np.abs(x-center-100))
        x=x[xstart:xfin]
        y=y[xstart:xfin]
        y/=np.max(y)
        
        bins=500
        
        yh,xh=np.histogram(x,bins=bins,weights=y)
        
        yh/=np.max(yh)
        yhg=np.gradient(yh)
        
        hbins=int(bins/2)
        fft=(np.abs((np.fft.fft(yhg))))[:hbins]
        
        fft=fft/fft[0]
        
        syl.append(fft)

xfreq=np.linspace(1/((x[-1]-x[0])*extx/pxx/zoom),bins*zoom*pxx/(x[-1]-x[0])/extx/2,hbins)

pk.dump([xfreq,syl],open(path+'data/mtf/far/pks/20xslantsyl.pk', "wb"))

# %%

xfreq,syl=pk.load(open(path+'data/mtf/far/pks/20xslantsyl.pk', "rb"))
lp_x,lp_y=pk.load(open(path+'data/mtf/far/pks/20xlp.pk', "rb"))

# brenn=75e-3
# radius=25.4e-3/2
# alpha=np.arctan(radius/brenn)
# NA=np.sin(alpha)*1.458

lam=632e-9
NA=0.4
# NA=0.25

cutoff=NA/lam

q=np.linspace(0,2*cutoff,1000)
otf_foc=focused_otf(q,NA,lam)


SIZE = 10

plt.rc('font', size=SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)    # legend fontsize
plt.rc('figure', titlesize=SIZE)  # fontsize of the figure title

fig=plt.figure('Figure_10.svg',figsize=(20,10))

# for i in range(len(syl)):
#     plt.plot(xfreq,syl[i])
    
plt.plot(xfreq,np.mean(syl,axis=0))
plt.plot(q,otf_foc)
plt.plot(lp_x,lp_y)


plt.xlabel(r"spatial frequency $[\mu m^{-1}]$")
plt.ylabel("power spectrum")
plt.yscale("log")
# plt.xlim(-0.05,1.5)
# plt.ylim(1e-7,1)
plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig(path+"Figure_10.svg",format="svg")



