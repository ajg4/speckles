# -*- coding: utf-8 -*-

"""
Created on Tue Feb  4 17:03:07 2020

@author: agoetz
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from scipy import ndimage, misc
from numpy import loadtxt
from scipy.optimize import curve_fit

def exp(x,a,b,c):
    temp=np.exp(-x/a)*b+c
    return(temp)

extx=7.2e-3
pxx=1626
exty=5.4e-3
pxy=1236
zoom=23.65


#%% manual projection
number_of_files=11
cut=30
dat=np.empty((number_of_files,pxy,pxx))
path='//cern.ch/dfs/Users/a/agoetz/Desktop/corona_work_place/thesis/data/classic_mtf/mtf_20x_slant/'

sl_y=np.zeros((300))

for m in range(0,number_of_files):
    im = np.array(Image.open(path+str(m)+".tif"))
    dat[m]=im
    
    img=dat[m]
    img=img/np.max(img)
    
    img2=np.where(img < 0.3, 0, 1)
    
    clist=np.empty((pxy))
    for i in range(pxy):
        clist[i]=img2[i].tolist().index(0)
    
    z = np.polyfit(np.arange(pxy), clist, 1)
    p=np.poly1d(z)
    
    alpha=np.arctan(z[0])
    print(m,alpha)

    xx=np.arange(pxx)
    yy=np.arange(pxy)
    xx,yy=np.meshgrid(xx,yy)
    xs=xx*np.cos(-alpha)+yy*np.sin(-alpha)
    x=xs.flatten()
    y=img.flatten()
    
    indices=np.argsort(x)
    x=x[indices]
    y=y[indices]
    
    center=clist[0]
    
    xstart=np.argmin(np.abs(x-center+100))
    xfin=np.argmin(np.abs(x-center-100))
    x=x[xstart:xfin]
    y=y[xstart:xfin]
    y/=np.max(y)
    
    binnum=len(sl_y)+1
    
    yh,xh=np.histogram(x,bins=binnum,weights=y)
    yhg=np.diff(yh)

    
    sl_x=np.fft.fftfreq(binnum,extx/zoom/pxx*(xh[1]-xh[0]))
    fft=np.abs((np.fft.fft(yhg)))
    sl_y+=fft
    

sl_y=sl_y[:cut]
sl_x=sl_x[:cut]

sl_y/=np.max(sl_y)


p0=[1e6,1,0.3]
sl_pars=curve_fit(exp,sl_x,sl_y,p0=p0)

#%%
#deviation from mean
samples=3

dat=np.empty((21,samples,pxy,pxx))
fdat=np.empty((21,samples,pxy,pxx))

path='//cern.ch/dfs/Users/a/agoetz/Desktop/corona_work_place/thesis/data/classic_mtf/mtf_20x/'

for i in range(1,21):
    for j in range(samples):
        im = np.array(Image.open(path+str(i*10)+"_"+str(j)+".tif"))
        dat[i,j]=im
        
std_y=np.empty((21))

for i in range(1,21):
    temp=0
    for j in range(samples):
        mean=np.std(dat[i,j])
        temp+=mean
    temp=temp/3
    std_y[i]=temp
std_y=std_y[1:]

std_x=np.empty((21))
for i in range(21):
    std_x[i]=i*10/0.001
std_x=std_x[1:]


std_y=std_y/np.max(std_y)

p0=[1e6,1,0.3]
std_pars=curve_fit(exp,std_x,std_y,p0=p0)


#%%anytic section

lam=632e-9
NA=0.4

rho0=NA/lam

# an_x=np.linspace(0,2*rho0,100)
an_x=sl_x

fac=an_x/2/rho0

an_y=2/np.pi*(np.arccos(fac)-fac*np.sqrt(1-fac**2))

p0=[1e6,1,0.3]
an_pars=curve_fit(exp,an_x,an_y,p0=p0)

#%%
SMALL_SIZE = 30
MEDIUM_SIZE = 30
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


fig=plt.figure('Figure_10.svg',figsize=(20,10))


plt.plot(1e-6*an_x,an_y,label="analytic formula")
# plt.plot(1e-6*an_x,exp(an_x,*an_pars[0]),label="analyti fit [exp(-q/a)] with a="+str(round(1e-6*an_pars[0][0],3))+r" $\mu m^{-1}$")

plt.plot(1e-6*std_x,std_y,label="linepair visibilty")
plt.plot(1e-6*sl_x,sl_y,label="slant edge")
# plt.plot(1e-6*sl_x,exp(sl_x,*sl_pars[0]),label="slant edge fit [exp(-q/a)] with a="+str(round(1e-6*sl_pars[0][0],3))+r" $\mu m^{-1}$")
plt.xlabel(r"spatial frequency $[\mu m^{-1}]$")
plt.ylabel("visibility")

# plt.yscale("log")
# plt.xscale("log")
plt.legend() 


plt.tight_layout()
plt.savefig("Figure_10.svg",format="svg")

