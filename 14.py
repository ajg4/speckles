# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 19:06:27 2020

@author: agoetz
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

NA=0.4 #numerical aperture
lam=550e-9 #maximum emission wavelength of scintillator
n=1.95 #refractive index of scintillator
dz=10e-6 #thickness of scintillator
magnification=20

qmax=10e6 #max frequency in the object plane (!)
zetas=200 #integration steps for the thickness
qs=100 #steps of the frequency
terms=10 #order of series approximation

qmax=2*NA/lam#/magnification #frequency in the image plane
q=np.linspace(qmax/2000,qmax,qs) #first frequency non zero

s=lam*q/NA


dz_list=[10e-6,20e-6,50e-6,100e-6]
otf_list=[]
for dz in dz_list:
    zeta=np.linspace(dz/zetas,dz,zetas)
    otf=np.zeros(len(s))
    
    def func1(k,a,b):
        temp=(-1)**(k+1) * np.sin(2*k*b)/(2*k) * (jv(2*k-1,a)-jv(2*k+1,a))
        return(temp)
    
    def func2(k,a,b):
        temp=(-1)**(k) * np.sin((2*k+1)*b)/(2*k+1) * (jv(2*k,a)-jv(2*k+2,a))
        return(temp)
    
    def freq_response(s,z):
        w20=NA**2*z/2/n
        a=4*np.pi*w20*s/lam
        b=np.arccos(s/2)
        
        temp1=b*jv(1,a)
        for i in range(1,terms):
            temp1+=func1(i,a,b)
    
        temp2=0
        for i in range(0,terms):
            temp2+=func2(i,a,b)
            
        out=4/np.pi/a*np.cos(a*s/2)*temp1-4/np.pi/a*np.sin(a*s/2)*temp2
        return(out)
    
    
    for i in range(len(s)):
        for j in range(len(zeta)):
            otf[i]+=freq_response(s[i],zeta[j])
        otf[i]=otf[i]/zetas
    otf_list.append(otf)
    
#%%
rho0=NA/lam
an_x=q
fac=an_x/2/rho0
an_y=2/np.pi*(np.arccos(fac)-fac*np.sqrt(1-fac**2))

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

fig=plt.figure('Figure_14.svg',figsize=(20,10))
plt.plot(q*1e-6,an_y,label="perfect focus")
for i in range(len(dz_list)):
    plt.plot(q*1e-6,otf_list[i],label=str(round(dz_list[i]*1e6))+r" $\mu m$ defocus")

plt.xlabel(r"spatial frequency $\mu m^{-1}$")
plt.ylabel("visibility")
# plt.yscale("log")
plt.legend()


plt.tight_layout()
plt.savefig("Figure_14.svg",format="svg")
    











