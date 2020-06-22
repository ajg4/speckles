# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 19:06:27 2020

@author: agoetz
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from helper import defocused_otf, focused_otf
path='/home/alex/Desktop/'

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

#%% defocues otf

dz_list=[10e-6,20e-6,50e-6,100e-6]
otf_list=[]
for dz in dz_list:   
    otf=defocused_otf(dz,zetas,q,NA,lam,n,terms)
    otf_list.append(otf)
    
#%% focused otf

otf_foc=focused_otf(q,NA,lam)

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
plt.plot(q*1e-6,otf_foc,label="perfect focus")
for i in range(len(dz_list)):
    plt.plot(q*1e-6,otf_list[i],label=str(round(dz_list[i]*1e6))+r" $\mu m$ defocus")

plt.xlabel(r"spatial frequency $\mu m^{-1}$")
plt.ylabel("visibility")
# plt.yscale("log")
plt.legend()


plt.tight_layout()
plt.savefig(path+"Figure_14.svg",format="svg")
    











