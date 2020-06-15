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
from os import listdir
from os.path import isfile, join
from multiprocessing.pool import ThreadPool
import gc

def radial_profile(data,center):
    y,x = np.indices((data.shape))
    r = np.sqrt((x-center[0])**2 + (y-center[1])**2)
    r = r.astype(np.int)
    tbin = np.bincount(r.ravel(), weights = data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin/nr
    return(radialprofile)

extx=5.4e-3#7.2e-3

pxx=1236#1626
exty=5.4e-3
pxy=1236
zoom=2.6
cores=4
colloidpath='//cern.ch/dfs/Users/a/agoetz/Desktop/corona_work_place/thesis/data/speckles/speckles/10/'


#%%avim and avimft of colloids
colloidfiles = [f for f in listdir(colloidpath) if isfile(join(colloidpath, f))]
colloidfiles=colloidfiles[:-1]

img=np.zeros((len(colloidfiles),pxy,pxx))

for i in range(len(colloidfiles)):
    if(colloidfiles[i][-3:]=="tif"):
        im = np.array(Image.open(colloidpath+colloidfiles[i]))
        im=im[:pxx,150:pxx+150]
        img[i]=im
        print("loaded colloidfile ",i,len(colloidfiles))

cut=len(colloidfiles)%cores
if(cut!=0):
    colloidfiles=colloidfiles[:-cut]

cavim=np.sum(img,axis=0)/len(colloidfiles)



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


fig=plt.figure('Figure_11.svg',figsize=(20,10))

extent=np.array([-extx/2,extx/2,-exty/2,exty/2])*1e3

plt.imshow(img[10]-cavim,extent=extent) 
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")


plt.tight_layout()
plt.savefig("Figure_11.svg",format="svg")

