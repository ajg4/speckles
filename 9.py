# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 12:45:06 2020

@author: agoetz
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

file_list=[]

pxx=1626
pxy=1236

# file_list.append('//cern.ch//dfs/Users/a/agoetz/Desktop/corona_work_place/thesis/data/spatialfilter/9mm-10um-50mm.tif')
file_list.append('//cern.ch//dfs/Users/a/agoetz/Desktop/corona_work_place/thesis/data/spatialfilter/9mm-20um-50mm.tif')
# file_list.append('//cern.ch//dfs/Users/a/agoetz/Desktop/corona_work_place/thesis/data/spatialfilter/9mm-30um-50mm.tif')
file_list.append('//cern.ch//dfs/Users/a/agoetz/Desktop/corona_work_place/thesis/data/spatialfilter/9mm-50um-50mm.tif')
file_list.append('//cern.ch//dfs/Users/a/agoetz/Desktop/corona_work_place/thesis/data/spatialfilter/9mm-50mm.tif')

ims=np.empty((len(file_list),pxy,pxx))

for i in range(len(file_list)):
    ims[i]=np.array(Image.open(file_list[i])).astype(np.float)
    ims[i]=ims[i]/np.max(ims[i])



#%%

from scipy.optimize import curve_fit
def gauss1(x,a,b,c,d):
    return(a*np.exp(-0.5*(x-b)**2/c**3)+d)
           
def gauss2(x,y,a,c,d,x0,y0):
    return(a*np.exp(-0.5*((x-x0)**2+(y-y0)**2)/c**3)+d)

#getting the central positions

y_pos=np.empty(len(file_list))

for i in range(len(file_list)):
    x_data=np.arange(pxy)
    y_data=ims[i,:,int(pxx/2)]
    p0=[0.6,pxy/2,40,0.1]
    par=curve_fit(gauss1,x_data,y_data,p0=p0)
    
    # plt.plot(x_data,y_data)
    # plt.plot(x_data,gauss1(x_data,*p0))
    # plt.plot(x_data,gauss1(x_data,*par[0]))
    
    y_pos[i]=par[0][1]

x_pos=np.empty(len(file_list))

for i in range(len(file_list)):
    x_data=np.arange(pxx)
    y_data=ims[i,int(pxy/2),:]
    p0=[0.6,pxx/2,40,0.1]
    par=curve_fit(gauss1,x_data,y_data,p0=p0)
    
    # plt.plot(x_data,y_data)
    # plt.plot(x_data,gauss1(x_data,*p0))
    # plt.plot(x_data,gauss1(x_data,*par[0]))
    
    x_pos[i]=par[0][1]
    

#getting the standard deviation (assuming symmetricity)

vals=np.empty((len(file_list),5))

labels=["20um","50um","no pinhole"]

    


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


fig=plt.figure('Figure_9.svg',figsize=(20,10))



for i in range(len(file_list)):
    x_data=np.arange(pxy)
    y_data=ims[i,:,int(x_pos[i])]
    p0=[0.6,pxy/2,40,0.1]
    par=curve_fit(gauss1,x_data,y_data,p0=p0)
    
    plt.plot(x_data/1235*6,y_data,alpha=1,linewidth=0.5,label=labels[i])
    # plt.plot(x_data,gauss1(x_data,*p0))
    plt.plot(x_data/1235*6,gauss1(x_data,*par[0]),label=labels[i]+" fit")
    
    # vals[i][0:3]=par[0][np.array([0,2,3])]
    # vals[i][3]=x_pos[i]
    # vals[i][4]=y_pos[i]
    
plt.legend() 
plt.xlabel("x [mm]")
plt.ylabel("intensity [a.u.]")

plt.tight_layout()
plt.savefig("Figure_9.svg",format="svg")

    





    
    


