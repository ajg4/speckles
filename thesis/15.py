# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 17:37:36 2020

@author: agoetz
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from PIL import Image
from scipy import interpolate
path='/home/alex/Desktop/'


image=Image.open(path+'data/yag/absen.png')
im_arr=np.array(image)
im_arr=im_arr[:,:,:]


x1=np.array([30,25,20,17.01,16.99,15,12.4,10,7.7,6.1,4.6,3,2.64,2.2,1.8,0])
y1=np.array([2.7,4.2,7.4,10.8,2.5,3,4.7,8.6,16.1,27.8,49.8,88,95.2,99,100,100])/100

x2=np.array([30,25,20,17.01,16.99,15,12.33,10,8.32,6.91,5,4.05,3.2,2.63,0])
y2=np.array([5.25,8.17,13.8,20,4.3,5.64,8.91,16.33,24.6,37.5,68.5,86.5,98,100,100])/100

x3=np.array([30,25,20,17.01,16.99,15,12.33,10,8.427,7,5.66,4.98,4.13,3.44,0])
y3=np.array([9.6,15.1,25.6,35.6,8.17,10.76,17.27,29.6,42,60.1,80.3,89.72,97.8,100,100])/100

x4=np.array([30,25,20,17.01,16.99,15.04,12.21,10,8.01,6.59,5.78,4.89,4.09,0])
y4=np.array([14.1,22.19,35.4,48.6,11.3,15.2,25.63,40.3,60,80,90,97.3,100,100])/100

x5=np.array([30,27.64,25,22.6,20,17.01,16.99,16,15,11.97,10,8.86,7.45,6.96,5.74,4.72,0])
y5=np.array([22.4,26.71,33.8,40.6,51.7,67.1,18.1,20.1,24.1,40,58.2,70.3,85,90,98.1,100,100])/100

x6=np.array([30,25,20,17.01,16.99,15,12.4,10,8.29,7.04,6.2,0])
y6=np.array([39.7,55.95,76.7,89.31,32.9,42.4,60,82.3,94.4,99.1,100,100])/100

x7=np.array([30,25,20,17.01,16.99,15.04,13,11.57,10,8.49,7.6,0])
y7=np.array([63,80,94.6,99.1,54.8,66.7,80,89.5,96.7,99.9,100,100])/100


x1=np.flip(x1)
x2=np.flip(x2)
x3=np.flip(x3)
x4=np.flip(x4)
x5=np.flip(x5)
x6=np.flip(x6)
x7=np.flip(x7)

y1=np.flip(y1)
y2=np.flip(y2)
y3=np.flip(y3)
y4=np.flip(y4)
y5=np.flip(y5)
y6=np.flip(y6)
y7=np.flip(y7)



i=0
names=[5,10,20,30,50,100,200]


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


fig=plt.figure('Figure_15.svg',figsize=(20,10))
extent=[0,30,0,1]
# plt.imshow(im_arr,extent=extent,aspect='auto')
plt.grid(True)

for x,y in zip([x1,x2,x3,x4,x5,x6,x7],[y1,y2,y3,y4,y5,y6,y7]):
    x=x[1:]
    y=y[1:]
    
    x_1_new=np.linspace(np.min(x),16.99,10000)
    x_2_new=np.linspace(17.01,30,10000)
    
    
    x_1=x[:np.sum(np.where(x<17,1,0))]
    x_2=x[np.sum(np.where(x<17,1,0)):] 
    y_1=y[:np.sum(np.where(x<17,1,0))]
    y_2=y[np.sum(np.where(x<17,1,0)):]     
    
    
    f1 = interpolate.interp1d(x_1, y_1,kind="cubic")
    
    f2 = interpolate.interp1d(x_2, y_2,kind="cubic")
        
    x_plot=np.concatenate((x_1_new,x_2_new))
    y_plot=np.concatenate((f1(x_1_new),f2(x_2_new)))
    
    plt.plot(x_plot,y_plot,label=str(names[i])+r' $\mu m$')  
    i+=1

    plt.legend()
plt.ylim(0,1)
plt.xlim(0,30)
plt.xlabel('energy [keV]')
plt.ylabel('absorption')


plt.tight_layout()
plt.savefig(path+"Figure_15.svg",format="svg")