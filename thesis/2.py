#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:36:20 2020

@author: agoetz
"""

import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.special import jv
import numpy as np
from scipy.optimize import curve_fit
import sys
from helper import bhmie
path='/home/alex/Desktop/'

points=1000
rad=np.pi/2/10000*1.3
radius=500e-9
lam=1e-10
refr=1 - 1.28e-6 + 2.49e-09*1j

#lam=632e-9
#refr=1.587/1.331
#rad=np.pi/2


a=bhmie(lam,radius,refr,points,rad)
mie=np.abs(a[0])
mie=mie/np.max(mie)

theta=np.linspace(0,rad,points)

colloid=radius*2
q=colloid*np.pi*np.sin(theta)/lam
q=np.abs(q)

airy_pattern=np.abs((2*jv(1,q)/q) )              
anom_pattern=np.abs(jv(3/2,q)*np.sqrt(1/q**3))
anom_pattern[0]=anom_pattern[1]
anom_pattern/=np.max(anom_pattern)

fac2=theta/(0.4793*lam/colloid*np.sqrt(2))
anom_pattern_gauss=np.exp(-0.5*(fac2**2))


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


plt.figure('Figure_2.svg',figsize=(20,10))

plt.plot(1e3*theta[::16],mie[::16],'r.',markersize=10,label='Mie theory')
plt.plot(1e3*theta,anom_pattern,label="Anomalous diffraction")
plt.plot(1e3*theta,anom_pattern_gauss,label='Gaussian fit')
plt.plot(1e3*theta,airy_pattern,label='Frauenhofer diffraction')

plt.xlabel(r'$\theta$ [mrad]')
plt.ylabel('amplitude [a.u.]')
plt.legend()


plt.tight_layout()
plt.savefig("../../Figure_2.svg",format="svg")

#%% Comparison of Mie and Van de Hulst exstinction cross section

x=2*np.pi*radius/lam
rho=2*x*(np.real(refr)-1)
Q=2-4/rho*np.sin(rho)+4/rho**2*(1-np.cos(rho))
geometric_cross=radius**2*np.pi

qext_theory=geometric_cross*Q
qext_mie=geometric_cross*a[2]

print("qext_theory :",qext_theory)
print("qext_mie :",qext_mie)
