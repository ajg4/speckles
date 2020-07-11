# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 17:37:38 2020

@author: agoetz
"""

import numpy as np
from scipy.special import kv
from scipy.constants import c
import matplotlib.pyplot as plt

z1=100
z2=1
sigma=15e-6
lam=10e-11

mina=0.4793*z2/(z1+z2)*np.pi*sigma*np.sqrt(2)
print("maximum colloid radius ",mina)

minsigkoverk=z2/(z1+z2)**2*2*np.pi/lam*sigma**2*np.sqrt(2/np.log(2))
print("maximum energy bandwith ",minsigkoverk)

s=1
sigmavcz=lam*(z1+z2)/2/np.pi/sigma
qmax=s*sigmavcz/np.sqrt(2)/lam/z2
print("maximum frequency ",qmax)

nmax=int(qmax**2*lam*z2)
print("maximum fringe number ",nmax)

deltaq=np.sqrt(nmax/lam/z2)-np.sqrt((nmax-1)/lam/z2)
print("maximal resolvable frequency ",deltaq)
s2=10
print("maximal resolvable frequency with ",s2," pixel for the last fringe ",deltaq/s2)

fov=1/deltaq*s2
print("suggested fov [mm] ",fov*1e3)
print("suggested px ",qmax*fov*np.sqrt(2))
print("suggested px size ",fov*1e3/(qmax*fov*np.sqrt(2)))

print("maximum distance ",sigmavcz*s*1e3)

nmax2=int((sigmavcz*s)**2/z2/lam)
deltax=np.sqrt(lam*nmax2*z2)-np.sqrt(lam*(nmax2-1)*z2)
print("suggested pixel size [mm]: ",deltax*1e3/s2)

#%%
ext=2e-3
px=2048

electron=0.5109989500015e6
energy=182.5e9
radius=10760
magnet=23.94
deltalambdaoverlambda=0.1
gamma=energy/electron
sigmaCx=lam*(z1+z2)/sigma/2/np.pi
sigmaCq=sigmaCx/np.sqrt(2)/lam/z2

ex=1.46e-9
ey=2.9e-12

#%%Puls length
v=np.sqrt(1-1/gamma**2)
deltac=magnet*(1/v-1)
print(deltac)

#%% temporal coherence
lc=lam/deltalambdaoverlambda
print(lc)

deltalambda=deltalambdaoverlambda*lam
deltanu=c/lam - c/(lam+deltalambda)
tc=1/(deltanu)
lc=c*tc
print(lc)

#%%fit factor for bandwidth
print(deltanu)
nu=c/lam
print(nu)
print((sigmaCx*3)**2/2/z2)

#%%beam size
beta=49
print(np.sqrt(beta*ey)*1e6)

#%% fit of sin(xx)/xx
deltalambdaoverlambda=0.2

deltalambda=deltalambdaoverlambda*lam
deltak=2*np.pi*(1/lam-1/(lam+deltalambda))

x=np.linspace(0,3*sigmaCx,1000)
fac=deltak/4/z2*x*x
i=np.sin(fac)/fac
i[0]=i[1]

def gauss(x,s):
    return(np.exp(-0.5*x*x/s/s))

from scipy.optimize import curve_fit

popt, pcov = curve_fit(gauss, x, i)
print(popt[0]*(deltak/z2)**(1/2))
print()


fac2=2*(deltak/z2)**(-1/2)

plt.plot(x/sigmaCx,gauss(x,fac2))
plt.plot(x/sigmaCx,gauss(x,popt[0]))
plt.plot(x/sigmaCx,i)  

#%% integration of cosine over an gaussian distribution
points=1000

k=2*np.pi/lam
ks=np.linspace(k-4*deltak,k+4*deltak,points)
ks_weight=np.exp(-0.5*(ks-k)**2/deltak**2)
x=np.linspace(0,3*sigmaCx,points)

def cos(k):
    return(np.cos(k*x**2/2/z2))
    
i=np.zeros(points)
for i in range(points):
    i+=cos(ks[i])*ks_weight[i]    
i=i/1000
icos=i/cos(k)

plt.plot(x,i,label='sum')
#plt.plot(x,icos,label='decay')
#plt.plot(x,cos(k),label='cosine')
plt.yscale('log')
plt.legend()

#%%  colloids density

lam=632e-9
radius=500e-9*2
# refrel=1+1e-1+1e-1j
refrel=1.457
d=1e-3

x=2*np.pi*radius/lam
rho=2*x*(np.real(refrel)-1)
Q=2-4/rho*np.sin(rho)+4/rho**2*(1-np.cos(rho))
# Q=3.9634450616970356
print(Q)
geo_cross=np.pi*radius**2
Cext=geo_cross/Q
print(Cext*1e13)

n=-np.log(0.95)/(d*Cext)
print(n*1e-14)

rat=4*np.pi*radius**3/3*n

print(rat**(-1))

#%% theoretical energy at sensor TODO

