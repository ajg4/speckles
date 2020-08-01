# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 17:37:38 2020

@author: agoetz
"""

from scipy.special import kv
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e,c,epsilon_0,h,m_e


z1=100
z2=3
sigma=15e-6
lam=10e-11

mina=0.4793*z2/(z1+z2)*np.pi*sigma*np.sqrt(2)
print("maximum colloid radius ",mina)

minsigkoverk=z2/(z1+z2)**2*2*np.pi/lam*sigma**2*np.sqrt(2/np.log(2))
print("maximum energy bandwith ",minsigkoverk*100)

s=1
s2=5
sigmavcz=lam*(z1+z2)/2/np.pi/sigma

print(" ")

nmax2=int((sigmavcz*s)**2/z2/lam)
print("maximum fringe: ",nmax2)
deltax=np.sqrt(lam*nmax2*z2)-np.sqrt(lam*(nmax2-1)*z2)
pxs=deltax/s2
print("suggested pixel size: ",pxs)
print("suggested magnification: ",2.4e-6/pxs)


print(" ")

qmax=s*sigmavcz/np.sqrt(2)/lam/z2
print("maximum fft frequency ",qmax)

nmax=int(qmax**2*lam*z2)
deltaq=np.sqrt(nmax/lam/z2)-np.sqrt((nmax-1)/lam/z2)
print("suggested fft resolution ",deltaq/s2)

fov=1/deltaq*s2
print("suggested fov [mm] ",fov*1e3)

print(" ")

print("suggested number of px ",fov/pxs)


#%%Puls length
electron=m_e*c**2/e #rest mass of electron
energy=182.5e9
radius=10760
magnet=23.94
deltalambdaoverlambda=0.1
gamma=energy/electron

v=np.sqrt(1-1/gamma**2)
deltac=magnet*(1/v-1)
print(deltac)


#%% Van de Hulst exstinction cross section  and colloids concentration
refr=1.457/1.331
lam=632e-9
radius=500e-9

x=2*np.pi*radius/lam
rho=2*x*(np.real(refr)-1)
Q=2-4/rho*np.sin(rho)+4/rho**2*(1-np.cos(rho))
geometric_cross=radius**2*np.pi

qext_theory=geometric_cross*Q


print("qext_theory :",qext_theory)


thickness=1e-3 #off the holder
solution=1e-1 #volume concentration of colloids

tau=0.1 #how much we want to absorb

n=tau/qext_theory/thickness
v=4*np.pi*(radius)**3/3
x=solution/(n*v)

print(x, " times diluted")

#%% theoretical energy at sensor TODO

ext=1e-3 #extension of the aperture
z1=100 #distance from the source
orad=np.tan(ext/z1/2) #opening angle in rad

electron=m_e*c**2/e #rest mass of electron
energy=45.6e9 #beam energy
radius=10760 #synchrotron radius
gamma=energy/electron #beam gamma

bunches=16640 #bunches per filling
pop=1.7e11 #electrons per bunch
revt=0.32e-3 #revolving time of an electron
frac=0.71 #fraction of how much the magnets account for the circumference of the synchrotron

ph_energy=12.4e3*e #central photon energy
DeltaEE=1e-4 #energy bandwidth
omega0=ph_energy/h*2*np.pi #central photon frequency

absorption_coeff=0.37 #conversion from xray to visible (for 50um YAG:Ce)
ph_yield=1.87e17 #photon yield per joule

n=1000 #points for riemann integration
theta=np.linspace(-orad,orad,n)
omega=np.rot90([np.linspace(omega0,omega0*(1+DeltaEE),n)])

def it(theta):
    out=7/16*e**2/radius
    out/=4*np.pi*epsilon_0
    out/=(gamma**(-2)+theta**2)**(5/2)
    out*=1+5/7*(theta**2/(gamma**(-2)+theta**2))
    return(out)

def itw(theta,omega):
    xi=omega*radius/3/c*(1/gamma**2+theta**2)**(3/2)
    
    out=1/4/np.pi/epsilon_0
    out*=e**2/3/np.pi**2/c
    out*=(omega*radius/c)**2*(gamma**(-2)+theta**2)**2*(kv(2/3,xi)**2+kv(1/3,xi)**2*(theta**2/(gamma**(-2)+theta**2)))*np.cos(theta)
    return(out)

e1=it(theta)
i1=np.sum(e1*(orad*2/n))
i1_2=7/24/epsilon_0*e**2/radius*gamma**4*(1+1/7)
e_flux=bunches/revt*pop
i1_tot=i1*e_flux*orad*2/2/np.pi*frac

e2=itw(theta,omega)
i2=np.sum(e2)*(DeltaEE*omega0)/n*(orad*2)/n
i2_tot=i2*bunches/revt*pop*orad*2/2/np.pi*frac
i2_tot_irr=i2_tot/(ext**2)
flux=i2_tot_irr*absorption_coeff*ph_yield

print("flux per electron on given cone for all frequencies [W]: ",i1)
print("total power per electron analytic [W]: ",i1_2)
print("electrons per second: ",e_flux)
print("flux on aperture for all frequencies [W]: ",i1_tot)
print("flux per electron on given cone for given frequencies [W]: ",i2)    
print("flux on aperture for given frequencies [mW]: ",i2_tot*1e3)
print("iraddiance for given frequencies [W]: ",i2_tot_irr)
print("photon flux on aperture for given frequencies [/m^2/s]: ", flux)

#%% ALBA Undulator K value
def getUndK(gap_um):
    min_valid_K=0.5
    a_0=-178.683137165;a_1=101031.437305031;a_2=-268554.955894147
    a_3=333043.58574148;a_4=-223412.253880588;a_5=78201.083309632
    a_6=-11222.656555176
    r=np.roots(np.flipud([a_0-gap_um,a_1,a_2,a_3,a_4,a_5,a_6]))
    r=r[np.isreal(r)];r=r[r>=min_valid_K]
    return r.real[0]
