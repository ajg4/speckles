import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing.pool import ThreadPool
from scipy.special import jv
sys.path.append('/eos/home-a/agoetz/scripts/helper/')
from mie import bhmie
from the_speckles import the_speckles,radial_profile, sector_profile,running_mean
import pickle as pk

#Beam
#sigma=[10e-6,10e-6]
#sigma=[150e-6,150e-6]
sigma=[150e-6,10e-6]
#sigma=[0,0]
#k=2*np.pi/1e-10
k=2*np.pi/632e-9
fwhmk=1e-20
numSource=int(2**0)  

#Setup
z1=100
z2=50e-3
fwhmz2=1e-20
ext=0.5e-3
px=int(2**10)
   
#Colloids
colloid_diameter=1e-6
numScat=int(2**10)

#Computing   
cores=16

#scattering airy (4s), anom (5s), or mie (1s)
scattertype="gauss"

#%%

a=the_speckles(sigma,k,fwhmk,numSource,z1,z2,fwhmz2,ext,px,colloid_diameter,numScat,scattertype,cores)
#pk.dump(a,open("/eos/home-a/agoetz/scripts/main/spherical.pk","wb"))

plt.imshow(a[0])

#imft=a[1]*1
#rp=radial_profile(imft,[int(px/2),int(px/2)])
#plt.plot(rp)

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


fig=plt.figure('Figure_8_3.svg',figsize=(20,10))


x=np.linspace(0,0.5*px/ext,px)

extent=np.array([-x[-1],x[-1],-x[-1],x[-1]])*1e-6
plt.imshow(a[1],extent=extent)
plt.xlabel(r"horizontal spatial frequency $[\mu m^{-1}]$")
plt.ylabel(r"vertical spatial frequency $[\mu m^{-1}]$")


plt.tight_layout()
plt.savefig("/eos/home-a/agoetz/scripts/thesis/Figure_8_3.svg",format="svg")   

#%%
imft=a[1]*1
rp,sec_data1=sector_profile(imft,[int(px/2),int(px/2)],[0,10])
imft=a[1]*1
rp,sec_data2=sector_profile(imft,[int(px/2),int(px/2)],[90,10])

sec_data=sec_data1+sec_data2


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


fig=plt.figure('Figure_8_4.svg',figsize=(20,10))

extent=np.array([-ext/2,ext/2,-ext/2,ext/2])*1e6
plt.imshow(sec_data,extent=extent)
plt.xlabel("x [um]")
plt.ylabel("y [um]")


plt.tight_layout()
plt.savefig("Figure_8_4.svg",format="svg")   

#%%
sigmax=sigma[0]

imft=a[1]*1
rp,sec_data=sector_profile(imft,[int(px/2),int(px/2)],[0,10])

rp=running_mean(rp,1)

x=np.linspace(0,0.5*np.sqrt(2)*px/ext,np.size(rp))

lam=2*np.pi/k
sigmaC=lam*(z1+z2)/2/np.pi/sigmax
sigmaS=0.4793*lam*z2/colloid_diameter*np.sqrt(2)

sigmak=k*fwhmk/2.35
sigmaT=np.sqrt(2*z2/sigmak)

sigma_total=np.sqrt(1/( 1/sigmaC**2 + 1/sigmaS**2 + 1/sigmaT**2 ))
sigma_total2=np.sqrt(1/( 1/sigmaC**2 + 1/sigmaS**2))

sigmaq=sigma_total/lam/z2/np.sqrt(2)
sigmaq2=sigma_total2/lam/z2/np.sqrt(2)
sigmaCq=sigmaC/lam/z2/np.sqrt(2)
sigmaSq=sigmaS/lam/z2/np.sqrt(2)
sigmaTq=sigmaT/lam/z2/np.sqrt(2)

leading=np.exp(-0.5*(x/sigmaq)**2)
top=leading[np.argmax(rp)]
rp=rp/np.max(rp)*top


from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
def gauss(x,sigma):
    out=np.exp(-0.5*(x/sigma)**2)
    return(out)

from scipy.signal import find_peaks

peaks=find_peaks(rp,distance=70)[0][1:]
#plt.plot(x,rp)
#plt.plot(x[peaks],rp[peaks])
#maxima=argrelextrema(rp,np.greater)[0]
#maxima2=argrelextrema(rp[maxima],np.greater)[0][:]
pars=curve_fit(gauss,x[peaks],rp[peaks],p0=[sigmaq])


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


fig=plt.figure('Figure_8_1.svg',figsize=(20,10))

plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaTq)**2),label="temporal")
plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaSq)**2),label="scattering")
plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaCq)**2),label="spatial")
plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaq)**2),label="total")
#plt.plot(x*1e-6,np.exp(-0.5*(x/pars[0])**2),label="fit")
#plt.plot(x*1e-6,np.exp(-0.5*(x/pars[0])**2),label="fit")
#plt.plot(x[peaks]*1e-6,rp[peaks],label="fit_helper")
plt.plot(x*1e-6,rp,label="simulation")

plt.xlabel(r'spatial radial frequency $[\mu m^{-1}]$')
plt.ylabel('power spectrum [a.u.]')
plt.legend()


plt.tight_layout()
plt.savefig("/eos/home-a/agoetz/scripts/thesis/Figure_8_1.svg",format="svg")    


print(pars[0]/sigmaq)

#%%
sigmay=sigma[1]

imft=a[1]*1
rp,sec_data=sector_profile(imft,[int(px/2),int(px/2)],[90,5])

#rp=running_mean(rp,1)


x=np.linspace(0,0.5*np.sqrt(2)*px/ext,np.size(rp))

lam=2*np.pi/k
sigmaC=lam*(z1+z2)/2/np.pi/sigmay
sigmaS=0.4793*lam*z2/colloid_diameter*np.sqrt(2)

sigmak=k*fwhmk/2.35
sigmaT=np.sqrt(2*z2/sigmak)

sigma_total=np.sqrt(1/( 1/sigmaC**2 + 1/sigmaS**2 + 1/sigmaT**2 ))
sigma_total2=np.sqrt(1/( 1/sigmaC**2 + 1/sigmaS**2))

sigmaq=sigma_total/lam/z2/np.sqrt(2)
sigmaq2=sigma_total2/lam/z2/np.sqrt(2)
sigmaCq=sigmaC/lam/z2/np.sqrt(2)
sigmaSq=sigmaS/lam/z2/np.sqrt(2)
sigmaTq=sigmaT/lam/z2/np.sqrt(2)

leading=np.exp(-0.5*(x/sigmaq)**2)
top=leading[np.argmax(rp)]
rp=rp/np.max(rp)*top


from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
def gauss(x,sigma):
    out=np.exp(-0.5*(x/sigma)**2)
    return(out)

from scipy.signal import find_peaks

peaks=find_peaks(rp,distance=100)[0][1:]
#plt.plot(x,rp)
#plt.plot(x[peaks],rp[peaks])
#maxima=argrelextrema(rp,np.greater)[0]
#maxima2=argrelextrema(rp[maxima],np.greater)[0][:]
pars=curve_fit(gauss,x[peaks],rp[peaks],p0=[sigmaq])


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



fig=plt.figure('Figure_8_2.svg',figsize=(20,10))

plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaTq)**2),label="temporal")
plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaSq)**2),label="scattering")
plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaCq)**2),label="spatial")
plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaq)**2),label="total")
#plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaq2)**2),label="total2")
#plt.plot(x*1e-6,np.exp(-0.5*(x/pars[0])**2),label="fit")
#plt.plot(x[peaks]*1e-6,rp[peaks],label="fit_helper")
plt.plot(x*1e-6,rp,label="simulation")

plt.xlabel(r'spatial radial frequency $[\mu m^{-1}]$')
plt.ylabel('power spectrum [a.u.]')
plt.legend()


plt.tight_layout()
plt.savefig("/eos/home-a/agoetz/scripts/thesis/Figure_8_2.svg",format="svg")  

print(pars[0]/sigmaq)
