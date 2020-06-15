import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing.pool import ThreadPool
from scipy.special import jv
from mie import bhmie

sys.path.append('//cern.ch/dfs/Users/a/agoetz/Desktop/corona_work_place/helper')
from the_speckles import the_speckles,radial_profile

#Beam
sigma=12e-6
k=2*np.pi/1e-11
fwhmk=0.5
numSource=int(2**0)  

#Setup
z1=70
z2=20
fwhmz2=1e-20
ext=0.5e-3 
px=int(2048)
   
#Colloids
colloid_diameter=8e-6
numScat=int(2**0)

#Computing   
cores=4

#scattering airy (4s), anom (5s), or mie (1s)
scattertype="anom"


a=the_speckles(sigma,k,fwhmk,numSource,z1,z2,fwhmz2,ext,px,colloid_diameter,numScat,scattertype,cores)

# plt.figure(2)
# plt.imshow(a[1])

imft=a[1]
rp=radial_profile(imft,[int(px/2),int(px/2)])
rp=rp/np.max(rp)


freqs=np.fft.fftshift(np.fft.fftfreq(int(np.sqrt((px/2)**2+(px/2)**2)),ext/px))
x=np.linspace(0,freqs[-1],np.size(rp))

lam=2*np.pi/k

sigmaC=lam*(z1+z1)/2/np.pi/sigma
sigmaS=0.4793*lam*z2/colloid_diameter
sigmak=fwhmk/2.35
sigmaT=np.sqrt(2*z2/sigmak)
sigma_total=np.sqrt(1/( 1/sigmaC**2 + 1/sigmaS**2 + 1/sigmaT**2 ))

sigmaq=sigma_total*k/np.sqrt(2)/np.pi/z2
sigmaCq=sigmaC*k/np.sqrt(2)/np.pi/z2
sigmaSq=sigmaS*k/np.sqrt(2)/np.pi/z2
sigmaTq=sigmaT*k/np.sqrt(2)/np.pi/z2


#%%

SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


fig=plt.figure('Figure_14',figsize=(20,10))

plt.plot(x,rp,label="simulation")
plt.plot(x,np.exp(-0.5*(x/sigmaq)**2),label="predicted envelope")
plt.plot(x,np.exp(-0.5*(x/sigmaSq)**2),label="scat")
plt.plot(x,np.exp(-0.5*(x/sigmaTq)**2),label="temp")
plt.plot(x,np.exp(-0.5*(x/sigmaCq)**2),label="spat")

plt.xlabel(r'spatial radial frequency $\left[\dfrac{1}{\mu m}\right]$')
plt.ylabel('power spectrum')
plt.legend()


plt.tight_layout()
plt.savefig("Figure_14",format="svg")

    
    
