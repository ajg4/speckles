import numpy as np
import pickle as pk
import helper
import matplotlib.pyplot as plt
import h5py
from helper import radial_profile, sector_profile,running_mean, defocused_otf
path='/home/alex/Desktop/speckles/thesis/'
#%%  
#jobId0=int(sys.argv[1]) #for using the batch cluster
jobId0=0

file="img_final.p"

#Beam
sigmax=130e-6
sigmay=6.5e-6
lam=6.2e-11
fwhmk=1e-20
numSource=int(2**8)     

#Setup
z1=33
z2=0.1
ext=4e-4        
px=int(2**10)
fwhmz2=0
   
#Colloids
colloid_radius=0.25e-6
numScat=10

k=2*np.pi/2

x=np.linspace(0,0.5*np.sqrt(2)*px/ext,px)

dz,zetas,q,NA,lam_v,n,terms=100e-6,30,x/23,0.4,500e-9,1.5,10

otf=defocused_otf(dz,zetas,q,NA,lam_v,n,terms)
   

#%%

sigmaC=lam*(z1+z2)/2/np.pi/sigmax
sigmaS=0.4793*lam*z2/(colloid_radius*2)*np.sqrt(2)

sigmak=k*fwhmk/2.35
sigmaT=np.sqrt(z2/sigmak*np.sqrt(2/np.log(2)))

sigma_total=np.sqrt(1/( 1/sigmaC**2 + 1/sigmaS**2 + 1/sigmaT**2 ))
sigma_total2=np.sqrt(1/( 1/sigmaC**2 + 1/sigmaS**2))

sigmaq=sigma_total/lam/z2/np.sqrt(2)
sigmaq2=sigma_total2/lam/z2/np.sqrt(2)
sigmaCq=sigmaC/lam/z2/np.sqrt(2)
sigmaSq=sigmaS/lam/z2/np.sqrt(2)
sigmaTq=sigmaT/lam/z2/np.sqrt(2)


fig=plt.figure('Figure_8_1.svg')

plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaTq)**2),label="temporal")
plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaSq)**2),label="scattering")
plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaCq)**2),label="spatial")
plt.plot(x*1e-6,otf,label="otf")
a=np.pi*lam*z2
plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaq)**2)*otf*np.sin(a*x**2)**2,label="total")
plt.legend()
plt.xlim(0,1)


#%%

sigmaC=lam*(z1+z2)/2/np.pi/sigmay
sigmaS=0.4793*lam*z2/(colloid_radius*2)*np.sqrt(2)

sigmak=k*fwhmk/2.35
sigmaT=np.sqrt(z2/sigmak*np.sqrt(2/np.log(2)))

sigma_total=np.sqrt(1/( 1/sigmaC**2 + 1/sigmaS**2 + 1/sigmaT**2 ))
sigma_total2=np.sqrt(1/( 1/sigmaC**2 + 1/sigmaS**2))

sigmaq=sigma_total/lam/z2/np.sqrt(2)
sigmaq2=sigma_total2/lam/z2/np.sqrt(2)
sigmaCq=sigmaC/lam/z2/np.sqrt(2)
sigmaSq=sigmaS/lam/z2/np.sqrt(2)
sigmaTq=sigmaT/lam/z2/np.sqrt(2)


fig=plt.figure('Figure_8_2.svg')

plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaTq)**2),label="temporal")
plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaSq)**2),label="scattering")
plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaCq)**2),label="spatial")
plt.plot(x*1e-6,otf,label="otf")
a=np.pi*lam*z2
plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaq)**2)*otf*np.sin(a*x**2)**2,label="total")
plt.legend()
plt.xlim(0,1)