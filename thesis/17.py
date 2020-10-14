import sys
import numpy as np
from helper import the_speckles,bhmie
import h5py
from scipy import interpolate
import matplotlib.pyplot as plt
path='./'
#%%  
#jobId0=int(sys.argv[1]) #for using the batch cluster
jobId0=0

file="img_final2.p"

#Beam
sigmax=105e-6
sigmay=15e-6
lam=1e-10
fwhmk=1e-3
numSource=int(2**0)     

#Setup
z1=100
z2=3
ext=0.75e-3        
px=int(2**10)
fwhmz2=0
   
#Colloids
colloid_radius=0.5e-6
numScat=5

#Mie
points=1000
rad=np.arctan(ext/z2)*2
refr=1 - 1.28e-6 + 2.49e-09*1j

a=bhmie(lam,colloid_radius,refr,points,rad)
mie=np.abs(a[0])
mie=mie/np.max(mie)
theta=np.linspace(0,rad,points)

mie_interpol = interpolate.interp1d(theta, mie)

#Computing   
cores=8


#%%
a=the_speckles(sigmax,sigmay,lam,fwhmk,100,z1,z2,fwhmz2,ext,px,colloid_radius,5,mie_interpol,cores)

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

plt.figure('Figure_17_1.pdf',figsize=(10,10))

plt.imshow(a)
plt.axis('off')

plt.tight_layout()
plt.savefig(path+"Figure_17_1.pdf",format="pdf")

#%%
a=the_speckles(sigmax,sigmay,lam,fwhmk,numSource,z1,z2,fwhmz2,ext,px,colloid_radius,1000,mie_interpol,cores)

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

plt.figure('Figure_17_2.pdf',figsize=(10,10))

plt.imshow(a)
plt.axis('off')

plt.tight_layout()
plt.savefig(path+"Figure_17_2.pdf",format="pdf")

#%%
a=the_speckles(sigmax,sigmay,lam,fwhmk,200,z1,z2,fwhmz2,ext,px,colloid_radius,1,mie_interpol,cores)

b=np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(a))))**2

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

plt.figure('Figure_17_3.pdf',figsize=(10,10))

plt.imshow(b)
plt.axis('off')

plt.tight_layout()
plt.savefig(path+"Figure_17_3.pdf",format="pdf")
