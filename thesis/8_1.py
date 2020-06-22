import sys
import numpy as np
from helper import the_speckles,bhmie
import h5py
from scipy import interpolate
#%%  
#jobId0=int(sys.argv[1]) #for using the batch cluster
jobId0=0

path='/home/alex/Desktop/temp_res/'
file="img_final2.p"

#Beam
sigmax=150e-6
sigmay=10e-6
lam=1e-10
fwhmk=1e-3
numSource=int(2**4)     

#Setup
z1=100
z2=1.1
ext=1e-3        
px=int(2**10)
fwhmz2=0
   
#Colloids
colloid_radius=1e-6
numScat=1

#Mie
points=1000
rad=np.arctan(ext/z2)*1.5
refr=1 - 1.28e-6 + 2.49e-09*1j

a=bhmie(lam,colloid_radius,refr,points,rad)
mie=np.abs(a[0])
mie=mie/np.max(mie)
theta=np.linspace(0,rad,points)

mie_interpol = interpolate.interp1d(theta, mie)

#Computing   
cores=4


#%%
a=the_speckles(sigmax,sigmay,lam,fwhmk,numSource,z1,z2,fwhmz2,ext,px,colloid_radius,numScat,mie_interpol,cores)

hf = h5py.File(path+"A_"+file, 'w')
hf.create_dataset('dataset_1', data=a)
hf.close()  
