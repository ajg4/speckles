import time
from scipy.constants import h,c,e
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from array import array

#jobId=int(sys.argv[1])
jobId=0

path='./'
path='/eos/home-a/agoetz/tempresults/'

#Beam
sigma=5e-6
lam=6e-11
fwhmlam=0.05    
numSource=1    

#Setup
z1=100
z2=10
ext=4e-3        
px=int(2**12)
   
#Colloids
colloid=1e-6
numScat=1000

#Computing   
cores=4
resize=4 #resizing within the SRW calculation of the wavefront

px_col=int(colloid/(ext/px))
print(px_col)
#%%

print('   Preparing the mask ... ', end='')
t0 = time.time()
a=np.ones((px,px),dtype='complex64')
mask=np.ones((px_col,px_col),dtype='complex64')
fwhm_window=px_col
sq=fwhm_window/2/np.sqrt(2*(np.log(2))**(1/16))
size_half=int(px_col/2)
xx,yy=np.meshgrid(np.linspace(-px_col/2,px_col/2,px_col),np.linspace(-px_col/2,px_col/2,px_col))
grid=xx**2+yy**2

beta=1
mask=1-beta*np.exp(-(0.5*grid/sq**2)**30)


for i in range(numScat):

    nx=int(np.random.rand()*0.6*px+0.2*px)
    ny=int(np.random.rand()*0.6*px+0.2*px)
    if(numScat==1):
        nx=int(0.5*px)
        ny=int(0.5*px)       
    a[nx-size_half:nx+size_half,ny-size_half:ny+size_half]*=mask

plt.imshow(np.real(a))

a=a.flatten()
a=np.array([np.real(a),np.real(a)])
a=np.rot90(a,-1)
a=a.flatten()

a=array('f', a)
f=open(path+'mask','wb');a.tofile(f);f.close()

print('done in', round(time.time() - t0), 's')
