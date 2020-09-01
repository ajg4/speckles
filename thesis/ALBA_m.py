import time
import numpy as np
import pickle as pk
from array import array

h=6.62607015e-34
c=299792458.0
e=1.602176634e-19
electron_mass=9.1093837015e-31

#jobId=int(sys.argv[1])
jobId=0

path='./'
# path='/eos/home-a/agoetz/tempresults/'

#Setup
z1=33
z2=0.2
ext=160e-6        
px=int(2**12)
   
#Colloids
colloid=0.5e-6
numScat=10000


px_col=int(colloid/(ext/px))
print(px_col)
#%%

print('   Preparing the mask ... ')
t0 = time.time()
a=np.ones((px,px),dtype='complex64')

beta=1
size_half=int(px_col/2)
size_real_half=ext/px*px_col/2

xx,yy=np.meshgrid(np.linspace(-size_real_half,size_real_half,px_col),np.linspace(-size_real_half,size_real_half,px_col))
grid=np.sqrt(xx**2+yy**2)
grid=np.where(grid>colloid/2,colloid/2,grid)
mask=1-beta*np.sqrt(1-(grid/colloid*2)**2)
# mask=np.where(grid<colloid/2,0,1)

for i in range(numScat):
    nx=int(np.random.rand()*0.9*px+0.05*px)
    ny=int(np.random.rand()*0.9*px+0.05*px)
    if(numScat==1):
        nx=int(0.5*px)
        ny=int(0.5*px)       
    a[nx-size_half:nx+size_half,ny-size_half:ny+size_half]*=mask

a=a.flatten()
a=np.array([np.real(a),np.real(a)])
a=np.rot90(a,-1)
a=a.flatten()

b=np.abs(a[::2]+a[1::2]*1j)**2
c=np.reshape(b,(px,px))

a=array('f', a)
f=open(path+'mask','wb');a.tofile(f);f.close()

print('done in', round(time.time() - t0), 's')
