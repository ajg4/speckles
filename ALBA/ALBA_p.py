import numpy as np
import pickle as pk
import helper
import matplotlib.pyplot as plt
import h5py
sys.path.append('../thesis/')
from helper import radial_profile, sector_profile,running_mean, defocused_otf
#jobId0=int(sys.argv[1]) #for using the batch cluster
jobId0=0

# file="an.h5"
file0="srw_test.h5"
path="./"
path='/home/alex/Desktop/speckles/ALBA/'

#Beam
sigmax=130e-6
sigmay=6.5-6
lam=1e-10
fwhmk=0.002
numSource=int(2**8)     

#Setup
z1=33
z2=0.2
ext=160e-6        
px=int(2**12)
fwhmz2=0
   
#Colloids
colloid_radius=0.25e-6
numScat=10


#%%
file=path+file0

hf = h5py.File(file, 'r')
img = np.array(hf.get('dataset_1'))
hf.close()  

# sh=np.shape(img)
# sh2=int(sh[0]/4)
# img=img[sh2:-sh2,sh2:-sh2]
# ext=ext/2
# px=int(px/2)

k=2*np.pi/lam
imft=np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img))))**2

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


fig=plt.figure('speckles.svg',figsize=(10,10))

extent=np.array([-ext/2,ext/2,-ext/2,ext/2])*1e6*1.2
plt.imshow(img,extent=extent)
plt.xlabel(r"x $[\mu m]$")
plt.ylabel(r"y $[\mu m]$")
plt.xlim(-80,80)
plt.ylim(-80,80)


plt.tight_layout()
plt.savefig(path+"speckles.svg",format="svg",bbox_inches='tight') 

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


fig=plt.figure('fft.svg',figsize=(10,10))

x=np.linspace(0,0.5*px/ext,px)

# imft2=np.where(imft>np.mean(imft)*300,np.mean(imft)*300,imft)
# imft2=np.log(imft)
imft2=imft

extent=np.array([-x[-1],x[-1],-x[-1],x[-1]])*1e-6
plt.imshow(imft2,extent=extent)
plt.xlabel(r"horizontal spatial frequency $[\mu m^{-1}]$")
plt.ylabel(r"vertical spatial frequency $[\mu m^{-1}]$")
plt.xlim(-2,2)
plt.ylim(-2,2)


plt.tight_layout()
plt.savefig(path+"fft.svg",format="svg")   


#%%

rp,sec_data=sector_profile(imft*1,[int(px/2),int(px/2)],[0,8])

rp=running_mean(rp,1)

x=np.linspace(0,0.5*np.sqrt(2)*px/ext,np.size(rp))*2*np.pi


rp[0]=rp[1]
rp=rp/np.max(rp)


last_cal=np.where(x*1e-6>3.9)[0][0]
sc_q=x[:last_cal]*1e-6


matplotlib_size=30

plt.rc('font', size=matplotlib_size)          # controls default text sizes
plt.rc('axes', titlesize=matplotlib_size)     # fontsize of the axes title
plt.rc('axes', labelsize=matplotlib_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=matplotlib_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=matplotlib_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=matplotlib_size)    # legend fontsize
plt.rc('figure', titlesize=matplotlib_size)  # fontsize of the figure title


cut=93

fig=plt.figure('cut.svg',figsize=(10,10))


plt.plot(x,rp)


plt.xlabel(r'q $[\mu m^{-1}]$')
plt.ylabel('I(q) [arb. units]')
# plt.legend(loc="upper right")

# plt.xlim([0,3.5])
# plt.ylim(2e-3,3e-1)

plt.tight_layout()
plt.yscale("log")
plt.savefig(path+"cut.svg",format="svg")    

