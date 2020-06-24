import sys
sys.path.insert(0, './srw')
import time
import numpy as np
import multiprocessing as mp
from srwlpy import *
from srwlib import *
import pickle as pk
import os
import h5py
from scipy.special import kv
from scipy.constants import c,h,e
import matplotlib.pyplot as plt
path='/home/alex/Desktop/'

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
#%%  

jobId=1


file="img_final.p"

#Beam
sigma=0
lam=6e-11
fwhmlam=0  
numSource=1   

#Setup
z1=100
ext=1e-3        
px=int(2**13)
   
#Computing   
cores=16
resize=2 #resizing within the SRW calculation of the wavefront
slices=16   #cuts the calculation in slices to reduces RAM peak
downsamplefactor=1 #downsample final image before combining of slices

electron=0.5109989500015e6
energy=45.6e9
radius=10760
magnet=23.94
magfield=14.1*1e-3
gamma=energy/electron
softedge=0.1

  
#%%
if((px/slices)%resize!=0):
    print("resize odd!")
if((px/slices)%downsamplefactor!=0):
    print("downsample odd!")
if(slices<cores):
    print("put more slices than cores..")

#%%       
k=2*np.pi/lam

np.random.seed(1*int(1e10*(time.time()%0.0001)))
lam=lam+np.random.standard_normal()*lam*(fwhmlam)/2.35
print("Wavelength [m] "+str(lam))   


xpos=np.random.standard_normal()*sigma
ypos=np.random.standard_normal()*sigma
print("Particle position [um] ",xpos*1e6,ypos*1e6)

np.random.seed(1)
#%%Functions
def wfrjob(xpos,ypos,ext,xStart,xFin,xpx,ypx,slicenum,resize):
    xpx=int(xpx/resize)
    ypx=int(ypx/resize)  
    #***********Bending Magnet
    B = magfield #Dipole magnetic field [T]
    LeffBM = magnet #Magnet length [m]
    Ledge=softedge #soft sedge
    R=radius #bending radius
    BM = SRWLMagFldM(B, 1, 'n', LeffBM,Ledge,R)
    magFldCnt = SRWLMagFldC([BM], [0], [0], [0]) #Container of magnetic field elements and their positions in 3D
    
    #***********Electron Beam
    eBeam = SRWLPartBeam()
    eBeam.Iavg = 0.5 #Average current [A]
    #1st order statistical moments:
    eBeam.partStatMom1.x = xpos #Initial horizontal position of central trajectory [m]
    eBeam.partStatMom1.y = ypos #Initial vertical position of central trajectory [m]
    eBeam.partStatMom1.z = 0. #Initial longitudinal position of central trajectory [m]
    eBeam.partStatMom1.xp = 0. #Initial horizontal angle of central trajectory [rad]
    eBeam.partStatMom1.yp = 0. #Initial vertical angle of central trajectory [rad]
    eBeam.partStatMom1.gamma = gamma #Relative energy
    #2nd order statistical moments:
    eBeam.arStatMom2[0] = 0#<(x-x0)^2> [m^2]
    eBeam.arStatMom2[1] = 0 #<(x-x0)*(x'-x'0)> [m]
    eBeam.arStatMom2[2] = 0 #<(x'-x'0)^2>
    eBeam.arStatMom2[3] = 0#<(y-y0)^2>
    eBeam.arStatMom2[4] = 0 #<(y-y0)*(y'-y'0)> [m]
    eBeam.arStatMom2[5] = 0 #<(y'-y'0)^2>
    eBeam.arStatMom2[10] = 0 #<(E-E0)^2>/E0^2
    
    #Precision parameters
    meth = 2 #SR calculation method: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"
    relPrec = 0.01 #Relative precision
    zStartInteg = 0 #Longitudinal position to start integration (effective if < zEndInteg)
    zEndInteg = 0 #Longitudinal position to finish integration (effective if > zStartInteg)
    npTraj = 20000 #Number of points for trajectory calculation 
    useTermin = 1 #Use "terminating terms" (i.e. asymptotic expansions at zStartInteg and zEndInteg) or not (1 or 0 respectively)
    
    
    wfr=SRWLWfr()
    wfr.allocate(1, xpx, ypx) #Numbers of points vs photon energy, horizontal and vertical positions (the last two will be modified in the process of calculation)   
    wfr.mesh.zStart = (z1) #Longitudinal position for initial wavefront [m]   
    wfr.mesh.eStart = h*c/e/lam #Initial photon energy [eV]
    wfr.mesh.eFin = wfr.mesh.eStart #Final photon energy [eV]   
    wfr.mesh.xStart = xStart #Initial horizontal position [m]
    wfr.mesh.xFin = xFin #Final horizontal position [m]
    wfr.mesh.yStart = -ext/2 #Initial vertical position [m]
    wfr.mesh.yFin = ext/2 #Final vertical position [m] 
    wfr.partBeam = eBeam #e-beam data is contained inside the wavefront struct
    
    par=0
    sampFactNxNyForProp=0
    
    arPrecSR = [meth, relPrec, zStartInteg, zEndInteg, npTraj, useTermin, sampFactNxNyForProp] 
    
    srwl.CalcElecFieldSR(wfr, par, magFldCnt, arPrecSR)   

    if(resize!=1):
        srwl.ResizeElecField(wfr,'c',[0,1,resize,1,resize])
    
    xpx=int(xpx*resize)
    ypx=int(ypx*resize)  
       
    arEx=np.array(wfr.arEx)
    e1=np.reshape((arEx[::2]+arEx[1::2]*1j),(ypx,xpx))
    
    arEy=np.array(wfr.arEy)
    e2=np.reshape((arEy[::2]+arEy[1::2]*1j),(ypx,xpx)) 
    
    pk.dump([e1,e2],open(path+str(jobId)+"_"+str(slicenum)+"_"+"wf.p","wb"))
    

#%% Wavefront calculation
a=time.time()
xxg=np.linspace(-ext/2,ext/2,px)
xpx=int(px/slices)

xlist=[]
for i in range(slices):
    xlist.append([xxg[i*xpx],xxg[(i+1)*xpx-1]])
    
for j in range(int(slices/cores)):
    pool = mp.Pool(cores)
    poolInputs=[]
    for i in range(cores):
        poolInputs.append((xpos,ypos,ext,xlist[j*cores+i][0],xlist[j*cores+i][1],xpx,px,(j*cores+i),resize))
    wfrSplits=pool.starmap(wfrjob,poolInputs)
    pool.close()
    pool.join()
        
print("WFs calculated in ",time.time()-a)

#%% Combining images and dummping
a=time.time()
imgs=np.zeros((2,px,px),dtype=np.complex128)
for k in range(slices):
    dat = pk.load(open(path+str(jobId)+"_"+str(k)+"_"+"wf.p","rb"))
    os.remove(path+str(jobId)+"_"+str(k)+"_"+"wf.p")
    
    imgs[:,:,k*xpx:(k+1)*xpx]=dat    

hf = h5py.File(path+str(jobId)+"_"+file, 'w')
hf.create_dataset('dataset_1', data=imgs)
hf.close()   
print("Collecting and dumping in ",time.time()-a)  

print(time.time()-a)

#%% plotting against LW
theta=np.arctan(xxg/(z1))

mid=int(px/2)
center=xxg[mid]

k=2*np.pi/lam

ih1=1/z1*np.exp(-1j*k*np.sqrt(xxg**2+z1**2)+1j*k*np.sqrt(center**2+z1**2))
ih1_angle=np.angle(ih1)
dubi=ih1_angle[mid]
ih1_angle=np.angle(ih1*np.exp(-1j*dubi))

srwx=np.angle(imgs[0,:,int(px/2)])
dubi=srwx[mid]
srwx=np.angle(imgs[0,:,int(px/2)]*np.exp(-1j*dubi))*(-1)


srwy=np.angle(imgs[1,:,int(px/2)])
dubi=srwy[mid]
srwy=np.angle(imgs[0,:,int(px/2)]*np.exp(-1j*dubi))*(-1)


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


fig=plt.figure('Figure_5',figsize=(20,10))

ax1 = fig.add_subplot(111)

#plt.plot(xxg,ih1_angle,label='LW horizontal',color="g")
#plt.plot(xxg,iv1,label='LW vertical')
ax1.plot(xxg*1e3,srwx,label='SRW horizontal',color="#1f77b4")
#ax1.plot(xxg,srwy,label='SRW vertical',color="#ff7f0e")
ax1.legend(loc="upper left")
ax1.set_ylabel(r"phase [rad]")
ax1.set_xlabel("x,y [mm]")
ax1.set_ylim(-4,4.5)

ax2=ax1.twinx()

ax1.set_zorder(1)  # default zorder is 0 for ax1 and ax2
ax1.patch.set_visible(False)  # prevents ax1 from hiding ax2

diff=np.where((srwx-ih1_angle)<-1,0,srwx-ih1_angle)
ax2.plot(xxg*1e3,1e3*diff,label="deviation from LW horizontal",color="#ff7f0e",alpha=0.7,linestyle="--")
###ax2.plot(xxg,100*(srwy-ih1)/srwy,label="SRW deviation from LW vertical",color="#ff7f0e",linestyle="--")
ax2.legend(loc="upper right")
ax2.set_ylabel("phase deviation [mrad]")



plt.tight_layout()
plt.savefig(path+"Figure_5.svg",format="svg") 
