import sys
sys.path.insert(0, './srw')
import time
import numpy as np
import multiprocessing as mp
# from srwlpy import *
from srwlib import *
import pickle as pk
import os
import h5py
from scipy.special import kv
from scipy.constants import c,h,e,electron_mass
import matplotlib.pyplot as plt

#%%  
jobId=1

path='./'

file="img_final.p"

#Beam
lam=6e-11

#Setup
z1=28.5
ext=1e-3        
px=int(2**10)
   
#Computing   
cores=4
resize=2 #resizing within the SRW calculation of the wavefront
slices=16   #cuts the calculation in slices to reduces RAM peak

electron=0.5109989500015e6
energy=45.6e9
radius=10760
magnet=23.94
magfield=14.1*1e-3
gamma=energy/electron
softedge=0.1

#Undulator params
def getUndK(gap_um):
    min_valid_K=0.5
    a_0=-178.683137165;a_1=101031.437305031;a_2=-268554.955894147
    a_3=333043.58574148;a_4=-223412.253880588;a_5=78201.083309632
    a_6=-11222.656555176
    r=np.roots(np.flipud([a_0-gap_um,a_1,a_2,a_3,a_4,a_5,a_6]))
    r=r[np.isreal(r)];r=r[r>=min_valid_K]
    return r.real[0]

ALBA_Energy=2.98*1e9
gamma=ALBA_Energy*e/(electron_mass*c**2)
ALBA_und_gap_um=6.05e3
harm=11
plane='v'
ALBA_und_Period=0.0216
ALBA_und_numPer=92
ALBA_und_K=getUndK(ALBA_und_gap_um)
ALBA_und_B= ALBA_und_K/(0.934*ALBA_und_Period*1e2)
ALBA_und_LambdaPeak_nm=(1+ALBA_und_K**2/2)/(2*gamma**2)*ALBA_und_Period*1e9

#%%       
k=2*np.pi/lam
print("Wavelength [m] "+str(lam))   

xpos=0
ypos=0
print("Particle inital position [um] ",xpos*1e6,ypos*1e6)

#%%Functions
def wfrjob(xpos,ypos,ext,xStart,xFin,xpx,ypx,slicenum,resize):
    xpx=int(xpx/resize)
    ypx=int(ypx/resize)  
    #***********Undulator
    
    U = SRWLMagFldU([SRWLMagFldH(harm, plane, ALBA_und_B)], ALBA_und_Period, ALBA_und_numPer) #Undulator Segment
    magFldCnt = SRWLMagFldC([U], [0], [0], [0]) #Container of magnetic field elements and their positions in 3D
    
    #***********Electron Beam
    eBeam = SRWLPartBeam()
    eBeam.Iavg = 0.5 #Average current [A]
    #1st order statistical moments:
    eBeam.partStatMom1.x = xpos #Initial horizontal position of central trajectory [m]
    eBeam.partStatMom1.y = ypos #Initial vertical position of central trajectory [m]
    eBeam.partStatMom1.z = -0.5*ALBA_und_Period*(ALBA_und_numPer *2) #Initial longitudinal position of central trajectory [m]
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
    meth = 1 #SR calculation method: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"
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
wavefront=np.zeros((2,px,px),dtype=np.complex128)
for k in range(slices):
    dat = pk.load(open(path+str(jobId)+"_"+str(k)+"_"+"wf.p","rb"))
    os.remove(path+str(jobId)+"_"+str(k)+"_"+"wf.p")
    
    wavefront[:,:,k*xpx:(k+1)*xpx]=dat    

# hf = h5py.File(path+str(jobId)+"_"+file, 'w')
# hf.create_dataset('dataset_1', data=imgs)
# hf.close()   
print("Collecting and dumping in ",time.time()-a)  

print(time.time()-a)