import sys
sys.path.append('/eos/home-a/agoetz/scripts/SRW/')
import time
import numpy as np
import multiprocessing as mp
from srwlpy import *
from srwlib import *
import pickle as pk
import os
import h5py
import matplotlib.pyplot as plt
#%%  
#jobId0=int(sys.argv[1]) #for using the batch cluster
numSource=500
jobId0=1
   
#path='/eos/home-a/agoetz/tempresults/'
path='/home/agoetz/Desktop/'
file="img_final.p"

for jobtask in range(numSource):
    ttotal=time.time()
    jobId=str(jobId0)+"_"+str(jobtask) 

    #Beam
    sigma=10e-6
    lam=1e-10
    skok=0
    
    #Setup
    z1=100
    ext=1.5e-3        
    px=int(2**11)     
    
    #Computing   
    cores=8
    resize=4 #resizing within the SRW calculation of the wavefront
    slices=16   #cuts the calculation in slices to reduces RAM peak
    downsamplefactor=1 #downsample final image before combining of slices
      
    
    #%%       
    k=2*np.pi/lam
    sigmaC=lam*(z1)/sigma/2/np.pi

    np.random.seed(jobId0*int(1e10*(time.time()%0.0001)))
    k=2*np.pi/lam
    ks=k+np.random.standard_normal()*k*skok
    print("Wavelength [m] "+str(lam))   

    
    xpos=np.random.standard_normal()*0
    ypos=np.random.standard_normal()*sigma
    print("Particle position [um] ",xpos*1e6,ypos*1e6)
    
    #%%Functions
    def wfrjob(xpos,ypos,ext,xStart,xFin,xpx,ypx,slicenum,resize):
        h=6.62607004e-34
        c=299792458.0
        e=1.6021766208e-19
        xpx=int(xpx/resize)
        ypx=int(ypx/resize)  
        #***********Bending Magnet
        B = 0.056586646847086634 #Dipole magnetic field [T]
        LeffBM = 23.94 #Magnet length [m]
        Ledge=0.1 #soft sedge
        R=10760 #bending radius
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
        eBeam.partStatMom1.gamma = 357212.7617929145 #Relative energy
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
    
        e=np.reshape((arEx[::2]+arEx[1::2]*1j),(ypx,xpx)) 
        pk.dump(e,open(path+str(jobId)+"_"+str(slicenum)+"_"+"wf.p","wb"))
          

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
    imgs=np.zeros((px,px),dtype=np.complex128)
    for k in range(slices):
        dat = pk.load(open(path+str(jobId)+"_"+str(k)+"_"+"wf.p","rb"))
        os.remove(path+str(jobId)+"_"+str(k)+"_"+"wf.p")
        
        imgs[:,k*xpx:(k+1)*xpx]=dat    

    hf = h5py.File(path+str(jobId)+"_"+file, 'w')
    hf.create_dataset('dataset_1', data=imgs)
    hf.close()   
    print("Collecting and dumping in ",time.time()-a)  
    
    print(time.time()-ttotal)

#%%
half=int(px/2)
coh=np.zeros((px,px),dtype=np.complex128)
imap=np.zeros((px,px))         

for i in range(numSource):
    jobId=str(jobId0)+"_"+str(i)
    hf = h5py.File(path+str(jobId)+"_"+file, 'r')
    data=hf.get('dataset_1')
    E=np.array(data)
    hf.close()
  
    coh+=E*np.conj(E[half,half])
    imap+=np.abs(E)**2


coh=np.abs(coh)
imap=np.abs(imap)

gams=(coh/imap)[:,half]

xMin,xMax,nx=-ext/2,ext/2,px
xx=np.linspace(xMin,xMax,nx)


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


fig=plt.figure('Figure_6',figsize=(20,10))


ax1 = fig.add_subplot(111)


ax1.plot(xx*1e3,gams,label="SRW",color="#1f77b4")
#ax1.plot(xx,np.exp(-0.5*(xx/sigmaC)**2),label="Van Citter Zerneke Theorem")
ax1.legend(loc="upper left")
ax1.set_zorder(1)  # default zorder is 0 for ax1 and ax2
ax1.patch.set_visible(False)  # prevents ax1 from hiding ax2
ax1.set_xlabel("y [mm]")
ax1.set_ylabel("spatial coherence")

ax2=ax1.twinx()
diff=(gams-np.exp(-0.5*(xx/sigmaC)**2))
ax2.plot(xx*1e3,diff,label="deviation from VCT Theorem",color="#ff7f0e",alpha=0.7,linestyle="--")
ax2.legend(loc="upper right")
ax2.set_ylabel("deviation")


plt.tight_layout()
plt.savefig("/eos/home-a/agoetz/scripts/thesis/Figure_6.svg",format="svg") 



    
