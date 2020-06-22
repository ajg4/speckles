import time
import sys
sys.path.insert(0, './srw')
import numpy as np
import multiprocessing as mp
# from srwlpy import *
from srwlib import *
import pickle as pk
import os
import h5py
from helper import bhmie
from scipy.constants import h,c,e
from scipy import interpolate
#%%  
#jobId0=int(sys.argv[1]) #for using the batch cluster
jobId0=1

path='/home/alex/Desktop/temp_res/'
file="img_final.p"

#Beam
sigmax=150e-6
sigmay=10e-6
lam=1e-10
fwhmk=1e-3     

#Setup
z1=100
z2=1.1
ext=1e-3        
px=int(2**10)
   
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
resize=4 #resizing within the SRW calculation of the wavefront
slices=4   #cuts the calculation in slices to reduces RAM peak
    
#%%  
for particle in range(2):
    ttotal=time.time()
    jobId=str(jobId0)+"_"+str(particle)   
     
    np.random.seed(jobId0*int(1e10*(time.time()%0.0001)))
    k=2*np.pi/lam
    k=k+np.random.standard_normal()*k*(fwhmk)/2.35
    lam=2*np.pi/k
    
    print("Wavelength [m] "+str(lam))   
    
    xpos=np.random.standard_normal()*sigmax
    ypos=np.random.standard_normal()*sigmay
    print("Particle position [um] ",xpos*1e6,ypos*1e6)
    
    np.random.seed(1)
    #scatlist=np.reshape((np.random.rand(numScat*2)*0.4*ext-0.2*ext),(numScat,2))
    scatlist=np.reshape((np.random.rand(numScat*2)*0.6*ext-0.3*ext),(numScat,2))
    #scatlist=np.reshape((np.random.rand(numScat*2)*1.0*ext-0.5*ext),(numScat,2))
    #%%Functions
    def wfrjob(xpos,ypos,ext,xStart,xFin,xpx,ypx,scatlist,slicenum,resize):
        from scipy.constants import h,c,e
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
        wfr.mesh.zStart = (z1+z2) #Longitudinal position for initial wavefront [m]   
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
        
        phases=np.zeros(len(scatlist),dtype='complex64')
        
        xx=np.linspace(xStart,xFin,xpx)
        yy=np.linspace(-ext/2,ext/2,ypx)
        yy=np.rot90([yy],k=3)
        
        for i in range(len(scatlist)):
            x=scatlist[i][0]
            y=scatlist[i][1]
            
            x1=(x-xpos)*(z1+z2)/z1+xpos
            y1=(y-ypos)*(z1+z2)/z1+ypos
            
            if(x1>=xStart and x1<=xFin):
                idx = (np.abs(xx - (x1))).argmin()
                idy = (np.abs(yy - (y1))).argmin()  
                phases[i]=e[idy,idx]
        
        pk.dump(phases,open(path+str(jobId)+"_"+str(slicenum)+"_"+"phases.p","wb"))    
                      

    def scjob(xpos,ypos,ext,xStart,xFin,xpx,ypx,scatlist,mie_interpol,slicenum,phases):
        e = pk.load(open(path+str(jobId)+"_"+str(slicenum)+"_"+"wf.p","rb") )
        os.remove(path+str(jobId)+"_"+str(slicenum)+"_"+"wf.p")
           
        xx=np.linspace(xStart,xFin,xpx)
        yy=np.linspace(-ext/2,ext/2,ypx)    
        yy=np.rot90([yy],k=3)
        
        es=np.zeros((ypx,xpx),dtype='complex64')
        for j in range(len(scatlist)):            
            x=scatlist[j][0]
            y=scatlist[j][1]
            
            x1=(x-xpos)*(z1+z2)/z1+xpos
            y1=(y-ypos)*(z1+z2)/z1+ypos
            
            diffx=x-x1
            diffy=y-y1        
            
            phase=(phases[j])/(np.exp(1j*k*np.sqrt(diffx**2+diffy**2+z2**2)))
           
            rs=np.sqrt((xx-x)**2+(yy-y)**2+z2**2)
            
            ra=np.sqrt((xx-x1)**2+(yy-y1)**2)
    
            # airy=np.exp(-0.5*(ra/sigmaS)**2)
            
            pattern=mie_interpol(np.arctan(ra/z2))
            
            es+=phase*np.exp(1j*k*rs)/rs*pattern
           
        img=np.abs(e+es)**2-np.abs(e)**2-np.abs(es)**2
        
        pk.dump(img,open(path+str(jobId)+"_"+str(slicenum)+"_"+"img.p","wb"))        
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
            poolInputs.append((xpos,ypos,ext,xlist[j*cores+i][0],xlist[j*cores+i][1],xpx,px,scatlist,(j*cores+i),resize))
        wfrSplits=pool.starmap(wfrjob,poolInputs)
        pool.close()
        pool.join()
            
    print("WFs calculated in ",time.time()-a)
    #%% Collecting of phases 
    a=time.time()
    phases=np.zeros(numScat,dtype='complex64')
    for i in range(slices):
        dat = pk.load(open(path+str(jobId)+"_"+str(i)+"_"+"phases.p","rb"))
        phases+=dat
        os.remove(path+str(jobId)+"_"+str(i)+"_"+"phases.p")
        
    print("Phases collected in ",time.time()-a)       
    #%% Scattering the colloids
       
    a=time.time()
    for j in range(int(slices/cores)):
    
        pool = mp.Pool(cores)
        poolInputs=[]
        for i in range(cores):
            poolInputs.append((xpos,ypos,ext,xlist[j*cores+i][0],xlist[j*cores+i][1],xpx,px,scatlist,mie_interpol,(j*cores+i),phases))
        wfrSplits=pool.starmap(scjob,poolInputs)
        pool.close()
        pool.join()
            
    print("Colloids scattered in ",time.time()-a)  
    #%% Combining images and dummping

    a=time.time()
    imgs=np.zeros((px,px))
    for k in range(slices):
        dat = pk.load(open(path+str(jobId)+"_"+str(k)+"_"+"img.p","rb"))
        os.remove(path+str(jobId)+"_"+str(k)+"_"+"img.p")
        
        imgs[:,k*xpx:(k+1)*xpx]=dat    
    
    hf = h5py.File(path+str(jobId)+"_"+file, 'w')
    hf.create_dataset('dataset_1', data=imgs)
    hf.close()   
    print("Collecting and dumping in ",time.time()-a)  
    
    print(time.time()-ttotal)