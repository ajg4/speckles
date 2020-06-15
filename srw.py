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
#%%  
#jobId0=int(sys.argv[1]) #for using the batch cluster
jobId0=1
for jobtask in range(1):
    ttotal=time.time()
    
    jobId=str(jobId0)+"_"+str(jobtask)    
#    path='./'
    path='/eos/home-a/agoetz/tempresults/'
#    path='/home/agoetz/Desktop/'
    file="img_final.p"

    #Beam
    sigma=5e-6
    lam=6e-11
    fwhmlam=0.2    
    numSource=1   
    
    #Setup
    z1=100
    z2=20
    ext=4e-3        
    px=int(2**10)
       
    #Colloids
    colloid=2e-6
    numScat=1
    
    #Computing   
    cores=16
    resize=4 #resizing within the SRW calculation of the wavefront
    slices=16   #cuts the calculation in slices to reduces RAM peak
    downsamplefactor=1 #downsample final image before combining of slices
      
    #%%
    if((px/slices)%resize!=0):
        print("resize odd!")
    if((px/slices)%downsamplefactor!=0):
        print("downsample odd!")
    if(slices<cores):
        print("put more slices than cores..")
    
    #%%       
    k=2*np.pi/lam
    sigmaC=lam*(z1+z2)/sigma/2/np.pi
    sigmaS=0.4793**lam*z2/colloid
    sigmaD=np.sqrt(1/(sigmaC**(-2)+sigmaS**(-2)))

    np.random.seed(jobId0*int(1e10*(time.time()%0.0001)))
    lam=lam+np.random.standard_normal()*lam*(fwhmlam)/2.35
    print("Wavelength [m] "+str(lam))   

    
    xpos=np.random.standard_normal()*sigma
    ypos=np.random.standard_normal()*sigma
    print("Particle position [um] ",xpos*1e6,ypos*1e6)
    
    np.random.seed(1)
    #scatlist=np.reshape((np.random.rand(numScat*2)*0.4*ext-0.2*ext),(numScat,2))
    scatlist=np.reshape((np.random.rand(numScat*2)*0.6*ext-0.3*ext),(numScat,2))
    #scatlist=np.reshape((np.random.rand(numScat*2)*1.0*ext-0.5*ext),(numScat,2))
    #%%Functions
    def wfrjob(xpos,ypos,ext,xStart,xFin,xpx,ypx,scatlist,sigmaS,slicenum,resize):
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
        
    def sampler(file,downsamplefactor,pxd,xpx2):
    
        dat = pk.load(open(file,"rb"))
        os.remove(file)
      
        img2=np.zeros((pxd,xpx2))
        for i in range(xpx2):
            for j in range(pxd):
                xstart=downsamplefactor*(i)
                xfin=downsamplefactor*(i+1)
                yfin=downsamplefactor*(j)
                ystart=downsamplefactor*(j+1)
                img2[j,i]=np.sum(dat[yfin:ystart,xstart:xfin])   
        pk.dump(img2,open(file,"wb"))                
    
    def scjob(xpos,ypos,ext,xStart,xFin,xpx,ypx,scatlist,sigmaS,slicenum,phases):
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

            airy=np.exp(-0.5*(ra/sigmaS)**2)
            
            es+=phase*np.exp(1j*k*rs)/rs*airy
           
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
            poolInputs.append((xpos,ypos,ext,xlist[j*cores+i][0],xlist[j*cores+i][1],xpx,px,scatlist,sigmaS,(j*cores+i),resize))
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
            poolInputs.append((xpos,ypos,ext,xlist[j*cores+i][0],xlist[j*cores+i][1],xpx,px,scatlist,sigmaS,(j*cores+i),phases))
        wfrSplits=pool.starmap(scjob,poolInputs)
        pool.close()
        pool.join()
            
    print("Colloids scattered in ",time.time()-a)  
    #%% Downsample the images               
    pxd=int(px/downsamplefactor)
    xpx2=int(xpx/downsamplefactor)
    
    if(downsamplefactor!=1):
    
        a=time.time()
        for j in range(int(slices/cores)):
            pool = mp.Pool(cores)
            poolInputs=[]
            for i in range(cores):
                print("slice ",j*cores+i)
                poolInputs.append((path+str(jobId)+"_"+str(j*cores+i)+"_"+"img.p"  ,downsamplefactor,pxd,xpx2))
            wfrSplits=pool.starmap(sampler,poolInputs)
            pool.close()
            pool.join()
                
        print("Downsampling in ",time.time()-a)
        
    if(downsamplefactor==1):
        print("No downsampling..")
    #%% Combining images and dummping
    a=time.time()
    imgs=np.zeros((pxd,pxd))
    for k in range(slices):
        dat = pk.load(open(path+str(jobId)+"_"+str(k)+"_"+"img.p","rb"))
        os.remove(path+str(jobId)+"_"+str(k)+"_"+"img.p")
        
        imgs[:,k*xpx2:(k+1)*xpx2]=dat    

    hf = h5py.File(path+str(jobId)+"_"+file, 'w')
    hf.create_dataset('dataset_1', data=imgs)
    hf.close()   
    print("Collecting and dumping in ",time.time()-a)  
    
    print(time.time()-ttotal)