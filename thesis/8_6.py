import sys
sys.path.append('./srw/')
from srwlib import *
import time
import numpy as np
import pickle as pk
import multiprocessing as mp
import gc
import os
import psutil
from array import array
import h5py

h=6.62607015e-34
c=299792458.0
e=1.602176634e-19
electron_mass=9.1093837015e-31

# jobId=int(sys.argv[1])
# jobId=0


def job(jobId):
    path='./'
    # path='/eos/home-a/agoetz/tempresults/'
    
    #Beam
    sigmax=130e-6
    sigmay=6.5e-6
    lam=6e-11
    fwhmlam=0    
    numSource=1
    xpos=0
    ypos=0 
    
    #Setup
    z1=32.5
    z2=0.5
    ext=1e-4        
    px=int(2**11)
    
    #Computing   
    cores=4
    resize=8 #resizing within the SRW calculation of the wavefront
    
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
    # print(ext/px/resize)
    # print(psutil.virtual_memory())
    
    px=int(px/resize)
    t00=time.time()
    
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
    
    
    print('Performing initial electric field wavefront calculation ... ')
    t0 = time.time()
    
    #%% single threading
    if(True):  
        wfr=SRWLWfr()
        wfr.allocate(1, px, px) #Numbers of points vs photon energy, horizontal and vertical positions (the last two will be modified in the process of calculation)   
        wfr.mesh.zStart = (z1) #Longitudinal position for initial wavefront [m]   
        wfr.mesh.eStart = h*c/e/lam #Initial photon energy [eV]
        wfr.mesh.eFin = wfr.mesh.eStart #Final photon energy [eV]   
        wfr.mesh.xStart = -ext/2 #Initial horizontal position [m]
        wfr.mesh.xFin = ext/2 #Final horizontal position [m]
        wfr.mesh.yStart = -ext/2 #Initial vertical position [m]
        wfr.mesh.yFin = ext/2 #Final vertical position [m] 
        wfr.partBeam = eBeam #e-beam data is contained inside the wavefront struct
        
        par=0
        sampFactNxNyForProp=0
        
        arPrecSR = [meth, relPrec, zStartInteg, zEndInteg, npTraj, useTermin, sampFactNxNyForProp] 
        
        srwl.CalcElecFieldSR(wfr, par, magFldCnt, arPrecSR) 
    
    #%% multi threading
    if(False):
        def thread(args):
            wfr,par,magFldCnt,arPrecSR=args
            srwl.CalcElecFieldSR(wfr, par, magFldCnt, arPrecSR) #Calculating electric field
            return(wfr)
            
        sl=ext/cores
        
        wfrs=[]
        
        a=time.time()
        for i in range(cores):
            wfrs.append(SRWLWfr()) #Wavefront structure (placeholder for data to be calculated)
            wfrs[-1].allocate(1, px, int(px/cores)) #Numbers of points vs photon energy, horizontal and vertical positions (the last two will be modified in the process of calculation)   
            wfrs[-1].mesh.zStart = z1 #Longitudinal position for initial wavefront [m]   
            wfrs[-1].mesh.eStart = h*c/e/lam #Initial photon energy [eV]
            wfrs[-1].mesh.eFin = wfrs[-1].mesh.eStart #Final photon energy [eV]   
            wfrs[-1].mesh.xStart = -ext/2 #Initial horizontal position [m]
            wfrs[-1].mesh.xFin = ext/2 #Final horizontal position [m]
            wfrs[-1].mesh.yStart = -ext/2+i*sl #Initial vertical position [m]
            wfrs[-1].mesh.yFin = -ext/2+i*sl+sl#Final vertical position [m] 
            wfrs[-1].partBeam = eBeam #e-beam data is contained inside the wavefront struct
        
        par=0
        sampFactNxNyForProp=0
        arPrecSR = [meth, relPrec, zStartInteg, zEndInteg, npTraj, useTermin, sampFactNxNyForProp]    
            
        pool = mp.Pool(cores)
        poolInputs=[]
        for i in range(cores):
            poolInputs.append((wfrs[i],par, magFldCnt, arPrecSR))
        wfrSplits=pool.map(thread,poolInputs)
        pool.close()
        pool.join()
        
        Ex=np.array([])
        Ey=np.array([])
        for i in range(cores):
            Ex=np.append(Ex,np.array(wfrSplits[i].arEx))
            Ey=np.append(Ey,np.array(wfrSplits[i].arEy))   
        
        temp_avgPhotEn=wfrSplits[0].avgPhotEn
        
        del(wfrSplits)
        gc.collect()
            
        wfr=SRWLWfr()
        wfr.allocate(1, px, px)
        #wfr.xc=wfrSplits[0].xc
        #wfr.yc=wfrSplits[0].yc
        #wfr.numTypeElFld=wfrSplits[0].numTypeElFld
        #wfr.dRx=wfrSplits[0].dRx
        #wfr.dRy=wfrSplits[0].dRy
        wfr.avgPhotEn=temp_avgPhotEn
        #wfr.arElecPropMatr=wfrSplits[0].arElecPropMatr
        #wfr.Rx=wfrSplits[0].Rx
        #wfr.Ry=wfrSplits[0].Ry
        wfr.mesh.zStart = z1 #Longitudinal position for initial wavefront [m]   
        wfr.mesh.eStart = h*c/e/lam #Initial photon energy [eV]
        wfr.mesh.eFin = wfr.mesh.eStart #Final photon energy [eV]   
        wfr.mesh.xStart = -ext/2 #Initial horizontal position [m]
        wfr.mesh.xFin = ext/2 #Final horizontal position [m]
        wfr.mesh.yStart = -ext/2 #Initial vertical position [m]
        wfr.mesh.yFin = ext/2#Final vertical position [m]   
        wfr.partBeam = eBeam #e-beam data is contained inside the wavefront struct
        wfr.arEx=array('f', Ex)
        wfr.arEy=array('f', Ey)
        
        del(Ex,Ey)
        gc.collect()
    
    print('done in', round(time.time() - t0))
    
    #%%
    
    print('Resizing wavefront ... ')
    t0 = time.time()
    srwl.ResizeElecField(wfr,'c',[0,1,resize,1,resize])
    px=px*resize
    print('done in', round(time.time() - t0))
    
    print('Saving wavefront ... ')
    t0 = time.time()
    f=open(path+'wfr_dat_x'+str(jobId),'wb');wfr.arEx.tofile(f);f.close()
    f=open(path+'wfr_dat_y'+str(jobId),'wb');wfr.arEy.tofile(f);f.close()
    print('done in', round(time.time() - t0))
    
    wfr.arEx=array('f', [])
    wfr.arEy=array('f', [])
    
    file = open(path+'wfr_struct'+str(jobId), 'wb');pk.dump(wfr, file);file.close()
    
    # print(psutil.virtual_memory())
    del (wfr)
    gc.collect()
    
    #%%
    
    optDrift = SRWLOptD(z2)
    
    #Propagation paramaters (SRW specific)
    #                [0][1][2] [3][4] [5] [6] [7] [8]
    propagParDrift = [0, 0, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
    #Wavefront Propagation Parameters:
    #[0]: Auto-Resize (1) or not (0) Before propagation
    #[1]: Auto-Resize (1) or not (0) After propagation
    #[2]: Relative Precision for propagation with Auto-Resizing (1. is nominal)
    #[3]: Allow (1) or not (0) for semi-analytical treatment of the quadratic (leading) phase terms at the propagation
    #[4]: Do any Resizing on Fourier side, using FFT, (1) or not (0)
    #[5]: Horizontal Range modification factor at Resizing (1. means no modification)
    #[6]: Horizontal Resolution modification factor at Resizing
    #[7]: Vertical Range modification factor at Resizing
    #[8]: Vertical Resolution modification factor at Resizing
    #[9]: Type of wavefront Shift before Resizing (not yet implemented)
    #[10]: New Horizontal wavefront Center position after Shift (not yet implemented)
    #[11]: New Vertical wavefront Center position after Shift (not yet implemented)
    
    optBL = SRWLOptC([optDrift], [propagParDrift])
    
    lenData=px*px
    lenData2=px*px*2
    
    print('Loading mask... ')
    t0 = time.time()
    
    mask=array('f', [])
    f=open(path+'mask','rb');mask.fromfile(f,lenData2);f.close
    mask=np.array(mask)
    
    print('done in', round(time.time() - t0))
    print('Loading data and masking... ')
    t0 = time.time()
    
    arEx=array('f', [])
    f=open(path+'wfr_dat_x'+str(jobId),'rb');arEx.fromfile(f,lenData2);f.close
    arEx=np.array(arEx)
    arEx*=mask
    
    arEy=array('f', [])
    f=open(path+'wfr_dat_y'+str(jobId),'rb');arEy.fromfile(f,lenData2);f.close
    arEy=np.array(arEy)
    arEy*=mask
    
    arEx=array('f',arEx)
    arEy=array('f',arEy)
    
    
    #%%
    # print(psutil.virtual_memory())
    del(mask)
    gc.collect()
    
    file = open(path+'wfr_struct'+str(jobId), 'rb');wfr = pk.load(file);file.close
    wfr.arEx=arEx
    wfr.arEy=arEy
    
    print('done in', round(time.time() - t0))
    
    print('Simulating masked single-electron electric field wavefront propagation ... ')
    t0 = time.time()
    srwl.PropagElecField(wfr, optBL)
    print('done in', round(time.time() - t0))    
    
    
    print('Extracting intensity and saving masked propagated img ... ')
    t0 = time.time()
    
    arEx=np.array(wfr.arEx)
    wfr.arEx=array('f', [])
    img=np.empty((lenData))
    img=np.abs(arEx[::2]+arEx[1::2]*1j)**2
    
    # print(psutil.virtual_memory())
    del(arEx)
    gc.collect()
    
    arEy=np.array(wfr.arEy)
    wfr.arEy=array('f', [])
    
    # print(psutil.virtual_memory())
    del(wfr)
    gc.collect()
    
    img+=np.abs(arEy[::2]+arEy[1::2]*1j)**2
    
    # print(psutil.virtual_memory())
    del(arEy)
    gc.collect()
    
    img = array('f', img) #"Flat" array to take 2D single-electron intensity data (vs X & Y)
    
    f=open(path+'arI1s'+str(jobId),'wb');img.tofile(f);f.close()
    print('done in', round(time.time() - t0))  
    
    # print(psutil.virtual_memory())
    del(img)
    gc.collect()
    
    
    print('Loading wfr data ... ')
    t0 = time.time()
    file = open(path+'wfr_struct'+str(jobId), 'rb');wfr = pk.load(file);file.close()
    f=open(path+'wfr_dat_x'+str(jobId),'rb');wfr.arEx.fromfile(f,lenData2);f.close()
    f=open(path+'wfr_dat_y'+str(jobId),'rb');wfr.arEy.fromfile(f,lenData2);f.close()
    print('done in', round(time.time() - t0))  
    
    os.remove(path+'wfr_dat_x'+str(jobId))
    os.remove(path+'wfr_dat_y'+str(jobId))
    
    print('Simulating single-electron electric field wavefront propagation ... ')
    t0 = time.time()
    srwl.PropagElecField(wfr, optBL)
    print('done in', round(time.time() - t0)) 
     
    print('Extracting intensity of nonmasked propagated wfr ... ')
    t0 = time.time()
    
    arEx=np.array(wfr.arEx)
    wfr.arEx=array('f', [])
    arI2s=np.empty((lenData))
    arI2s=np.abs(arEx[::2]+arEx[1::2]*1j)**2
    
    # print(psutil.virtual_memory())
    del(arEx)
    gc.collect()
    
    arEy=np.array(wfr.arEy)
    wfr.arEy=array('f', [])
    
    # print(psutil.virtual_memory())
    del(wfr)
    gc.collect()
    
    arI2s+=np.abs(arEy[::2]+arEy[1::2]*1j)**2
    
    # print(psutil.virtual_memory())
    del(arEy)
    gc.collect()
    
    arI2s = array('f', arI2s) #"Flat" array to take 2D single-electron intensity data (vs X & Y)
    print('done in', round(time.time() - t0)) 
    
    print('Loading data of masked img ... ')
    t0 = time.time()
    arI1s=array('f', [])
    f=open(path+'arI1s'+str(jobId),'rb');arI1s.fromfile(f,lenData);f.close()
    arI1s=np.array(arI1s)
    print('done in', round(time.time() - t0))  
    
    os.remove(path+'arI1s'+str(jobId))
    
    print('adding, reshaping and saving ... ')
    t0 = time.time()
    arI1s-=arI2s
    
    # print(psutil.virtual_memory())
    del(arI2s)
    gc.collect()
    
    file = open(path+'wfr_struct'+str(jobId), 'rb');wfr = pk.load(file);file.close()    
    arI1s=np.reshape(arI1s,(wfr.mesh.ny,wfr.mesh.nx))
    
    hf = h5py.File(path+str(jobId)+"_"+"img_final.p", 'w')
    hf.create_dataset('dataset_1', data=arI1s)
    hf.close()
    print('done in',round(time.time() - t0)) 
    print('total',round(time.time() - t00))
    # print(psutil.virtual_memory())


from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count

img=np.zeros((px,px))

for j in range(3):
    proc=[]
    thr_count=cpu_count()
    pool=ThreadPool(processes=thr_count)
    for i in range(thr_count):
        proc.append(pool.apply_async(job, (i)))
        
    for i in range(thr_count):
        proc[i].get()
        
    print("summming up")
    first=0
    for i in range(thr_count):
        try:
            if(first==0):
                hf = h5py.File(path+str(i)+"_"+file, 'r')
                img = np.array(hf.get('dataset_1'))
                hf.close()
                first=1
            hf = h5py.File(path+str(i)+"_"+file, 'r')
            img2 = np.array(hf.get('dataset_1'))
            hf.close()
            img+=img2
            print(i,'done')
        except:
            print(i, 'file not found')

    out = h5py.File("./final", 'w')
    out.create_dataset('dataset_1'   , data=img)
    out.close()
    os.system('aws s3 cp ./final s3://mocd/final.h5')
    
