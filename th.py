import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing.pool import ThreadPool
from scipy.special import jv
from mie import bhmie

def radial_profile(data,center):
    y,x = np.indices((data.shape))
    r = np.sqrt((x-center[0])**2 + (y-center[1])**2)
    r = r.astype(np.int)
    tbin = np.bincount(r.ravel(), weights = data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin/nr
    return(radialprofile) 
    
def the_speckles(sigma,k,fwhmk,numSource,z1,z2,fwhmz2,ext,px,colloid_diameter,numScat,scattertype,cores):
    
    #if mie, call the mie function an fit it with a polynomial
    if(scattertype=="mie"):
        polyorder=20
        points=10000
        radius=colloid_diameter/2    
        
        refr=1.587/1.331    
        rads=int(np.arctan(np.sqrt(2)*ext/z2*2)) #the rads to be calculated
        
    #    refr=1-1.28e-6+2.49e-09*1j
    #    rads=int(np.arctan(np.sqrt(2)*ext/z2))
        
        a=bhmie(lam,radius,refr,points,polyorder,rads)
        scattertype=a[1]
    
        #check the polynomial fit before using it:
    #    plt.plot(a[0])
    #    plt.plot(a[1](np.linspace(0,rads,points)))    
    
    #%%    
    #replace zeros by small numbers
    if(sigma==0):
        sigma=1e-20
    if(fwhmz2==0):
        fwhmz2=1e-20
    
    
    #scatlist contains the positions of the scatterers
    np.random.seed(1)
    x=(np.random.rand(numScat)*0.8-0.4)*ext
    y=(np.random.rand(numScat)*0.8-0.4)*ext
    #x=(np.random.rand(numScat))*ext
    #y=(np.random.rand(numScat))*ext
    z=np.random.rand(numScat)*fwhmz2
    scatlist=np.rot90([x,y,z])
    
    #beamlist has all the information about the particles
    #one particle emits one specific wavelength, not physical but ok
    beamx=np.random.standard_normal(numSource)*sigma
    beamy=np.random.standard_normal(numSource)*sigma
    ks=k*(1+np.random.standard_normal(numSource)*fwhmk)
    
    beamlist=np.rot90([beamx,beamy,ks])
    
    
    
    #taking a set of beamparticles, looping each over all the scatteres
    def thread(beamlist,ext,px,z1,z2,scatlist,scattertype):
        img=np.zeros((px,px))
        for j in range(len(beamlist)):
            k=beamlist[j][2]
            lam=2*np.pi/k
    
    #calculating the field at the detection plane     
            Min,Max,n=-ext/2,ext/2,px   
            
            xx=np.linspace(Min-beamlist[j][0],Max-beamlist[j][0],n)
            yy=np.linspace(Min-beamlist[j][1],Max-beamlist[j][1],n)
            yy=np.rot90([yy])
            RR2=np.sqrt(xx**2+yy**2+(z1+z2)**2)
            e2=np.exp(1j*k*RR2)/RR2        
            
            e10s=np.zeros((px,px),dtype='complex128')
            for i in range(len(scatlist)):
                x=scatlist[i][0]
                y=scatlist[i][1]
                z2c=scatlist[i][2]
    
    #calculating the phase of the field at the position of a scatterer
                e10=np.exp(1j*k*np.sqrt(x**2+y**2+(z1-z2c)**2)) 
               
    #calculating the scattering efficiency of the scatterer
    #taking into account its displaced central point according to the divergence of the beam            
                scale=(z1+z2)/(z1-z2c)
                xx=np.linspace(Min-x*scale,Max-x*scale,n)    
                yy=np.linspace(Min-y*scale,Max-y*scale,n)  
                yy=np.rot90([yy])
                
                if(isinstance(scattertype,str)):
                    q=colloid_diameter*np.pi*np.sin(np.tan(np.sqrt(xx**2+yy**2)/(z2+z2c)))/lam
                    if(scattertype=="airy"):
                        pattern=(2*jv(1,q)/q)
                    if(scattertype=="anom"):           
                        pattern=jv(3/2,q)*np.sqrt(1/q**3)
                    if(scattertype=="gauss"):
                        sigma=0.4793*lam*(z2+z2c)/colloid_diameter*np.sqrt(2)
                        pattern=np.exp(-0.5*(xx**2+yy**2)/sigma**2)
                if(isinstance(scattertype,str)==False): 
                    q=np.sqrt(xx**2+yy**2)/(z2+z2c)
                    pattern=scattertype(q)
        
                pattern/=np.max(pattern)                 
    
    #calculating the field of the scattering particle and shapping it by the patter of the scattering efficiency
                xx=np.linspace(Min-x,Max-x,n)
                yy=np.linspace(Min-y,Max-y,n)
                yy=np.rot90([yy])
                RR3=np.sqrt(xx**2+yy**2+(z2+z2c)**2)
                
                e10s+=e10*np.exp(1j*k*RR3)/RR3*pattern
    
    #letting interfere the inital field and the scattered field, subtracting their absolute values         
            add=np.abs(e10s+e2)**2-np.abs(e2)**2-np.abs(e10s)**2
            img+=add
        return(img)
    
    #same thing, but multithreading the scatteres instead of the particles (for one central particle)
    def cthread(scatlist,ext,px,z1,z2,k,scattertype):
        Min,Max,n=-ext/2,ext/2,px
        e10s=np.zeros((px,px),dtype='complex128')
        lam=2*np.pi/k
        for i in range(len(scatlist)):
            x=scatlist[i][0]
            y=scatlist[i][1]
            z2c=scatlist[i][2]  
                             
            e10=np.exp(1j*k*np.sqrt(x**2+y**2+(z1-z2c)**2)) 
            
            scale=(z1+z2)/(z1-z2c)
            xx=np.linspace(Min-x*scale,Max-x*scale,n)    
            yy=np.linspace(Min-y*scale,Max-y*scale,n)  
            yy=np.rot90([yy])
            
            if(isinstance(scattertype,str)):
                q=colloid_diameter*np.pi*np.sin(np.tan(np.sqrt(xx**2+yy**2)/(z2+z2c)))/lam
                if(scattertype=="airy"):
                    pattern=(2*jv(1,q)/q)
                if(scattertype=="anom"):           
                    pattern=jv(3/2,q)*np.sqrt(1/q**3)
                if(scattertype=="gauss"):
                    sigma=0.4793*lam*(z2+z2c)/colloid_diameter*np.sqrt(2)
                    pattern=np.exp(-0.5*(xx**2+yy**2)/sigma**2)
            if(isinstance(scattertype,str)==False): 
                q=np.sqrt(xx**2+yy**2)/(z2+z2c)
                pattern=scattertype(q)
    
            pattern/=np.max(pattern)                 
    
            xx=np.linspace(Min-x,Max-x,n)
            yy=np.linspace(Min-y,Max-y,n)
            yy=np.rot90([yy])
            RR3=np.sqrt(xx**2+yy**2+(z2+z2c)**2)
            
            e10s+=e10*np.exp(1j*k*RR3)/RR3*pattern
        return(e10s)      
        
    
    if(numSource!=1):
        a=time.time()
        if(cores>numSource):
            cores=numSource
        pool=ThreadPool(processes=cores)
        sl=int(numSource/cores)
        res=np.zeros((cores,px,px))
        proc=[]
        for i in range(cores):
            print(i*sl,(i+1)*sl)
            proc.append(pool.apply_async(thread, (beamlist[i*sl:(i+1)*sl],
                                                        ext,
                                                        px,
                                                        z1,
                                                        z2,
                                                        scatlist,
                                                        scattertype)))
        for i in range(cores):
            res[i]=proc[i].get()
        
        img=np.sum(res,axis=0)
        img/=np.max(img)    
        print(time.time()-a)
        
    if(numSource==1):
        a=time.time()
        if(cores>numScat):
            cores=numScat
        pool=ThreadPool(processes=cores)
        sl=int(numScat/cores)
        res=np.zeros((cores,px,px),dtype='complex128')
        proc=[]
        for i in range(cores):
            proc.append(pool.apply_async(cthread, (scatlist[i*sl:(i+1)*sl],
                                                        ext,
                                                        px,
                                                        z1,
                                                        z2,
                                                        k,
                                                        scattertype)))
        for i in range(cores):
            res[i]=proc[i].get()
        
        e10s=np.sum(res,axis=0)
    
    #calculing the initial field of the one particle, interference with all the scattered waves      
        Min,Max,n=-ext/2,ext/2,px   
        xx=np.linspace(Min,Max,n)
        yy=np.linspace(Min,Max,n)
        yy=np.rot90([yy])
        RR2=np.sqrt(xx**2+yy**2+(z1+z2)**2)
        e2=np.exp(1j*k*RR2)/RR2 
        
    #    e10s=e10s/np.max(e10s)*np.sum(e2)*0.05
        
        img=np.abs(e10s+e2)**2-np.abs(e2)**2-np.abs(e10s)**2
        
        img/=np.max(img)

          
    imft=np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img))))**2
    
    return(img,imft)


      

#Beam
sigma=12e-6
k=2*np.pi/1e-11
fwhmk=0.5
numSource=int(2**0)  

#Setup
z1=70
z2=20
fwhmz2=1e-20
ext=0.5e-3 
px=int(2048)
   
#Colloids
colloid_diameter=8e-6
numScat=int(2**0)

#Computing   
cores=4

#scattering airy (4s), anom (5s), or mie (1s)
scattertype="anom"


a=the_speckles(sigma,k,fwhmk,numSource,z1,z2,fwhmz2,ext,px,colloid_diameter,numScat,scattertype,cores)

# plt.figure(2)
# plt.imshow(a[1])

imft=a[1]
rp=radial_profile(imft,[int(px/2),int(px/2)])
rp=rp/np.max(rp)


freqs=np.fft.fftshift(np.fft.fftfreq(int(np.sqrt((px/2)**2+(px/2)**2)),ext/px))
x=np.linspace(0,freqs[-1],np.size(rp))

lam=2*np.pi/k

sigmaC=lam*(z1+z1)/2/np.pi/sigma
sigmaS=0.4793*lam*z2/colloid_diameter
sigmak=fwhmk/2.35
sigmaT=np.sqrt(2*z2/sigmak)
sigma_total=np.sqrt(1/( 1/sigmaC**2 + 1/sigmaS**2 + 1/sigmaT**2 ))

sigmaq=sigma_total*k/np.sqrt(2)/np.pi/z2
sigmaCq=sigmaC*k/np.sqrt(2)/np.pi/z2
sigmaSq=sigmaS*k/np.sqrt(2)/np.pi/z2
sigmaTq=sigmaT*k/np.sqrt(2)/np.pi/z2


#%%

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


fig=plt.figure('Figure_14',figsize=(20,10))

plt.plot(x,rp,label="simulation")
plt.plot(x,np.exp(-0.5*(x/sigmaq)**2),label="predicted envelope")
plt.plot(x,np.exp(-0.5*(x/sigmaSq)**2),label="scat")
plt.plot(x,np.exp(-0.5*(x/sigmaTq)**2),label="temp")
plt.plot(x,np.exp(-0.5*(x/sigmaCq)**2),label="spat")

plt.xlabel(r'spatial radial frequency $\left[\dfrac{1}{\mu m}\right]$')
plt.ylabel('power spectrum')
plt.legend()


plt.tight_layout()
plt.savefig("Figure_14",format="svg")

    
    
