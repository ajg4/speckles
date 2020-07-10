import numpy as np
import pickle as pk
import helper
import matplotlib.pyplot as plt
import h5py
from helper import radial_profile, sector_profile,running_mean
path='/home/alex/Desktop/'
path='C:/Users/Alex/Desktop/'
#%%  
#jobId0=int(sys.argv[1]) #for using the batch cluster
jobId0=0

file="img_final2.p"

#Beam
sigmax=105e-6
sigmay=15e-6
lam=1e-10
fwhmk=1e-3     

#Setup
z1=100
z2=2
ext=2.25e-3        
px=int(2**11)
   
#Colloids
colloid_radius=0.25e-6
numScat=1


#%%
# print("summming up")
# first=0
# for i in range(10):
#     try:
#         if(first==0):
#             hf = h5py.File(path+str(1)+"_"+str(i)+"_"+file, 'r')
#             img = np.array(hf.get('dataset_1'))
#             hf.close()
#             first=1
#         hf = h5py.File(path+str(1)+"_"+str(i)+"_"+file, 'r')
#         img2 = np.array(hf.get('dataset_1'))
#         hf.close()
#         img+=img2
#         print(i,'done')
#     except:
#         print(i, 'file not found')

# out = h5py.File(path+str("A")+"_"+file, 'w')
# out.create_dataset('dataset_1'   , data=img)
# out.close()

#%%
file=path+str("A")+"_"+file

hf = h5py.File(file, 'r')
img = np.array(hf.get('dataset_1'))
hf.close()  

k=2*np.pi/lam

#%%
imft=np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img))))**2

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


fig=plt.figure('Figure_8_3.svg',figsize=(20,10))

x=np.linspace(0,0.5*px/ext,px)

extent=np.array([-x[-1],x[-1],-x[-1],x[-1]])*1e-6
plt.imshow(imft,extent=extent)
plt.xlabel(r"horizontal spatial frequency $[\mu m^{-1}]$")
plt.ylabel(r"vertical spatial frequency $[\mu m^{-1}]$")


plt.tight_layout()
plt.savefig(path+"Figure_8_3.svg",format="svg")   


#%%
rp,sec_data=sector_profile(imft*1,[int(px/2),int(px/2)],[0,10])

rp=running_mean(rp,1)

x=np.linspace(0,0.5*np.sqrt(2)*px/ext,np.size(rp))

sigmaC=lam*(z1+z2)/2/np.pi/sigmax
sigmaS=0.4793*lam*z2/(colloid_radius*2)*np.sqrt(2)

sigmak=k*fwhmk/2.35
sigmaT=np.sqrt(2*z2/sigmak)

sigma_total=np.sqrt(1/( 1/sigmaC**2 + 1/sigmaS**2 + 1/sigmaT**2 ))
sigma_total2=np.sqrt(1/( 1/sigmaC**2 + 1/sigmaS**2))

sigmaq=sigma_total/lam/z2/np.sqrt(2)
sigmaq2=sigma_total2/lam/z2/np.sqrt(2)
sigmaCq=sigmaC/lam/z2/np.sqrt(2)
sigmaSq=sigmaS/lam/z2/np.sqrt(2)
sigmaTq=sigmaT/lam/z2/np.sqrt(2)

leading=np.exp(-0.5*(x/sigmaq)**2)
top=leading[np.argmax(rp)]
rp=rp/np.max(rp)*top


from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
def gauss(x,sigma):
    out=np.exp(-0.5*(x/sigma)**2)
    return(out)

from scipy.signal import find_peaks

peaks=find_peaks(rp,distance=70)[0][1:]
#plt.plot(x,rp)
#plt.plot(x[peaks],rp[peaks])
#maxima=argrelextrema(rp,np.greater)[0]
#maxima2=argrelextrema(rp[maxima],np.greater)[0][:]
pars=curve_fit(gauss,x[peaks],rp[peaks],p0=[sigmaq])


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


fig=plt.figure('Figure_8_1.svg',figsize=(20,10))

plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaTq)**2),label="temporal")
plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaSq)**2),label="scattering")
plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaCq)**2),label="spatial")
plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaq)**2),label="total")
#plt.plot(x*1e-6,np.exp(-0.5*(x/pars[0])**2),label="fit")
#plt.plot(x*1e-6,np.exp(-0.5*(x/pars[0])**2),label="fit")
#plt.plot(x[peaks]*1e-6,rp[peaks],label="fit_helper")
plt.plot(x*1e-6,rp,label="simulation")

plt.xlabel(r'spatial radial frequency $[\mu m^{-1}]$')
plt.ylabel('power spectrum [a.u.]')
plt.legend()


plt.tight_layout()
plt.savefig(path+"Figure_8_1.svg",format="svg")    

print(pars[0]/sigmaq)

#%%
rp,sec_data=sector_profile(imft*1,[int(px/2),int(px/2)],[90,5])

x=np.linspace(0,0.5*np.sqrt(2)*px/ext,np.size(rp))

lam=2*np.pi/k
sigmaC=lam*(z1+z2)/2/np.pi/sigmay
sigmaS=0.4793*lam*z2/(colloid_radius*2)*np.sqrt(2)

sigmak=k*fwhmk/2.35
sigmaT=np.sqrt(2*z2/sigmak)

sigma_total=np.sqrt(1/( 1/sigmaC**2 + 1/sigmaS**2 + 1/sigmaT**2 ))
sigma_total2=np.sqrt(1/( 1/sigmaC**2 + 1/sigmaS**2))

sigmaq=sigma_total/lam/z2/np.sqrt(2)
sigmaq2=sigma_total2/lam/z2/np.sqrt(2)
sigmaCq=sigmaC/lam/z2/np.sqrt(2)
sigmaSq=sigmaS/lam/z2/np.sqrt(2)
sigmaTq=sigmaT/lam/z2/np.sqrt(2)

leading=np.exp(-0.5*(x/sigmaq)**2)
top=leading[np.argmax(rp)]
rp=rp/np.max(rp)*top


from scipy.optimize import curve_fit
def gauss(x,sigma):
    out=np.exp(-0.5*(x/sigma)**2)
    return(out)

from scipy.signal import find_peaks

peaks=find_peaks(rp,distance=100)[0][1:]
#plt.plot(x,rp)
#plt.plot(x[peaks],rp[peaks])
#maxima=argrelextrema(rp,np.greater)[0]
#maxima2=argrelextrema(rp[maxima],np.greater)[0][:]
pars=curve_fit(gauss,x[peaks],rp[peaks],p0=[sigmaq])


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



fig=plt.figure('Figure_8_2.svg',figsize=(20,10))

plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaTq)**2),label="temporal")
plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaSq)**2),label="scattering")
plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaCq)**2),label="spatial")
plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaq)**2),label="total")
#plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaq2)**2),label="total2")
#plt.plot(x*1e-6,np.exp(-0.5*(x/pars[0])**2),label="fit")
#plt.plot(x[peaks]*1e-6,rp[peaks],label="fit_helper")
plt.plot(x*1e-6,rp,label="simulation")

plt.xlabel(r'spatial radial frequency $[\mu m^{-1}]$')
plt.ylabel('power spectrum [a.u.]')
plt.legend()


plt.tight_layout()
plt.savefig(path+"Figure_8_2.svg",format="svg")  

print(pars[0]/sigmaq)
