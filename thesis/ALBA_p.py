import numpy as np
import pickle as pk
import helper
import matplotlib.pyplot as plt
import h5py
from helper import radial_profile, sector_profile,running_mean, defocused_otf
path='/home/alex/Desktop/speckles/thesis/'
#%%  
#jobId0=int(sys.argv[1]) #for using the batch cluster
jobId0=0

file="img_final2.p"

#Beam
sigmax=130e-6
sigmay=6.5e-6
lam=6.2e-11
fwhmk=0.002
numSource=int(2**8)     

#Setup
z1=33
z2=0.1
ext=2e-4        
px=int(2**11)
fwhmz2=0
   
#Colloids
colloid_radius=0.25e-6
numScat=10
   

#%%
# print("summming up")
# first=0
# for i in range(10):
#     try:
#         if(first==0):
#             hf = h5py.File(path+str(i)+"_"+file, 'r')
#             img = np.array(hf.get('dataset_1'))
#             hf.close()
#             first=1
#         hf = h5py.File(path+str(i)+"_"+file, 'r')
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

# sh=np.shape(img)
# sh2=int(sh[0]/4)
# img=img[sh2:-sh2,sh2:-sh2]
# ext=ext/2
# px=int(px/2)

k=2*np.pi/lam
imft=np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img))))**2

#%%
plt.figure(123)
plt.imshow(img)

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


fig=plt.figure('Figure_8_3.svg',figsize=(10,10))

x=np.linspace(0,0.5*px/ext,px)

extent=np.array([-x[-1],x[-1],-x[-1],x[-1]])*1e-6
plt.imshow(imft,extent=extent)
plt.xlabel(r"horizontal spatial frequency $[\mu m^{-1}]$")
plt.ylabel(r"vertical spatial frequency $[\mu m^{-1}]$")


plt.tight_layout()
plt.savefig(path+"Figure_8_3.svg",format="svg")   


#%%
rp,sec_data=sector_profile(imft*1,[int(px/2),int(px/2)],[0,4])

rp=running_mean(rp,1)

x=np.linspace(0,0.5*np.sqrt(2)*px/ext,np.size(rp))

dz,zetas,q,NA,lam_v,n,terms=100e-6,30,x/23,0.4,500e-9,1.5,10

otf=defocused_otf(dz,zetas,q,NA,lam_v,n,terms)

sigmaC=lam*(z1+z2)/2/np.pi/sigmax
sigmaS=0.4793*lam*z2/(colloid_radius*2)*np.sqrt(2)

sigmak=k*fwhmk/2.35
sigmaT=np.sqrt(z2/sigmak*np.sqrt(2/np.log(2)))

sigma_total=np.sqrt(1/( 1/sigmaC**2 + 1/sigmaS**2 + 1/sigmaT**2 ))
sigma_total2=np.sqrt(1/( 1/sigmaC**2 + 1/sigmaS**2))

sigmaq=sigma_total/lam/z2/np.sqrt(2)
sigmaq2=sigma_total2/lam/z2/np.sqrt(2)
sigmaCq=sigmaC/lam/z2/np.sqrt(2)
sigmaSq=sigmaS/lam/z2/np.sqrt(2)
sigmaTq=sigmaT/lam/z2/np.sqrt(2)

rp=rp/np.max(rp)
rp=rp*otf
rp[0]=rp[1]

from scipy.optimize import curve_fit
def gauss(x,sigma,a,b):
    out=b*np.exp(-0.5*(x/sigma)**2)*np.sin(a*x**2)**2
    return(out)

pars=curve_fit(gauss,x,rp,p0=[sigmaq,np.pi*lam*z2,1])


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


fig=plt.figure('Figure_8_1.svg',figsize=(10,10))

plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaTq)**2),label="C(q) (temporal)")
plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaCq)**2),label="C(q) (spatial)")
plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaSq)**2),label="S(q)")
plt.plot(x*1e-6,otf,label="H(q)")
plt.plot(x*1e-6,np.exp(-0.5*(x/sigmaq)**2)*otf,label="I(q)")
# plt.plot(x*1e-6,np.exp(-0.5*(x/pars[0][0])**2),label="fit")
plt.plot(x*1e-6,rp/pars[0][2],label="simulation")

plt.xlabel(r'spatial radial frequency $[\mu m^{-1}]$')
plt.ylabel('power spectrum [a.u.]')
plt.legend(loc="upper right")

plt.xlim([0,1])
plt.tight_layout()
plt.savefig(path+"Figure_8_1.svg",format="svg")    

print(pars[0][0]/sigmaq)

#%%
rp,sec_data=sector_profile(imft*1,[int(px/2),int(px/2)],[90,5])

rp=running_mean(rp,4)

x=np.linspace(0,0.5*np.sqrt(2)*px/ext,np.size(rp))

lam=2*np.pi/k
sigmaC=lam*(z1+z2)/2/np.pi/sigmay
sigmaS=0.4793*lam*z2/(colloid_radius*2)*np.sqrt(2)

sigmak=k*fwhmk/2.35
sigmaT=np.sqrt(z2/sigmak*np.sqrt(2/np.log(2)))

sigma_total=np.sqrt(1/( 1/sigmaC**2 + 1/sigmaS**2 + 1/sigmaT**2 ))
sigma_total2=np.sqrt(1/( 1/sigmaC**2 + 1/sigmaS**2))

sigmaq=sigma_total/lam/z2/np.sqrt(2)
sigmaq2=sigma_total2/lam/z2/np.sqrt(2)
sigmaCq=sigmaC/lam/z2/np.sqrt(2)
sigmaSq=sigmaS/lam/z2/np.sqrt(2)
sigmaTq=sigmaT/lam/z2/np.sqrt(2)

rp=rp/np.max(rp)

from scipy.optimize import curve_fit
def gauss(x,sigma,a):
    out=a*np.exp(-0.5*(x/sigma)**2)
    return(out)

from scipy.signal import find_peaks

peaks=find_peaks(rp,distance=10)[0][np.array([0,1,2,3])]

pars=curve_fit(gauss,x[peaks],rp[peaks],p0=[sigmaq,1])

fitted_curve=gauss(x,pars[0][0],1)
top=fitted_curve[np.argmax(rp)]


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
plt.plot(x*1e-6,fitted_curve,label="fit")
# plt.plot(x[peaks]*1e-6,rp[peaks]*top,label="fit helper")
plt.plot(x*1e-6,rp*top,label="simulation")

plt.xlabel(r'spatial radial frequency $[\mu m^{-1}]$')
plt.ylabel('power spectrum [a.u.]')
plt.legend(loc="upper right")

plt.xlim([0,1])
plt.tight_layout()
plt.savefig(path+"Figure_8_2.svg",format="svg")  

print(pars[0][0]/sigmaq)
