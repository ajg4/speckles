import numpy as np
import time
from pypylon import pylon
import matplotlib.pyplot as plt
import pickle as pk

def radial_profile(data,center):
    y,x = np.indices((data.shape))
    r = np.sqrt((x-center[0])**2 + (y-center[1])**2)
    r = r.astype(np.int)
    tbin = np.bincount(r.ravel(), weights = data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin/nr
    return(radialprofile)
       
def work_frame(img,i,imgcount,fftcount):
    j1=i%imgcount
    j2=(i+1)%imgcount    
    l1=i%fftcount
    l2=(i+1)%fftcount
       
    avimg[j1]=img    
    avimg2[0]=avimg2[0]+avimg[j1]-avimg[j2]      
    
    speckles=img-avimg2[0]/imgcount
    
    speckles=np.fft.fftshift(speckles)
    temp=np.fft.fft2(speckles)
    avfft[l1]=np.abs(np.fft.fftshift(temp))**2
    
    avfft2[0]=avfft2[0]+avfft[l1]-avfft[l2]  
    
    rft=radial_profile(avfft2[0],[int(pxx/2),int(pxy/2)])    
    rft=rft/np.max(rft)    
    
    specklesfft=avfft2[0]
    specklesfft=np.abs(np.log(specklesfft))
    specklesfft=specklesfft/np.max(specklesfft) 
         
    return(img,speckles,specklesfft,rft)
    
#%%
path="./"

pxx=int(1626)
pxy=int(1236)
pxx=pxy #for cropping to a square

imgcount=10
fftcount=10

zoom=2.7
z1=10
z2=6e-3
lam=632e-9
k=2*np.pi/lam
colloid=1e-6

extx=7.2e-3/zoom
exty=5.4e-3/zoom
extx=exty #for cropping to a square

ext=np.sqrt((extx)**2+(exty)**2)

#%%
global avimg
global avfft
global avimg2
global avfft2

avimg=np.zeros((imgcount,pxy,pxx))
avfft=np.zeros((fftcount,pxy,pxx))
avimg2=np.zeros((1,pxy,pxx))
avfft2=np.zeros((1,pxy,pxx))
    
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()    
camera.ExposureTimeAbs.SetValue(25)
camera.AcquisitionFrameRateEnable.SetValue(True)
camera.AcquisitionFrameRateAbs.SetValue(10)
camera.BinningHorizontal = 1
camera.BinningVertical = 1
camera.Gamma.SetValue(1)
camera.BlackLevelRaw.SetValue(0)
camera.GainRaw.SetValue(230)

#%%
fig, axs = plt.subplots(2,2)  
  
im1=axs[0,0].imshow(np.random.rand(pxy,pxx)*255)
axs[0,0].set_title('single frame')

im2=axs[0,1].imshow(np.random.rand(pxy,pxx))
axs[0,1].set_title('averaged frames')

im3=axs[1,0].imshow(np.random.rand(pxy,pxx))
axs[1,0].set_title('averaged fft')

pxxs=len(radial_profile(np.random.rand(pxy,pxx),[int(pxy/2),int(pxx/2)]))*2

freqs=(np.fft.fftshift(np.fft.fftfreq(pxxs,exty/pxy/zoom*np.sqrt(2)))*2*np.pi)[int(pxxs/2):]

im6,=axs[1,1].plot(freqs,np.random.rand(len(freqs)))
im6.set_label('measured signal')

axs[1,1].set_title('radial average')
axs[1,1].set_xlabel('radial spatial frequency 1/m')
axs[1,1].set_ylabel('log power spectrum')

global b
imglist=[]
  
plt.ion()
i=0      
camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
b=time.time()
while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded():
        i+=1
        img = grabResult.Array
        img=img[:,195:-195]
                       
        img,speckles,specklesfft,rft=work_frame(img,i,imgcount,fftcount)
        speckles=speckles/np.max(speckles)  
        
        im1.set_array(img)
        im2.set_array(speckles)
        im3.set_array(specklesfft)
        im6.set_data(np.linspace(0,freqs[-1],len(freqs)),rft)
        plt.pause(5)           
        print('total: ',(time.time()-b)/i,i)
        # imgslist.append(img)

grabResult.Release()
camera.StopGrabbing()
        
plt.ioff()
plt.show()
camera.Close()

#pk.dump(imglist, open(path+'imglist.pk', 'wb'))





