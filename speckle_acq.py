import numpy as np
import time
from pypylon import pylon
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from fft2d import fft2d

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
#    speckles=avimg[j1]-avimg[j2]
    
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
    
def get_frame(dummy,pxy,pxx):
    camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            img = grabResult.Array
            break
    grabResult.Release()
    camera.StopGrabbing()    
    return(img)

def update(i,imgcount,fftcount,pxx,pxy):
    i=int(i+3)
    print('total: ',(time.time()-b)/i,i)             
    img=get_frame('dummy',pxy,pxx)
    img,speckles,specklesfft,rft=work_frame(img,i,imgcount,fftcount)    
    im1.set_array(img)
    im2.set_array(speckles)
    im3.set_array(specklesfft)
    im4.set_data(np.arange(len(rft)),rft)     
    return(im1,im2,im3,im4,)
    
    
def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

#%%
bintype='analog'

binx=1
biny=binx

pxx=int(1626/binx)
pxy=int(1236/biny)
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

sigmaSx=0.4253*lam*z2/colloid
sigmaSTq=np.sqrt((1/sigmaSx**2/2+2*np.pi**2*sigmaSx**2/lam**2*(1/(z1+z2)-1/z2)**2))



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


if(bintype=='analog'):
    camera.BinningHorizontal = binx
    camera.BinningVertical = biny
#    camera.BinningHorizontalMode='Average'
#    camera.BinningHorizontalMode='Average'
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
scft=np.exp(-0.5*(freqs/sigmaSTq)**2)

im4,=axs[1,1].plot(freqs,scft)
im4.set_label('theoretical model')
im5,=axs[1,1].plot(freqs,rftsim)
im5.set_label('simulation')
im6,=axs[1,1].plot(freqs,scft)
im6.set_label('measured signal')
im7,=axs[1,1].plot(freqs,scft)
im7.set_label('optical transfer function')
axs[1,1].legend()

axs[1,1].set_yscale('log')
axs[1,1].set_title('radial average')
axs[1,1].set_xlabel('radial spatial frequency 1/m')
axs[1,1].set_ylabel('log power spectrum')

global b
  

#rmslist=[]
  
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

                          
        if(bintype=='digital'):
            img=rebin(img,[pxy,pxx])*binx*biny
                       
        img,speckles,specklesfft,rft=work_frame(img,i,imgcount,fftcount)
        speckles=speckles/np.max(speckles)  
        
        im1.set_array(img)
        im2.set_array(speckles)
        im3.set_array(specklesfft)
        im6.set_data(np.linspace(0,freqs[-1],len(scft)),rft)
        im7.set_data(np.linspace(0,freqs[-1],len(scft)),rft/rftsim)
        plt.pause(5)           
        print('total: ',(time.time()-b)/i,i)
#            rmslist.append(np.sum(img))
#            print(np.sum(rmslist)/i,np.std(rmslist),i)
        
#            if(i==1000):
#                break

grabResult.Release()
camera.StopGrabbing()
        
plt.ioff()
plt.show()
camera.Close()



#import pickle as pk
#pk.dump(imglist, open('G:/Users/a/agoetz/Desktop/lab/speckle/run2/5mm/imglist.pk', 'wb'))





