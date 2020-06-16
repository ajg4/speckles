import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
from os.path import isfile, join
from multiprocessing.pool import ThreadPool
import gc
from helper import radial_profile, focused_otf
from scipy.optimize import curve_fit

#%%
extx=5.4e-3#7.2e-3
pxx=1236#1626

exty=5.4e-3
pxy=1236

cores=4

lam=632e-9
k=2*np.pi/lam

# brenn=75e-3
# radius=25.4e-3/2
# alpha=np.arctan(radius/brenn)
# NA=np.sin(alpha)*1.458

# NA=0.4
NA=0.25

#%% magnification

def fsin(x,w,a,b,c):
    out=np.sin(x*w+c)*a+b
    return(out)
    
x=np.linspace(0,extx,pxx)   

paths=[]
# paths.append(['/home/alex/Desktop/Lab/20x_mag/6/',60])
# paths.append(['/home/alex/Desktop/Lab/20x_mag/7/',70])
# paths.append(['/home/alex/Desktop/Lab/20x_mag/10/',100])

paths.append(['/home/alex/Desktop/Lab/10x_mag/9/',90])
paths.append(['/home/alex/Desktop/Lab/10x_mag/10/',100])

zoom_list=[]

for i in range(len(paths)):
    files = [f for f in listdir(paths[i][0]) if isfile(join(paths[i][0], f))]
    
    img=np.zeros((len(files),pxy,pxx))
    for j in range(len(files)):
        if(files[i][-4:]=="tiff"):
            im = np.array(Image.open(paths[i][0]+files[j]))
            im=im[:pxx,150:pxx+150]
            img[j]=im
            # print("lp/mm:",paths[i][1]," number:",j)
            
            est_zoom=21
            
            cross=im[:,0]
            lps=extx*paths[i][1]*1e3/est_zoom
            
            est_freq=2*np.pi*lps/extx  
            est_mean=np.mean(cross)
            
           
            p0=[est_freq,20,est_mean,5e3]
            par=curve_fit(fsin,x,cross.astype(np.float),p0=p0)
            # plt.plot(x,fsin(x,*par[0]))
            # plt.plot(x,fsin(x,*p0))
            # plt.plot(x,cross)
            
            freq=par[0][0]
            
            linepair_object=2*np.pi/freq
            linepair_image=1/paths[i][1]*1e-3
            zoom=linepair_object/linepair_image
            zoom_list.append(zoom)

zoom=np.mean(zoom_list)    
print("magnification: ",zoom)

#%% speckles mtf

def furierer(path,files,av,pxx,pxy,coreid):
    imft=np.zeros((pxy,pxx))
    for i in range(len(files)):
        im = np.array(Image.open(path+files[i]))[:pxx,150:pxx+150]
        cim=im-av
        imft+=np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(cim))))**2
        if(coreid==0):
            print("fft ",i,len(files))
    return(imft)
        

def exp_speckles(path, pxx,pxy,cores):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    
    img=np.zeros((pxy,pxx))
    for i in range(len(files)):
        if(files[i][-4:]=="tiff"):
            im = np.array(Image.open(path+files[i]))
            img+=im[:pxx,150:pxx+150]
            print("loaded file ",i,len(files))
    
    cut=len(files)/cores
    cut2=len(files)-int((cut%1)*4)
    waterfiles=files[:cut2]
    
    wavim=img/len(waterfiles)
    
    pool=ThreadPool(processes=cores)
    sl=int(len(files)/cores)    
    imft=np.zeros((sl,pxy,pxx))
    proc=[]
    for i in range(cores):
        proc.append(pool.apply_async(furierer,(path,files[i*sl:(i+1)*sl],wavim,pxy,pxx,i)))
    for i in range(cores):
        imft[i:(i+1)]=proc[i].get()
    
    wavimft=np.sum(imft,axis=0)/len(waterfiles)
    del(imft)
    gc.collect()   
    
    return(wavim,wavimft)

# colloidpath='/home/alex/Desktop/Lab/20x_1um/2/'
# waterpath='/home/alex/Desktop/Lab/20x_1um/w2/'
colloidpath='/home/alex/Desktop/Lab/10x_1um/5/'
waterpath='/home/alex/Desktop/Lab/10x_1um/w5/'

cavim,cavimft=exp_speckles(colloidpath,pxx,pxy,cores)
cy=radial_profile(cavimft,[int(pxx/2),int(pxy/2)])

wavim,wavimft=exp_speckles(waterpath,pxx,pxy,cores)
wy=radial_profile(wavimft,[int(pxx/2),int(pxy/2)])

q=np.linspace(0,pxx*np.sqrt(2)/(extx/zoom)/2,np.size(cy)) #nyquist frequency for radial (sqrt(2)) extension and zoom factor

cy=cy/np.max(cy)
wy=wy/np.max(wy)


cy2=cavimft[int(pxx/2):,int(pxx/2)]
cy2=cy2/np.max(cy2)
q2=np.linspace(0,pxx/(extx/zoom)/2,int(pxx/2))

cy3=cavimft[int(pxx/2),int(pxx/2):]
cy3=cy3/np.max(cy3)
q3=np.linspace(0,pxx/(extx/zoom)/2,int(pxx/2))
    
#%% slant edge

# path='/home/alex/Desktop/Lab/20x_slant/'
path='/home/alex/Desktop/Lab/10x_slant/'

files = [f for f in listdir(path) if isfile(join(path, f))]
sy_list=[]

for j in range(len(files)):
    if(files[j][-4:]=="tiff"):
        print("slant edge ",j)
        im = np.array(Image.open(path+files[j]))
        im=im[:pxx,150:pxx+150]
           
        img=im/np.max(im)
        
        img2=np.where(img < np.mean(img), 0, 1)       
        
        if(np.mean(img2[:,0])>0.5):
            print("flipped ",j)
            img=np.flip(img)
            img2=np.flip(img2)

        
        clist=np.empty((pxy))
        for i in range(pxy):
            clist[i]=img2[i].tolist().index(1)
        
        z = np.polyfit(np.arange(pxy), clist, 1)
        p=np.poly1d(z)
        
        alpha=np.arctan(z[0])
        
        pr=np.empty((2,pxx*pxy))
        k=0
        for i in range(pxx):
            for m in range(pxy):
                l=np.sqrt((i+1)**2+(m+1)**2)
                beta=np.arctan((m+1)/(i+1))
                x=np.cos(alpha+beta)*l
                pr[0,k]=x
                pr[1,k]=img[m,i]
                k+=1
        
        center=clist[0]
        
        indices=np.argsort(pr[0])
        x=pr[0][indices]
        y=pr[1][indices]
        
        xstart=np.argmin(np.abs(x-center+100))
        xfin=np.argmin(np.abs(x-center-100))
        x=x[xstart:xfin]
        y=y[xstart:xfin]
        y/=np.max(y)
        
        bins=500
        
        yh,xh=np.histogram(x,bins=bins,weights=y)
        
        yh/=np.max(yh)
        yhg=np.gradient(yh)
        
        hbins=int(bins/2)
        fft=(np.abs((np.fft.fft(yhg))))[:hbins]
        
        sy_list.append(fft)

xfreq=np.linspace(0,bins*zoom*pxx/(x[-1]-x[0])/extx/2,hbins)

#%% theoretical mtf
N=1/(2*NA)
print("cutoff at ",1/(lam*N))

otf_foc=focused_otf(q,NA,lam)

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

fig=plt.figure('Figure_12.svg',figsize=(20,10))


idx=2
norm=cy[idx]

otf_foc_plot=otf_foc/otf_foc[idx]*norm

plt.plot(q*1e-6,cy,label="measurement colloids")
# plt.plot(q2*1e-6,cy2,label="measurement colloids2")
# plt.plot(q3*1e-6,cy3,label="measurement colloids3")
plt.plot(q*1e-6,wy,label="measurement water")
plt.plot(q*1e-6,otf_foc_plot,label="analytic with NA="+str(round(NA,3)))

idx = (np.abs(q - 0.1*1e6)).argmin()
norm=cy[idx]
for i in range(len(sy_list)):
    cut=80
    sx=xfreq[:cut]
    idx = (np.abs(sx - 0.1*1e6)).argmin()
    sy=sy_list[i][:cut]
    sy=sy/sy[idx]*norm
    plt.plot(sx*1e-6,sy,label="slant edge "+str(i))

plt.xlabel(r"spatial frequency $[\mu m^{-1}]$")
plt.ylabel("power spectrum")
plt.legend()
plt.yscale("log")
# plt.xscale("log")
plt.legend(loc="upper right")


plt.tight_layout()
plt.savefig("Figure_13.svg",format="svg")


