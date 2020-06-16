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

NA=0.4


#%% magnification

def fsin(x,w,a,b,c):
    out=np.sin(x*w+c)*a+b
    return(out)
    
x=np.linspace(0,extx,pxx)   

paths=[]
paths.append(['/home/alex/Desktop/Lab/20x_mag/6/',60])
paths.append(['/home/alex/Desktop/Lab/20x_mag/7/',70])
paths.append(['/home/alex/Desktop/Lab/20x_mag/10/',100])


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

def furierer(files,av,pxx,pxy,coreid):
    imft=np.zeros((len(files),pxy,pxx))
    for i in range(len(files)):
        cim=files[i]-av
        imft[i]=np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(cim))))**2
        if(coreid==0):
            print("fft ",i,len(files))
    out=np.sum(imft,axis=0)
    return(out)
        

def exp_speckles(path, pxx,pxy,cores):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    # files=files[:-1]
    
    img=np.zeros((len(files),pxy,pxx))
    for i in range(len(files)):
        if(files[i][-4:]=="tiff"):
            im = np.array(Image.open(path+files[i]))
            im=im[:pxx,150:pxx+150]
            img[i]=im
            print("loaded file ",i,len(files))
    
    cut=len(files)/cores
    cut2=len(files)-int((cut%1)*4)
    waterfiles=files[:cut2]
    
    wavim=np.sum(img,axis=0)/len(waterfiles)
    
    pool=ThreadPool(processes=cores)
    sl=int(len(files)/cores)    
    imft=np.zeros((sl,pxy,pxx))
    proc=[]
    for i in range(cores):
        proc.append(pool.apply_async(furierer,(img[i*sl:(i+1)*sl],wavim,pxy,pxx,i)))
    for i in range(cores):
        imft[i:(i+1)]=proc[i].get()
    
    wavimft=np.sum(imft,axis=0)/len(waterfiles)
    del(imft)
    gc.collect()   
    
    return(wavim,wavimft)

colloidpath='/home/alex/Desktop/Lab/20x_1um/0/'
waterpath='/home/alex/Desktop/Lab/20x_1um/w0/'

cavim,cavimft=exp_speckles(colloidpath,pxx,pxy,cores)
cy=radial_profile(cavimft,[int(pxx/2),int(pxy/2)])

wavim,wavimft=exp_speckles(waterpath,pxx,pxy,cores)
wy=radial_profile(wavimft,[int(pxx/2),int(pxy/2)])

freqs=np.fft.fftshift(np.fft.fftfreq(int(np.sqrt((pxx/2)**2+(pxy/2)**2)),exty/pxy/zoom))
q=np.linspace(0,freqs[-1],np.size(cy))

y=cy-wy
y=y/np.max(y)


# from scipy.optimize import curve_fit
# def expd(x,a,b,c):
#     return(np.exp(-x/a)*b+c)

# cy_cut=cy[2:-300]
# x_cut=x[2:-300]

# cut_for_fit_x=x_cut[:300]
# cut_for_fit_y=cy_cut[:300]

# p0=[1e5,4e9,1e9]
# par,err=curve_fit(expd,cut_for_fit_x,cut_for_fit_y,p0=p0)

# cy_fitted=expd(cut_for_fit_x,*par)

    
#%% slant edge

path='/home/alex/Desktop/Lab/20x_slant/'

files = [f for f in listdir(path) if isfile(join(path, f))]

img=np.zeros((len(files),pxy,pxx))
for j in range(len(files)):
    if(files[j][-4:]=="tiff"):
        im = np.array(Image.open(path+files[j]))
        im=im[:pxx,150:pxx+150]
        img[j]=im
    
im=im/np.max(im)

img2=np.where(im < np.mean(im), 0, 1)

clist=np.empty((pxy))
for i in range(pxy):
    clist[i]=img2[i].tolist().index(img[0,-1])

z = np.polyfit(np.arange(pxy), clist, 1)
p=np.poly1d(z)

alpha=np.arctan(z[0])

pr=np.empty((2,pxx*pxy))
k=0
for i in range(pxx):
    for j in range(pxy):
        l=np.sqrt((i+1)**2+(j+1)**2)
        beta=np.arctan((j+1)/(i+1))
        x=np.cos(alpha+beta)*l
        pr[0,k]=x
        pr[1,k]=img[j,i]
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


#def func(x,u,s,d,e):
#    fermi=e/(np.exp((x-u)/s)+1)+d
#    return(fermi)  
#
#popt, pcov = curve_fit(func, x, y,p0=[center,1,0.1,1])
#plt.scatter(x,y,s=0.5)
#plt.plot(x,func(x,*popt),color='r')

yh,xh=np.histogram(x,bins=500,weights=y)
yh/=np.max(yh)
plt.plot(xh[1:],yh,color='g')
yhg=np.gradient(yh)


xfreq=np.fft.fftfreq(500,(x[-1]/x[0])/pxx*extx*23.56/500)*2*np.pi

fft=np.abs((np.fft.fft(yhg)))**2

cut=40

plt.figure("manual projection")
plt.plot(xfreq[:cut],fft[:cut])


#%% theoretical mtf

# q=x_cut
otf_foc=focused_otf(q,NA,lam)
# otf_foc=otf_foc/np.max(otf_foc)*np.max(cy_fitted)


#%% theoretical mtf defocused
# dz=1e-3
# zetas=100
# n=1
# terms=10

# otf=defocused_otf(dz,zetas,q,NA,lam,n,terms)

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
plt.plot(q*1e-6,y,label="measurement")
# plt.plot(cut_for_fit_x*1e-6,cy_fitted,label=r"exponential fit [exp(-q/a)] with a="+str(round(par[0]*1e-6,3))+r" $\mu m^{-1}$")
plt.plot(q*1e-6,otf_foc,label="focused with NA="+str(round(NA,3)))
# plt.plot(q*1e-6,otf,label="defocused with NA="+str(round(NA,3)))
# plt.plot(x_cut*1e-6,expd(x_cut,*an_par),label="exponential fit with a="+str(round(an_par[0]*1e-6,3))+r" $\mu m^{-1}$")
plt.xlabel(r"spatial frequency $[\mu m^{-1}]$")
plt.ylabel("power spectrum")
plt.legend()
# plt.yscale("log")
# plt.xscale("log")
plt.legend()


plt.tight_layout()
plt.savefig("Figure_13.svg",format="svg")


