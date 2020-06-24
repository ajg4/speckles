import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
from os.path import isfile, join
from multiprocessing.pool import ThreadPool
import gc
from helper import radial_profile, focused_otf,bhmie,bhmie2
from scipy import interpolate
from scipy.optimize import curve_fit
import pickle as pk
path='/home/alex/Desktop/'

#%%
cores=4
lam=632e-9
k=2*np.pi/lam

#%% magnification

extx=7.2e-3
pxx=1626
exty=5.4e-3
pxy=1236

def fsin(x,w,a,b,c):
    out=np.sin(x*w+c)*a+b
    return(out)
    
x=np.linspace(0,extx,pxx)   

paths=[]

# est_zoom=21
# paths.append([path+'data/mtf/far/20x_mag/6/',60])
# paths.append([path+'data/mtf/far/20x_mag/7/',70])
# paths.append([path+'data/mtf/far/20x_mag/10/',100])

# est_zoom=21
# paths.append([path+'data/mtf/near/20x_mag/9/',90])
# paths.append([path+'data/mtf/near/20x_mag/10/',100])


# est_zoom=11
# paths.append([path+'data/mtf/far/10x_mag/9/',90])
# paths.append([path+'data/mtf/far/10x_mag/10/',100])

est_zoom=11
paths.append([path+'data/mtf/near/10x_mag/6/',60])
paths.append([path+'data/mtf/near/10x_mag/7/',70])


zoom_list=[]
for i in range(len(paths)):
    files = [f for f in listdir(paths[i][0]) if isfile(join(paths[i][0], f))]
    
    img=np.zeros((len(files),pxy,pxx))
    for j in range(len(files)):
        if(files[i][-4:]=="tiff"):
            im = np.array(Image.open(paths[i][0]+files[j]))
            img[j]=im
            # print("lp/mm:",paths[i][1]," number:",j)   
            
            cross=im[0,:]
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

extx=5.4e-3
pxx=1236
exty=5.4e-3
pxy=1236

def furierer(path,files,av,pxx,pxy,coreid):
    imft=np.zeros((pxy,pxx))
    for i in range(len(files)):
        file=Image.open(path+files[i])
        im = np.array(file)[:pxx,150:pxx+150]
        file.close()
        cim=im-av
        imft+=np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(cim))))**2
        # del(im)
        # gc.collect()  
        # if(coreid==0):
            # print("fft ",i,len(files))
    return(imft)
        

def exp_speckles(path, pxx,pxy,cores):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    
    img=np.zeros((pxy,pxx))
    for i in range(len(files)):
        if(files[i][-4:]=="tiff"):
            file=Image.open(path+files[i])
            im = np.array(file)[:pxx,150:pxx+150]
            file.close()
            img+=im
            # del(im)
            # gc.collect() 
            # print("loaded file ",i,len(files))
    
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

# base_path='/home/alex/Desktop/Lab/10x_1um/'
# base_path='/home/alex/Desktop/Lab/10x_05um/'
# base_path='/home/alex/Desktop/Lab/20x_1um/'
# base_path='/home/alex/Desktop/Lab/20x_05um/'
# base_path='/home/alex/Desktop/Lab/20x_1um/w'
# base_path='/home/alex/Desktop/Lab/10x_1um/w'

# dsts=[0]
# dsts=[0,1,2,3,4,5]
# dsts=[0,1,2,3,4,5,6,7,8,9,10,15,18]


# base_path=path+'data/mtf/near/10x_1um/'
base_path=path+'data/mtf/near/10x_05um/'
# base_path=path+'data/mtf/near/20x_1um/'
# base_path=path+'data/mtf/near/20x_05um/'
dsts=[-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.2]
# dsts=[0,0.25,0.5,0.75,1,1.2]

cpaths=[]
for i in range(len(dsts)):
    cpaths.append(base_path+str(dsts[i])+"/")

cyl=[]
for i in range(len(cpaths)):
    print(cpaths[i])

    cavim,cavimft=exp_speckles(cpaths[i],pxx,pxy,cores)
    cy=radial_profile(cavimft,[int(pxx/2),int(pxy/2)])   
    q=np.linspace(1/(extx/zoom),pxx*np.sqrt(2)/(extx/zoom)/2,np.size(cy))  
    cy=cy/np.max(cy)    
    cyl.append(cy)

# for i in range(len(wpaths)): 
#     print(wpaths[i])
    
#     wavim,wavimft=exp_speckles(wpaths[i],pxx,pxy,cores)
#     wy=radial_profile(wavimft,[int(pxx/2),int(pxy/2)])    
#     q=np.linspace(0,pxx*np.sqrt(2)/(extx/zoom)/2,np.size(cy))   
#     wy=wy/np.max(wy)    
#     cyl.append(cy)

# cy2=cavimft[int(pxx/2):,int(pxx/2)]
# cy2=cy2/np.max(cy2)
# q2=np.linspace(0,pxx/(extx/zoom)/2,int(pxx/2))

# cy3=cavimft[int(pxx/2),int(pxx/2):]
# cy3=cy3/np.max(cy3)
# q3=np.linspace(0,pxx/(extx/zoom)/2,int(pxx/2))


# pk.dump([q,cyl,dsts],open(path+'data/mtf/near/pks/10x1umcyl.pk', "wb"))
pk.dump([q,cyl,dsts],open(path+'data/mtf/near/pks/10x05umcyl.pk', "wb"))
# pk.dump([q,cyl,dsts],open(path+'data/mtf/near/pks/20x1umcyl.pk', "wb"))


# pk.dump([q,cyl,dsts],open('/home/alex/Desktop/Lab/pks/10x1umcyl.pk', "wb"))
# pk.dump([q,cyl,dsts],open('/home/alex/Desktop/Lab/pks/10x05umcyl.pk', "wb"))

# pk.dump([q,cyl,dsts],open('/home/alex/Desktop/Lab/pks/20x1umcyl.pk', "wb"))
# pk.dump([q,cyl,dsts],open('/home/alex/Desktop/Lab/pks/20x05umcyl.pk', "wb"))

# pk.dump([q,cyl,dsts],open('/home/alex/Desktop/Lab/pks/10x1umwyl.pk', "wb"))
# pk.dump([q,cyl,dsts],open('/home/alex/Desktop/Lab/pks/20x1umwyl.pk', "wb"))


# %%

extx=5.4e-3
pxx=1236
exty=5.4e-3
pxy=1236

from scipy.special import jv

# brenn=75e-3
# radius=25.4e-3/2
# alpha=np.arctan(radius/brenn)
# NA=np.sin(alpha)*1.458

lam=632e-9
# NA=0.4
NA=0.25

cutoff=NA/lam

lam=632e-9
k=2*np.pi/lam

points=int(pxx*np.sqrt(2)/2)
z2=1e-3
x=np.linspace(0,extx/2/zoom,points)
theta=np.arctan(x/z2)

colloid_radius=0.5e-6
rad=theta[-1]
refr=1.5
a=bhmie(lam,colloid_radius,refr,points,rad)
mie=np.abs(a[0])
mie=mie/np.max(mie)

pat=mie*np.cos(x**2*k/2/z2)
pat2=np.abs(np.fft.fft(pat))[:int(points/2)]
pat3=pat2/np.max(pat2)
xpat=np.linspace(1/(extx/zoom),pxx*np.sqrt(2)/(extx/zoom)/2,int(pxx*np.sqrt(2)/4)) 

#%%

q,cyl,dsts=pk.load(open(path+'data/mtf/near/pks/10x1umcyl.pk', "rb"))
# q2,cyl2,dsts2=pk.load(open(path+'data/mtf/far/pks/10x1umcyl.pk', "rb"))
SIZE = 10

plt.rc('font', size=SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)    # legend fontsize
plt.rc('figure', titlesize=SIZE)  # fontsize of the figure title

fig=plt.figure('Figure_12.svg',figsize=(20,10))


plt.plot(1e-6*xpat,pat3)

norm_pos=0.28
plt.plot([cutoff*1e-6,cutoff*1e-6],[1,0],label="analytic cutoff with NA="+str(round(NA,3)),color="tab:orange")
plt.plot([norm_pos,norm_pos],[1,0],label="normalisation position",color="tab:pink")
idx_c = (np.abs(q - norm_pos*1e6)).argmin()
norm_c=cyl[0][idx_c]


linerange=[9]
linerange2=[3]
# linerange=np.arange(len(cyl))
# linerange2=np.arange(len(cyl2))

for i in linerange:
    norm_temp=cyl[i][idx_c]
    # plt.plot(q*1e-6,cyl[i]/norm_temp*norm_c,label="1um colloids at z="+str(dsts[i]+0.7)+"mm",color="tab:green",alpha=np.linspace(1,0.2,len(cyl))[i])
    plt.plot(q*1e-6,cyl[i]/norm_temp*norm_c,label="1um colloids at z="+str(dsts[i])+"mm")

    # break

# for i in linerange2:
#     norm_temp=cyl2[i][idx_c]
#     # plt.plot(q2*1e-6,cyl2[i]/norm_temp*norm_c,label="0.5um colloids at z="+str(dsts2[i]+0.7)+"mm",color="tab:red",alpha=np.linspace(1,0.2,len(cyl2))[i])
#     plt.plot(q2*1e-6,cyl2[i]/norm_temp*norm_c,label="0.5um colloids at z="+str(dsts2[i])+"mm",color="tab:red",alpha=np.linspace(1,0.2,len(cyl2))[i])
#     # break

# idx_w = (np.abs(q - 0.1*1e6)).argmin()
# norm_w=wyl[0][idx_w]

# for i in linerange:
#     norm_temp=wyl[i][idx_w]
#     plt.plot(q*1e-6,wyl[i]/norm_temp*norm_w,label="water at z="+str(dsts[i]+0.7)+"mm",color="tab:blue",alpha=np.linspace(1,0.2,len(wyl))[i])
#     # break



plt.xlabel(r"spatial frequency $[\mu m^{-1}]$")
plt.ylabel("power spectrum")
plt.yscale("log")
# plt.xscale("log")
plt.xlim(-0.05,1.5)
# plt.ylim(1e-7,1)
plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig(path+"Figure_12.svg",format="svg")


