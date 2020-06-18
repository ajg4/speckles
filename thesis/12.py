import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
from os.path import isfile, join
from multiprocessing.pool import ThreadPool
import gc
from helper import radial_profile, focused_otf
from scipy.optimize import curve_fit
import pickle as pk

#%%
extx=5.4e-3#7.2e-3
pxx=1236#1626

exty=5.4e-3
pxy=1236

cores=4

lam=632e-9
k=2*np.pi/lam

#%% magnification

def fsin(x,w,a,b,c):
    out=np.sin(x*w+c)*a+b
    return(out)
    
x=np.linspace(0,extx,pxx)   

paths=[]

# est_zoom=21
# paths.append(['/home/alex/Desktop/Lab/20x_mag/6/',60])
# paths.append(['/home/alex/Desktop/Lab/20x_mag/7/',70])
# paths.append(['/home/alex/Desktop/Lab/20x_mag/10/',100])

est_zoom=11
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
        # if(coreid==0):
            # print("fft ",i,len(files))
    return(imft)
        

def exp_speckles(path, pxx,pxy,cores):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    
    img=np.zeros((pxy,pxx))
    for i in range(len(files)):
        if(files[i][-4:]=="tiff"):
            im = np.array(Image.open(path+files[i]))
            img+=im[:pxx,150:pxx+150]
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
base_path='/home/alex/Desktop/Lab/10x_05um/'
# base_path='/home/alex/Desktop/Lab/20x_1um/'
# base_path='/home/alex/Desktop/Lab/20x_05um/'
# base_path='/home/alex/Desktop/Lab/20x_1um/w'
# base_path='/home/alex/Desktop/Lab/10x_1um/w'

# dsts=[0]
# dsts=[0,1,2,3,4,5]
dsts=[0,1,2,3,4,5,6,7,8,9,10,15,18]
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


# pk.dump([q,cyl,dsts],open('/home/alex/Desktop/Lab/pks/10x1umcyl.pk', "wb"))
pk.dump([q,cyl,dsts],open('/home/alex/Desktop/Lab/pks/10x05umcyl.pk', "wb"))

# pk.dump([q,cyl,dsts],open('/home/alex/Desktop/Lab/pks/20x1umcyl.pk', "wb"))
# pk.dump([q,cyl,dsts],open('/home/alex/Desktop/Lab/pks/20x05umcyl.pk', "wb"))

# pk.dump([q,cyl,dsts],open('/home/alex/Desktop/Lab/pks/10x1umwyl.pk', "wb"))
# pk.dump([q,cyl,dsts],open('/home/alex/Desktop/Lab/pks/20x1umwyl.pk', "wb"))

#%% line pairs
samples=3

pxx=1626
pxy=1236

dat=np.empty((21,samples,pxy,pxx))
fdat=np.empty((21,samples,pxy,pxx))

path='/home/alex/Desktop/Lab/20x_lp/'

for i in range(1,21):
    for j in range(samples):
        im = np.array(Image.open(path+str(i*10)+"_"+str(j)+".tif"))
        dat[i,j]=im
        
lp_y=np.empty((21))

for i in range(1,21):
    temp=0
    for j in range(samples):
        mean=np.std(dat[i,j])
        temp+=mean
    temp=temp/3
    lp_y[i]=temp
lp_y=lp_y[1:]

lp_x=np.empty((21))
for i in range(21):
    lp_x[i]=i*10*1e3
lp_x=lp_x[1:]

lp_y=lp_y/np.max(lp_y)

pk.dump([lp_x,lp_y],open('/home/alex/Desktop/Lab/pks/20xlp.pk', "wb"))
    
#%% slant edge

path='/home/alex/Desktop/Lab/10x_slant/'
# path='/home/alex/Desktop/Lab/20x_slant/'

files = [f for f in listdir(path) if isfile(join(path, f))]
syl=[]

for j in range(len(files)):
    if(files[j][-4:]=="tiff"):
        print("slant edge ",j)
        im = np.array(Image.open(path+files[j]))
        im=im[:pxx,150:pxx+150]
           
        img=im/np.max(im)
        
        img2=np.where(img < np.mean(img), 0, 1)       
        
        if(np.mean(img2[:,0])>0.5):
            # print("flipped ",j)
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
        
        syl.append(fft)

xfreq=np.linspace(1/((x[-1]-x[0])*extx/pxx/zoom),bins*zoom*pxx/(x[-1]-x[0])/extx/2,hbins)

pk.dump([xfreq,syl],open('/home/alex/Desktop/Lab/pks/10xslantsyl.pk', "wb"))
# pk.dump([xfreq,syl],open('/home/alex/Desktop/Lab/pks/20xslantsyl.pk', "wb"))

# %%
# q,cyl,dsts=pk.load(open('/home/alex/Desktop/Lab/pks/10x1umcyl.pk', "rb"))
# q2,cyl2,dsts2=pk.load(open('/home/alex/Desktop/Lab/pks/10x05umcyl.pk', "rb"))
# xfreq,syl=pk.load(open('/home/alex/Desktop/Lab/pks/10xslantsyl.pk', "rb"))

# q,wyl,dsts=pk.load(open('/home/alex/Desktop/Lab/pks/10x1umwyl.pk', "rb"))

q,cyl,dsts=pk.load(open('/home/alex/Desktop/Lab/pks/20x1umcyl.pk', "rb"))
q2,cyl2,dsts2=pk.load(open('/home/alex/Desktop/Lab/pks/20x05umcyl.pk', "rb"))
xfreq,syl=pk.load(open('/home/alex/Desktop/Lab/pks/20xslantsyl.pk', "rb"))

q,wyl,dsts=pk.load(open('/home/alex/Desktop/Lab/pks/20x1umwyl.pk', "rb"))

lp_x,lp_y=pk.load(open('/home/alex/Desktop/Lab/pks/20xlp.pk', "rb"))

# brenn=75e-3
# radius=25.4e-3/2
# alpha=np.arctan(radius/brenn)
# NA=np.sin(alpha)*1.458

NA=0.4
# NA=0.25

cutoff=NA/lam

SIZE = 30

plt.rc('font', size=SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)    # legend fontsize
plt.rc('figure', titlesize=SIZE)  # fontsize of the figure title

fig=plt.figure('Figure_12.svg',figsize=(20,10))

plt.plot([cutoff*1e-6,cutoff*1e-6],[1,0],label="analytic cutoff with NA="+str(round(NA,3)),color="tab:orange")
plt.plot([0.15,0.15],[1,0],label="normalisation position",color="tab:pink")

idx_c = (np.abs(q - 0.15*1e6)).argmin()
norm_c=cyl[0][idx_c]


linerange=[1]
linerange2=[1]
# linerange=np.arange(len(cyl))
# linerange2=np.arange(len(cyl2))

for i in linerange:
    norm_temp=cyl[i][idx_c]
    plt.plot(q*1e-6,cyl[i]/norm_temp*norm_c,label="1um colloids at z="+str(dsts[i]+0.7)+"mm",color="tab:green",alpha=np.linspace(1,0.2,len(cyl))[i])
    # break

for i in linerange2:
    norm_temp=cyl2[i][idx_c]
    plt.plot(q2*1e-6,cyl2[i]/norm_temp*norm_c,label="0.5um colloids at z="+str(dsts2[i]+0.7)+"mm",color="tab:red",alpha=np.linspace(1,0.2,len(cyl2))[i])
    # break

idx_w = (np.abs(q - 0.1*1e6)).argmin()
norm_w=wyl[0][idx_w]

for i in linerange:
    norm_temp=wyl[i][idx_w]
    plt.plot(q*1e-6,wyl[i]/norm_temp*norm_w,label="water at z="+str(dsts[i]+0.7)+"mm",color="tab:blue",alpha=np.linspace(1,0.2,len(wyl))[i])
    # break

idx_s = (np.abs(xfreq - 0.15*1e6)).argmin()
norm_s=syl[0][idx_s]
sy_tot=[]
for i in range(len(syl)):
    cut=80
    sx=xfreq[:cut]
    idx = (np.abs(sx - 0.15*1e6)).argmin()
    sy=syl[i][:cut]
    sy=sy/sy[idx_s]*norm_s
    sy_tot.append(sy)
    # plt.plot(sx*1e-6,sy)

sy_plot=np.mean(sy_tot,axis=0)
sy_plot=sy_plot/sy_plot[idx_s]*norm_c


plt.plot(sx*1e-6,sy_plot,label="slanted edge",color="tab:purple")

idx_lp = (np.abs(lp_x - 0.15*1e6)).argmin()
norm_lp=lp_y[idx_lp]
lp_y_plot=lp_y/norm_lp*norm_c

plt.plot(lp_x*1e-6,lp_y_plot,label="line pairs",color="black")

plt.xlabel(r"spatial frequency $[\mu m^{-1}]$")
plt.ylabel("power spectrum")
plt.yscale("log")
# plt.xscale("log")
plt.xlim(-0.05,1.5)
plt.ylim(1e-7,1)
plt.legend(loc="upper right")

plt.tight_layout()
# plt.savefig("Figure_12.svg",format="svg")


