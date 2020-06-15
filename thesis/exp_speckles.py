from PIL import Image
from os import listdir
from os.path import isfile, join
from multiprocessing.pool import ThreadPool
import gc
import sys
import numpy as np

def furierer(files,av,pxx,pxy,coreid):
    imft=np.zeros((len(files),pxy,pxx))
    for i in range(len(files)):
        cim=files[i]-av
        imft[i]=np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(cim))))**2
        if(coreid==0):
            print("fft ",i,len(files))
    return(imft)
        

def exp_speckles(path, pxx,pxy,cores):
    waterfiles = [f for f in listdir(path) if isfile(join(path, f))]
    waterfiles=waterfiles[:-1]
    
    img=np.zeros((len(waterfiles),pxy,pxx))
    for i in range(len(waterfiles)):
        if(waterfiles[i][-3:]=="tif"):
            im = np.array(Image.open(path+waterfiles[i]))
            im=im[:pxx,150:pxx+150]
            img[i]=im
            print("loaded file ",i,len(waterfiles))
    
    cut=len(waterfiles)/cores
    cut2=len(waterfiles)-int((cut%1)*4)
    waterfiles=waterfiles[:cut2]
    
    wavim=np.sum(img,axis=0)/len(waterfiles)
    
    imft=np.zeros((len(waterfiles),pxy,pxx))
    
    pool=ThreadPool(processes=cores)
    sl=int(len(waterfiles)/cores)
    proc=[]
    for i in range(cores):
        proc.append(pool.apply_async(furierer,(img[i*sl:(i+1)*sl],wavim,pxy,pxx,i)))
    for i in range(cores):
        imft[i*sl:(i+1)*sl]=proc[i].get()
    
    wavimft=np.sum(imft,axis=0)/len(waterfiles)
    del(imft)
    gc.collect()
    
    
    return(wavim,wavimft)