import EdfFile
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from helper import radial_profile, sector_profile

path0='/home/alex/Desktop/alba_data/K020/'

path=['WD500mm_ET50ms_CouplingSmallest_2019-10-22_00h35m58/',
 'WD860mm_ET50ms_CouplingSmallest_2019-10-22_00h47m52/',
 'WD100mm_ET50ms_CouplingSmallest_2019-10-22_00h27m42/',
 'WD300mm_ET50ms_CouplingSmallest_2019-10-22_00h32m29/',
 'WD200mm_ET50ms_CouplingSmallest_2019-10-22_00h30m56/',
 'WD960mm_ET50ms_CouplingSmallest_2019-10-22_00h43m36/',
 'WD100mm_ET50ms_CouplingSmallest_2019-10-22_00h27m42/',
 'WD760mm_ET50ms_CouplingSmallest_2019-10-22_00h49m33/',
 'WD1060mm_ET50ms_CouplingSmallest_2019-10-22_00h41m46/',
 'WD660mm_ET50ms_CouplingSmallest_2019-10-22_00h51m37/',
 'WD400mm_ET50ms_CouplingSmallest_2019-10-22_00h34m10/']

distances=np.array([500,860,100,300,200,960,100,760,1060,660,400])

so=np.argsort(distances)

distances=distances[so]

pxx=1296
pxy=966

imfts=[]


for i in range(len(path)):    
    files = [f for f in listdir(path0+path[so[i]]) if isfile(join(path0+path[so[i]], f))]
    img0=np.zeros((pxy,pxx))
    for j in range(len(files)):
        img = EdfFile.EdfFile(path0+path[so[i]]+files[j]).GetData(0).astype(np.float)
        img0+=img  
    img0=img0/len(files) 
    
    imft0=np.zeros((pxy,pxx))
    for j in range(len(files)):
        img = EdfFile.EdfFile(path0+path[so[i]]+files[j]).GetData(0).astype(np.float)
        imft=np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img-img0))))**2
        imft0+=imft     
    imft0=imft0/len(files)         

    imfts.append(imft0)
    # plt.figure(distances[i])
    # plt.imshow(np.log(imft0))
    
#%%
rps=[]
ext=1.6e-4
x=np.linspace(0,0.5*pxy/ext,pxy)

for i in range(len(imfts)):
    rp,sec_data=sector_profile(imfts[i][:,165:-165]*1,[int(pxy/2),int(pxy/2)],[90,10])
    rp=rp/np.max(rp)
    rps.append(rp)
    plt.plot(rp,label=distances[i])
plt.legend()
# plt.xlim(0,1)
plt.yscale("log")
#%%

a=imfts[-1][:,165:-165]

# plt.imshow(np.log(a))

rp,sec_data=sector_profile(a*1,[int(pxy/2),int(pxy/2)],[0,20])

# rp[1]=rp[2]

plt.plot(rp)
plt.yscale("log")
#%%

for i in range(len(imfts)):
    plt.figure(distances[i])
    plt.imshow(np.log(imfts[i]))

