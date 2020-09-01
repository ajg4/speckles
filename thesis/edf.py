import EdfFile
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from helper import radial_profile, sector_profile


path='/home/alex/Desktop/__myprojects/speckles/Measurements_Raw_Data/ALBA/NCD/2018_10_16/F1_WD200_ET50ms_Col1_2018-10-15_20h02m59/'
# path='/home/alex/Desktop/__myprojects/speckles/Measurements_Raw_Data/ALBA/NCD/2018_10_16/F2_WD200_ET50ms_Col1_2018-10-15_20h16m57/'
# path='/home/alex/Desktop/__myprojects/speckles/Measurements_Raw_Data/ALBA/NCD/2018_10_16/F1_WD200_ET50ms_Water_2018-10-15_20h06m54/'
# path='/home/alex/Desktop/__myprojects/speckles/Measurements_Raw_Data/ALBA/NCD/2018_10_16/F2_WD200_ET50ms_Water_2018-10-15_20h14m24/'


distances=np.array([200])

so=np.argsort(distances)

distances=distances[so]

pxx=1296
pxy=966
ext=160e-6
#
#%%

   
files = [f for f in listdir(path) if isfile(join(path, f))]
img0=np.zeros((pxy,pxx))
for j in range(len(files)):
    img = EdfFile.EdfFile(path+files[j]).GetData(0).astype(np.float)
    img0+=img  
img0=img0/len(files)

show=img-img0
show2=show[:,165:-165]

path='/home/alex/Desktop/speckles/thesis/'
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


fig=plt.figure('C_speckles.svg',figsize=(10,10))

extent=np.array([-ext/2,ext/2,-ext/2,ext/2])*1e6
plt.imshow(show2,extent=extent)
plt.xlabel(r"x $[\mu m]$")
plt.ylabel(r"y $[\mu m]$")
plt.xlim(-80,80)
plt.ylim(-80,80)


plt.tight_layout()
plt.savefig(path+"C_speckles.svg",format="svg") 

