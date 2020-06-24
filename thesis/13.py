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
path='/home/alex/Desktop/'

#%%
extx=5.4e-3#7.2e-3
pxx=1236#1626

exty=5.4e-3
pxy=1236

cores=4

lam=632e-9
k=2*np.pi/lam



# %%
slx,sly=pk.load(open(path+'data/mtf/run0/pks/20xslantsyl.pk', "rb"))
lpx,lpy=pk.load(open(path+'data/mtf/far/pks/20xlp.pk', "rb"))
cox,coy,dsts=pk.load(open(path+'data/mtf/near/pks/20x05umcyl.pk', "rb"))


lam=632e-9
NA=0.4
# NA=0.25

cutoff=NA/lam

sly_mean=np.mean(sly,axis=0)
sly_plot=syl_mean/focused_otf(slx,NA,lam)
lpy_plot=lpy/focused_otf(lpx,NA,lam)

co_number=1
norm_pos=0.2

idx_c = (np.abs(cox - norm_pos*1e6)).argmin()
norm_c=cyl[co_number][idx_c]

idx_s = (np.abs(slx - norm_pos*1e6)).argmin()
sly_normed=sly_plot/sly_plot[idx_s]*norm_c

idx_l = (np.abs(lpx - norm_pos*1e6)).argmin()
lpy_normed=lpy_plot/lpy_plot[idx_l]*norm_c

SIZE = 30

plt.rc('font', size=SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)    # legend fontsize
plt.rc('figure', titlesize=SIZE)  # fontsize of the figure title

fig=plt.figure('Figure_13.svg',figsize=(20,10))

plt.plot([cutoff*1e-6,cutoff*1e-6],[1,0],label="analytic cutoff with NA="+str(round(NA,3)))
plt.plot([norm_pos,norm_pos],[1,0],label="normalisation position")


plt.plot(cox*1e-6,coy[co_number],label="1um colloids at z="+str(dsts[co_number])+"mm")
plt.plot(slx*1e-6,sly_normed,label="slant edge")
plt.plot(lpx*1e-6,lpy_normed,label="line pairs")
# plt.plot(q2*1e-6,otf_foc,label="theoretical MTF")


plt.xlabel(r"spatial frequency $[\mu m^{-1}]$")
plt.ylabel("power spectrum")
plt.yscale("log")
# plt.xscale("log")
plt.xlim(-0.05,0.8)
plt.ylim(4e-4,2)
plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig(path+"Figure_13.svg",format="svg")


