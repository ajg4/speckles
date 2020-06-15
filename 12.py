import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('//cern.ch/dfs/Users/a/agoetz/Desktop/corona_work_place/helper')
from exp_speckles import *
from the_speckles import the_speckles,radial_profile


#Beam
sigma=5e-6
k=2*np.pi/632e-9
fwhmk=0
numSource=int(2**0)  

#Setup
z1=10
z2=10e-3
fwhmz2=1e-3
   
#Colloids
colloid_diameter=1e-6
numScat=int(2**0)

#Computing   
cores=4

#scattering airy (4s), anom (5s), or mie (1s)
scattertype="airy"

extx=5.4e-3#7.2e-3
pxx=1236#1626
exty=5.4e-3
pxy=1236
zoom=2.6
cores=4

ext=extx/zoom
px=pxx

waterpath='//cern.ch/dfs/Users/a/agoetz/Desktop/corona_work_place/thesis/data/speckles/water/10/'
colloidpath='//cern.ch/dfs/Users/a/agoetz/Desktop/corona_work_place/thesis/data/speckles/speckles/1um,2000us,10000,2x/'

#%%
cavim,cavimft=exp_speckles(colloidpath, pxx,pxy,cores)
cy=radial_profile(cavimft,[int(pxx/2),int(pxy/2)])
freqs=np.fft.fftshift(np.fft.fftfreq(int(np.sqrt((pxx/2)**2+(pxy/2)**2)),exty/pxy/zoom))
x=np.linspace(0,freqs[-1],np.size(cy))

from scipy.optimize import curve_fit
def expd(x,a,b,c):
    return(np.exp(-x/a)*b+c)

cy_cut=cy[2:-300]
x_cut=x[2:-300]

cut_for_fit_x=x_cut[:300]
cut_for_fit_y=cy_cut[:300]

p0=[1e5,4e9,1e9]
par,err=curve_fit(expd,cut_for_fit_x,cut_for_fit_y,p0=p0)

cy_fitted=expd(cut_for_fit_x,*par)

#%%
brenn=75e-3
radius=25.4e-3/2
alpha=np.arctan(radius/brenn)
NA=np.sin(alpha)*1.458
rho0=NA/(2*np.pi/k)
an_x=x_cut
fac=an_x/2/rho0
an_y=2/np.pi*(np.arccos(fac)-fac*np.sqrt(1-fac**2))
an_y=an_y/np.max(an_y)*np.max(cy_fitted)

p0=[1e5,4e9,1e9]
an_par,err=curve_fit(expd,an_x,an_y,p0=p0)


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

fig=plt.figure('Figure_13.svg',figsize=(20,10))
plt.plot(x_cut*1e-6,cy_cut,label="measurement")
plt.plot(cut_for_fit_x*1e-6,cy_fitted,label=r"exponential fit [exp(-q/a)] with a="+str(round(par[0]*1e-6,3))+r" $\mu m^{-1}$")
plt.plot(an_x*1e-6,an_y,label="analytic with NA="+str(round(NA,3)))
# plt.plot(x_cut*1e-6,expd(x_cut,*an_par),label="exponential fit with a="+str(round(an_par[0]*1e-6,3))+r" $\mu m^{-1}$")
plt.xlabel(r"spatial frequency $[\mu m^{-1}]$")
plt.ylabel("power spectrum")
plt.legend()
# plt.yscale("log")
# plt.xscale("log")
plt.legend()


plt.tight_layout()
plt.savefig("Figure_13.svg",format="svg")



#%%
better=4

wavim,wavimft=exp_speckles(waterpath, pxx,pxy,cores)
tavim,tavimft=the_speckles(sigma,k,fwhmk,numSource,z1,z2,fwhmz2,ext,px,colloid_diameter,numScat,scattertype,cores)
tavim2,tavimft2=the_speckles(sigma,k,fwhmk,numSource,z1,z2,0,ext,px,colloid_diameter,numScat,scattertype,cores)
tavim3,tavimft3=the_speckles(sigma,k,fwhmk,numSource,z1,z2,0,ext*better,px*better,colloid_diameter,numScat,scattertype,cores)
tavim4,tavimft4=the_speckles(sigma,k,fwhmk,numSource,z1,z2,fwhmz2,ext*better,px*better,colloid_diameter,numScat,scattertype,cores)

wy=radial_profile(wavimft,[int(pxx/2),int(pxy/2)])
ty=radial_profile(tavimft,[int(pxx/2),int(pxy/2)])
ty2=radial_profile(tavimft2,[int(pxx/2),int(pxy/2)])
ty3=radial_profile(tavimft3,[int(pxx*better/2),int(pxy*better/2)])
ty4=radial_profile(tavimft4,[int(pxx*better/2),int(pxy*better/2)])

freqs3=np.fft.fftshift(np.fft.fftfreq(int(np.sqrt((pxx*better/2)**2+(pxy*better/2)**2)),exty/pxy/zoom))
x3=np.linspace(0,freqs3[-1],np.size(ty4))
#%%

fig=plt.figure('Figure_12.svg',figsize=(20,10))

plt.plot(x,wy,label='water')
plt.plot(x,cy,label='measurement')
plt.plot(x,ty,label='simulation, real cuvette, real resolution')
plt.plot(x,ty2*1e-2,label='simulation, thin cuvette, real resolution')
plt.plot(x3,ty4*1e-4,label='simulation, real cuvette, better resolution')
plt.plot(x3,ty3*1e-6,label='simulation, thin cuvette, better resolution')

plt.xlabel(r'spatial radial frequency $\left[\dfrac{1}{\mu m}\right]$')
plt.ylabel('power spectrum')
plt.yscale('log')
plt.legend()


plt.tight_layout()
plt.savefig("Figure_12.svg",format="svg")


