import numpy as np
from scipy.special import kv
from scipy.constants import c,h,epsilon_0,e
import matplotlib.pyplot as plt


z1=100
z2=20
sigma=60e-6
lam=10e-11
extx=3e-3
electron=0.5109989500015e6
energy=45.6e9
radius=10760
magnet=23.94
umfang=97756
deltalambdaoverlambda=0.1
gamma=energy/electron
sigmaCx=lam*(z1+z2)/sigma/2/np.pi
v=c*np.sqrt(1-1/gamma)
print(sigmaCx)

ex=1.46e-9
ey=2.9e-12

#%%
#engery per wavelength for fixed angle
theta=0
energy=45.6e9
gamma=energy/electron

ev=np.logspace(1.5,6.5,100000,base=10)

omega=ev/h*2*np.pi*e
fac=(1/gamma**2+theta**2)
xi=omega*radius/3/c*fac**(3/2)


ih1=omega*fac*kv(2/3,xi)
iv1=omega*np.sqrt(fac)*theta*kv(1/3,xi)
i1=ih1+iv1

# i1=i1/np.sum(i1)*13


# mc=np.ones(len(i1))
# dee=0.5e-3
# ar=ev<20000+20000*dee
# ar2= ev>20000-20000*dee
# mc=np.where(ar,1,0)*np.where(ar2,1,0)


theta=0
energy=80e9
gamma=energy/electron


omega=ev/h*2*np.pi*e
fac=(1/gamma**2+theta**2)
xi=omega*radius/3/c*fac**(3/2)


ih2=omega*fac*kv(2/3,xi)
iv2=omega*np.sqrt(fac)*theta*kv(1/3,xi)
i2=ih2+iv2



# print(np.sum(i1))
# print(np.sum(i1*mc)*1e3)
#%%
SMALL_SIZE = 30
MEDIUM_SIZE = 30
BIGGER_SIZE = 30


# plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['font.family'] = 'STIXGeneral'
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


plt.figure('Figure_1.svg',figsize=(20,10))

plt.plot(ev,i1,label=r'45.6 GeV')
# plt.plot(ev,i1*mc,label=r'45.6 GeV')
plt.plot(ev,i2,label=r'80 GeV')
plt.xscale('log')
# plt.yscale("log")
plt.xlabel("energy of synchrotron radiation [eV]")
plt.ylabel(r"intensity [a.u.]")
plt.legend()


plt.tight_layout()
plt.savefig("Figure_1.svg",format="svg")






