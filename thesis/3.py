import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.special import kv
from scipy.constants import c,h,e
path='/home/alex/Desktop/'


dat = np.genfromtxt('../../data/absorption/SiO2_2.csv', delimiter=';')
x=dat[:,0]*1e3
omega=x/h*2*np.pi*e


f=omega/2/np.pi
lam=c/f

radius=500e-9

x_fac=2*np.pi*radius/lam
rho=2*x_fac*(np.real(dat[:,2]))
Q=2-4/rho*np.sin(rho)+4/rho**2*(1-np.cos(rho))

#%%

z1=100
z2=20
sigma=5e-6
lam=6e-11
extx=3e-3
electron=0.5109989500015e6
energy=45.6e9
radius=10760
magnet=23.94
deltalambdaoverlambda=0.1
gamma=energy/electron
sigmaCx=lam*(z1+z2)/sigma/2/np.pi
a=1e-6

ex=1.46e-9
ey=2.9e-12

#%%

theta=0

fac=(1/gamma**2+theta**2)
xi=omega*radius/3/c*fac**(3/2)
i=omega**2*fac**2*(kv(2/3,xi)**2 + theta**2/fac * kv(1/3,xi)**2)

i=i/np.max(i)

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


fig=plt.figure('Figure_3.svg',figsize=(20,10))

ax2 = fig.add_subplot(111)

ax2.set_ylabel('exstinction efficiency')  # we already handled the x-label with ax1
ax2.plot(x*1e-3, Q, color="#1f77b4",label='SiO2')
ax2.tick_params(axis='y')
ax2.legend(loc="upper left")
ax2.set_xlabel("energy [keV]")



ax1=ax2.twinx()

ax1.plot(x*1e-3,i, color="#ff7f0e",label="intensity of synchrotron radiation")
ax1.set_ylabel(r"intensity [a.u.]")
ax1.legend(loc="upper right")


plt.tight_layout()
plt.savefig("../../Figure_3.svg",format="svg")




