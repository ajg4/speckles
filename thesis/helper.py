from numpy import *
import numpy as np
from scipy.special import jv
from multiprocessing.pool import ThreadPool

def focused_otf(q,NA,lam):
    """

    Parameters
    ----------
    q : array
        The spatial frequencies.
    NA : float
        The numberical aperture.
    lam : float
        The working wavelength.

    Returns
    -------
    otf : array
        The optical transfer function.

    """
    
    s=lam*q/NA/2
    
    otf=2/np.pi*(np.arccos(s)-s*np.sqrt(1-s**2))
    otf=np.nan_to_num(otf)
    return(otf)


def defocused_otf(dz,zetas,q,NA,lam,n,terms):
    """
    Parameters
    ----------
    dz : float
        The depth of the unfocused area.
    zetas : float
        The number of points within the unfocused area, which are to be evaluated.
    q : array
        The spatial frequencies.
    NA : float
        The numberical aperture.
    lam : float
        The working wavelength.
    n : float
        The refractive index of the area of the defocuse (for example the index of the scintillator.
    terms : int
        Max order of the series approximation.

    Returns
    -------
    otf : array
        The optical transfer function.

    """        
    
    s=lam*q/NA
    
    zeta=np.linspace(dz/zetas,dz,zetas)
    otf=np.zeros(len(s))
    
    def func1(k,a,b):
        temp=(-1)**(k+1) * np.sin(2*k*b)/(2*k) * (jv(2*k-1,a)-jv(2*k+1,a))
        return(temp)
    
    def func2(k,a,b):
        temp=(-1)**(k) * np.sin((2*k+1)*b)/(2*k+1) * (jv(2*k,a)-jv(2*k+2,a))
        return(temp)
    
    def freq_response(s,z):
        w20=NA**2*z/2/n
        a=4*np.pi*w20*s/lam
        b=np.arccos(s/2)
        
        temp1=b*jv(1,a)
        for i in range(1,terms):
            temp1+=func1(i,a,b)
    
        temp2=0
        for i in range(0,terms):
            temp2+=func2(i,a,b)
            
        out=4/np.pi/a*np.cos(a*s/2)*temp1-4/np.pi/a*np.sin(a*s/2)*temp2
        return(out)
    
    
    for i in range(len(s)):
        for j in range(len(zeta)):
            otf[i]+=freq_response(s[i],zeta[j])
        otf[i]=otf[i]/zetas
    return(otf)



def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return((cumsum[N:] - cumsum[:-N]) / np.float128(N))

def sector_profile(data,center,deg_range): 
    """
    

    Parameters
    ----------
    data : 2darray
        data.
    center : list
        X and Y coordinate of center. [X,Y]
    deg_range : list
        Central angle and width of angles. [center,width]

    Returns
    -------

    """
    
#    data=np.reshape(np.random.rand(2500),(50,50))
#    center=[10,5]
#    deg_range=[0,20]
#    deg_range=[90,20]
    
    y,x = np.indices((data.shape))
    
    deg_map=np.arctan((x-center[0])/(y-center[1]))*180/np.pi
    deg_map[center[1]:]+=90
    deg_map[:center[1]]+=90
    
    if(deg_range[0]-deg_range[1]>0):
        deg_map_sector=np.where(deg_map>deg_range[0]-deg_range[1],1,0)*np.where(deg_map<deg_range[0]+deg_range[1],1,0)
    if(deg_range[0]-deg_range[1]<0):
        start=deg_range[0]-deg_range[1]+180
        end=deg_range[0]+deg_range[1]
        deg_map_sector=np.where(deg_map>start,1,0)+np.where(deg_map<end,1,0)        
#    plt.imshow(np.where(deg_map>deg_range[0],1,0))
#    plt.imshow(np.where(deg_map<end,1,0))
#    plt.imshow(deg_map)
#    plt.imshow(deg_map_sector)
    deg_map_sector=np.where(deg_map_sector==1,1,0)
    data*=deg_map_sector
    
    r = np.sqrt((x-center[0])**2 + (y-center[1])**2)
    r = r.astype(np.int)
    tbin = np.bincount(r.ravel(), weights = data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin/nr
    return(radialprofile,data)

def radial_profile(data,center):
    y,x = np.indices((data.shape))
    r = np.sqrt((x-center[0])**2 + (y-center[1])**2)
    r = r.astype(np.int)
    tbin = np.bincount(r.ravel(), weights = data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin/nr
    return(radialprofile) 


def bhmie(lam,radius,refrel,nang,rad):
    """   

    Parameters
    ----------
    lam : float
        Working wavelength.
    radius : float
        Colloid radius.
    refrel : complex
        Refractive index.
    nang : int
        Number of angles to be calculated.
    polyorder : int
        Max order of the fit.
    rad : float
        Maximum angle to be calculated.


    Returns
    -------
    i : array
        Scattered intensity.
    p : array
        Polynom coefficients of the fit.
    s1 : array
        Horizontal pol. amplitude.
    s2 : array
        Vertical pol. amplitude.
    qext : float
        Exstinction coefficient: exstinction cross section / geometrical cross section.
    qsca : float
        Refractive part.
    qback : float
        Diffractive part.
    gsca : float
        ?.

    """
    nmxx=150000
    
    s1_1=zeros(nang,dtype=complex128)
    s1_2=zeros(nang,dtype=complex128)
    s2_1=zeros(nang,dtype=complex128)
    s2_2=zeros(nang,dtype=complex128)
    pi=zeros(nang)
    tau=zeros(nang)
    
    
    # Require NANG>1 in order to calculate scattering intensities
    if (nang < 2):
        nang = 2
    
    pii = 4.*arctan(1.)
    x=2*pii*radius/lam
    dx = x
      
    drefrl = refrel
    y = x*drefrl
    ymod = abs(y)
    
    
    #    Series expansion terminated after NSTOP terms
    #    Logarithmic derivatives calculated from NMX on down
    
    xstop = x + 4.*x**0.3333 + 2.0
    nmx = max(xstop,ymod) + 15.0
    nmx=fix(nmx)
     
    # BTD experiment 91/1/15: add one more term to series and compare resu<s
    #      NMX=AMAX1(XSTOP,YMOD)+16
    # test: compute 7001 wavelen>hs between .0001 and 1000 micron
    # for a=1.0micron SiC grain.  When NMX increased by 1, only a single
    # computed number changed (out of 4*7001) and it only changed by 1/8387
    # conclusion: we are indeed retaining enough terms in series!
    
    nstop = int(xstop)
    
    # if (nmx > nmxx):
    #     print ( "error: nmx > nmxx=%f for |m|x=%f" % ( nmxx, ymod) )
    #     return
    
    dang = rad/ (nang-1)
    

    amu=arange(0.0,nang,1)
    amu=cos(amu*dang)

    pi0=zeros(nang)
    pi1=ones(nang)
    
    # Logarithmic derivative D(J) calculated by downward recurrence
    # beginning with initial value (0.,0.) at J=NMX
    
    nn = int(nmx)-1
    d=zeros(nn+1)
    for n in range(0,nn):
        en = nmx - n
        d[nn-n-1] = (en/y) - (1./ (d[nn-n]+en/y))
    
    #*** Riccati-Bessel functions with real argument X
    #    calculated by upward recurrence
    
    psi0 = cos(dx)
    psi1 = sin(dx)
    chi0 = -sin(dx)
    chi1 = cos(dx)
    xi1 = psi1-chi1*1j
    qsca = 0.
    gsca = 0.
    p = -1
    
    for n in range(0,nstop):
        en = n+1.0
        fn = (2.*en+1.)/(en* (en+1.))
    
    # for given N, PSI  = psi_n        CHI  = chi_n
    #              PSI1 = psi_{n-1}    CHI1 = chi_{n-1}
    #              PSI0 = psi_{n-2}    CHI0 = chi_{n-2}
    # Calculate psi_n and chi_n
        psi = (2.*en-1.)*psi1/dx - psi0
        chi = (2.*en-1.)*chi1/dx - chi0
        xi = psi-chi*1j
    
    #*** Store previous values of AN and BN for use
    #    in computation of g=<cos(theta)>
        if (n > 0):
            an1 = an
            bn1 = bn
    
    #*** Compute AN and BN:
        an = (d[n]/drefrl+en/dx)*psi - psi1
        an = an/ ((d[n]/drefrl+en/dx)*xi-xi1)
        bn = (drefrl*d[n]+en/dx)*psi - psi1
        bn = bn/ ((drefrl*d[n]+en/dx)*xi-xi1)

    #*** Augment sums for Qsca and g=<cos(theta)>
        qsca += (2.*en+1.)* (abs(an)**2+abs(bn)**2)
        gsca += ((2.*en+1.)/ (en* (en+1.)))*( real(an)* real(bn)+imag(an)*imag(bn))
    
        if (n > 0):
            gsca += ((en-1.)* (en+1.)/en)*( real(an1)* real(an)+imag(an1)*imag(an)+real(bn1)* real(bn)+imag(bn1)*imag(bn))
    
    
    #*** Now calculate scattering intensity pattern
    #    First do angles from 0 to 90
        pi=0+pi1    # 0+pi1 because we want a hard copy of the values
        tau=en*amu*pi-(en+1.)*pi0
        s1_1 += fn* (an*pi+bn*tau)
        s2_1 += fn* (an*tau+bn*pi)
    
    #*** Now do angles greater than 90 using PI and TAU from
    #    angles less than 90.
    #    P=1 for N=1,3,...% P=-1 for N=2,4,...
    #   remember that we have to reverse the order of the elements
    #   of the second part of s1 and s2 after the calculation
        p = -p
        s1_2+= fn*p* (an*pi-bn*tau)
        s2_2+= fn*p* (bn*pi-an*tau)

        psi0 = psi1
        psi1 = psi
        chi0 = chi1
        chi1 = chi
        xi1 = psi1-chi1*1j
    
    #*** Compute pi_n for next value of n
    #    For each angle J, compute pi_n+1
    #    from PI = pi_n , PI0 = pi_n-1
        pi1 = ((2.*en+1.)*amu*pi- (en+1.)*pi0)/ en
        pi0 = 0+pi   # 0+pi because we want a hard copy of the values
    
    #*** Have summed sufficient terms.
    #    Now compute QSCA,QEXT,QBACK,and GSCA

    #   we have to reverse the order of the elements of the second part of s1 and s2
    s1=concatenate((s1_1,s1_2[-2::-1]))
    s2=concatenate((s2_1,s2_2[-2::-1]))
    gsca = 2.*gsca/qsca
    qsca = (2./ (dx*dx))*qsca
    qext = (4./ (dx*dx))* real(s1[0])

    # more common definition of the backscattering efficiency,
    # so that the backscattering cross section really
    # has dimension of length squared
    qback = 4*(abs(s1[2*nang-2])/dx)**2    
    #qback = ((abs(s1[2*nang-2])/dx)**2 )/pii  #old form


    return(s1_1,s2_1,qext,qsca,qback,gsca)


def the_speckles(sigmax,sigmay,lam,fwhmk,numSource,z1,z2,fwhmz2,ext,px,colloid_radius,numScat,mie_interpol,cores):
    k=2*np.pi/lam
     
    #replace zeros by small numbers
    if(sigmax==0):
        sigmax=1e-20
    if(sigmay==0):
        sigmay=1e-20
    if(fwhmz2==0):
        fwhmz2=1e-20
    
    
    #scatlist contains the positions of the scatterers
    np.random.seed(1)
    x=(np.random.rand(numScat)*0.8-0.4)*ext
    y=(np.random.rand(numScat)*0.8-0.4)*ext
    #x=(np.random.rand(numScat))*ext
    #y=(np.random.rand(numScat))*ext
    z=np.random.rand(numScat)*fwhmz2
    
    x[0]=0
    y[0]=0
    scatlist=np.rot90([x,y,z])
    
    #beamlist has all the information about the particles
    #one particle emits one specific wavelength, not physical but ok
    beamx=np.random.standard_normal(numSource)*sigmax
    beamy=np.random.standard_normal(numSource)*sigmay
    ks=k*(1+np.random.standard_normal(numSource)*fwhmk)
    
    beamlist=np.rot90([beamx,beamy,ks])
    
       
    #taking a set of beamparticles, looping each over all the scatteres
    def thread(beamlist,ext,px,z1,z2,scatlist,mie_interpol):
        img=np.zeros((px,px))
        for j in range(len(beamlist)):
            k=beamlist[j][2]
    
    #calculating the field at the detection plane     
            Min,Max,n=-ext/2,ext/2,px   
            
            xx=np.linspace(Min-beamlist[j][0],Max-beamlist[j][0],n)
            yy=np.linspace(Min-beamlist[j][1],Max-beamlist[j][1],n)
            yy=np.rot90([yy])
            RR2=np.sqrt(xx**2+yy**2+(z1+z2)**2)
            e2=np.exp(1j*k*RR2)/RR2        
            
            e10s=np.zeros((px,px),dtype='complex128')
            for i in range(len(scatlist)):
                x=scatlist[i][0]
                y=scatlist[i][1]
                z2c=scatlist[i][2]
    
    #calculating the phase of the field at the position of a scatterer
                e10=np.exp(1j*k*np.sqrt(x**2+y**2+(z1-z2c)**2)) 
               
    #calculating the scattering efficiency of the scatterer
    #taking into account its displaced central point according to the divergence of the beam            
                scale=(z1+z2)/(z1-z2c)
                xx=np.linspace(Min-x*scale,Max-x*scale,n)    
                yy=np.linspace(Min-y*scale,Max-y*scale,n)  
                yy=np.rot90([yy])
                
                theta=np.arctan(np.sqrt(xx**2+yy**2)/(z2+z2c))               
                pattern=mie_interpol(theta)
        
                pattern/=np.max(pattern)                 
    
    #calculating the field of the scattering particle and shapping it by the patter of the scattering efficiency
                xx=np.linspace(Min-x,Max-x,n)
                yy=np.linspace(Min-y,Max-y,n)
                yy=np.rot90([yy])
                RR3=np.sqrt(xx**2+yy**2+(z2+z2c)**2)
                
                e10s+=e10*np.exp(1j*k*RR3)/RR3*pattern
    
    #letting interfere the inital field and the scattered field, subtracting their absolute values         
            add=np.abs(e10s+e2)**2-np.abs(e2)**2-np.abs(e10s)**2
            img+=add
        return(img)
    
    #same thing, but multithreading the scatteres instead of the particles (for one central particle)
    def cthread(scatlist,ext,px,z1,z2,k,mie_interpol):
        Min,Max,n=-ext/2,ext/2,px
        e10s=np.zeros((px,px),dtype='complex128')
        for i in range(len(scatlist)):
            x=scatlist[i][0]
            y=scatlist[i][1]
            z2c=scatlist[i][2]  
                             
            e10=np.exp(1j*k*np.sqrt(x**2+y**2+(z1-z2c)**2)) 
            
            scale=(z1+z2)/(z1-z2c)
            xx=np.linspace(Min-x*scale,Max-x*scale,n)    
            yy=np.linspace(Min-y*scale,Max-y*scale,n)  
            yy=np.rot90([yy])
            
            theta=np.arctan(np.sqrt(xx**2+yy**2)/(z2+z2c))               
            pattern=mie_interpol(theta)
    
            pattern/=np.max(pattern)                 
    
            xx=np.linspace(Min-x,Max-x,n)
            yy=np.linspace(Min-y,Max-y,n)
            yy=np.rot90([yy])
            RR3=np.sqrt(xx**2+yy**2+(z2+z2c)**2)
            
            e10s+=e10*np.exp(1j*k*RR3)/RR3*pattern
        return(e10s)      
        
    
    if(numSource!=1):
        if(cores>numSource):
            cores=numSource
        pool=ThreadPool(processes=cores)
        sl=int(numSource/cores)
        res=np.zeros((cores,px,px))
        proc=[]
        for i in range(cores):
            print(i*sl,(i+1)*sl)
            proc.append(pool.apply_async(thread, (beamlist[i*sl:(i+1)*sl],
                                                        ext,
                                                        px,
                                                        z1,
                                                        z2,
                                                        scatlist,
                                                        mie_interpol)))
        for i in range(cores):
            res[i]=proc[i].get()
        
        img=np.sum(res,axis=0)
        img/=np.max(img)    
        
    if(numSource==1):
        if(cores>numScat):
            cores=numScat
        pool=ThreadPool(processes=cores)
        sl=int(numScat/cores)
        res=np.zeros((cores,px,px),dtype='complex128')
        proc=[]
        for i in range(cores):
            proc.append(pool.apply_async(cthread, (scatlist[i*sl:(i+1)*sl],
                                                        ext,
                                                        px,
                                                        z1,
                                                        z2,
                                                        k,
                                                        mie_interpol)))
        for i in range(cores):
            res[i]=proc[i].get()
        
        e10s=np.sum(res,axis=0)
    
    #calculing the initial field of the one particle, interference with all the scattered waves      
        Min,Max,n=-ext/2,ext/2,px   
        xx=np.linspace(Min,Max,n)
        yy=np.linspace(Min,Max,n)
        yy=np.rot90([yy])
        RR2=np.sqrt(xx**2+yy**2+(z1+z2)**2)
        e2=np.exp(1j*k*RR2)/RR2 

        
        img=np.abs(e10s+e2)**2-np.abs(e2)**2-np.abs(e10s)**2
        
        img/=np.max(img)

    
    return(img)


def bhmie2(x,refrel,nang):
# This file is converted from mie.m, see http://atol.ucsd.edu/scatlib/index.htm
# Bohren and Huffman originally published the code in their book on light scattering

# Calculation based on Mie scattering theory  
# input:
#      x      - size parameter = k*radius = 2pi/lambda * radius   
#                   (lambda is the wavelength in the medium around the scatterers)
#      refrel - refraction index (n in complex form for example:  1.5+0.02*i;
#      nang   - number of angles for S1 and S2 function in range from 0 to pi/2
# output:
#        S1, S2 - funtion which correspond to the (complex) phase functions
#        Qext   - extinction efficiency
#        Qsca   - scattering efficiency 
#        Qback  - backscatter efficiency
#        gsca   - asymmetry parameter


    nmxx=150000
    
    s1_1=zeros(nang,dtype=complex128)
    s1_2=zeros(nang,dtype=complex128)
    s2_1=zeros(nang,dtype=complex128)
    s2_2=zeros(nang,dtype=complex128)
    pi=zeros(nang)
    tau=zeros(nang)
    
    if (nang > 1000):
        print ('error: nang > mxnang=1000 in bhmie')
        return
    
    # Require NANG>1 in order to calculate scattering intensities
    if (nang < 2):
        nang = 2
    
    pii = 4.*arctan(1.)
    dx = x
      
    drefrl = refrel
    y = x*drefrl
    ymod = abs(y)
    
    
    #    Series expansion terminated after NSTOP terms
    #    Logarithmic derivatives calculated from NMX on down
    
    xstop = x + 4.*x**0.3333 + 2.0
    nmx = max(xstop,ymod) + 15.0
    nmx=fix(nmx)
     
    # BTD experiment 91/1/15: add one more term to series and compare resu<s
    #      NMX=AMAX1(XSTOP,YMOD)+16
    # test: compute 7001 wavelen>hs between .0001 and 1000 micron
    # for a=1.0micron SiC grain.  When NMX increased by 1, only a single
    # computed number changed (out of 4*7001) and it only changed by 1/8387
    # conclusion: we are indeed retaining enough terms in series!
    
    nstop = int(xstop)
    
    if (nmx > nmxx):
        print ( "error: nmx > nmxx=%f for |m|x=%f" % ( nmxx, ymod) )
        return
    
    dang = .5*pii/ (nang-1)
    

    amu=arange(0.0,nang,1)
    amu=cos(amu*dang)

    pi0=zeros(nang)
    pi1=ones(nang)
    
    # Logarithmic derivative D(J) calculated by downward recurrence
    # beginning with initial value (0.,0.) at J=NMX
    
    nn = int(nmx)-1
    d=zeros(nn+1)
    for n in range(0,nn):
        en = nmx - n
        d[nn-n-1] = (en/y) - (1./ (d[nn-n]+en/y))
    
    #*** Riccati-Bessel functions with real argument X
    #    calculated by upward recurrence
    
    psi0 = cos(dx)
    psi1 = sin(dx)
    chi0 = -sin(dx)
    chi1 = cos(dx)
    xi1 = psi1-chi1*1j
    qsca = 0.
    gsca = 0.
    p = -1
    
    for n in range(0,nstop):
        en = n+1.0
        fn = (2.*en+1.)/(en* (en+1.))
    
    # for given N, PSI  = psi_n        CHI  = chi_n
    #              PSI1 = psi_{n-1}    CHI1 = chi_{n-1}
    #              PSI0 = psi_{n-2}    CHI0 = chi_{n-2}
    # Calculate psi_n and chi_n
        psi = (2.*en-1.)*psi1/dx - psi0
        chi = (2.*en-1.)*chi1/dx - chi0
        xi = psi-chi*1j
    
    #*** Store previous values of AN and BN for use
    #    in computation of g=<cos(theta)>
        if (n > 0):
            an1 = an
            bn1 = bn
    
    #*** Compute AN and BN:
        an = (d[n]/drefrl+en/dx)*psi - psi1
        an = an/ ((d[n]/drefrl+en/dx)*xi-xi1)
        bn = (drefrl*d[n]+en/dx)*psi - psi1
        bn = bn/ ((drefrl*d[n]+en/dx)*xi-xi1)

    #*** Augment sums for Qsca and g=<cos(theta)>
        qsca += (2.*en+1.)* (abs(an)**2+abs(bn)**2)
        gsca += ((2.*en+1.)/ (en* (en+1.)))*( real(an)* real(bn)+imag(an)*imag(bn))
    
        if (n > 0):
            gsca += ((en-1.)* (en+1.)/en)*( real(an1)* real(an)+imag(an1)*imag(an)+real(bn1)* real(bn)+imag(bn1)*imag(bn))
    
    
    #*** Now calculate scattering intensity pattern
    #    First do angles from 0 to 90
        pi=0+pi1    # 0+pi1 because we want a hard copy of the values
        tau=en*amu*pi-(en+1.)*pi0
        s1_1 += fn* (an*pi+bn*tau)
        s2_1 += fn* (an*tau+bn*pi)
    
    #*** Now do angles greater than 90 using PI and TAU from
    #    angles less than 90.
    #    P=1 for N=1,3,...% P=-1 for N=2,4,...
    #   remember that we have to reverse the order of the elements
    #   of the second part of s1 and s2 after the calculation
        p = -p
        s1_2+= fn*p* (an*pi-bn*tau)
        s2_2+= fn*p* (bn*pi-an*tau)

        psi0 = psi1
        psi1 = psi
        chi0 = chi1
        chi1 = chi
        xi1 = psi1-chi1*1j
    
    #*** Compute pi_n for next value of n
    #    For each angle J, compute pi_n+1
    #    from PI = pi_n , PI0 = pi_n-1
        pi1 = ((2.*en+1.)*amu*pi- (en+1.)*pi0)/ en
        pi0 = 0+pi   # 0+pi because we want a hard copy of the values
    
    #*** Have summed sufficient terms.
    #    Now compute QSCA,QEXT,QBACK,and GSCA

    #   we have to reverse the order of the elements of the second part of s1 and s2
    s1=concatenate((s1_1,s1_2[-2::-1]))
    s2=concatenate((s2_1,s2_2[-2::-1]))
    gsca = 2.*gsca/qsca
    qsca = (2./ (dx*dx))*qsca
    qext = (4./ (dx*dx))* real(s1[0])

    # more common definition of the backscattering efficiency,
    # so that the backscattering cross section really
    # has dimension of length squared
    qback = 4*(abs(s1[2*nang-2])/dx)**2    
    #qback = ((abs(s1[2*nang-2])/dx)**2 )/pii  #old form

    return s1,s2,qext,qsca,qback,gsca