# Repository for speckles measurements and simulation at CERN BE-BI-PM

##Acquisition
A script to stream the the output of the Basler camera for live analysis of the speckle pattern. It includes a running mean of the last pictures for subtracting noise from the actual picture, a Fourier transformation, averaged over the last noiseless speckle pattern and a radial average of the power spectrum.

##Thesis
Several scripts to reproduce the plots of my master thesis, including the SRW simulations of the speckles. Further documentation inside.
-1.py ... photon energy distribution of synchrotron radiation
-2.py ... comparision of Mie theory, Anomalous Diffraction and Frauenhofer Diffraction
-3.py ... exstinction coefficient for SiO2 for different photon energies
-4.py ... proof of the applicability of the Van Cittert Zerneke Theorem for an FCC arcdipole with SRW (horizontal)
-5.py ... proof of the applicability of the Van Cittert Zerneke Theorem for an FCC arcdipole with SRW (vertical)
-6.py ... comparison of the phase of synchtrotron radiation with an analytic model (horizontal)
-7.py ... comparison of the phase of synchtrotron radiation with an analytic model (vertical)
-8_0.py ... simulation of speckles with SRW and an analytic scattering model
-8_1.py ... simulation of speckles with an analytic synchrotron radiation model and an analytic scattering model
-9.py ... effect of spatial filtering
-10.py ... classic methods of MTF evaluation
-11.py ... a typical speckle pattern
-12.py ... comparison of classic methods and the speckles method for MTF evaluation
-13.py ... none
-14.py ... the effect of a defocused system (scinitillator) on the MTF in comparision with a perfectly focused system
-15.m ... a CELES Matlab code, for numerically calculating the near field of the scattering at a colloid for xrays.


##SRW
The SRW repository copied from @ochubar. Tested in a conda=4.8.2 python=3.7 environment on Ubuntu 18.04.3.
