import os
import sys
import glob
import argparse
import signal

import numpy as np
import astropy.units as u
import matplotlib as mpl
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
from astropy.constants import c
from astropy.io import fits
from astropy.time import Time
from dynspectools import read_psrflux, rfifilter_median
from ScintArcs import compute_nutSS, compute_staufD, ParabolicFitter, LineFinder

def signal_handler(sig, frame):
    print('Exiting, writing results in {0}'.format(outfile))
    results.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

"""
create three numpy arrays:
t : 1d over time in seconds
nu : 1d over frequency in Hz
DS : 2d over time and frequency in this order
"""

parser = argparse.ArgumentParser(description='Run interactive arc fitter on directory of dynspecs')
parser.add_argument("-fpath", type=str)
parser.add_argument("-fittype", nargs='?', default='parabola', type=str)
parser.add_argument("-fref", default=1400., type=float)
parser.add_argument("-ylim", default=1.0, type=float)
parser.add_argument("-xlim", default=1.0, type=float)
parser.add_argument("-npad", default=0, type=int)
parser.add_argument("-tmin", default=0., type=float)
parser.add_argument("-vm", default=3., type=float)
parser.add_argument("-outfile", default='curv.txt', type=str)

a = parser.parse_args()
fpath = a.fpath
npad = a.npad
ylim = a.ylim
xlim = a.xlim
fref = a.fref
tmin = a.tmin
fittype = a.fittype
vm = a.vm
outfile = a.outfile
lam = c / (fref*u.MHz)

dspecfiles = np.sort(glob.glob('{0}*fit'.format(fpath) ))
results = open(outfile, 'a')
results.write('name,mjd,freq,bw,tobs,dt,df,betaeta,betaetaerr\n')

for fn in dspecfiles:
    print("Fitting {0}".format(fn))

    a = fits.open(fn)

    DS = a[0].data
    dyn_med, mask = rfifilter_median(DS, xs=4, ys=16, sigma=5)
    DS[mask<0.5] = dyn_med[mask<0.5]

    nchan = a[0].header['NCHANS']
    F0 = a[0].header['FREQ']
    BW = a[0].header['BW']
    Tobs = a[0].header['TTOT']

    dt = (Tobs / DS.shape[0]) * u.s
    df = a[0].header['CDELT1']
    F = (F0 + np.linspace(0, BW, nchan, endpoint=False))*u.MHz
    T0 = a[0].header['MJD']
    T = Time(T0, format='mjd') + np.arange(DS.shape[0]) * dt  
    t = T.unix - T[0].unix

    Funits = F
    nu = F.to(u.Hz).value

    if Tobs < tmin:
        print("Skipping {0}, Tobs of {1} shorter than Tmin of {2}".format(fn, Tobs, tmin))
        continue

    nu0 = fref*1e+6 #or np.mean(nu)
    if npad:
        pad_width= ((npad*DS.shape[0], npad*DS.shape[0]), (0, 0))
        DS = np.pad(DS, pad_width, mode='constant')
        t = np.arange(DS.shape[0])*dt.value

    fD,tau,SS = compute_nutSS(t,nu,DS,nu0=nu0) #or just do FFT power spectrum

    bintau = int(DS.shape[1]//1024)
    bintau = 1
    SS = SS.reshape(SS.shape[0], -1, bintau).mean(-1)
    tau = tau.reshape(-1, bintau).mean(-1)
    if BW < 0:
        SS = SS[:,::-1]
        tau = tau[::-1]

    #range parameters
    xmin = -xlim*max(fD)*1000.
    xmax = xlim*max(fD)*1000.
    ymin = -ylim*max(tau)*1e6
    ymax = ylim*max(tau)*1e6
    

    if fittype == 'parabola':
        logSS = np.log10(SS)
        vmin = np.median(logSS)-0.2
        vmax = vmin + vm
        eta,eta_err = ParabolicFitter(fD,tau,SS,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,vmin=vmin,vmax=vmax)
        if eta == 0:
            continue

    elif fittype == 'line':
        N_stau = 256 #alter to change resolution (lower resolution increases S/N)
        stau, fD, staufD = compute_staufD(fD,tau,SS, N_stau=N_stau)
        vmin = np.median(stau)-0.2
        vmax = vmin + vm
        #range parameters
        zeta_max = 4.0e-9 #covers curvatures down to 0.03 s^3
        zeta,zeta_err = LineFinder(stau, fD, staufD, nu0=nu0, zeta_max=zeta_max, 
                          xmin=xmin,xmax=xmax,ymin=-np.sqrt(ymax),ymax=np.sqrt(ymax))

        if zeta == 0:
            continue
        eta = 1./(2.*nu0*zeta)**2
        eta_err = 2.*eta*zeta_err/zeta

    else:
        print("fittype must be either 'parabola' or 'line' ")

    print("eta={0} +- {1}".format(eta,eta_err))

    lamcurv = (eta*u.s**3* fref*u.MHz / lam).to(u.m**-1 * u.mHz**-2)
    lamcurverr = (eta_err*u.s**3* fref*u.MHz / lam).to(u.m**-1 * u.mHz**-2)
    results.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}\n'.format(
        fn, T[0].mjd, fref, BW, t[-1], dt.value, df, lamcurv.value, lamcurverr.value))
    results.flush()

results.close()
