import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib.dates import date2num

import xarray as xr

from scipy.signal import welch

def dispNewtonTH(f, dep):
    """
    Purpose: inverts the linear dispersion relation (2*pi*f)^2=g*k*tanh(k*dep) to get 
    ----------
    k from f and depth. 
    inputs:
    --------
    f: frequencies [Hz]
    dep: water depth [m]
    output:
    --------
    dispNewtonTH: wavenumber
    """
    eps=0.000001
    g=9.81
    sig=2*np.pi*f
    Y=dep*sig**2./g 

    X=np.sqrt(Y)
    I=1
    F=1
    while abs(np.amax(F)) > eps:
        H=np.tanh(X)
        F=Y-X*H
        FD=-H-X/(np.cosh(X)**2)
        X=X-F/FD

    dispNewtonTH=X/dep

    return dispNewtonTH

def ep_to_ef(ep, depth, wavenumber, nfft):
    
    """
    Purpose: Convert the pressure spectrum into elevation spectrum. The transfer function is based on Airy linear theory
    ----------
    inputs:
    --------
    ep: pressure spectrum [m2/Hz]
    wavenumber: the wavenumber [m-1]
    nfft: the number of point for the spectral analysis
    output:
    --------
    Ef: the elevation spectrum
    """
        
        
    offbot = 0.1 # height of pressure sensor over bottom [m]

    M = np.cosh(wavenumber*depth)/np.cosh(wavenumber*offbot) # The transfert function from pressure to elevation (Airy theroy)
    Nf = int(nfft/2+1) # The half-frequency axis

    fmax=0.109 # The highest frequency resolved by the pressure sensor
    res_f = 1/nfft
    ifmax = int(fmax/res_f)  
    cosur = M**2
    cosur[0] = 1.
    cosur[ifmax+1 : Nf] = 1.

    Ef = ep[1:] * cosur[1:]
    return Ef


def compute_periodogram(ds, nrec, recfac, ntr, nt, nfft, fs):
    """
    Purpose: Perform the periodogram: E(frequency)= f(time)
    ----------
    inputs:
    --------
    ds: the xarray dataset with the pressure data
    nrec: number of record (how many data per record?)
    recfac: Number of spectra per pressure record (in 12h)
    ntr: number of seconds per record (e.g in 3 hours for 12h spectrum)
    fs: the sampling frequency
    output:
    --------
    Ef_all: the 2D periodogram
    time_4_spec: The associated time axis in days (2 spectra per days)
    """
        
        
    offbot = 0.1
    
    fmax=0.109 # The highest frequency resolved by the pressure sensor
    res_f = 1/nfft
    ifmax = int(fmax/res_f)


    Nf=int(nfft/2+1)
    Epf_all=np.zeros((Nf, nrec))  # Initialized the pressure spectra
    Ef_all =np.zeros((Nf, nrec))  # Initialized the elevation spectra

    time_p=np.zeros(nrec)
    irec=0
    recmax=nrec

    pres_new = np.array(ds.pressure.values).reshape(208, nt)  # Reshape the dataset in order to compute N spectra per 12 hours
    time_4_spec = []

    for t in range(0, len(ds.time.values), 43180): # Reshape the time axis
        time_4_spec.append(ds.time[t])

# Loop over time 
    while irec < recmax: 
        jrec=int(np.floor(irec/recfac)) # index for pressure record
        krec=int((irec-jrec*recfac))
        i1=krec*ntr
        i2=min([nt, ntr*(krec+1)])
        #print ('processing record',jrec, 'part', krec+1, 'out of', recfac, 'using indices',i1, i2)
        elevation = pres_new[jrec, i1:i2] # the signal 

        ##########
        # ---  Repeat the steps provided above
        ##########
        freq, E_p=welch(elevation, fs=1, window='hanning', nperseg=nfft, detrend='linear', return_onesided=True, scaling='density')

        dep = np.mean(elevation)
        k = dispNewtonTH(freq,dep)
        k[0] = 0 # height of pressure sensor over bottom 

        M = np.cosh(k*dep)/np.cosh(k*offbot)
        cosur = M**2
        cosur[0] = 1.
        cosur[ifmax+1:Nf]=1.
        Ef = E_p*cosur
        Epf_all[:,irec] = E_p
        Ef_all[:,irec] = Ef

        #time_p[irec] =  date2num(time_4_spec[jrec].values)+krec/(recfac*2)
        irec = irec+1
        
    return Ef_all, time_4_spec
    
