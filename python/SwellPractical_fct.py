import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import detrend, welch


import xarray as xr

g = 9.81  # gravitational acceleration

def create_UTC_axis(ds_xyz, arrayIndex):
    
    arrayIndex = xr.DataArray(arrayIndex, attrs={'units': 's'}).astype('timedelta64[s]')

    sample_rate_sec = xr.DataArray(ds_xyz.xyzSampleRate.values, attrs={'units': 's'}).astype('timedelta64[s]')
    filter_delay_sec = xr.DataArray(ds_xyz.xyzFilterDelay.values, attrs={'units': 's'}).astype('timedelta64[s]')
    factor = xr.DataArray(arrayIndex/sample_rate_sec, attrs={'units': 's'}).astype('timedelta64[s]')

    time_array = ds_xyz.xyzStartTime.values + factor - filter_delay_sec
    return time_array


def wave_frequency_spectrum(elevation, Fs = 1.2):
    
    nfft = len(elevation)  # The number of point for the spectral analysis
    
    Nf = int(nfft/2 + 1)  # The number of frequencies

    df = Fs/nfft  # The spectral resolution
    print(f'The spectral resolution is {df} Hz')
    print('\n')
    freq = np.linspace(df, df*(Nf-1), (Nf-1))  # Define the frequency axis

    #########
    # --- The heart of the spectral analysis
    #########
    
    hanningt = 0.5 * (1-np.cos(2*np.pi*np.linspace(0, nfft-1, nfft)/(nfft-1)))  # Hanning window
    wc2t = 1/np.mean(hanningt**2)  # Correction factor
    Zw = (elevation-np.mean(elevation)) * hanningt  # Apply the window to avoid spectral leakage (put to 0 the extrema) + detrend
    Zf = np.fft.fft(Zw, nfft, axis = 0)/nfft  # Apply the Fourier Transform
    spec = abs(Zf)**2 / df  # Compute the power spectral density
    spec_folded = 2 * spec[1:Nf]

    hs_spec = 4 * np.sqrt(np.sum(spec_folded)*df)
    #print(f'The significant wave height associated to the spectrum is {hs_spec} m')
    #print('\n')
    print(f'The significant wave height associated to the elevation is {4 * np.sqrt(elevation.var())} m')

    spec_corr =  spec_folded * wc2t  # Correct the spectrum
    hs_spec_corr = 4 * np.sqrt(np.sum(spec_corr)*df)
    print(f'The significant wave height associated to the spectrum is {hs_spec_corr} m')

    return freq, spec_corr



def wave_frequency_spectrum_overlap(elevation, overlap, nfft,  Fs = 1.2):
    
   
    Nf = int(nfft/2 + 1)
    NS1 = len(elevation)//nfft
    ov = overlap/100
    NS = NS1 * (1 + 2*ov) - 2 * ov
    
    Eh = np.ones((int(NS), 1))
    hanningt = 0.5 * (1-np.cos(2*np.pi*np.linspace(0, nfft-1, nfft)/(nfft-1)))

        
        
    H = hanningt * Eh
    elevmat = np.zeros((nfft, int(NS)))
    vec_i = np.arange(0, NS, 1)

    df = Fs/nfft
    freq = np.linspace(df, df*(Nf-1), (Nf - 1))

    for iw in range(len(vec_i)):
        nstart = np.floor((iw-1+1)*(1 - ov) * nfft)
        nend = nstart + nfft
        elevmat[:, iw] = elevation[int(nstart):int(nend)]

    elevmat2 = elevmat.T
    
    wc2t = 1/np.mean(hanningt**2)
    Zw = (elevmat2-np.mean(elevmat2)) * hanningt

    Zf = np.fft.fft(Zw, nfft, axis = 1)/nfft

    spec = abs(Zf)**2  / (df)
    spec_folded = 2 * spec[:, 0:np.size(spec, axis = 1)//2]
    spec_corr = spec_folded * wc2t
    spec_corr_mean = np.mean(spec_corr, axis = 0).T

    return freq, spec_corr_mean



def buoy_spectrum2d(sf, a1, a2, b1, b2, dirs = np.arange(0,360,10)):
    """
    Purpose: perform the Maximum entropy method to estimate the Directional Distribution
    ---------
    # Maximum Entropy Method - Lygre & Krogstad (1986 - JPO)
    # Eqn. 13:
    # phi1 = (c1 - c2c1*)/(1 - abs(c1)^2)
    # phi2 = c2 - c1phi1
    # 2piD = (1 - phi1c1* - phi2c2*)/abs(1 - phi1exp(-itheta) -phi2exp(2itheta))^2
    # c1 and c2 are the complex fourier coefficients
    
    Inputs:
    -------
    sf, a1, a2, b1, b2, coeficients. Provided by the Buoy
    dirs. in degrees
    Outputs:
    -------    
    The diretional energy wave spectrum and the directional distribution of the waves
    """
    nfreq = np.size(sf)
    nbin = np.size(dirs)
    
    c1 = a1+1j*b1
    c2 = a2+1j*b2
    p1 = (c1-c2*np.conj(c1))/(1.-abs(c1)**2)
    p2 = c2-c1*p1
    
    # numerator(2D): x
    x = 1.-p1*np.conj(c1)-p2*np.conj(c2)
    x = np.tile(np.real(x),(nbin,1)).T
    
    # denominator(2D): y
    a = dirs*np.pi/180.
    e1 = np.tile(np.cos(a)-1j*np.sin(a),(nfreq,1))
    e2 = np.tile(np.cos(2*a)-1j*np.sin(2*a),(nfreq,1))
    
    p1e1 = np.tile(p1,(nbin,1)).T*e1
    p2e2 = np.tile(p2,(nbin,1)).T*e2
    
    y = abs(1-p1e1-p2e2)**2
    
    D = x/(y*2*np.pi)
    
    # normalizes the spreading function,
    # so that int D(theta,f) dtheta = 1 for each f  
    
    dth = dirs[1]-dirs[0]

    tot = np.tile(np.sum(D, axis=1)*dth/180.*np.pi,(nbin,1)).T
    D = D/tot
    
    sp2d = np.tile(sf,(nbin,1)).T*D
    
    return sp2d, D



def plot_polar_spectrum(dirspec, DIRS, FREQS, time_spec, vmin0 = 0, vmax0 = 30, cbar = 0, date_on = 0):
    
    
    fig,ax=plt.subplots(subplot_kw={'projection': 'polar'})
    p1 = plt.pcolor(DIRS, FREQS, dirspec.T, vmin = vmin0, vmax = vmax0, cmap = 'Blues')
    plt.ylim([0,0.3])
    ax.set_rmax(.20)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    plt.tight_layout()
    angle_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W','NW']
    ax.set_rlabel_position(180 + 45)
    #ax.set_ylim(0, 30)#set_rmax(30)
    #ax.set_ylabels('Wave period [s]', labelpad = 20)
    ax.set_thetagrids(angles = range(0, 360, 45),
                          labels = angle_labels,fontsize=12)


    #############,
    #---White circles
    #############
    radius_050m=np.sqrt(g*2*np.pi/(50.*4*np.pi**2))
    radius_100m=np.sqrt(g*2*np.pi/(100.*4*np.pi**2))
    radius_200m=np.sqrt(g*2*np.pi/(200.*4*np.pi**2))
    radius_500m=np.sqrt(g*2*np.pi/(500.*4*np.pi**2))
    #radius_300m=(2*np.pi)/300.


    circle1 = plt.Circle((0.0, 0.0), radius_050m, color='k',transform=ax.transData._b,linestyle='--',fill=False)
    ax.add_patch(circle1)
    circle2 = plt.Circle((0.0, 0.0), radius_100m, color='k',transform=ax.transData._b,linestyle='--',fill=False)
    ax.add_patch(circle2)
    circle3 = plt.Circle((0.0, 0.0), radius_200m, color='k',transform=ax.transData._b,linestyle='--',fill=False)
    ax.add_patch(circle3)
    circle4 = plt.Circle((0.0, 0.0), radius_500m, color='k',transform=ax.transData._b,linestyle='--',fill=False)
    ax.add_patch(circle4)
    plt.text(0,radius_050m,'50 m',color='k')
    plt.text(0,radius_100m,'100 m',color='k')
    plt.text(0,radius_200m,'200 m',color='k')
    plt.text(0,radius_500m,'500 m',color='k')

    ax.set_yticklabels([])

    ax.text(150*np.pi/180,.3,'CDIP', fontsize=16)
    ax.set_title(time_spec)
    if cbar==1:
        cbar_ax = fig.add_axes([0.2, .08, .2, 0.05])
        cbar=fig.colorbar(p1, cax=cbar_ax, shrink=0.5,aspect=155,extend='max',orientation='horizontal')
        cbar.ax.tick_params(labelsize=12) 
        cbar.ax.set_xlabel(' [m²/rad/Hz]',rotation=0,fontsize=15)
    ax.set_aspect('equal', 'box')
    plt.tight_layout()


    