import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

def air2vac2(wl_air):
    
    """
    Convert wavelengths in air to vacuum wavelength
    Author: Nikolai Piskunov
    """
    wl_vac = np.copy(wl_air)
    ii = np.where(wl_air > 1999.352)

    sigma2 = (1e4 / wl_air[ii]) ** 2  # Compute wavenumbers squared
    fact = (
        1e0
        + 8.336624212083e-5
        + 2.408926869968e-2 / (1.301065924522e2 - sigma2)
        + 1.599740894897e-4 / (3.892568793293e1 - sigma2)
    )
    wl_vac[ii] = wl_air[ii] * fact  # Convert to vacuum wavelength
    return wl_vac
        
def vac2air(wl_air):
    """
    Convert wavelengths vacuum to air wavelength
    Author: Nikolai Piskunov
    """
    wl_vac = np.copy(wl_air)
    ii = np.where(wl_air > 1999.352)

    sigma2 = (1e4 / wl_air[ii]) ** 2  # Compute wavenumbers squared
    fact = (
        1e0
        + 8.336624212083e-5
        + 2.408926869968e-2 / (1.301065924522e2 - sigma2)
        + 1.599740894897e-4 / (3.892568793293e1 - sigma2)
    )
    wl_vac[ii] = wl_air[ii] / fact  # Convert to vacuum wavelength
    return wl_vac

file = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/1118/reduced/bgoH202211180013.fits"

hdul = fits.open(file)
arc_data = hdul['Fibre_P'].data
arc_wave = hdul['WAVE_P'].data
n_ord = arc_data.shape[0]
hdul.close

tharf = "/Users/daniel/Documents/Work/SALT_Pipeline/PyReduce-HRS/DLH_Codes_combined/2025_Mar/thar_best.fits"
thar_hdu = fits.open(tharf)
thar_data = thar_hdu[0].data
thar_wave = np.arange(len(thar_data))*0.02 + 3004.8623046875
thar_wave = air2vac2(thar_wave)
thar_hdu.close

linesf = "/Users/daniel/Documents/Work/SALT_Pipeline/PyReduce-HRS/DLH_Codes_combined/2025_Mar/Thorium_mask_031921.mas"
lines = np.loadtxt(linesf,usecols=0)

lines3 = np.loadtxt("./AA468_1115_linelist.dat",usecols=0)

lines5 = np.loadtxt("./AJSS211_linelist.dat",usecols=0)*10.

lines7 = np.loadtxt("thar_atlas_uves.dat",usecols=0)

plt.plot(thar_wave,thar_data,'k')

plt.vlines(lines7,0,1e6,'c')
plt.vlines(lines,0,1e6,'r')
plt.vlines(lines3,0,1e6,'b')
plt.vlines(lines5,0,1e6,'g')


for i in range(n_ord):
    plt.plot(air2vac2(arc_wave[i]),arc_data[i])

plt.show()
