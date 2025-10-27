import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.special import erf

from scipy.optimize import curve_fit

with fits.open('./thar_best.fits') as hdu:
    header = hdu[0].header
    known_spec = (hdu[0].data)

    specaxis = str(1)
    flux = known_spec
    wave_step = header['CDELT%s' % (specaxis)]
    wave_base = header['CRVAL%s' % (specaxis)]
    reference_pixel = header['CRPIX%s' % (specaxis)]
    xconv = lambda v: ((v-reference_pixel+1)*wave_step+wave_base)
    known_waveobs = xconv(np.arange(len(flux)))
    
#plt.plot(known_waveobs, flux,'b')

file2 = "/Users/daniel/Desktop/SALT_HRS_DATA/Red/2022/0729/reduced/bgoR202207290042.fits"

with fits.open(file2) as hdu:

    Fibre_o_2022 = hdu['FIBRE_O'].data
    wave_o_2022 = hdu[16].data

    
file3 = "/Users/daniel/Desktop/SALT_HRS_DATA/Red/2025/1011/reduced/bgoR202510110034.fits"
with fits.open(file3) as hdu:

    Fibre_o_2025 = hdu['FIBRE_O'].data
    wave_o_2025 = hdu[32].data

file4 = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0729/reduced/bgoH202207290042.fits"
with fits.open(file4) as hdu:

    Fibre_o_2022_b = hdu['FIBRE_O'].data
    wave_o_2022_b = hdu['WAVE_O'].data
    
atlas_w,atlas_int = np.loadtxt("./New_Th_linelist_air.list",usecols=(0,1),unpack=True)

for ord in range(20,33):
    ii=np.where(np.logical_and(atlas_w > np.min(wave_o_2022[ord])-5, atlas_w < np.max(wave_o_2022[ord])+5))[0]
    plt.vlines(atlas_w[ii],0,atlas_int[ii]*50,'g')
    plt.plot(wave_o_2022[ord], Fibre_o_2022[ord],'r')
#    plt.plot(wave_o_2022[ord-1], Fibre_o_2022[ord-1],'r',alpha = 0.5)
    plt.plot(wave_o_2022[ord+1], Fibre_o_2022[ord+1],'r',alpha = 0.5)
    plt.plot(wave_o_2022_b[41],Fibre_o_2022_b[41],'b',alpha = 0.5)
    plt.plot(wave_o_2022_b[40],Fibre_o_2022_b[40],'b',alpha = 0.5)
#    plt.plot(wave_o_2022_b[39],Fibre_o_2022_b[39],'b',alpha = 0.5)
#    plt.plot(wave_o_2025[ord],Fibre_o_2025[ord]/50,'g')

    plt.show()

    for j in range(len(wave_o_2022[ord])-1):
        plt.plot(wave_o_2022[ord][j],wave_o_2022[ord][j+1]-wave_o_2022[ord][j],'.')
    plt.show()
