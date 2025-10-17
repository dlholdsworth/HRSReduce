from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

blu_file = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0730/reduced/H202207300018_product.fits"

red_file="/Users/daniel/Desktop/SALT_HRS_DATA/Red/2022/0730/reduced/R202207300018_product.fits"

with fits.open(blu_file) as hdu:
    blu_data = hdu[7].data
    blu_wave = np.arange(len(blu_data))
    blu_wave = blu_wave * float(hdu[7].header['CDELT1']) + float(hdu[7].header['CRVAL1'])
    
#plt.plot(blu_wave,blu_data,'b')

with fits.open(red_file) as hdu:
    red_data = hdu[7].data
    red_wave = np.arange(len(red_data))
    red_wave = red_wave * float(hdu[7].header['CDELT1']) + float(hdu[7].header['CRVAL1'])
    
#    for ord in range(33):
#        plt.plot(hdu[ord*2+10].data['Wave'],hdu[ord*2+10].data['Flux'])
#        plt.plot(hdu['WAVE_O'].data[ord],hdu['FIBRE_O'].data[ord]/hdu['BLAZE_O'].data[ord])
    
plt.plot(red_wave,red_data,'r')

plt.show()
