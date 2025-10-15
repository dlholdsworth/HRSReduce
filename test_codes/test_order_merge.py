import numpy as np
import logging,os
import hrsreduce
from astropy.io import fits
import matplotlib.pyplot as plt

from hrsreduce.order_merge.order_merge import OrderMerge

file = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0730/reduced/bgoH202207300018.fits'
flat = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0729/reduced/HR_Master_Flat_H20220729.fits'

#file = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/1118/reduced/bgoH202211180015.fits'
#flat = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/1115/reduced/HR_Master_Flat_H20221115.fits'

hdu = fits.open(file)
FIBRE_O = hdu['FIBRE_O'].data
WAVE_O = hdu['WAVE_O'].data
WAVE_P = hdu['WAVE_P'].data
#hdu.close
#
#FIBRE_O += np.abs(np.min(FIBRE_O))
#sigma = np.sqrt(FIBRE_O)
#sigma[np.isnan(sigma)] = 1000
#hdu=fits.open(flat)
#BLAZE_O = hdu['BLAZE_O'].data
#hdu.close
#
#
nord = FIBRE_O.shape[0]
#
#BLAZE_O[20:,482:500] = 0
#BLAZE_O[:,851:856] = 0
#BLAZE_O[37:,536:542] = 0
#for ord in range(42):
#    plt.plot(BLAZE_O[ord])
#plt.show()
#
#for ord in range(nord):
#    plt.plot(WAVE_O[ord])
#    plt.plot(WAVE_P[ord])
#plt.show()
#
wave,spectrum = OrderMerge(file,flat,'H',plot=False).execute()
#
plt.plot(wave,spectrum)
plt.show()
out_put= np.array([wave,spectrum,np.sqrt(spectrum)])
#
np.savetxt('test_out_O.txt',out_put.T)
