import numpy as np
import logging
import hrsreduce
from astropy.io import fits
import matplotlib.pyplot as plt

from hrsreduce.order_merge.order_merge import OrderMerge

file = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0717/reduced/bgoH202207170035.fits'
flat = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0717/reduced/HR_Master_Flat_H20220717.fits'

hdu = fits.open(file)
FIBRE_O = hdu['FIBRE_O'].data
WAVE_O = hdu['WAVE_O'].data
hdu.close
sigma = np.sqrt(FIBRE_O)
sigma[np.isnan(sigma)] = 1000
hdu=fits.open(flat)
BLAZE_O = hdu['BLAZE_O'].data
hdu.close

wave,spectrum = OrderMerge(FIBRE_O,WAVE_O,BLAZE_O,sigma).splice_orders()

out_put= np.array([wave,spectrum,np.sqrt(spectrum)])

np.savetxt('test_out_O.txt',out_put.T)
