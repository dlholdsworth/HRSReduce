import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy import constants as const
import scipy.constants as conts
from astropy.modeling import models, fitting
import math
import dlh_RV_calc

#sci2 = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/1008/reduced/bgoH202210080017.fits"
#sci = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1204/reduced/bgoH202312040022.fits"
#arc ="/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/0112/reduced/bgoH202301120013.fits"
files = ["/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1204/reduced/bgoH202312040022.fits",
"/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1204/reduced/bgoH202312040023.fits",
"/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1204/reduced/bgoH202312040024.fits",
"/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1204/reduced/bgoH202312040025.fits",
"/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1204/reduced/bgoH202312040026.fits",
"/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1204/reduced/bgoH202312040027.fits",
"/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1204/reduced/bgoH202312040028.fits",
"/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1204/reduced/bgoH202312040029.fits",
"/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1204/reduced/bgoH202312040030.fits",
"/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1204/reduced/bgoH202312040031.fits",
#"/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1205/reduced/bgoH202312050018.fits",
#"/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1205/reduced/bgoH202312050019.fits",
#"/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1205/reduced/bgoH202312050020.fits",
#"/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1205/reduced/bgoH202312050021.fits",
#"/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1205/reduced/bgoH202312050022.fits",
"/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1206/reduced/bgoH202312060014.fits",
"/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1206/reduced/bgoH202312060015.fits",
"/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1206/reduced/bgoH202312060016.fits",
"/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1206/reduced/bgoH202312060017.fits",
"/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1206/reduced/bgoH202312060018.fits"]
files = ['/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/0705/reduced/bgoH202307050029.fits']
#hdu = fits.open(arc)
#waves = hdu['WAVE_P'].data
#hdu.close

#flat="/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1201/reduced/HS_Master_Flat_H20231201.fits"
#with fits.open(flat) as hdu2:
#    blaze = hdu2['FIBRE_O'].data
    
for sci in files:
    hdu = fits.open(sci)
    spectrum = hdu['FIBRE_O'].data
    waves = hdu['WAVE_O'].data
    arc = hdu['FIBRE_P'].data
    blaze = hdu['BLAZE_O'].data
    header=hdu[0].header
    hdu.close

    known_rv=-16597
    #known_rv = 31695

    rv,rv_err,BCV,BJD=dlh_RV_calc.execute(header,waves,spectrum,blaze,known_rv)
    print(rv,rv_err, known_rv-rv, BCV)
    plt.plot(BJD,rv-known_rv,'o')
#    plt.plot(BJD,BCV+22555+known_rv,'x')
plt.show()

    #hdu = fits.open(sci2)
    #arc2 = hdu['FIBRE_O'].data
    #hdu.close

    #plt.plot(arc[20])
    #plt.plot(arc2[20])
    #plt.show()

