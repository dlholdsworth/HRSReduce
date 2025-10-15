from hrsreduce.wave_cal.wave_cal import WavelengthCalibration
from hrsreduce.wave_cal.build_wavemodel import BuildWaveModel

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
thar_file = "/Users/daniel/Documents/Work/SALT_Pipeline/PyReduce-HRS/DLH_Codes_combined/2025_Mar/thar_best.fits"
arc_file = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0729/reduced/bgoH202207290042.fits"
super_arc= "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/Super_Arcs/HR_Super_Arc_H20220701.fits"
arc_file2= "/Users/daniel/Desktop/SALT_HRS_DATA/Red/2022/1115/reduced/bgoR202211150026.fits"
arm = "H"
m="HR"
cal_type= "ThAr"
cal_line_list= './hrsreduce/wave_cal/thar_best.txt'
base_dir = "/Users/daniel/Desktop/SALT_HRS_DATA/"
plot= True

line_list = np.load("./hrsreduce/wave_cal/HR_H_linelist_P.npy",allow_pickle=True).item()

#pixels = np.arange(2048)
#for ord in range(42):
#    wav = line_list[ord]['known_wavelengths_air']
#    pix = line_list[ord]['line_positions']
#    
#    fit = np.polyfit(pix,wav,6)
#    wls = np.polyval(fit,pixels)
#
#    plt.plot(wls)
#
#with fits.open(arc_file) as hdu:
#    wave = hdu[46].data
#    flux= hdu['Fibre_P'].data
#
#with fits.open(super_arc) as hdu:
#    flux= hdu['Fibre_P'].data
#plt.plot(flux[6])
#plt.show()
#for ord in range(42):
#    plt.plot(wave[ord],'--')
#plt.show()


WavelengthCalibration(arc_file, super_arc, arm, m, base_dir,cal_type,cal_line_list,plot).execute()

