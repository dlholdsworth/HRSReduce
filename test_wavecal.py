from hrsreduce.wave_cal.wave_cal import WavelengthCalibration
from hrsreduce.wave_cal.build_wavemodel import BuildWaveModel

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
thar_file = "/Users/daniel/Documents/Work/SALT_Pipeline/PyReduce-HRS/DLH_Codes_combined/2025_Mar/thar_best.fits"
arc_file= "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/0704/reduced/bgoH202307040013.fits"
arc_file1= "/Users/daniel/Desktop/SALT_HRS_DATA/Red/2022/1115/reduced/bgoR202211150027.fits"
arc_file2= "/Users/daniel/Desktop/SALT_HRS_DATA/Red/2022/1115/reduced/bgoR202211150026.fits"
arm = "H"
m="HS"
cal_type= "ThAr"
base_dir = "/Users/daniel/Desktop/SALT_HRS_DATA/"
plot= True

with fits.open(arc_file) as hdul:
    O_fluxs = hdul['Fibre_O'].data
    ord_lens = np.zeros(O_fluxs.shape[0])
with fits.open(arc_file1) as hdul:
    O_fluxs1 = hdul['Fibre_O'].data
with fits.open(arc_file2) as hdul:
    O_fluxs2 = hdul['Fibre_O'].data
    
plt.plot(O_fluxs[20])
plt.plot(O_fluxs1[20])
plt.plot(O_fluxs2[20])
plt.show()

for o in range(len(ord_lens)):
    ord_lens[o] = len(O_fluxs[o])
rough_wls, echelle_ord = BuildWaveModel(arm,m,ord_lens).execute()

with fits.open(thar_file) as hdul:
    thar = hdul[0].data
    specaxis = str(1)
    header = hdul[0].header
    wave_step = header['CDELT%s' % (specaxis)]
    wave_base = header['CRVAL%s' % (specaxis)]
    reference_pixel = header['CRPIX%s' % (specaxis)]
    xconv = lambda v: ((v-reference_pixel+1)*wave_step+wave_base)
    wave = xconv(np.arange(len(thar)))


#plt.plot(O_fluxs[10][102:],'r')
#plt.plot(O_fluxs1[10],'g')
#plt.plot(O_fluxs2[10][162:],'b')
#plt.show()
#for ord in range(O_fluxs.shape[0]):
#    plt.plot(wave,thar,'k')
#    plt.plot(rough_wls[ord],O_fluxs[ord])
#    plt.xlim(np.min(rough_wls[ord])-100,np.max(rough_wls[ord])+100)
#    plt.ylim(np.min(O_fluxs[ord]),np.max(O_fluxs[ord])*2)
#    plt.show()

#WavelengthCalibration(arc_file, arm, m, base_dir,cal_type,plot).execute()
