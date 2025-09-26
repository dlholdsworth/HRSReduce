import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

ref_file = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/Super_Arcs/HR_Super_Arc_H20220701.fits'
#ref_file = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0717/reduced/bgoH202207170027.fits'

with fits.open('thar_best.fits') as hdu:
    header = hdu[0].header
    known_spec = (hdu[0].data)

    specaxis = str(1)
    flux = known_spec
    wave_step = header['CDELT%s' % (specaxis)]
    wave_base = header['CRVAL%s' % (specaxis)]
    reference_pixel = header['CRPIX%s' % (specaxis)]
    xconv = lambda v: ((v-reference_pixel+1)*wave_step+wave_base)
    known_waveobs = xconv(np.arange(len(flux)))

ii = np.where(np.logical_and(known_waveobs>3700, known_waveobs<5600))
plt.plot(known_waveobs[ii],known_spec[ii],'k')

file2="/Users/daniel/Documents/Work/SALT_Pipeline/Alexei Solutions/HR_arcs/mbgphH202207170027_2wm.fits"

with fits.open(file2) as hdu:
    header = hdu[0].header
    spec2=hdu[0].data
    wave_step = header['CDELT%s' % (specaxis)]
    wave_base = header['CRVAL%s' % (specaxis)]
    reference_pixel = header['CRPIX%s' % (specaxis)]
    xconv = lambda v: ((v-reference_pixel+1)*wave_step+wave_base)
    wave2 = xconv(np.arange(len(spec2)))

plt.plot(wave2,spec2*50,'c')

with fits.open(ref_file) as hdu:
    for i in range(0,42):
    
        try:
            lines = np.load('./HR_H_linelist_P_cl_'+str(i)+'.npy',allow_pickle=True).item()
            x=np.arange(len(hdu['FIBRE_P'].data[i]))
        
            FIBRE_P=hdu['FIBRE_P'].data[i]
            FIBRE_P[np.isnan(FIBRE_P)] = 0
        

        
            fit = np.polyfit(lines[i]['line_positions'],lines[i]['known_wavelengths_air'],6)
            fit_wave = np.polyval(fit,x)
        
            plt.plot(fit_wave,FIBRE_P*10000)
        except:
            pass

    plt.show()
