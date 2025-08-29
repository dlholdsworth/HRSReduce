from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from lmfit import  Model,Parameter
from scipy.signal import find_peaks

##################################################################################
# FUNCTION: find nearest value in array, return array value and index
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    #return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))
    #return (1./(wid*np.sqrt(2*np.pi))) * np.exp(-(x-cen)**2 / (2*wid**2))
    return amp*np.exp(-(x-cen)**2/(2*wid**2))
    
def air2vac(wl_air):
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
            
arm = 'H'
ref_file1 = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2025/0610/reduced/bgoH202506100032.fits'
ref_file = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0717/reduced/bgoH202207170027.fits'
ref_flat = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2025/0610/reduced/HR_Master_Flat_H20250610.fits'
#ref_file = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2025/Super_Arcs/HR_Super_Arc_H20250101.fits'
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

    
file2="/Users/daniel/Documents/Work/SALT_Pipeline/Alexei Solutions/HR_arcs/mbgphH202207170027_2wm.fits"

with fits.open(file2) as hdu:
    header = hdu[0].header
    spec2=hdu[0].data
    wave_step = header['CDELT%s' % (specaxis)]
    wave_base = header['CRVAL%s' % (specaxis)]
    reference_pixel = header['CRPIX%s' % (specaxis)]
    xconv = lambda v: ((v-reference_pixel+1)*wave_step+wave_base)
    wave2 = xconv(np.arange(len(spec2)))

HS_sol=np.load('/Users/daniel/Documents/Work/SALT_Pipeline/HRSReduce/hrsreduce/wave_cal/New_HS_H_linelist_P_clean.npy',allow_pickle=True).item()
fig, axs = plt.subplots(2,1,figsize=(20,7))



#axs[0].set_xlim(np.min(HS_sol[0]['known_wavelengths_vac'])-100,np.max(HS_sol[41]['known_wavelengths_vac'])+100)

axs[0].set_xlim(5518,5583)

ii=np.where(np.logical_and(known_waveobs>5517,known_waveobs<5583))[0]

axs[0].plot(known_waveobs[ii], known_spec[ii]/10,'k')
axs[0].plot(wave2,spec2,'g')
atlas = np.loadtxt('thar_list_orig.txt',usecols=(0),unpack=True)
axs[0].vlines(atlas,0,1000,'r')
axs[0].set_ylim(0,np.max(known_spec[ii])/10)

if arm == 'H':
    with fits.open(ref_file) as hdu:
        with fits.open(ref_file1) as hdu1:
            for i in range(41,42):
            
                peak_cens = []
            
                x=np.arange(len(hdu['FIBRE_P'].data[i]))#+len(hdu['FIBRE_P'].data[i])*i
                    
                #axs[0].plot(known_waveobs[ii], known_spec[ii],'k')
                #axs[0].plot(wave2,spec2*10,'r')

                axs[1].plot(x+6,hdu['FIBRE_P'].data[i],'g')
                axs[1].plot(x,hdu1['FIBRE_P'].data[i], 'b')
                #axs[1].plot(hdu1['FIBRE_P'].data[i])
                #fig.suptitle("ORDER "+str(i))
                
                peaks,_ = find_peaks(hdu1['FIBRE_P'].data[i],height=100,distance=5)
                plt.plot(peaks,hdu1['FIBRE_P'].data[i][peaks],'cx')
                
                #fit gausians to the peaks and over plot in red.
                for peak in peaks:
                    cut_x = x[peak-10:peak+10]
                    cut_y =hdu1['FIBRE_P'].data[i][peak-10:peak+10]
                    gmod = Model(gaussian)
                    result_ref = gmod.fit(cut_y, x=cut_x, amp=Parameter('amp',value=np.max(cut_y)),cen=Parameter('cen',value=peak),wid=Parameter('wid',value=2.,min=0.5))

                    axs[1].plot(cut_x,result_ref.best_fit,'r--')
                    peak_cens.append(result_ref.params['cen'])
    plt.show(block=False)
    plt.pause(0.1)
    
    cont='y'

    while (cont =='y'):
        click = fig.ginput(1, timeout=0, show_clicks=False)
        init_pix_x = list(map(lambda x: x[0], click))
        peak_px_sp, tmp_init = find_nearest(peak_cens, init_pix_x[0])

        #print ("\n     Input known lambda?")
        #known_lam = float((input("(y/n) = ")))

        click = fig.ginput(1, timeout=0, show_clicks=False)
        init_pix_x = list(map(lambda x: x[0], click))
        peak_wave, tmp_init2 = find_nearest(atlas, init_pix_x[0])
        #peak_wave, tmp_init2 = find_nearest(peak_wave_atl, known_lam)

        print(peak_wave,peak_px_sp)

        print ("\n     Continue with this order?")
        cont = str((input("(y/n) = ")))
    

    wave_sol = 2.5798E-10*x**3 - 4.5517E-06*x**2 + 4.3146E-02*x + 5.5118E+03
    
    plt.plot(wave_sol,hdu['FIBRE_P'].data[i])
    plt.plot(wave2,spec2,'r')
    plt.show()
