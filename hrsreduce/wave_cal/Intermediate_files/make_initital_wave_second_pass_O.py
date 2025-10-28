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
ref_file = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/Super_Arcs/HR_Super_Arc_H20220701.fits'

ref_file = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0717/reduced/bgoH202207170027.fits'

ref_flat = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2025/0610/reduced/HR_Master_Flat_H20250610.fits'
#ref_file = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2025/Super_Arcs/HR_Super_Arc_H20250101.fits'

with fits.open('../thar_best.fits') as hdu:
    header = hdu[0].header
    known_spec = (hdu[0].data)

    specaxis = str(1)
    flux = known_spec
    wave_step = header['CDELT%s' % (specaxis)]
    wave_base = header['CRVAL%s' % (specaxis)]
    reference_pixel = header['CRPIX%s' % (specaxis)]
    xconv = lambda v: ((v-reference_pixel+1)*wave_step+wave_base)
    known_waveobs = xconv(np.arange(len(flux)))

    
file2="/Users/daniel/Documents/Work/SALT_Pipeline/Alexei Solutions/HR_arcs/mbgphH202207170027_1wm.fits"

with fits.open(file2) as hdu:
    header = hdu[0].header
    spec2=hdu[0].data
    wave_step = header['CDELT%s' % (specaxis)]
    wave_base = header['CRVAL%s' % (specaxis)]
    reference_pixel = header['CRPIX%s' % (specaxis)]
    xconv = lambda v: ((v-reference_pixel+1)*wave_step+wave_base)
    wave2 = xconv(np.arange(len(spec2)))
    
fig, axs = plt.subplots(1,1,figsize=(17,7))

    #Order 00 range 3720-3755
    #Order 01 range 3740-3795
    #Order 02 range 3760-3815
    #Order 03 range 3790-3850
    #Order 04 range 3825-3880
    #Order 05 range 3855-3910
    #Order 06 range 3885-3945
    #Order 07 range 3920-3982
    #Order 08 range 3960-4012
    #Order 09 range 3990-4045
    #Order 10 range 4030-4085
    #Order 11 range 4061-4116
    #Order 12 range 4095-4155
    #Order 13 range 4133-4190
    #Order 14 range 4170-4225
    #Order 15 range 4205-4265
    #Order 16 range 4245-4307
    #Order 17 range 4285-4347
    #Order 18 range 4325-4390
    #Order 19 range 4365-4430
    #Order 20 range 4405-4470
    #Order 21 range 4450-4510
    #Order 22 range 4495-4560
    #Order 23 range 4539-4600
    #Order 24 range 4583-4645
    #Order 25 range 4630-4692
    #Order 26 range 4675-4740
    #Order 27 range 4724-4790
    #Order 28 range 4772-4840
    #Order 29 range 4823-4890
    #Order 30 range 4874-4940
    #Order 31 range 4925-5000
    #Order 32 range 4975-5045
    #Order 33 range 5035-5100
    #Order 34 range 5088-5160
    #Order 35 range 5140-5215
    #Order 36 range 5195-5275
    #Order 37 range 5250-5335
    #Order 38 range 5315-5395
    #Order 39 range 5380-5455
    #Order 40 range 5446-5520
    #Order 41 range 5510-5580


ord=41
order_min = 5510
order_max = 5580
height = 0.02

ii=np.where(np.logical_and(known_waveobs>order_min-5,known_waveobs<order_max+5))[0]
plt.plot(known_waveobs[ii], (known_spec[ii]/np.max(known_spec[ii])),'c')

#jj=np.where(np.logical_and(wave2>order_min-5,wave2<order_max+5))[0]
#plt.plot(wave2[jj], spec2[jj]/np.max(spec2[jj]),'b')

plt.xlim(order_min,order_max)
plt.ylim(-0.05,0.5)
atlas = np.loadtxt('../thar_list_orig.txt',usecols=(0),unpack=True)


if arm == 'H':
    line_wave = []
    line_pix = []
    with fits.open(ref_file) as hdu:
        for i in range(ord,ord+1):
        
            x=np.arange(len(hdu['FIBRE_O'].data[i]))
            
            lines = np.load('../HR_H_linelist_O_500s.npy',allow_pickle=True).item()
            fit = np.polyfit(lines[i]['line_positions'],lines[i]['known_wavelengths_air'],3)
            #fit = ([-1.52192313e-18,  9.33330773e-15, -2.12069425e-11,  1.92488133e-08, -1.88106710e-06,  1.72081971e-02,  3.70900000e+03])
            fit_wave = np.polyval(fit,x)
        
            peak_cens_wave = []
            peak_cens_pix = []
            peaks2 = []

            FIBRE_O=hdu['FIBRE_O'].data[i]
            FIBRE_O[np.isnan(FIBRE_O)] = 0
            FIBRE_O -=np.median(FIBRE_O)
            FIBRE_O/=(np.max(FIBRE_O))
            plt.plot(fit_wave,FIBRE_O,'k')


            peaks,_ = find_peaks(FIBRE_O,height=height,distance=8)
            plt.plot(fit_wave[peaks],FIBRE_O[peaks],'kx')
            
            #fit gausians to the peaks and over plot in green.
            for peak in peaks:
                if np.logical_and(peak > 8, peak < 2040):
                    #Do fit in Wavelength space
                    cut_x = fit_wave[peak-5:peak+5]
                    cut_y =FIBRE_O[peak-5:peak+5]
                    gmod = Model(gaussian)
                    result_ref = gmod.fit(cut_y, x=cut_x, amp=Parameter('amp',value=np.max(cut_y)),cen=Parameter('cen',value=fit_wave[peak]),wid=Parameter('wid',value=0.05,min=0.001))

                    plt.plot(cut_x,result_ref.best_fit,'g--')
                    if np.logical_and(result_ref.params['cen'] > order_min, result_ref.params['cen'] <order_max):
                        peak_cens_wave.append(result_ref.params['cen'])
                        peaks2.append(peak)
                        plt.plot(result_ref.params['cen'],np.max(cut_y),'bx')
                    
                        #Do fit in pixel space
                        cut_x = x[peak-6:peak+6]
                        cut_y =FIBRE_O[peak-6:peak+6]
                        gmod = Model(gaussian)
                        result_ref = gmod.fit(cut_y, x=cut_x, amp=Parameter('amp',value=np.max(cut_y)),cen=Parameter('cen',value=x[peak]),wid=Parameter('wid',value=2,min=1))

                        peak_cens_pix.append(result_ref.params['cen'].value)

                    
    plt.vlines(atlas,0,0.05,'r')
    plt.show(block=False)
    plt.pause(0.1)
    
    cont='y'

    while (cont =='y'):
        click = fig.ginput(1, timeout=0, show_clicks=False)
        init_pix_x = list(map(lambda x: x[0], click))
        peak_px_sp, tmp_init = find_nearest(peak_cens_wave, init_pix_x[0])

        #print ("\n     Input known lambda?")
        #known_lam = float((input("(y/n) = ")))

        click = fig.ginput(1, timeout=0, show_clicks=False)
        init_pix_x = list(map(lambda x: x[0], click))
        peak_wave, tmp_init2 = find_nearest(atlas, init_pix_x[0])

        print(peak_wave,peak_cens_pix[tmp_init])
        line_wave.append(peak_wave)
        line_pix.append(peak_cens_pix[tmp_init])

        print ("\n     Continue with this order?")
        cont = str((input("(y/n) = ")))
    
    plt.close()
    
    line_wave = np.asarray(line_wave, dtype=float)
    line_pix = np.asarray(line_pix, dtype=float)
    
    output = {}
    output[ord] = {}
    output[ord]['line_positions'] = line_pix
    output[ord]['known_wavelengths_air'] = line_wave

    np.save("./HR_H_linelist_O_500s_"+str(ord),output)
    fit = np.polyfit(line_pix,line_wave,3)
    fit_wave = np.polyval(fit,x)

    plt.plot(known_waveobs[ii], known_spec[ii]/np.max(known_spec[ii]),'c')
    plt.plot(fit_wave,FIBRE_O,'k')
    plt.xlim(order_min,order_max)
    plt.show()
    
    plt.plot(x,fit_wave)
    plt.plot(line_pix,line_wave,'rx')
    plt.show()

    
