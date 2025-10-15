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
            
arm = 'R'

if arm =='R':

    ref_file1 = '/Users/daniel/Desktop/SALT_HRS_DATA/Red/2022/0717/reduced/bgoR202207170027.fits'
    ref_file = '/Users/daniel/Desktop/SALT_HRS_DATA/Red/2022/Super_Arcs/HR_Super_Arc_R20220701.fits'
    ref_flat = '/Users/daniel/Desktop/SALT_HRS_DATA/Red/2022/0729/reduced/HR_Master_Flat_R20220729.fits'
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

        
    file2="/Users/daniel/Documents/Work/SALT_Pipeline/HRSReduce/MIDAS_Reductions/mbgphR202207170027_2wm.fits"

    with fits.open(file2) as hdu:
        header = hdu[0].header
        spec2=hdu[0].data
        wave_step = header['CDELT%s' % (specaxis)]
        wave_base = header['CRVAL%s' % (specaxis)]
        reference_pixel = header['CRPIX%s' % (specaxis)]
        xconv = lambda v: ((v-reference_pixel+1)*wave_step+wave_base)
        wave2 = xconv(np.arange(len(spec2)))
        
        print(np.min(wave2))

    fig, axs = plt.subplots(2,1,figsize=(20,7))

    #axs[0].set_xlim(np.min(HS_sol[0]['known_wavelengths_vac'])-100,np.max(HS_sol[41]['known_wavelengths_vac'])+100)

    #Order 00 range 5440-5530
    #Order 01 range 5510-5600
    #Order 02 range 5580-5670
    #Order 03 range 5590-5740
    #Order 04 range 5690-5810
    #Order 05 range 5750-5880
    #Order 06 range 5820-5950
    #Order 07 range 5890-6020
    #Order 08 range 5960-6110
    #Order 09 range 6040-6190
    #Order 10 range 6120-6265
    #Order 11 range 6220-6350
    #Order 12 range 6305-6445
    #Order 13 range 6395-6535
    #Order 14 range 6485-6635
    #Order 15 range 6575-6735
    #Order 16 range 6665-6835
    #Order 17 range 6755-6935
    #Order 18 range 6845-7035
    #Order 19 range 6970-7127
    #Order 20 range 7080-7240
    #Order 21 range 7190-7350
    #Order 22 range 7300-7460
    #Order 23 range 7425-7590
    #Order 24 range 7540-7710
    #Order 25 range 7670-7840
    #Order 26 range 7800-7980
    #Order 27 range 7930-8120
    #Order 28 range 8060-8260
    #Order 29 range 8190-8400
    #Order 30 range 8360-8570
    #Order 31 range 8530-8740
    #Order 32 range 8700-8910

    ord=32
    order_min = 8700
    order_max = 8910

    axs[0].set_xlim(order_min,order_max)
    jj=np.where(np.logical_and(wave2>order_min,wave2<order_max))[0]
    ii=np.where(np.logical_and(known_waveobs>order_min,known_waveobs<order_max))[0]

    atlas = np.loadtxt('thar_list_orig.txt',usecols=(0),unpack=True)
    axs[0].plot(known_waveobs[ii], known_spec[ii]/np.max(known_spec[ii]),'k')
    axs[0].vlines(atlas,0,0.05,'r')
    #axs[0].plot(known_waveobs,known_spec,'g')
    axs[0].plot(wave2[jj],spec2[jj]/np.max(spec2[jj]*2),'b',alpha=0.7)
    
    #axs[0].set_ylim(-200,np.max(spec2[jj])-0000)
    axs[0].set_ylim(-0.02,0.1)

    if arm == 'R':
        line_wave = []
        line_pix = []
        with fits.open(ref_file) as hdu:
            with fits.open(ref_file1) as hdu1:
                for i in range(ord,ord+1):
                
                    peak_cens = []
                    peaks2 = []
                
                    x=np.arange(len(hdu['FIBRE_P'].data[i]))
                    
                    FIBRE_P=hdu['FIBRE_P'].data[i]
                    FIBRE_P -=np.nanmedian(FIBRE_P)
                    FIBRE_P[np.isnan(FIBRE_P)] = 0
                    FIBRE_P/=(np.max(FIBRE_P))
                    
                    FIBRE_P1=hdu1['FIBRE_P'].data[i]
                    FIBRE_P1 -=np.nanmedian(FIBRE_P1)
                    FIBRE_P1[np.isnan(FIBRE_P1)] = 0
                    FIBRE_P1/=(np.max(FIBRE_P1))
                        
                    #axs[0].plot(known_waveobs[ii], known_spec[ii],'k')
                    #axs[0].plot(wave2,spec2*10,'r')

                    axs[1].plot(x,FIBRE_P1,'c',alpha=.8)
                    axs[1].plot(x,FIBRE_P,'g')
                    axs[1].set_ylim(-0.002,np.max(FIBRE_P)-0.8)
                    
                    #fig.suptitle("ORDER "+str(i))
                    
                    peaks,_ = find_peaks(FIBRE_P,height=0.001,distance=6)
                    #plt.plot(peaks,FIBRE_P[peaks],'cx')
                    
                    #fit gausians to the peaks and over plot in red.
                    for peak in peaks:
                        if np.logical_and(peak > 8, peak < 4090):
                            cut_x = x[peak-6:peak+4]
                            cut_y =FIBRE_P[peak-6:peak+4]
                            gmod = Model(gaussian)
                            result_ref = gmod.fit(cut_y, x=cut_x, amp=Parameter('amp',value=np.max(cut_y)),cen=Parameter('cen',value=peak),wid=Parameter('wid',value=2.,min=0.5))

                            axs[1].plot(cut_x,result_ref.best_fit,'r--')
                            if np.logical_and(result_ref.params['cen'] > 0, result_ref.params['cen'] <4090):
                                peak_cens.append(result_ref.params['cen'])
                                peaks2.append(peak)
                                plt.plot(result_ref.params['cen'],np.max(cut_y),'kx')
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

            line_wave.append(peak_wave)
            line_pix.append(peak_px_sp)

            print ("\n     Continue with this order?")
            cont = str((input("(y/n) = ")))
        

        line_wave = np.asarray(line_wave, dtype=float)
        line_pix = np.asarray(line_pix, dtype=float)
        fit = np.polyfit(line_pix,line_wave,3)
        
        output = {}
        output[ord] = {}
        output[ord]['line_positions'] = line_pix
        output[ord]['known_wavelengths_air'] = line_wave

        np.save("./HR_R_linelist_P_"+str(ord),output)
        
        plt.close()
        
        fit_wave = np.polyval(fit,x)
        
        #plt.plot(wave2,spec2,'r')
        plt.plot(known_waveobs[ii],known_spec[ii]/np.max(known_spec[ii]),'k')
        plt.plot(fit_wave,FIBRE_P/np.max(FIBRE_P),'r')
        plt.show()


if arm =='H':

    ref_file1 = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2025/0610/reduced/bgoH202506100032.fits'
    ref_file = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0717/reduced/bgoH202207170027.fits'
    ref_file = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/Super_Arcs/HR_Super_Arc_H20220701.fits'
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

    #Order 00 range 3700-3760
    #Order 01 range 3730-3790
    #Order 02 range 3765-3815
    #Order 03 range 3795-3845
    #Order 04 range 3840-3870
    #Order 05 range 3850-3910
    #Order 06 range 3890-3940
    #Order 07 range 3930-3982
    #Order 08 range 3960-4012
    #Order 09 range 3990-4045
    #Order 10 range 4035-4085
    #Order 11 range 4065-4115
    #Order 12 range 4095-4145
    #Order 13 range 4135-4185
    #Order 14 range 4175-4225
    #Order 15 range 4205-4265
    #Order 16 range 4245-4307
    #Order 17 range 4285-4347
    #Order 18 range 4325-4390
    #Order 19 range 4365-4430
    #Order 20 range 4415-4470
    #Order 21 range 4455-4510
    #Order 22 range 4495-4560
    #Order 23 range 4539-4600
    #Order 24 range 4583-4644
    #Order 25 range 4630-4690
    #Order 26 range 4675-4740
    #Order 27 range 4725-4790
    #Order 28 range 4775-4840
    #Order 29 range 4825-4890
    #Order 30 range 4875-4940
    #Order 31 range 4925-5000
    #Order 32 range 4975-5045
    #Order 33 range 5035-5100
    #Order 34 range 5088-5160
    #Order 35 range 5150-5215
    #Order 36 range 5195-5275
    #Order 37 range 5250-5335
    #Order 38 range 5315-5395
    #Order 39 range 5380-5455
    #Order 40 range 5446-5516
    #Order 41 range 5510-5580

    ord=5
    order_min = 3850
    order_max = 3910

    axs[0].set_xlim(order_min,order_max)
    jj=np.where(np.logical_and(wave2>order_min,wave2<order_max))[0]
    ii=np.where(np.logical_and(known_waveobs>order_min,known_waveobs<order_max))[0]

    axs[0].plot(known_waveobs[ii], known_spec[ii]/np.max(known_spec[ii]),'k')
    axs[0].plot(wave2[jj],spec2[jj]/np.max(spec2[jj]*2),'b')
    #axs[0].plot(known_waveobs,known_spec,'g')
    atlas = np.loadtxt('thar_list_orig.txt',usecols=(0),unpack=True)
    axs[0].vlines(atlas,0,0.005,'r')
    #axs[0].set_ylim(-200,np.max(spec2[jj])-0000)
    axs[0].set_ylim(-0.02,0.2)

    if arm == 'H':
        line_wave = []
        line_pix = []
        with fits.open(ref_file) as hdu:
            with fits.open(ref_file1) as hdu1:
                for i in range(ord,ord+1):
                
                    peak_cens = []
                    peaks2 = []
                
                    x=np.arange(len(hdu['FIBRE_P'].data[i]))
                    
                    FIBRE_P=hdu['FIBRE_P'].data[i]
                    FIBRE_P[np.isnan(FIBRE_P)] = 0
                        
                    #axs[0].plot(known_waveobs[ii], known_spec[ii],'k')
                    #axs[0].plot(wave2,spec2*10,'r')

                    axs[1].plot(x,FIBRE_P,'g')
                    axs[1].set_ylim(-0.2,np.max(FIBRE_P)-0.0)
                    #axs[1].plot(x,hdu1['FIBRE_P'].data[i], 'b')
                    #axs[1].plot(hdu1['FIBRE_P'].data[i])
                    #fig.suptitle("ORDER "+str(i))
                    
                    peaks,_ = find_peaks(FIBRE_P,height=0.05,distance=6)
                    #plt.plot(peaks,FIBRE_P[peaks],'cx')
                    
                    #fit gausians to the peaks and over plot in red.
                    for peak in peaks:
                        if np.logical_and(peak > 8, peak < 2040):
                            cut_x = x[peak-6:peak+4]
                            cut_y =FIBRE_P[peak-6:peak+4]
                            gmod = Model(gaussian)
                            result_ref = gmod.fit(cut_y, x=cut_x, amp=Parameter('amp',value=np.max(cut_y)),cen=Parameter('cen',value=peak),wid=Parameter('wid',value=2.,min=0.5))

                            axs[1].plot(cut_x,result_ref.best_fit,'r--')
                            if np.logical_and(result_ref.params['cen'] > 0, result_ref.params['cen'] <2048):
                                peak_cens.append(result_ref.params['cen'])
                                peaks2.append(peak)
                                plt.plot(result_ref.params['cen'],np.max(cut_y),'kx')
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
            line_wave.append(peak_wave)
            line_pix.append(peak_px_sp)

            print ("\n     Continue with this order?")
            cont = str((input("(y/n) = ")))
        

        line_wave = np.asarray(line_wave, dtype=float)
        line_pix = np.asarray(line_pix, dtype=float)
        fit = np.polyfit(line_pix,line_wave,3)
        
        output = {}
        output[ord] = {}
        output[ord]['line_positions'] = line_pix
        output[ord]['known_wavelengths_vac'] = line_wave

        np.save("./HR_H_linelist_P_"+str(ord),output)
        
        plt.close()
        
        fit_wave = np.polyval(fit,x)
        
        plt.plot(wave2,spec2,'r')
        plt.plot(known_waveobs[ii],known_spec[ii]/100,'k')
        plt.plot(fit_wave,FIBRE_P*1000)
        plt.show()
