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

def gaussian(x, amp, cen, wid, offset):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    #return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))
    #return (1./(wid*np.sqrt(2*np.pi))) * np.exp(-(x-cen)**2 / (2*wid**2))
    return amp*np.exp(-(x-cen)**2/(2*wid**2)) + offset
            
arm = 'R'


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

if arm == 'R':

    ref_file1 = '/Users/daniel/Desktop/SALT_HRS_DATA/Red/2025/0610/reduced/bgoR202506100032.fits'
    ref_file = '/Users/daniel/Desktop/SALT_HRS_DATA/Red/2022/0717/reduced/bgoHR02207170027.fits'
    ref_file = '/Users/daniel/Desktop/SALT_HRS_DATA/Red/2022/Super_Arcs/HR_Super_Arc_R20220701.fits'
    ref_flat = '/Users/daniel/Desktop/SALT_HRS_DATA/Red/2025/0610/reduced/HR_Master_Flat_R20250610.fits'
    #ref_file = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2025/Super_Arcs/HR_Super_Arc_H20250101.fits'
    
    file2="/Users/daniel/Documents/Work/SALT_Pipeline/Alexei Solutions/HR_arcs/mbgphR202207170027_2wm.fits"

    with fits.open(file2) as hdu:
        header = hdu[0].header
        spec2=hdu[0].data
        wave_step = header['CDELT%s' % (specaxis)]
        wave_base = header['CRVAL%s' % (specaxis)]
        reference_pixel = header['CRPIX%s' % (specaxis)]
        xconv = lambda v: ((v-reference_pixel+1)*wave_step+wave_base)
        wave2 = xconv(np.arange(len(spec2)))
        
    fig, axs = plt.subplots(1,1,figsize=(40,7))

    #Order 00 range 5420-5540
    #Order 01 range 5480-5600
    #Order 02 range 5545-5670
    #Order 03 range 5610-5740
    #Order 04 range 5680-5810
    #Order 05 range 5750-5880
    #Order 06 range 5820-5955
    #Order 07 range 5900-6030
    #Order 08 range 5960-6110
    #Order 09 range 6040-6190
    #Order 10 range 6120-6275
    #Order 11 range 6220-6360
    #Order 12 range 6305-6445
    #Order 13 range 6390-6535
    #Order 14 range 6480-6635
    #Order 15 range 6575-6735
    #Order 16 range 6665-6835
    #Order 17 range 6755-6935
    #Order 18 range 6845-7035
    #Order 19 range 6970-7127
    #Order 20 range 7080-7240
    #Order 21 range 7190-7350
    #Order 22 range 7300-7470
    #Order 23 range 7420-7590
    #Order 24 range 7540-7710
    #Order 25 range 7670-7840
    #Order 26 range 7800-7980
    #Order 27 range 7930-8120
    #Order 28 range 8060-8260
    #Order 29 range 8190-8400
    #Order 30 range 8360-8570
    #Order 31 range 8520-8740
    #Order 32 range 8685-8910


    ord=0
    order_min = 5420
    order_max = 5540

#    jj=np.where(np.logical_and(wave2>order_min-5,wave2<order_max+5))[0]
#    plt.plot(wave2[jj], spec2[jj]/np.max(spec2[jj]),'m',alpha=0.5)

#    with fits.open(ref_file) as hdu:
#        i=ord
#        x=np.arange(len(hdu['FIBRE_P'].data[i]))
#        lines = np.load('./HR_R_linelist_P_'+str(ord)+'.npy',allow_pickle=True).item()
#        try:
#            fit = np.polyfit(lines[i]['line_positions'],lines[i]['known_wavelengths_vac'],3)
#        except:
#            fit = np.polyfit(lines[i]['line_positions'],lines[i]['known_wavelengths_air'],3)
#        fit_wave = np.polyval(fit,x)
#        FIBRE_P=hdu['FIBRE_P'].data[i]
#        FIBRE_P[np.isnan(FIBRE_P)] = 0
#        FIBRE_P -=np.median(FIBRE_P)
#        FIBRE_P -=np.min(FIBRE_P[10:-10])
#        FIBRE_P/=(np.max(FIBRE_P))
#        plt.plot(fit_wave,FIBRE_P,'k')
#        plt.show()
        
        
    ii=np.where(np.logical_and(known_waveobs>order_min-5,known_waveobs<order_max+5))[0]
    plt.plot(known_waveobs[ii], (known_spec[ii]/np.max(known_spec[ii])),'c')

    plt.xlim(order_min,order_max)
    #Change these for scaling
#    plt.ylim(-0.001,0.06)
    atlas, atlas_int = np.loadtxt('../New_Th_linelist_air.list',usecols=(0,1),unpack=True)

    gmod = Model(gaussian)
    
    line_wave = []
    line_pix = []
    with fits.open(ref_file) as hdu:
        for i in range(ord,ord+1):
        
            if ord == 32:
                x=np.arange(len(hdu['FIBRE_P'].data[i][:-1800]))
            else:
                x=np.arange(len(hdu['FIBRE_P'].data[i]))
            
            lines = np.load('./HR_R_linelist_P_cl_'+str(ord)+'.npy',allow_pickle=True).item()
            try:
                fit = np.polyfit(lines[i]['line_positions'],lines[i]['known_wavelengths_vac'],3)
            except:
                fit = np.polyfit(lines[i]['line_positions'],lines[i]['known_wavelengths_air'],3)
            #fit = ([-1.52192313e-18,  9.33330773e-15, -2.12069425e-11,  1.92488133e-08, -1.88106710e-06,  1.72081971e-02,  3.70900000e+03])
            fit_wave = np.polyval(fit,x)
        
            peak_cens_wave = []
            peak_cens_pix = []
            peaks2 = []

            FIBRE_P=hdu['FIBRE_P'].data[i]
            FIBRE_P[np.isnan(FIBRE_P)] = 0
            FIBRE_P -=np.median(FIBRE_P)
            FIBRE_P -=np.min(FIBRE_P[10:-10])
#            FIBRE_P/=(np.max(FIBRE_P))
            FIBRE_P *= 1000.
            if ord == 31:
                FIBRE_P -= 0.0405
            if ord == 32:
                FIBRE_P -= 0.635
                FIBRE_P = FIBRE_P[:-1800]
            plt.plot(fit_wave,FIBRE_P,'k')

            #Change height for when there are very strong lines and need to detect weaker ones.
            peaks,_ = find_peaks(FIBRE_P,height=100,distance=6)
            plt.plot(fit_wave[peaks],FIBRE_P[peaks],'kx')
            
            #fit gausians to the peaks and over plot in green.
            for peak in peaks:
                if np.logical_and(peak > 8, peak < 4092):
                    #Do fit in Wavelength space
                    cut_x = fit_wave[peak-5:peak+5]
                    cut_y =FIBRE_P[peak-5:peak+5]

                    result_ref = gmod.fit(cut_y, x=cut_x, amp=Parameter('amp',value=np.max(cut_y)),cen=Parameter('cen',value=fit_wave[peak]),wid=Parameter('wid',value=0.05,min=0.001), offset=10)

                    plt.plot(cut_x,result_ref.best_fit,'g--')
                    if np.logical_and(result_ref.params['cen'] > order_min, result_ref.params['cen'] <order_max):
                        peak_cens_wave.append(result_ref.params['cen'])
                        peaks2.append(peak)
                        plt.plot(result_ref.params['cen'],np.max(cut_y),'bx')
                    
                        #Do fit in pixel space
                        cut_x = x[peak-6:peak+6]
                        cut_y =FIBRE_P[peak-6:peak+6]
                        result_ref = gmod.fit(cut_y, x=cut_x, amp=Parameter('amp',value=np.max(cut_y)),cen=Parameter('cen',value=x[peak]),wid=Parameter('wid',value=2,min=1), offset=10)

                        peak_cens_pix.append(result_ref.params['cen'].value)

    aa = np.where(np.logical_and(atlas > np.min(fit_wave)-5, atlas < np.max(fit_wave)+5))[0]
    plt.vlines(atlas[aa],0,atlas_int[aa],'r')
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

    np.save("./HR_R_linelist_P_cl1_"+str(ord),output)
    fit = np.polyfit(line_pix,line_wave,6)
    fit_wave = np.polyval(fit,x)

    plt.plot(known_waveobs[ii], known_spec[ii]/np.max(known_spec[ii]),'c')
    plt.plot(fit_wave,FIBRE_P,'k')
    plt.xlim(order_min,order_max)
    plt.show()
    
    plt.plot(x,fit_wave)
    plt.plot(line_pix,line_wave,'rx')
    plt.show()

    

if arm == 'H':

    ref_file1 = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2025/0610/reduced/bgoH202506100032.fits'
    ref_file = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0717/reduced/bgoH202207170027.fits'
    ref_file = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/Super_Arcs/HR_Super_Arc_H20220701.fits'
    ref_flat = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2025/0610/reduced/HR_Master_Flat_H20250610.fits'
    #ref_file = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2025/Super_Arcs/HR_Super_Arc_H20250101.fits'
    
    file2="/Users/daniel/Documents/Work/SALT_Pipeline/Alexei Solutions/HR_arcs/mbgphH202207170027_2wm.fits"

    with fits.open(file2) as hdu:
        header = hdu[0].header
        spec2=hdu[0].data
        wave_step = header['CDELT%s' % (specaxis)]
        wave_base = header['CRVAL%s' % (specaxis)]
        reference_pixel = header['CRPIX%s' % (specaxis)]
        xconv = lambda v: ((v-reference_pixel+1)*wave_step+wave_base)
        wave2 = xconv(np.arange(len(spec2)))
        
    fig, axs = plt.subplots(1,1,figsize=(20,7))

    #Order 00 range 3720-3755
    #Order 01 range 3745-3795
    #Order 02 range 3760-3815
    #Order 03 range 3790-3850
    #Order 04 range 3825-3880
    #Order 05 range 3850-3910
    #Order 06 range 3885-3945
    #Order 07 range 3930-3982
    #Order 08 range 3960-4012
    #Order 09 range 3990-4045
    #Order 10 range 4030-4085
    #Order 11 range 4065-4115
    #Order 12 range 4095-4155
    #Order 13 range 4135-4190
    #Order 14 range 4175-4225
    #Order 15 range 4205-4265
    #Order 16 range 4245-4307
    #Order 17 range 4285-4347
    #Order 18 range 4325-4390
    #Order 19 range 4365-4430
    #Order 20 range 4405-4470
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
    #Order 35 range 5140-5215
    #Order 36 range 5195-5275
    #Order 37 range 5250-5335
    #Order 38 range 5315-5395
    #Order 39 range 5380-5455
    #Order 40 range 5446-5516
    #Order 41 range 5510-5580

    ord=2
    order_min = 3760
    order_max = 3815

    ii=np.where(np.logical_and(known_waveobs>order_min-5,known_waveobs<order_max+5))[0]
    plt.plot(known_waveobs[ii], (known_spec[ii]/np.max(known_spec[ii])),'c')

    #jj=np.where(np.logical_and(wave2>order_min-5,wave2<order_max+5))[0]
    #plt.plot(wave2[jj], spec2[jj]/np.max(spec2[jj]),'b')

    plt.xlim(order_min,order_max)
    plt.ylim(-0.05,0.5)
    atlas = np.loadtxt('thar_list_orig.txt',usecols=(0),unpack=True)

    
    line_wave = []
    line_pix = []
    with fits.open(ref_file) as hdu:
        for i in range(ord,ord+1):
        
            x=np.arange(len(hdu['FIBRE_P'].data[i]))
            
            lines = np.load('./HR_H_linelist_P_cl4_'+str(ord)+'.npy',allow_pickle=True).item()
            fit = np.polyfit(lines[i]['line_positions'],lines[i]['known_wavelengths_air'],6)
            #fit = ([-1.52192313e-18,  9.33330773e-15, -2.12069425e-11,  1.92488133e-08, -1.88106710e-06,  1.72081971e-02,  3.70900000e+03])
            fit_wave = np.polyval(fit,x)
        
            peak_cens_wave = []
            peak_cens_pix = []
            peaks2 = []

            FIBRE_P=hdu['FIBRE_P'].data[i]
            FIBRE_P[np.isnan(FIBRE_P)] = 0
            FIBRE_P -=np.median(FIBRE_P)
            FIBRE_P/=(np.max(FIBRE_P))
            plt.plot(fit_wave,FIBRE_P,'k')


            peaks,_ = find_peaks(FIBRE_P,height=0.012,distance=6)
            plt.plot(fit_wave[peaks],FIBRE_P[peaks],'kx')
            
            #fit gausians to the peaks and over plot in green.
            for peak in peaks:
                if np.logical_and(peak > 8, peak < 2040):
                    #Do fit in Wavelength space
                    cut_x = fit_wave[peak-5:peak+5]
                    cut_y =FIBRE_P[peak-5:peak+5]
                    result_ref = gmod.fit(cut_y, x=cut_x, amp=Parameter('amp',value=np.max(cut_y)),cen=Parameter('cen',value=fit_wave[peak]),wid=Parameter('wid',value=0.05,min=0.001), offset=10)

                    plt.plot(cut_x,result_ref.best_fit,'g--')
                    if np.logical_and(result_ref.params['cen'] > order_min, result_ref.params['cen'] <order_max):
                        peak_cens_wave.append(result_ref.params['cen'])
                        peaks2.append(peak)
                        plt.plot(result_ref.params['cen'],np.max(cut_y),'bx')
                    
                        #Do fit in pixel space
                        cut_x = x[peak-6:peak+6]
                        cut_y =FIBRE_P[peak-6:peak+6]
                        result_ref = gmod.fit(cut_y, x=cut_x, amp=Parameter('amp',value=np.max(cut_y)),cen=Parameter('cen',value=x[peak]),wid=Parameter('wid',value=2,min=1), offset=10)

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

    np.save("./HR_H_linelist_P_cl5_"+str(ord),output)
    fit = np.polyfit(line_pix,line_wave,6)
    fit_wave = np.polyval(fit,x)

    plt.plot(known_waveobs[ii], known_spec[ii]/np.max(known_spec[ii]),'c')
    plt.plot(fit_wave,FIBRE_P,'k')
    plt.xlim(order_min,order_max)
    plt.show()
    
    plt.plot(x,fit_wave)
    plt.plot(line_pix,line_wave,'rx')
    plt.show()

    
