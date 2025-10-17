import numpy as np
from lmfit import  Model
from scipy.signal import chirp, find_peaks, peak_widths
from astropy.io import fits
import matplotlib.pyplot as plt

def gaussian(x, amp, cen, wid,offset):
    "1-d gaussian: gaussian(x, amp, cen, wid)"
    return (amp) * np.exp(-(x-cen)**2 /(2*wid**2))+offset

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

##### BLUE ###########################

#This holds the science spectrum and arc
file_blu = ("/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0729/reduced/bgoH202207290042.fits")
with fits.open(file_blu) as hdu:
    Fibre_O_Blu = hdu['FIBRE_O'].data
    Wave_O_Blu = hdu['WAVE_O'].data

gmod = Model(gaussian)
colour=0

all_ave_res = []
all_ave_wav = []
ord_min = []
fig, axs = plt.subplots(2,figsize=(10,8))
plt.subplots_adjust(hspace=0.5)


linelist = np.load("/Users/daniel/Documents/Work/SALT_Pipeline/HRSReduce/hrsreduce/wave_cal/HR_H_linelist_O.npy",allow_pickle=True).item()

for ord in range(42):

    arc = Fibre_O_Blu[ord]
    wave = Wave_O_Blu[ord]
    arc[np.isnan(arc)] = 0
    peaks = linelist[ord]['known_wavelengths_air']


    ord_ave_res = []
    ord_ave_wav = []
    ord_ave_res_err = []
    for j in range(len(peaks)):
        jj=np.where(np.logical_and(wave > peaks[j]-1, wave < peaks[j]+1))
        result = gmod.fit(arc[jj], x=wave[jj], amp=np.max(arc[jj]), cen=peaks[j], wid=0.03,offset=10)
        res = result.params['cen'].value/(result.params['wid'].value*2.3548)
            #print(result.fit_report())
        if res > 2000 and res < 900000 and result.params['wid'].value*2.3548 > 0 and result.params['wid'].value*2.3548 < 0.2 :
            if result.params['amp'].value > 500:# 0.0544 0.1and result.params['wid'].stderr*2.3548 <0.01:
                axs[0].plot(wave[jj],arc[jj])
                axs[0].plot(wave[jj], result.best_fit,'--')


#Comapre with Alexei

#This holds the science spectrum and arc
file_AK = ("/Users/daniel/Documents/Work/SALT_Pipeline/HRSReduce/MIDAS_Reductions/20220729/mbgphH202207290042_1wm.fits")
with fits.open(file_AK) as hdu:
    specaxis = 1
    header = hdu[0].header
    AK_spec=hdu[0].data
    wave_step = header['CDELT%s' % (specaxis)]
    wave_base = header['CRVAL%s' % (specaxis)]
    reference_pixel = header['CRPIX%s' % (specaxis)]
    xconv = lambda v: ((v-reference_pixel+1)*wave_step+wave_base)
    AK_wave = xconv(np.arange(len(AK_spec)))

all_ave_res_AK = []
all_ave_wav_AK = []
    
for ord in range(42):
    peaks = linelist[ord]['known_wavelengths_air']

    ord_ave_res_AK = []
    ord_ave_res_err_AK = []
    ord_min = []
    colour=0
    for j in range(len(peaks)):
        if peaks[j] > np.min(AK_wave):
            jj=np.where(np.logical_and(AK_wave > peaks[j]-1, AK_wave < peaks[j]+1))
            result = gmod.fit(AK_spec[jj], x=AK_wave[jj], amp=np.max(AK_spec[jj]), cen=peaks[j], wid=0.03,offset=10)
            res = result.params['cen'].value/(result.params['wid'].value*2.3548)

            if res > 2000 and res < 900000 and result.params['wid'].value*2.3548 > 0.0 and result.params['wid'].value*2.3548 < 0.2 :
                if result.params['amp'].value > 500:# and result.params['wid'].stderr*2.3548 <0.01:
                    if result.params['wid'].stderr is not None:
                        axs[0].plot(AK_wave[jj],AK_spec[jj])
                        axs[0].plot(AK_wave[jj], result.best_fit,':')
                    



##### Red ###########################

#This holds the science spectrum and arc
file_red = ("/Users/daniel/Desktop/SALT_HRS_DATA/Red/2022/0729/reduced/bgoR202207290042.fits")
with fits.open(file_red) as hdu:
    Fibre_O_Red = hdu['FIBRE_O'].data
    Wave_O_Red = hdu['WAVE_O'].data

colour=0

all_ave_res = []
all_ave_wav = []
ord_min = []

linelist = np.load("/Users/daniel/Documents/Work/SALT_Pipeline/HRSReduce/hrsreduce/wave_cal/HR_R_linelist_O.npy",allow_pickle=True).item()

for ord in range(33):

    arc = Fibre_O_Red[ord]
    wave = Wave_O_Red[ord]
    arc[np.isnan(arc)] = 0
    peaks = linelist[ord]['known_wavelengths_air']
    
    ord_ave_res = []
    ord_ave_wav = []
    ord_ave_res_err = []
    for j in range(len(peaks)):
        jj=np.where(np.logical_and(wave > peaks[j]-1, wave < peaks[j]+1))[0]
        if len(jj) > 10:
            result = gmod.fit(arc[jj], x=wave[jj], amp=np.max(arc[jj]), cen=peaks[j], wid=0.03,offset=10)
            res = result.params['cen'].value/(result.params['wid'].value*2.3548)
            if res > 3000 and res < 900000 and result.params['wid'].value*2.3548 > 0 and result.params['wid'].value*2.3548 < 0.2 :
                if result.params['amp'].value > 500:# and result.params['wid'].stderr*2.3548 <0.01:
                    if result.params['wid'].stderr is not None:
                        axs[1].plot(wave[jj],arc[jj])
                        axs[1].plot(wave[jj], result.best_fit,'--')

#Comapre with Alexei

#This holds the science spectrum and arc
file_AK = ("/Users/daniel/Documents/Work/SALT_Pipeline/HRSReduce/MIDAS_Reductions/20220729/mbgphR202207290042_1wm.fits")
with fits.open(file_AK) as hdu:
    specaxis = 1
    header = hdu[0].header
    AK_spec=hdu[0].data
    wave_step = header['CDELT%s' % (specaxis)]
    wave_base = header['CRVAL%s' % (specaxis)]
    reference_pixel = header['CRPIX%s' % (specaxis)]
    xconv = lambda v: ((v-reference_pixel+1)*wave_step+wave_base)
    AK_wave = xconv(np.arange(len(AK_spec)))

all_ave_res_AK = []
all_ave_wav_AK = []
    
for ord in range(33):
    peaks = linelist[ord]['known_wavelengths_air']

    ord_ave_res_AK = []
    ord_ave_res_err_AK = []
    ord_min = []
    colour=0
    for j in range(len(peaks)):
        if peaks[j] < np.max(AK_wave):
            jj=np.where(np.logical_and(AK_wave > peaks[j]-1, AK_wave < peaks[j]+1))[0]
            result = gmod.fit(AK_spec[jj], x=AK_wave[jj], amp=np.max(AK_spec[jj]), cen=peaks[j], wid=0.03,offset=10)
            res = result.params['cen'].value/(result.params['wid'].value*2.3548)

            if res > 3000 and res < 900000 and result.params['wid'].value*2.3548 > 0.0 and result.params['wid'].value*2.3548 < 0.2 :# and result.params['wid'].stderr*2.3548 <0.01:
                if result.params['wid'].stderr is not None:
                    if result.params['amp'].value > 500:
                        axs[1].plot(AK_wave[jj],AK_spec[jj])
                        axs[1].plot(AK_wave[jj], result.best_fit,':')

plt.show()



