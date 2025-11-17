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

##This holds the science spectrum and arc
#file_blu = ("/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/Super_Arcs/HR_Super_Arc_H20220701.fits")
#with fits.open(file_blu) as hdu:
#    Fibre_O_Blu = hdu['FIBRE_O'].data


gmod = Model(gaussian)
colour=0

all_ave_res = []
all_ave_wav = []
ord_min = []
fig, axs = plt.subplots(2,figsize=(10,8))
plt.subplots_adjust(hspace=0.5)
fig.suptitle("Determination of the HR Resolving power")
axs[0].set_title("Calculated Resolution")
axs[1].set_title("FWHM of Arc lines")
axs[1].set_xlabel("Wavelength Å")
axs[0].set_xlabel("Wavelength Å")
axs[0].set_ylabel("Resolution (λ/Δλ)")
axs[1].set_ylabel("FWHM (Å)")

peak_waves = []

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
        jj=np.where(np.logical_and(wave > peaks[j]-0.5, wave < peaks[j]+0.5))
        result = gmod.fit(arc[jj], x=wave[jj], amp=np.max(arc[jj]), cen=peaks[j], wid=0.03,offset=10)
#        plt.plot(wave[jj],arc[jj])
#        plt.plot(wave[jj],result.best_fit,'--')
#        plt.show()
            #print(wave[peaks[j]],result.params['cen'].value,result.params['wid'].value*2.3548,result.params['cen'].value/(result.params['wid'].value*2.3548))
        res = result.params['cen'].value/(result.params['wid'].value*2.3548)
            #print(result.fit_report())
        if res > 20000 and res < 90000 and result.params['wid'].value*2.3548 > 0 and result.params['wid'].value*2.3548 < 0.2 :
            if result.params['amp'].value > 0:

#                plt.plot(wave, result.init_fit, 'g-')
#                plt.plot(wave[peaks[j]-10:peaks[j]+10], result.best_fit, 'r-')
#                plt.show()
                axs[0].plot(peaks[j],res,'o',alpha=0.5,color=str("C"+str(colour)))
                axs[1].plot(peaks[j],result.params['wid'].value*2.3548,'o',alpha=0.5,c=str("C"+str(colour)))
                ord_ave_res.append(res)
                ord_ave_res_err.append(result.params['wid'].stderr*2.3548)
            
                #all_ave_res.append(result.params['wid'].value*2.3548)
                ord_ave_wav.append(peaks[j])
                #all_ave_wav.append(wave[peaks[j]])
                peak_waves.append(result.params['cen'].value)
    if len(ord_ave_wav)>0:
        #ord_ave_res_err = np.array(ord_ave_res_err)
        #ord_fit_P = np.polyfit(ord_ave_wav,ord_ave_res,deg=2,w=1./ord_ave_res_err)
        #ord_fit = np.polyval(ord_fit_P,ord_ave_wav)
        #axs[1].plot(ord_ave_wav,ord_fit,'k')
        #ord_min.append(np.min(ord_fit))
        all_ave_wav.append(np.mean(ord_ave_wav))
        all_ave_res.append(np.median(ord_ave_res))
    else:
        print ("No lines fitted in Blue Order:",ord)
                
    ord_ave_wav=np.array(ord_ave_wav)
    ord_ave_res=np.array(ord_ave_res)
#    print("MAX",np.max(ord_ave_res))
    #axs[0].plot(np.mean(ord_ave_wav),np.median(ord_ave_res),'ro')
    #axs[1].plot(np.mean(ord_ave_wav),ord_min,'go')
    colour += 1
#    plt.show()

P=np.polyfit(all_ave_wav,all_ave_res,deg=2)
#axs[1].plot(all_ave_wav,all_ave_res,'ro')
wave2=np.arange(3720,5550,1)

yfit=np.polyval(P,all_ave_wav)
yfit2=np.polyval(P,wave2)
#axs[1].plot(all_ave_wav,yfit,'k')

axs[1].plot(wave2,wave2/yfit2,linestyle='-',color='blue',label='HRSReduce',lw=5)
axs[0].plot(wave2,yfit2,linestyle='-',color='blue',label='HRSReduce',lw=5)
print("MIN MAX of Blue HRS Reduce",np.min(yfit2),np.max(yfit2))

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
            jj=np.where(np.logical_and(AK_wave > peaks[j]-0.5, AK_wave < peaks[j]+0.5))
            result = gmod.fit(AK_spec[jj], x=AK_wave[jj], amp=np.max(AK_spec[jj]), cen=peaks[j], wid=0.03,offset=10)
            res = result.params['cen'].value/(result.params['wid'].value*2.3548)
#            plt.plot(wave2[idx-10:idx+10],spec2[idx-10:idx+10])
#            plt.plot(wave2[idx-10:idx+10],result.best_fit,'g--')

            if res > 20000 and res < 90000 and result.params['wid'].value*2.3548 > 0.0 and result.params['wid'].value*2.3548 < 0.2 :
                if result.params['amp'].value > 500:# and result.params['wid'].stderr*2.3548 <0.01:
                    if result.params['wid'].stderr is not None:
        #                plt.plot(wave2[idx-10:idx+10],spec2[idx-10:idx+10])
        #                plt.plot(wave2[idx-10:idx+10],result.best_fit,'g--')
        #                plt.show()
    #                    axs[0].plot(peaks[j],res,'x',color=str("C"+str(colour)))
    #                    axs[1].plot(peaks[j],result.params['wid'].value*2.3548,'x',c=str("C"+str(colour)))
                        ord_ave_res_AK.append(res)
                        ord_ave_res_err_AK.append(result.params['wid'].stderr*2.3548)
                    
                        all_ave_res_AK.append(res)
                        #ord_ave_wav_AK.append(peaks[j])
                        all_ave_wav_AK.append(peaks[j])


#ord_ave_wav=np.array(ord_ave_wav)
#ord_ave_res=np.array(ord_ave_res)
#
    colour += 1
#
P=np.polyfit(all_ave_wav_AK,all_ave_res_AK,deg=2)
#axs[1].plot(all_ave_wav,all_ave_res,'ro')
wave2=np.arange(3720,5550,1)

yfit=np.polyval(P,all_ave_wav_AK)
yfit2=np.polyval(P,wave2)
#axs[1].plot(all_ave_wav,yfit,'k')
axs[1].plot(wave2,wave2/yfit2,'b',label='MIDAS Blu')
axs[0].plot(wave2,yfit2,'b',label='MIDAS Blu')
print("MIN MAX of Blue MIDAS ",np.min(yfit2),np.max(yfit2))
axs[0].legend()
axs[1].legend()


##### Red ###########################

#This holds the science spectrum and arc
file_red = ("/Users/daniel/Desktop/SALT_HRS_DATA/Red/2022/0729/reduced/bgoR202207290042.fits")
with fits.open(file_red) as hdu:
    Fibre_O_Red = hdu['FIBRE_O'].data
    Wave_O_Red = hdu['WAVE_O'].data
#file_red = ("/Users/daniel/Desktop/SALT_HRS_DATA/Red/2022/Super_Arcs/HR_Super_Arc_R20220701.fits")
#with fits.open(file_red) as hdu:
#    Fibre_O_Red = hdu['FIBRE_O'].data

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
        jj=np.where(np.logical_and(wave > peaks[j]-0.5, wave < peaks[j]+0.5))[0]
        if len(jj) > 10:
            result = gmod.fit(arc[jj], x=wave[jj], amp=np.max(arc[jj]), cen=peaks[j], wid=0.03,offset=10)
    #            plt.plot(wave[peaks[j]-10:peaks[j]+10],arc[peaks[j]-10:peaks[j]+10])
                #print(wave[peaks[j]],result.params['cen'].value,result.params['wid'].value*2.3548,result.params['cen'].value/(result.params['wid'].value*2.3548))
            res = result.params['cen'].value/(result.params['wid'].value*2.3548)
                #print(result.fit_report())
            if res > 30000 and res < 100000 and result.params['wid'].value*2.3548 > 0 and result.params['wid'].value*2.3548 < 0.2 :
                if result.params['amp'].value > 0:# and result.params['wid'].stderr*2.3548 <0.01:
                    if result.params['wid'].stderr is not None:

            #                plt.plot(wave, result.init_fit, 'g-')
            #                plt.plot(wave[peaks[j]-10:peaks[j]+10], result.best_fit, 'r-')
            #                plt.show()
                        axs[0].plot(peaks[j],res,'o',alpha=0.5,color=str("C"+str(colour)))
                        axs[1].plot(peaks[j],result.params['wid'].value*2.3548,'o',alpha=0.5,c=str("C"+str(colour)))
                        ord_ave_res.append(res)
                        ord_ave_res_err.append(result.params['wid'].stderr*2.3548)
                        
                        #all_ave_res.append(result.params['wid'].value*2.3548)
                        ord_ave_wav.append(peaks[j])
                        #all_ave_wav.append(wave[peaks[j]])
                        peak_waves.append(result.params['cen'].value)
    if len(ord_ave_wav)>0:
        #ord_ave_res_err = np.array(ord_ave_res_err)
        #ord_fit_P = np.polyfit(ord_ave_wav,ord_ave_res,deg=2,w=1./ord_ave_res_err)
        #ord_fit = np.polyval(ord_fit_P,ord_ave_wav)
        #axs[1].plot(ord_ave_wav,ord_fit,'k')
        #ord_min.append(np.min(ord_fit))
        all_ave_wav.append(np.mean(ord_ave_wav))
        all_ave_res.append(np.median(ord_ave_res))
    else:
        print ("No lines fitted in Red Order:",ord)
                
    ord_ave_wav=np.array(ord_ave_wav)
    ord_ave_res=np.array(ord_ave_res)
#    print("MAX",np.max(ord_ave_res))
    #axs[0].plot(np.mean(ord_ave_wav),np.median(ord_ave_res),'ro')
    #axs[1].plot(np.mean(ord_ave_wav),ord_min,'go')
    colour += 1
#    plt.show()


P=np.polyfit(all_ave_wav,all_ave_res,deg=2)
#axs[1].plot(all_ave_wav,all_ave_res,'ro')
wave2=np.arange(5550,8900,1)

yfit=np.polyval(P,all_ave_wav)
yfit2=np.polyval(P,wave2)
#axs[1].plot(all_ave_wav,yfit,'k')

axs[1].plot(wave2,wave2/yfit2,linestyle='-',color='red',label='HRSReduce',lw=5)
axs[0].plot(wave2,yfit2,linestyle='-',color='red',label='HRSReduce',lw=5)
print("MIN MAX of Red HRS Reduce",np.min(yfit2),np.max(yfit2))
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
            jj=np.where(np.logical_and(AK_wave > peaks[j]-0.5, AK_wave < peaks[j]+0.5))[0]
            result = gmod.fit(AK_spec[jj], x=AK_wave[jj], amp=np.max(AK_spec[jj]), cen=peaks[j], wid=0.03,offset=10)
            res = result.params['cen'].value/(result.params['wid'].value*2.3548)
#            axs[0].plot(AK_wave[jj],AK_spec[jj])
#            axs[0].plot(AK_wave[jj],result.best_fit,'g--')

            if res > 30000 and res < 100000 and result.params['wid'].value*2.3548 > 0.0 and result.params['wid'].value*2.3548 < 0.2 :
                if result.params['amp'].value > 500:# and result.params['wid'].stderr*2.3548 <0.01:
                    if result.params['wid'].stderr is not None:
        #                plt.plot(wave2[idx-10:idx+10],spec2[idx-10:idx+10])
        #                plt.plot(wave2[idx-10:idx+10],result.best_fit,'g--')
        #                plt.show()
    #                    axs[0].plot(peaks[j],res,'x',color=str("C"+str(colour)))
    #                    axs[1].plot(peaks[j],result.params['wid'].value*2.3548,'x',c=str("C"+str(colour)))
                        ord_ave_res_AK.append(res)
                        ord_ave_res_err_AK.append(result.params['wid'].stderr*2.3548)
                    
                        all_ave_res_AK.append(res)
                        #ord_ave_wav_AK.append(peaks[j])
                        all_ave_wav_AK.append(peaks[j])


#ord_ave_wav=np.array(ord_ave_wav)
#ord_ave_res=np.array(ord_ave_res)
#
    colour += 1
#
P=np.polyfit(all_ave_wav_AK,all_ave_res_AK,deg=2)
#axs[1].plot(all_ave_wav,all_ave_res,'ro')
wave2=np.arange(5550,8900,1)

yfit=np.polyval(P,all_ave_wav_AK)
yfit2=np.polyval(P,wave2)
#axs[1].plot(all_ave_wav,yfit,'k')
axs[1].plot(wave2,wave2/yfit2,'r',label='MIDAS Red')
axs[0].plot(wave2,yfit2,'r',label='MIDAS Red')
print("MIN MAX of Red MIDAS",np.min(yfit2),np.max(yfit2))

axs[0].legend()
axs[1].legend()

plt.savefig('Resolution_2.png', dpi=900)
plt.show()



