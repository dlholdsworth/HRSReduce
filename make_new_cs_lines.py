import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.signal import find_peaks
import scipy.constants as conts
from scipy.optimize import curve_fit
import time
import warnings
from scipy.special import erf

def integrate_gaussian(x, a, mu, sig, const, int_width=0.5):
    """
    Returns the integral of a Gaussian over a specified symmetric range.
    Gaussian given by:
    g(x) = a * exp(-(x - mu)**2 / (2 * sig**2)) + const
    Args:
        x (float): the central value over which the integral will be calculated
        a (float): the amplitude of the Gaussian
        mu (float): the mean of the Gaussian
        sig (float): the standard deviation of the Gaussian
        const (float): the Gaussian's offset from zero (i.e. the value of
            the Gaussian at infinity).
        int_width (float): the width of the range over which the integral will
            be calculated (i.e. if I want to calculate from 0.5 to 1, I'd set
            x = 0.75 and int_width = 0.25).
    Returns:
        float: the integrated value
    """

    integrated_gaussian_val = a * 0.5 * (
        erf((x - mu + int_width) / (np.sqrt(2) * sig)) -
        erf((x - mu - int_width) / (np.sqrt(2) * sig))
        ) + (const * 2 * int_width)
    
    return integrated_gaussian_val
    
def fit_gaussian_integral(x, y):
    """
    Fits a continuous Gaussian to a discrete set of x and y datapoints
    using scipy.curve_fit
    
    Args:
        x (np.array): x data to be fit
        y (np.array): y data to be fit
    Returns a tuple of:
        list: best-fit parameters [a, mu, sigma**2, const]
        line_dict: dictionary of best-fit parameters, wav, flux, model, etc.
    """
    
    line_dict = {} # initialize dictionary to store fit parameters, etc.

    x = np.ma.compressed(x)
    y = np.ma.compressed(y)
    i = np.argmax(y[len(y) // 4 : len(y) * 3 // 4]) + len(y) // 4
    
    p0 = [y[i], x[i], 1.5, np.min(y)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, pcov = curve_fit(integrate_gaussian, x, y, p0=p0, maxfev=1000000)
        pcov[np.isinf(pcov)] = 0 # convert inf to zero
        pcov[np.isnan(pcov)] = 0 # convert nan to zero
        line_dict['amp']   = popt[0] # optimized parameters
        line_dict['mu']    = popt[1] # "
        line_dict['sig']   = popt[2] # "
        line_dict['const'] = popt[3] # ""
        line_dict['covar'] = pcov    # covariance
        line_dict['data']  = y
        line_dict['model'] = integrate_gaussian(x, *popt)
        line_dict['quality'] = 'good' # fits are assumed good until marked bad elsewhere
        
    cal_type = 'ThAr'
    if cal_type == 'ThAr':
        # Quality Checks for Gaussian Fits
        
        if max(y) == 0:
            print('Amplitude is 0')
            return(None, line_dict)
        
        chi_squared_threshold = int(2)

        # Calculate chi^2
        predicted_y = integrate_gaussian(x, *popt)
        chi_squared = np.sum(((y - predicted_y) ** 2) / np.var(y))
        line_dict['chi2'] = chi_squared

        # Calculate RMS of residuals for Gaussian fit
        rms_residual = np.sqrt(np.mean(np.square(y - predicted_y)))
        line_dict['rms'] = np.sqrt(np.mean(np.square(rms_residual)))

        #rms_threshold = 1000 # RMS Quality threshold
        #disagreement_threshold = 1000 # Disagreement with initial guess threshold
        #asymmetry_threshold = 1000 # Asymmetry in residuals threshold

        # Calculate disagreement between Gaussian fit and initial guess
        #disagreement = np.abs(popt[1] - p0[1])
        line_dict['mu_diff'] = popt[1] - p0[1] # disagreement between Gaussian fit and initial guess

        ## Check for asymmetry in residuals
        #residuals = y - predicted_y
        #left_residuals = residuals[:len(residuals)//2]
        #right_residuals = residuals[len(residuals)//2:]
        #asymmetry = np.abs(np.mean(left_residuals) - np.mean(right_residuals))
        
        # Run checks against defined quality thresholds
        if (chi_squared > chi_squared_threshold):
            print("Chi squared exceeded the threshold for this line. Line skipped")
            return None, line_dict

        # Check if the Gaussian amplitude is positive, the peak is higher than the wings, or the peak is too high
        if popt[0] <= 0 or popt[0] <= popt[3] or popt[0] >= 500*max(y):
            line_dict['quality'] = 'bad_amplitude'  # Mark the fit as bad due to bad amplitude or U shaped gaussian
            print('Bad amplitude detected')
            return None, line_dict

    return (popt, line_dict)


def gaussval2(x, a, mu, sig, const):
    return a * np.exp(-((x - mu) ** 2) / (2 * sig)) + const
    
def gaussfit3(x, y):
    """A very simple (and relatively fast) gaussian fit
    gauss = A * exp(-(x-mu)**2/(2*sig**2)) + offset

    Parameters
    ----------
    x : array of shape (n,)
        x data
    y : array of shape (n,)
        y data

    Returns
    -------
    popt : list of shape (4,)
        Parameters A, mu, sigma**2, offset
    """
    mask = np.ma.getmaskarray(x) | np.ma.getmaskarray(y)
    x, y = x[~mask], y[~mask]

    gauss = gaussval2
    i = np.argmax(y[len(y) // 4 : len(y) * 3 // 4]) + len(y) // 4
    p0 = [y[i], x[i], 1, np.min(y)]

#    with np.warnings.catch_warnings():
#        np.warnings.simplefilter("ignore")
    popt, _ = curve_fit(gauss, x, y, p0=p0)

    return popt
    
def find_peaks_dlh( comb):
    # Find peaks in the comb spectrum
    # Run find_peak twice
    # once to find the average distance between peaks
    # once for real (disregarding close peaks)
    c = comb - np.ma.min(comb)
    width = 2
    height = np.ma.median(c)
    #DLH Mod -- my mods currently commented.
    height = 100
    peaks, _ = find_peaks(c, height=height, width=width)
    width = 2
    distance = np.median(np.diff(peaks)) // 16
    distance=7
    peaks, _ = find_peaks(c, height=height, distance=distance, width=width)


    # Fit peaks with gaussian to get accurate position
    new_peaks = peaks.astype(float)
    peaks_fwhm = peaks.astype(float)
    width = np.mean(np.diff(peaks)) // 2
    for j, p in enumerate(peaks):
        idx = p + np.arange(-width, width + 1, 1)
        idx = np.clip(idx, 0, len(c) - 1).astype(int)
        try:
            coef = gaussfit3(np.arange(len(idx)), c[idx])
            new_peaks[j] = coef[1] + p - width
            peaks_fwhm[j] = 2.355*np.sqrt(coef[2])
        except RuntimeError:
            new_peaks[j] = p

    n = np.arange(len(peaks))

    # keep peaks within the range
    mask = (new_peaks > 0) & (new_peaks < len(c))
    n, new_peaks,peaks_fwhm = n[mask], new_peaks[mask],peaks_fwhm[mask]

    return n, new_peaks,peaks_fwhm,peaks


# FUNCTION: find nearest value in array, return array value and index
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

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

file = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/1118/reduced/bgoH202211180013.fits"

hdul = fits.open(file)
arc_data = hdul['Fibre_P'].data
arc_wave = hdul['WAVE_P'].data
n_ord = arc_data.shape[0]
hdul.close



thar_file = "HRS_HS_ThAr_vac.list"
thar = np.loadtxt(thar_file,unpack=True, usecols=(0))
ii=np.where(thar < 3000)[0]
thar[ii] = thar[ii]*10.


out_ord = []
out_wav = []
out_pix = []

for order in range(41):
    print(order)
    spectrum = arc_data[order]
    spectrum[np.isnan(spectrum)] = 0
    #spectrum /=np.nanmax(spectrum)
    wave = air2vac(arc_wave[order])
    
    fig, axs = plt.subplots(2)
    ii=np.where(np.logical_and(thar >= np.min(wave) ,thar <= np.max(wave)))[0]
    axs[0].vlines(thar[ii], 0,100000,'r')
    #axs[0].set_ylim(-1000,np.max(spectrum))
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[0].plot(wave,spectrum)
    

    #DLH below for peak ID
    _,peaks_fit,fwhms,peaks = find_peaks_dlh(spectrum)

    axs[1].plot(spectrum)
    axs[1].plot(peaks,spectrum[peaks],'b+')
    ones=np.ones(len(ii))
    #plt.plot(thar[ii],ones,'r+')
    
    gaussian_fit_width = 5
    best_x = []
    for pk in peaks:
        if(pk - gaussian_fit_width)>1:
            first_fit_pixel = pk - gaussian_fit_width
        else:
            first_fit_pixel = 1
        if (pk + gaussian_fit_width)<2047:
            last_fit_pixel = pk + gaussian_fit_width
        else:
            last_fit_pixel = 2047
        result, line_dict = fit_gaussian_integral(np.arange(first_fit_pixel,last_fit_pixel),spectrum[first_fit_pixel:last_fit_pixel])
        if result is not None:
            coefs = result
            best_x.append(coefs[1])
            xs = np.floor(coefs[1]) - gaussian_fit_width + \
                            np.linspace(
                                0,
                                2 * gaussian_fit_width,
                                2 * gaussian_fit_width
                            )
            gaussian_fit = integrate_gaussian(xs, coefs[0], coefs[1], coefs[2], coefs[3])
            axs[1].plot(xs,gaussian_fit,alpha=0.5,color='red')
    plt.draw()

#    ii=np.where(np.logical_and(thar >= np.min(wave) -4 ,thar <= np.max(wave)+4))[0]
#    #y1=np.max(spectrum)*(atlas.flux[ii]/np.max(atlas.flux[ii]))
#    #y = np.interp(wave, atlas.wave[ii], atlas.flux[ii])
#    #y = gaussian_filter1d(y, 1)
#    #y /= (np.max(y))#/5.)
#    #y=y+0.001
#    #atlas_wave = atlas.wave[ii]
#
#    fig1, ax = plt.subplots(figsize=(16, 8), num=1)
#    ax.set_ylim(-300,20000)
#    plt.title('Observed Blue, Known Red')
#    plt.plot(wave, spectrum,'b-')
#    plt.vlines(thar[ii], 0,100000,'r')
#    plt.show(block=False)
#    plt.pause(0.1)
#
#
#    plt.draw()
    #print(peak_wave_atl)
    cont='y'

    state= 'start'

    while (cont =='y'):

        click = fig.ginput(1, timeout=0, show_clicks=False)
        init_pix_x = list(map(lambda x: x[0], click))
        peak_x, tmp_init = find_nearest(best_x, init_pix_x[0])
        
        #print ("\n     Input known lambda?")
        #known_lam = float((input("(y/n) = ")))

        click = fig.ginput(1, timeout=0, show_clicks=False)
        init_pix_x = list(map(lambda x: x[0], click))
        peak_wave, tmp_init2 = find_nearest(thar[ii], init_pix_x[0])
        #peak_wave, tmp_init2 = find_nearest(peak_wave_atl, known_lam)

        print(peak_wave,peak_x,fwhms[tmp_init],spectrum[peaks[tmp_init]])

        print ("\n     Continue with this order?")
        cont = str((input("(y/n) = ")))
        
        out_ord.append(order)
        out_wav.append(peak_wave)
        out_pix.append(peak_x)
        
#        linelist3.add_line(
#            peak_wave,
#            order,
#            peaks_fit[tmp_init],
#            fwhms[tmp_init],
#            spectrum[peaks[tmp_init]],
#            True,
#        )
 
    plt.close(fig)
    output=np.array([out_ord,out_pix,out_wav])
    np.save("./line_list_order_"+str(order),output)
    #linelist3.save("./line_list_order_"+str(order))
    

