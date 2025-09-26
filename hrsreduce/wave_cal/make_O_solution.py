import numpy as np
from astropy.io import fits
from scipy.special import erf
import warnings
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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

gaussian_fit_width = 5

linelist_path = "./HR_H_linelist_P.npy"
arc = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/Super_Arcs/HR_Super_Arc_H20220701.fits'
peak_wavelengths_ang = np.load(linelist_path,allow_pickle=True).item()

hdul = fits.open(arc)
O_data=hdul['FIBRE_O'].data
hdul.close

ord_out = []
lines_out = []
wave_out = []

for ord in range(42):
    print("DOING ORDER",ord)
    spectrum = O_data[ord] - np.nanmedian(O_data[ord])
    plt.plot(spectrum,'b')
    
    lines = peak_wavelengths_ang[ord]['line_positions']
    waves = peak_wavelengths_ang[ord]['known_wavelengths_air']
    plt.vlines(lines,0,100000,'r')
    
    for idx,line in enumerate(lines):
        line +=2.5
        
        if(line - gaussian_fit_width)>1:
            first_fit_pixel = int(line - gaussian_fit_width)
        else:
            first_fit_pixel = 1
        if (line + gaussian_fit_width)<2047:
            last_fit_pixel = int(line + gaussian_fit_width)
        else:
            last_fit_pixel = 2047
        result, line_dict = fit_gaussian_integral(np.arange(first_fit_pixel,last_fit_pixel),spectrum[first_fit_pixel:last_fit_pixel])
        
        if result is not None:
            coefs = result
            xs = np.floor(coefs[1]) - gaussian_fit_width + \
                            np.linspace(
                                0,
                                2 * gaussian_fit_width,
                                2 * gaussian_fit_width
                            )
            ord_out.append(ord)
            lines_out.append(coefs[1])
            wave_out.append(waves[idx])
            
            gaussian_fit = integrate_gaussian(xs, coefs[0], coefs[1], coefs[2], coefs[3])

lines_out = np.array(lines_out)
ord_out=np.array(ord_out)
wave_out=np.array(wave_out)
plt.vlines(lines_out,0,100000,'g')
plt.show()

output = {}

for ord in range(42):
    output[ord] = {}
    ii=np.where(ord_out == ord)[0]
    output[ord]['line_positions'] = lines_out[ii]
    output[ord]['known_wavelengths_air'] = wave_out[ii]

#output=np.array([ord_out,lines_out,wave_out])
np.save("./HR_H_linelist_O",output)
    
        
