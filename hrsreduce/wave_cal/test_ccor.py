import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.special import erf

from scipy.optimize import curve_fit

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

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
    """

    x = np.ma.compressed(x)
    y = np.ma.compressed(y)
    i = np.argmax(y[len(y) // 4 : len(y) * 3 // 4]) + len(y) // 4
    
    p0 = [y[i], 0, 1.5, np.min(y)]

    popt, pcov = curve_fit(integrate_gaussian, x, y, p0=p0, maxfev=1000000)
    pcov[np.isinf(pcov)] = 0 # convert inf to zero
    pcov[np.isnan(pcov)] = 0 # convert nan to zero

    return (popt)
    

ref = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/Super_Arcs/HR_Super_Arc_H20220701.fits"

obs = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0729/reduced/bgoH202207290042.fits"

with fits.open(ref) as hdu:
    ref_P = np.float32(hdu['FIBRE_P'].data)
    ref_O = np.float32(hdu['FIBRE_O'].data)
with fits.open(obs) as hdu:
    obs_P = np.float32(hdu['FIBRE_P'].data)
    obs_O = np.float32(hdu['FIBRE_O'].data)
    

for ord in range(42):

    obs = obs_O[ord][:-2]-np.nanmedian(obs_O[ord][:-2])
    obs /= np.nanmax(obs)
    ref = ref_O[ord][:-2]-np.nanmedian(ref_O[ord][:-2])
    ref /= np.nanmax(ref)

    obs[np.isnan(obs)] = 0
    ref[np.isnan(ref)] = 0

    diff =obs-ref
    ii=np.where(diff > 0.2)[0]
    
    for idx in ii:
        if idx > 2 and idx < len(obs)-2:
            obs[idx-2:idx+2] = 0

#    plt.plot(obs)
#    plt.plot(ref)
#    diff = obs - ref
#    plt.plot(diff)
#    plt.show()

    obs= obs[:-10]
    ref = ref[10:]
    
    corr = signal.correlate(obs, ref)
    lags = signal.correlation_lags(len(ref), len(obs))
    corr /= np.max(corr)
    
    plt.plot(lags,corr)
    plt.plot(obs,'r')
    plt.plot(ref,'b')
    plt.show()
    
    popt,pcov = curve_fit(gaus,lags,corr,p0=[1,0,3])
    fit_params = fit_gaussian_integral(lags,corr)
    plt.plot(lags,gaus(lags,*popt),'ro:',label='fit1')
    gaussian_fit = integrate_gaussian(lags, fit_params[0], fit_params[1], fit_params[2], fit_params[3])

    plt.plot(lags, gaussian_fit, alpha=0.5, color='green')
    plt.plot(lags,corr)
    plt.show()
    #plt.plot(ord,popt[1],'x')
    print(popt[1],fit_params[1])
    print(popt[1])
    
    x=np.arange(len(ref))
    plt.plot(x,ref,'b')
    plt.plot(x-10,obs,'r')
    plt.show()
    
for ord in range(42):

    obs = obs_P[ord]-np.nanmedian(obs_P[ord])
    obs /= np.nanmax(obs)
    ref = ref_P[ord]-np.nanmedian(ref_P[ord])
    ref /= np.nanmax(ref)

    obs[np.isnan(obs)] = 0
    ref[np.isnan(ref)] = 0

    diff =obs-ref
    ii=np.where(diff > 0.3)[0]
    obs[ii] = 0

    corr = signal.correlate(obs, ref)
    lags = signal.correlation_lags(len(ref), len(obs))
    corr /= np.max(corr)
    
    popt,pcov = curve_fit(gaus,lags,corr,p0=[1,0,3])
    fit_params = fit_gaussian_integral(lags,corr)
    plt.plot(lags,gaus(lags,*popt),'ro:',label='fit1')
    gaussian_fit = integrate_gaussian(lags, fit_params[0], fit_params[1], fit_params[2], fit_params[3])

    plt.plot(lags, gaussian_fit, alpha=0.5, color='green')
    plt.plot(lags,corr)
    plt.show()
    #plt.plot(ord,popt[1],'x')
    print(popt[1],fit_params[1])
plt.show()
