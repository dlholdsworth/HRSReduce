import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import scipy.interpolate as inter
from scipy.signal import savgol_filter
from scipy.interpolate import make_smoothing_spline

from hrsreduce.norm.norm import ContNorm
import hrsreduce.norm.continuum_normalization as pyreduce_cont_norm

def spline_fit(x,y,window):
    """Perform spline fit.

    Args:
        x (np.array): X-data to be splined (wavelength).
        y (np.array): Y-data to be splined (flux).
        window (float): Window value for breakpoint's number of samples.

    Returns:
        ss (LSQUnivariateSpline): Spline-fit of data.
    """
    breakpoint = np.linspace(np.min(x),np.max(x),int((np.max(x)-np.min(x))/window))
    ss = inter.LSQUnivariateSpline(x,y,breakpoint[1:-1])
    return ss
    

def flatspec_spline(x,rawspec,weight,n_iter,ffrac):
    """Performs spline fit for specified number of iterations.

    Args:
        x (np.array): Wavelength data.
        rawspec (np.array): Flux data.
        weight (np.array): Weighting for fitting.

    Returns:
        normspec (np.array): Normalized flux data.
        yfit (np.array): Spline fit.
    """
    pos = np.where((np.isnan(rawspec)==False) & (np.isnan(weight)==False))[0]

    ss = spline_fit(x[pos],rawspec[pos],5.)
    yfit = ss(x)

    for i in range(n_iter):
        normspec = rawspec / yfit

        pos = np.where((normspec >= ffrac) & (yfit > 0))[0]#& (normspec <= 2.)

        ss = spline_fit(x[pos],rawspec[pos],5.)
        yfit = ss(x)

    normspec = rawspec / yfit

    return normspec,yfit

n_iter = 2
ffrac = 0.98

sci_file = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/0228/reduced/bgoH202302280035.fits"
arc_file ="/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/0228/reduced/bgoH202302280028.fits"
flat = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/1006/reduced/HS_Master_Flat_H20221006.fits"

with fits.open(sci_file) as hdul:
    data_P = hdul['FIBRE_P'].data
    data_O = hdul['FIBRE_O'].data

with fits.open(arc_file) as hdul:
    wave_P = hdul['WAVE_P'].data
    wave_O = hdul['WAVE_O'].data
    
with fits.open(flat)as hdul:
    flat_P = hdul['FIBRE_P'].data
    flat_O = hdul['FIBRE_O'].data
    
ContNorm(sci_file,arc_file,flat).execute()

#    
#np.savetxt('test_1_order.dat', np.array([wave_P[20],data_P[20]]).T)
#    
#weight = (data_P.copy())
#weight = np.sqrt(weight)
#
#spline_flat = flat_P.copy()
#sm_flat_P = flat_P.copy()
#
#for i in range(42):
#
#    _,trend = flatspec_spline(wave_P[i],flat_P[i],weight[i],n_iter,ffrac)
#    spline_flat[i]=trend
#    
##    plt.plot(flat_P[i])
#    #norm_flat_P[i] = savgol_filter(flat_P[i],window_length=50, polyorder=3)
#    spl = make_smoothing_spline(wave_P[i], flat_P[i], lam=1)
#    sm_flat_P[i]=spl(wave_P[i])
##    plt.plot(spl(wave_P[i]))
##plt.show()
#
#
#s_spec, s_wave, s_blaze,s_sigma = pyreduce_cont_norm.splice_orders(data_P,wave_P,sm_flat_P,weight,scaling=False)
#
#trend, full_wave, full_spec, full_cont = pyreduce_cont_norm.continuum_normalize(np.ma.array(s_spec), np.ma.array(s_wave), np.ma.array(s_blaze), np.ma.array(s_sigma), iterations=n_iter, scale_vert=1,smooth_initial=1e4, smooth_final=5e6, plot=False, plot_title=None)
#
#fig, (ax, ax1) = plt.subplots(2,1, sharex=True,figsize=(12,8))
#                
##for i in range(42):
##    ax.plot(wave_P[i],data_P[i])
##    ax.plot(wave_P[i],trend[i])
##    ax.plot(wave_P[i],flat_P[i])
##    ax1.plot(wave_P[i],data_P[i]/trend[i],)
##plt.show()
#
##plt.plot(full_wave)
##plt.show()
#
##for i in range(len(full_wave)-1):
##    plt.plot(full_wave[i], full_wave[i+1]-full_wave[i],'.')
##plt.show()
#
