import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit
from lmfit import Model,Parameter
from astropy.io import fits
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

class CCF():

    def __init__(self, sci_frame,ccf_mask):
    
        self.frame = sci_frame
        self.ccf_mask = ccf_mask
        self.logger = logger
        
        # start a logger
        self.logger.info('Started CCF')

    def fit_gaussian_integral(self,x, y,x0=None,do_test=True):
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
        
        if x0 is not None:
            p0 = [y[i], x0, 1.5, np.min(y)]
        else:
            p0 = [y[i], x[i], 1.5, np.min(y)]

        popt, _ = curve_fit(self.integrate_gaussian, x, y, p0=p0, maxfev=1000000)

        return popt
        

    def integrate_gaussian(self,x, a, mu, sig, const, int_width=0.5):
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

    # ============================================================
    # Helper 1: find_nearest
    # ============================================================

    def find_nearest(self,vector, value):
        """Return the index of the element in 'vector' closest to 'value'."""
        return np.argmin(np.abs(vector - value))
    
    def gaussian(self,x, amp, cen, wid,offset):
        """1-d gaussian: gaussian(x, amp, cen, wid)"""
        return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))+offset
        
    def vac2air(self,wl_air):
        """
        Convert wavelengths vacuum to air wavelength
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
        wl_vac[ii] = wl_air[ii] / fact  # Convert to vacuum wavelength
        return wl_vac
        


    # ============================================================
    # Helper 2: logdelfn_2_logdelfn_array
    # ============================================================

    def logdelfn_2_logdelfn_array(self,ldelfn, logwavelength):
        """
        Convert (log wavelength, weight) pairs into an array aligned with the log-wavelength axis.
        Equivalent to MATLAB logdelfn_2_logdelfn_array(ldelfn, logwavelength)
        """
        ldelfn_array = np.zeros_like(logwavelength)
        for s in range(ldelfn.shape[0]):
            T = self.find_nearest(logwavelength, ldelfn[s, 0])
            ldelfn_array[T] += ldelfn[s, 1]
        return ldelfn_array


    # ============================================================
    # Helper 3: testing_deconvolution
    # ============================================================

    def testing_deconvolution(self,a, b):
        """NumPy-only equivalent of MATLAB testing_deconvolution(a, b)."""
        nfft = 2 ** 14

        s1 = np.fft.fft(a - np.mean(a), nfft)
        s2 = np.fft.fft(b - np.mean(b), nfft)

        # Normalize and divide
        s2_norm = s2 / np.max(np.abs(s2))
        X = s1 / (np.abs(s2_norm))
        X[np.isinf(X)] = 0  # handle Inf

        res = np.real(np.fft.ifft(X))
        res = res[:len(a)]
        res = res + np.mean(a)
        return res


    # ============================================================
    # Main Function
    # ============================================================
    def do_ccf_from_spec_wavelengths_and_widths(self,wave, intmtx, wa, wi, numbins):
        """
        NumPy-only version of the MATLAB function:
        [v, vel, LSD, CCF, dfn_auto, wcw] = do_ccf_from_spec_wavelengths_and_widths(wave, intmtx, wa, wi, numbins)
        """

        c = 299792.458  # speed of light (km/s)
        wcw = np.sum((wi / np.sum(wi)) * wa)

        # Convert to log space
        logwave = np.log(wave)
        minlogaxis = np.floor(np.min(logwave) * 100000) / 100000
        maxlogaxis = np.ceil(np.max(logwave) * 100000) / 100000
        logaxisstep = (maxlogaxis - minlogaxis) / len(wave)
        logaxis = np.arange(minlogaxis, maxlogaxis + logaxisstep, logaxisstep)

        velstep = (np.exp(np.log(wcw) + logaxisstep) - wcw) / wcw * c

        lwa = np.log(wa)
        ldelfn = np.column_stack((lwa, wi))
        ldelfn_array = self.logdelfn_2_logdelfn_array(ldelfn, logaxis)

        # Auto-correlation using numpy.correlate
        dfn_auto_full = np.correlate(ldelfn_array, ldelfn_array, mode='full') / len(ldelfn_array)
        center = len(ldelfn_array) - 1
        dfn_auto = dfn_auto_full[center - numbins // 2 : center + numbins // 2 + 1]

        xax = np.arange(-len(ldelfn_array) + 1, len(ldelfn_array))
        v = xax * velstep
        vel = np.arange(np.ceil(np.min(v)), np.floor(np.max(v)), velstep)

        CCF = []
        LSD = []

        # Interpolate spectrum to log scale using np.interp (linear)
        logspecint = np.interp(logaxis, logwave, intmtx)

        # Cross-correlation
        ccf_full = np.correlate(logspecint, ldelfn_array, mode='full') / len(ldelfn_array) #1 - logspecint
        ccf = ccf_full[center - numbins // 2 : center + numbins // 2 + 1]
        CCF.append(ccf)

        # Deconvolution step
        tresult = self.testing_deconvolution(ccf, dfn_auto)
        tresult =  tresult # 1 - tresult
        tresult /= np.max(tresult)

        # Interpolate LSD back onto velocity grid
        LSD_interp = np.interp(vel, v[:len(tresult)], tresult)
        LSD.append(LSD_interp)

        CCF = np.array(CCF)
        LSD = np.array(LSD)

        return v, vel, LSD, CCF, dfn_auto, wcw
        


    # ============================================================
    # Example Usage
    # ============================================================
    def execute(self):
    
        wa, wi  = np.loadtxt(self.ccf_mask, usecols=(0,1),unpack=True,skiprows=1)
        numbins = 500
    
        with fits.open(self.frame) as hdu:
            
            nord = 37
            ord_min=84
            
            BVC = float(hdu[0].header["BARYRV"])
            star_RV = -13.441
            #star_RV = -21.138
            #star_RV = 1.866
            star_RV=0
            #BVC=0
            RV = []
            RV_err = []
            
            for ord in range(1):
            
                ext = "ORD"+str(ord_min+ord)+"_O"
                
                #wave = hdu['WAVE_O'].data[ord][10:-10]
                #flux = hdu['FIBRE_O'].data[ord][10:-10]/hdu['BLAZE_O'].data[ord][10:-10]
                wave = hdu[ext].data['Wave'][10:-10]
                flux = hdu[ext].data['Flux'][10:-10]/hdu[ext].data['Blaze'][10:-10]
                #uni_wave = np.arange(np.min(wave), np.max(wave),0.011689)
                #uni_flux = np.interp(uni_wave,wave,flux)
                
                flux = np.nan_to_num(flux,nan=np.nanmedian(flux))
                flux /=np.nanmax(flux)
                #wave = uni_wave
                
                CRPIX1 =int(hdu['FIBRE_O_SPEC_MERGED'].header['CRPIX1'])
                CRVAL1 =float(hdu['FIBRE_O_SPEC_MERGED'].header['CRVAL1'])
                CDELT1 =float(hdu['FIBRE_O_SPEC_MERGED'].header['CDELT1'])
                xconv = lambda v: ((v-CRPIX1+1)*CDELT1+CRVAL1)
                flux =hdu['FIBRE_O_SPEC_MERGED'].data
                wave = xconv(np.arange(len(flux)))
                plt.plot(wave,flux)
                plt.show()
                ii = np.where(np.logical_and(wave > np.min(wa)-1, wave < np.max(wa)+1))[0]
                wave = wave[ii]
                flux = flux[ii]
                
                
                peaks,_ = find_peaks(flux,height=0.5,distance=int(len(flux)/20))
#                plt.plot(wave,flux)
#                plt.plot(wave[peaks],flux[peaks],'x')
                cont_params = np.polyfit(wave[peaks],flux[peaks],4)
                cont_fit = np.polyval(cont_params,wave)
#                plt.plot(wave,cont_fit,'r')
#                plt.plot(wave,flux/cont_fit)
#                plt.show()

                flux /=cont_fit
                
                ii = np.where(np.logical_and(wa > np.min(wave)-1, wa < np.max(wave)+1))[0]
                wa_cut = wa[ii]
                wi_cut = wi[ii]

#                plt.plot(wave,flux)
#                #plt.vlines(wa_cut,1,1-wi_cut,'r')
#                plt.plot(wa_cut,wi_cut)
#                plt.show()
                
                v, vel, LSD, CCF, dfn_auto, wcw = self.do_ccf_from_spec_wavelengths_and_widths(wave, flux, wa_cut, wi_cut, numbins)
        
                out_vels = (np.arange(numbins+1)-(numbins-1)/2.)*(vel[1] - vel[0])
        
                corr = CCF[0] - np.min(CCF[0])
                #corr /= np.max(corr)
    
                #Fit a gaussian to the result
                fit_params = self.fit_gaussian_integral(out_vels,corr,x0=0,do_test=False)
                gaussian_fit = self.integrate_gaussian(out_vels, fit_params[0], fit_params[1], fit_params[2], fit_params[3])
                shift = fit_params[1]

                cen = np.where(corr == np.max(corr))[0][0]

                gmodel = Model(self.gaussian)
                result = gmodel.fit(corr, x=out_vels, amp=np.max(corr), cen=out_vels[cen], wid=Parameter('wid',value=2,max=6),offset=0)

                if (np.abs(result.params['cen'].value+BVC - star_RV)) < 500:
                    RV.append(result.params['cen'].value+BVC)
                    if result.params['cen'].stderr is not None:
                        RV_err.append(result.params['cen'].stderr*3.)
                    else:
                        RV_err.append(100000)
                    
                    print(result.params['cen'].value+BVC-star_RV,result.params['cen'].stderr)
                
#                if result.params['cen'].value < 0:
#                    plt.plot(wave,flux)
#                    plt.plot(hdu[ext].data['Wave'][10:-10],hdu[ext].data['Flux'][10:-10])
#                    plt.plot(hdu[ext].data['Wave'][10:-10],hdu[ext].data['Blaze'][10:-10])
#                    plt.show()
                
#                plt.plot(out_vels, corr)
#                plt.plot(out_vels,gaussian_fit)
#                plt.show()
                
#                plt.plot(ord,result.params['cen'].value+BVC-star_RV,'o')
                #plt.plot(ord,(result.params['cen'].value/299792.458)*np.median(wave),'o')



            weights = 1./np.array(RV_err)
            w_prime =  weights / np.sum(weights)
            RV_mean = np.sum(RV*w_prime)
            var = np.sum((RV-RV_mean)**2)/ len(RV)
            err = var*np.sum(w_prime**2)
            
            print(RV_mean-star_RV-0.5, err)
#
#            plt.title(str(RV_mean-star_RV)+"±"+str(err))
#            plt.hlines(0,0,35,'r')
#
#            plt.show()
        
        

