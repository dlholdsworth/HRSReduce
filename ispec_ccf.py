from astropy.io import fits
import matplotlib.pyplot as plt
import glob
from astroquery.simbad import Simbad
import numpy as np
import logging
import scipy
from hrsreduce.ccf.mpfitmodels import GaussianModel
from hrsreduce.ccf.mpfitmodels import VoigtModel
import copy
from scipy.fftpack import fft
from scipy.fftpack import ifft

def find_local_min_values(x):
    """
    For an array of values, find the position of local maximum values considering only
    the next and previous elements, except they have the same value.
    In that case, the next/previous different value is checked. Therefore,
    ::

        find_local_max([10,9,3,3,9,10,4,30])

    would return:
    ::

        [2, 3, 6]
    """
    ret = []
    n = len(x)
    m = 0;
    for i in np.arange(n):
        l_min = np.max([i-1, 0])
        #l_max = i-1
        #r_min = i+1
        #r_max = np.min([i+1, n-1])
        r_min = np.min([i+1, n-1])
        is_min = True
        # left side
        j = l_min
        # If value is equal, search for the last different value
        while j >= 0 and x[j] == x[i]:
            j -= 1

        if j < 0 or x[j] < x[i]:
            is_min = False

        # right side
        if is_min:
            j = r_min
            # If value is equal, search for the next different value
            while j < n and x[j] == x[i]:
                j += 1

            if j >= n or x[j] < x[i]:
                is_min = False

        if is_min:
            ret.append(i)
    return np.asarray(ret)
    
def find_local_max_values(x):
    """
    For an array of values, find the position of local maximum values considering only
    the next and previous elements, except they have the same value.
    In that case, the next/previous different value is checked. Therefore,
    ::

        find_local_max([1,2,3,3,2,1,4,3])

    would return:
    ::

        [2, 3, 6]
    """
    ret = []
    n = len(x)
    m = 0;
    for i in np.arange(n):
        l_min = np.max([i-1, 0])
        #l_max = i-1
        #r_min = i+1
        #r_max = np.min([i+1, n-1])
        r_min = np.min([i+1, n-1])
        is_max = True

        # left side
        j = l_min
        # If value is equal, search for the last different value
        while j >= 0 and x[j] == x[i]:
            j -= 1

        if (j < 0 or x[j] > x[i]) and i > 0:
            is_max = False

        # right side
        if is_max:
            j = r_min
            # If value is equal, search for the next different value
            while j < n and x[j] == x[i]:
                j += 1
            if (j >= n or x[j] > x[i]) and i < n-1:
                is_max = False

        if is_max:
            ret.append(i)
    return np.asarray(ret)
        
def __remove_consecutives_features(features):
    """
    Remove features (i.e. peaks or base points) that are consecutive, it makes
    no sense to have two peaks or two base points together.
    """
    if len(features) >= 2:
        duplicated_features = (np.abs(features[:-1] - features[1:]) == 1)
        duplicated_features = np.array([False] + duplicated_features.tolist())
        cleaned_features = features[~duplicated_features]
        return cleaned_features
    else:
        return features

def __assert_structure(xcoord, yvalues, peaks, base_points):
    """
    Given a group of peaks and base_points with the following assumptions:

    - base_points[i] < base_point[i+1]
    - peaks[i] < peaks[i+1]
    - base_points[i] < peaks[i] < base_points[i+1]

    The function returns peaks and base_points where:

    - The first and last feature is a base point: base_points[0] < peaks[0] < base_points[1] < ... < base_points[n-1] < peaks[n-1] < base_points[n] where n = len(base_points)
    - len(base_points) = len(peaks) + 1
    """
    if len(peaks) == 0 or len(base_points) == 0:
        return [], []

    # Limit the base_points array to the ones that are useful, considering that
    # the first and last peak are always removed
    first_wave_peak = xcoord[peaks][0]
    first_wave_base = xcoord[base_points][0]
    if first_wave_peak > first_wave_base:
        if len(base_points) - len(peaks) == 1:
            ## First feature found in spectrum: base point
            ## Last feature found in spectrum: base point
            base_points = base_points
            peaks = peaks
        elif len(base_points) - len(peaks) == 0:
            ## First feature found in spectrum: base point
            ## Last feature found in spectrum: peak
            # - Remove last peak
            #base_points = base_points
            #peaks = peaks[:-1]
            # - Add a base point (last point in the spectrum)
            base_points = np.hstack((base_points, [len(xcoord)-1]))
            peaks = peaks
        else:
            raise Exception("This should not happen")
    else:
        if len(base_points) - len(peaks) == -1:
            ## First feature found in spectrum: peak
            ## Last feature found in spectrum: peak
            # - Remove first and last peaks
            #base_points = base_points
            #peaks = peaks[1:-1]
            # - Add two base points (first and last point in the spectrum)
            base_points = np.hstack(([0], base_points))
            base_points = np.hstack((base_points, [len(xcoord)-1]))
            peaks = peaks
        elif len(base_points) - len(peaks) == 0:
            ## First feature found in spectrum: peak
            ## Last feature found in spectrum: base point
            # - Remove first peak
            #base_points = base_points
            #peaks = peaks[1:]
            # - Add a base point (first point in the spectrum)
            base_points = np.hstack(([0], base_points))
            peaks = peaks
        else:
            raise Exception("This should not happen")

    return peaks, base_points

def __improve_linemask_edges(xcoord, yvalues, base, top, peak):
    """
    Given a spectrum, the position of a peak and its limiting region where:

    - Typical shape: concave + convex + concave region.
    - Peak is located within the convex region, although it is just in the border with the concave region (first element).
    """
    # Try to use two additional position, since we are going to lose them
    # by doing the first and second derivative
    original_top = top
    top = np.min([top+2, len(xcoord)])
    y = yvalues[base:top+1]
    x = xcoord[base:top+1]
    # First derivative (positive => flux increases, negative => flux decreases)
    dy_dx = (y[:-1] - y[1:])/ (x[:-1] - x[1:])
    # Second derivative (positive => convex, negative => concave)
    d2y_dx2 = (dy_dx[:-1] - dy_dx[1:])/ (x[:-2] - x[2:])
    # Peak position inside the linemask region
    peak_relative_pos = peak - base
    # The peak should be in a convex region => the second derivative should be positive
    # - It may happen that the peak falls in the beginning/end of a concave region, accept also this cases
    if peak_relative_pos < len(d2y_dx2)-1 and peak_relative_pos > 0 and (d2y_dx2[peak_relative_pos-1] > 0 or d2y_dx2[peak_relative_pos] > 0 or d2y_dx2[peak_relative_pos+1] > 0):
        # Find the concave positions at both sides of the peak
        concave_pos = np.where(d2y_dx2<0)[0]
        if len(concave_pos) == 0:
            # This should not happen, but just in case...
            new_base = base
            new_top = original_top
        else:
            # Concave regions to the left of the peak
            left_concave_pos = concave_pos[concave_pos-peak_relative_pos < 0]
            if len(left_concave_pos) == 0:
                # This should not happen, but just in case...
                new_base = base
            else:
                # Find the edges of the concave regions to the left of the peak
                left_concave_pos_diff = left_concave_pos[:-1] - left_concave_pos[1:]
                left_concave_pos_diff = np.asarray([-1] + left_concave_pos_diff.tolist())
                left_concave_edge_pos = np.where(left_concave_pos_diff != -1)[0]
                if len(left_concave_edge_pos) == 0:
                    # There is only one concave region, we use its left limit
                    new_base = left_concave_pos[0] + base
                else:
                    # There is more than one concave region, select the nearest edge to the peak
                    new_base = np.max([left_concave_pos[np.max(left_concave_edge_pos)] + base, base])

            # Concave regions to the right of the peak
            right_concave_pos = concave_pos[concave_pos-peak_relative_pos > 0]
            if len(right_concave_pos) == 0:
                # This should not happen, but just in case...
                new_top = original_top
            else:
                # Find the edges of the concave regions to the right of the peak
                right_concave_pos_diff = right_concave_pos[1:] - right_concave_pos[:-1]
                right_concave_pos_diff = np.asarray(right_concave_pos_diff.tolist() + [1])
                right_concave_edge_pos = np.where(right_concave_pos_diff != 1)[0]
                if len(right_concave_edge_pos) == 0:
                    # There is only one concave region, we use its right limit
                    new_top = right_concave_pos[-1] + base
                else:
                    # There is more than one concave region, select the one with the nearest edge to the peak
                    new_top = np.min([right_concave_pos[np.min(right_concave_edge_pos)] + base, original_top])

    else:
        # This will happen very rarely (only in peaks detected at the extreme of a spectrum
        # and one of its basepoints has been "artificially" added and it happens to be
        # just next to the peak)
        new_base = base
        new_top = original_top
        #plt.plot(x, y)
        #l = plt.axvline(x = x[peak_relative_pos], linewidth=1, color='red')
        #print d2y_dx2, d2y_dx2[peak_relative_pos]
        #plt.show()

    return new_base, new_top


def __find_peaks_and_base_points(xcoord, yvalues):
    """
    Find peaks and base points. It works better with a smoothed spectrum (i.e. convolved using 2*resolution).
    """
    if len(yvalues[~np.isnan(yvalues)]) == 0 or len(yvalues[~np.isnan(xcoord)]) == 0:
        #raise Exception("Not enough data for finding peaks and base points")
        print("WARNING: Not enough data for finding peaks and base points")
        peaks = []
        base_points = []
    else:
        # Determine peaks and base points (also known as continuum points)
        peaks = find_local_min_values(yvalues)
        base_points = find_local_max_values(yvalues)

        # WARNING: Due to three or more consecutive values with exactly the same flux
        # find_local_max_values or find_local_min_values will identify all of them as peaks or bases,
        # where only one of the should be marked as peak or base.
        # These cases break the necessary condition of having the same number of
        # peaks and base_points +/-1
        # It is necessary to find those "duplicates" and remove them:
        peaks = __remove_consecutives_features(peaks)
        base_points = __remove_consecutives_features(base_points)

        if not (len(peaks) - len(base_points)) in [-1, 0, 1]:
            raise Exception("This should not happen")

        # Make sure that
        peaks, base_points = __assert_structure(xcoord, yvalues, peaks, base_points)

    return peaks, base_points



def __model_velocity_profile(ccf, nbins, only_one_peak=False, peak_probability=0.55, model='2nd order polynomial + gaussian fit'):
    """
    Fits a model ('Gaussian' or 'Voigt') to the deepest peaks in the velocity
    profile. If it is 'Auto', a gaussian and a voigt will be fitted and the best
    one used.

    In all cases, the peak is located by fitting a 2nd degree polynomial. Afterwards,
    the gaussian/voigt fitting is done for obtaining more info (such as sigma, etc.)

    * For Radial Velocity profiles, more than 1 outlier peak implies that the star is a spectroscopic binary.

    WARNING: fluxes and errors are going to be modified by a linear normalization process

    Detected peaks are evaluated to discard noise, a probability is assigned to each one
    in function to a linear model. If more than one peak is found, those with a peak probability
    lower than the specified by the argument will be discarded.

    :returns:
        Array of fitted models and an array with the margin errors for model.mu() to be able to know the interval
        of 99% confiance.

    """
    models = []
    if len(ccf) == 0:
        return models
    xcoord = ccf['x']
    fluxes = ccf['y']
    errors = ccf['err']

    # Smooth flux
    sig = 1
    smoothed_fluxes = scipy.ndimage.gaussian_filter1d(fluxes, sig)

    #smoothed_fluxes = fluxes
    # Finding peaks and base points
    peaks, base_points = __find_peaks_and_base_points(xcoord, smoothed_fluxes)

    if len(peaks) == 0 or len(base_points) == 0:
        return models

    if len(peaks) != 0:
        base = base_points[:-1]
        top = base_points[1:]
        # Adjusting edges
        new_base = np.zeros(len(base), dtype=int)
        new_top = np.zeros(len(base), dtype=int)
        for i in np.arange(len(peaks)):
            new_base[i], new_top[i] = __improve_linemask_edges(xcoord, smoothed_fluxes, base[i], top[i], peaks[i])
            #new_base[i] = base[i]
            #new_top[i] = top[i]
        base = new_base
        top = new_top

        if only_one_peak:
            # Just try with the deepest line
            selected_peaks_indices = []
        else:
            import statsmodels.api as sm
            #x = np.arange(len(peaks))
            #y = fluxes[peaks]
            x = xcoord
            y = fluxes
            # RLM (Robust least squares)
            # Huber's T norm with the (default) median absolute deviation scaling
            # - http://en.wikipedia.org/wiki/Huber_loss_function
            # - options are LeastSquares, HuberT, RamsayE, AndrewWave, TrimmedMean, Hampel, and TukeyBiweight
            x_c = sm.add_constant(x, prepend=False) # Add a constant (1.0) to have a parameter base
            huber_t = sm.RLM(y, x_c, M=sm.robust.norms.HuberT())
            linear_model = huber_t.fit()
            selected_peaks_indices = np.where(linear_model.weights[peaks] < 1. - peak_probability)[0]

        if len(selected_peaks_indices) == 0:
            # Try with the deepest line
            sorted_peak_indices = np.argsort(fluxes[peaks])
            selected_peaks_indices = [sorted_peak_indices[0]]
        else:
            # Sort the interesting peaks from more to less deep
            sorted_peaks_indices = np.argsort(fluxes[peaks[selected_peaks_indices]])
            selected_peaks_indices = selected_peaks_indices[sorted_peaks_indices]
            
    else:
        # If no peaks found, just consider the deepest point and mark the base and top
        # as the limits of the whole data
        sorted_fluxes_indices = np.argsort(fluxes)
        peaks = sorted_fluxes_indices[0]
        base = 0
        top = len(xcoord) - 1
        selected_peaks_indices = [0]

    for i in np.asarray(selected_peaks_indices):
        #########################################################
        ####### 2nd degree polinomial fit to determine the peak
        #########################################################
        poly_step = 0.0001
        # Use only 9 points for fitting (4 + 1 + 4)
        diff_base = peaks[i] - base[i]
        diff_top = top[i] - peaks[i]
        if diff_base > 4 and diff_top > 4:
            poly_base = peaks[i] - 4
            poly_top = peaks[i] + 4
        else:
            # There are less than 9 points but let's make sure that there are
            # the same number of point in each side to avoid asymetries that may
            # affect the fitting of the center
            if diff_base >= diff_top:
                poly_base = peaks[i] - diff_top
                poly_top = peaks[i] + diff_top
            elif diff_base < diff_top:
                poly_base = peaks[i] - diff_base
                poly_top = peaks[i] + diff_base
        p = np.poly1d(np.polyfit(xcoord[poly_base:poly_top+1], fluxes[poly_base:poly_top+1], 2))
        
        poly_vel = np.arange(xcoord[poly_base], xcoord[poly_top]+poly_step, poly_step)
        poly_ccf = p(poly_vel)
        mu = poly_vel[np.argmin(poly_ccf)]
        # Sometimes the polynomial fitting can give a point that it is not logical
        # (far away from the detected peak), so we do a validation check
        if mu < xcoord[peaks[i]-1] or mu > xcoord[peaks[i]+1]:
            mu = xcoord[peaks[i]]
            poly_step = xcoord[peaks[i]+1] - xcoord[peaks[i]] # Temporary just to the next iteration

        #########################################################
        ####### Gaussian/Voigt fit to determine other params.
        #########################################################
        # Models to fit
        gaussian_model = GaussianModel()
        voigt_model = VoigtModel()

        # Parameters estimators
        baseline = np.median(fluxes[base_points])
        A = fluxes[peaks[i]] - baseline
        sig = np.abs(xcoord[top[i]] - xcoord[base[i]])/3.0

        parinfo = [{'value':0., 'fixed':False, 'limited':[False, False], 'limits':[0., 0.]} for j in np.arange(5)]
        parinfo[0]['value'] = 1.0 #fluxes[base[i]] # baseline # Continuum
        parinfo[0]['fixed'] = True
        #parinfo[0]['limited'] = [True, True]
        #parinfo[0]['limits'] = [fluxes[peaks[i]], 1.0]
        parinfo[1]['value'] = A # Only negative (absorption lines) and greater than the lowest point + 25%
        parinfo[1]['limited'] = [False, True]
        parinfo[1]['limits'] = [0., 0.]
        parinfo[2]['value'] = sig # Only positives (absorption lines)
        parinfo[2]['limited'] = [True, False]
        parinfo[2]['limits'] = [1.0e-10, 0.]
        parinfo[3]['value'] = mu # Peak only within the xcoord slice
        #parinfo[3]['fixed'] = True
        parinfo[3]['fixed'] = False
        parinfo[3]['limited'] = [True, True]
        #parinfo[3]['limits'] = [xcoord[base[i]], xcoord[top[i]]]
        #parinfo[3]['limits'] = [xcoord[peaks[i]-1], xcoord[peaks[i]+1]]
        parinfo[3]['limits'] = [mu-poly_step, mu+poly_step]

        # Only used by the voigt model (gamma):
        parinfo[4]['value'] = (xcoord[top[i]] - xcoord[base[i]])/2.0 # Only positives (not zero, otherwise its a gaussian) and small (for nm, it should be <= 0.01 aprox but I leave it in relative terms considering the spectrum slice)
        parinfo[4]['fixed'] = False
        parinfo[4]['limited'] = [True, True]
        parinfo[4]['limits'] = [0.0, xcoord[top[i]] - xcoord[base[i]]]

        f = fluxes[base[i]:top[i]+1]
        min_flux = np.min(f)
        # More weight to the deeper fluxes
        if min_flux < 0:
            weights = f + -1*(min_flux) + 0.01 # Above zero
            weights = np.min(weights)/ weights
        else:
            weights = min_flux/ f
        weights -= np.min(weights)
        weights = weights/np.max(weights)

        try:
            # Fit a gaussian and a voigt, but choose the one with the best fit
            if model in ['2nd order polynomial + auto fit', '2nd order polynomial + gaussian fit']:
                gaussian_model.fitData(xcoord[base[i]:top[i]+1], fluxes[base[i]:top[i]+1], parinfo=copy.deepcopy(parinfo[:4]), weights=weights)
                #gaussian_model.fitData(xcoord[base[i]:top[i]+1], fluxes[base[i]:top[i]+1], parinfo=copy.deepcopy(parinfo[:4]))
                rms_gaussian = np.sqrt(np.sum(np.power(gaussian_model.residuals(), 2)) / len(gaussian_model.residuals()))
            if model in ['2nd order polynomial + auto fit', '2nd order polynomial + voigt fit']:
                voigt_model.fitData(xcoord[base[i]:top[i]+1], fluxes[base[i]:top[i]+1], parinfo=copy.deepcopy(parinfo), weights=weights)
                #voigt_model.fitData(xcoord[base[i]:top[i]+1], fluxes[base[i]:top[i]+1], parinfo=copy.deepcopy(parinfo))
                rms_voigt = np.sqrt(np.sum(np.power(voigt_model.residuals(), 2)) / len(voigt_model.residuals()))

            if model == '2nd order polynomial + voigt fit' or (model == '2nd order polynomial + auto fit' and rms_gaussian > rms_voigt):
                final_model = voigt_model
                #logging.info("Voigt profile fitted with RMS %.5f" % (rms_voigt))
            else:
                final_model = gaussian_model
                #logging.info("Gaussian profile fitted with RMS %.5f" % (rms_gaussian))
            
            # Calculate velocity error based on:
            # Zucker 2003, "Cross-correlation and maximum-likelihood analysis: a new approach to combining cross-correlation functions"
            # http://adsabs.harvard.edu/abs/2003MNRAS.342.1291Z
            inverted_fluxes = 1-fluxes
            distance = xcoord[1] - xcoord[0]
            first_derivative = np.gradient(inverted_fluxes, distance)
            second_derivative = np.gradient(first_derivative, distance)
            ## Using the exact velocity, the resulting error are less coherents (i.e. sometimes you can get lower errors when using bigger steps):
            #second_derivative_peak = np.interp(final_model.mu(), xcoord, second_derivative)
            #inverted_fluxes_peak = final_model.mu()
            ## More coherent results:
            peak = xcoord.searchsorted(final_model.mu())
            inverted_fluxes_peak = inverted_fluxes[peak]
            second_derivative_peak = second_derivative[peak]
            if inverted_fluxes_peak == 0:
                inverted_fluxes_peak = 1e-10
            if second_derivative_peak == 0:
                second_derivative_peak = 1e-10
            sharpness = second_derivative_peak/ inverted_fluxes_peak
            line_snr = np.power(inverted_fluxes_peak, 2) / (1 - np.power(inverted_fluxes_peak, 2))
            # Use abs instead of a simple '-1*' because sometime the result is negative and the sqrt cannot be calculated
            error = np.sqrt(np.abs(1 / (nbins * sharpness * line_snr)))

            final_model.set_emu(error)
#            print("Peak found at %.2f km/s (fitted at %.5f +/- %.5f km/s)" % (xcoord[peaks[i]], final_model.mu(), final_model.emu()))
            models.append(final_model)
        except Exception as e:
            print(type(e), e)


    return np.asarray(models)


def report_progress(current_work_progress, last_reported_progress):
    """

    :returns:

        True every 10% of progress.

    """
    return (int(current_work_progress) % 10 == 0 and current_work_progress - last_reported_progress > 10) or last_reported_progress < 0 or current_work_progress == 100


def cross_correlate_with_template(spectrum, template, lower_velocity_limit=-200, upper_velocity_limit=200, velocity_step=1.0, fourier=True, only_one_peak=False, model='2nd order polynomial + gaussian fit', peak_probability=0.75, frame=None):
    """
    Determines the velocity profile by cross-correlating the spectrum with
    a spectrum template.

    :returns:
        - Array with fitted gaussian models sorted by depth (deepest at position 0)
        - CCF structure with 'x' (velocities), 'y' (relative intensities), 'err'
    """
    return __cross_correlate(spectrum, linelist=None, template=template, \
            lower_velocity_limit=lower_velocity_limit, upper_velocity_limit = upper_velocity_limit, \
            velocity_step=velocity_step, \
            mask_size=None, mask_depth=None, fourier=fourier, \
            only_one_peak=only_one_peak, peak_probability=peak_probability, model=model, \
            frame=None)
            
def __cross_correlate(spectrum, linelist=None, template=None, lower_velocity_limit = -200, upper_velocity_limit = 200, velocity_step=1.0, mask_size=2.0, mask_depth=0.01, fourier=True, only_one_peak=False, peak_probability=0.75, model='2nd order polynomial + gaussian fit', frame=None):

    ccf, nbins = __build_velocity_profile(spectrum, \
            linelist = linelist, template = template, \
            lower_velocity_limit = lower_velocity_limit, upper_velocity_limit = upper_velocity_limit, \
            velocity_step=velocity_step, \
            mask_size=mask_size, mask_depth=mask_depth, \
            fourier=fourier, frame=frame)
            
    models = __model_velocity_profile(ccf, nbins, only_one_peak=only_one_peak, \
                                            peak_probability=peak_probability, model=model)
    # We have improved the peak probability detection using RLM, a priori it is not needed
    # this best selection:
    #best = select_good_velocity_profile_models(models, ccf)
    #return models[best], ccf
    
    return models, ccf
    
def determine_radial_velocity_with_template(star_spectrum,template,model = '2nd order polynomial + gaussian fit'):
    #mu_cas_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_muCas.txt.gz")
    #--- Radial Velocity determination with template -------------------------------
    logging.info("Radial velocity determination with template...")
    # - Read synthetic template
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Arcturus.372_926nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Sun.372_926nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/NARVAL.Sun.370_1048nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Synth.Sun.300_1100nm/template.txt.gz")
    models, ccf = cross_correlate_with_template(star_spectrum, template, \
                            lower_velocity_limit=-200, upper_velocity_limit=200, \
                            velocity_step=0.10, model = model, fourier=True)

    # Number of models represent the number of components
    components = len(models)
    # First component:
    rv = np.round(models[0].mu(), 6) # km/s
    rv_err = np.round(models[0].emu(), 6) # km/s
    return rv,rv_err
    
def __build_velocity_profile(spectrum, linelist=None, template=None, lower_velocity_limit = -200, upper_velocity_limit = 200, velocity_step=1.0, mask_size=2.0, mask_depth=0.01, fourier=True, frame=None):
    """
    Determines the velocity profile by cross-correlating the spectrum with:

    * a mask built from a line list if linelist is specified
    * a spectrum template if template is specified

    :returns:
        CCF structure with 'x' (velocities), 'y' (relative intensities), 'err'
        together with the number of spectrum's bins used in the cross correlation.

    """
    if linelist is not None:
        if template is not None:
            logging.warning("Building velocity profile with mask (ignoring template)")

        linelist = linelist[linelist['depth'] > 0.01]
        lfilter = np.logical_and(linelist['wave_peak'] >= np.min(spectrum['waveobs']), linelist['wave_peak'] <= np.max(spectrum['waveobs']))
        linelist = linelist[lfilter]

        velocity, ccf, ccf_err, nbins = __cross_correlation_function_uniform_in_velocity(spectrum, linelist, lower_velocity_limit, upper_velocity_limit, velocity_step, mask_size=mask_size, mask_depth=mask_depth, fourier=fourier, frame=frame)
    elif template is not None:
        ## Obtain the cross-correlate function by shifting the template
        velocity, ccf, ccf_err, nbins = __cross_correlation_function_uniform_in_velocity(spectrum, template, lower_velocity_limit, upper_velocity_limit, velocity_step, template=True, fourier=True, frame=frame)
        #velocity, ccf, ccf_err = __cross_correlation_function_template(spectrum, template, lower_velocity_limit = lower_velocity_limit, upper_velocity_limit=upper_velocity_limit, velocity_step = velocity_step, frame=frame)

    else:
        raise Exception("A linelist or template should be specified")

    ccf_struct = np.recarray((len(velocity), ), dtype=[('x', float),('y', float), ('err', float)])
    ccf_struct['x'] = velocity
    ccf_struct['y'] = ccf
    ccf_struct['err'] = ccf_err
    return ccf_struct, nbins

def __cross_correlation_function_uniform_in_velocity(spectrum, mask, lower_velocity_limit, upper_velocity_limit, velocity_step, mask_size=2.0, mask_depth=0.01, template=False, fourier=True, frame=None):
    """
    Calculates the cross correlation value between the spectrum and the specified mask
    by shifting the mask from lower to upper velocity.

    - The spectrum and the mask should be uniformly spaced in terms of velocity (which
      implies non-uniformly distributed in terms of wavelength).
    - The velocity step used for the construction of the mask should be the same
      as the one specified in this function.
    - The lower/upper/step velocity is only used to determine how many shifts
      should be done (in array positions) and return a velocity grid.

    If fourier is set, the calculation is done in the fourier space. More info:

        VELOCITIES FROM CROSS-CORRELATION: A GUIDE FOR SELF-IMPROVEMENT
        CARLOS ALLENDE PRIETO
        http://iopscience.iop.org/1538-3881/134/5/1843/fulltext/205881.text.html
        http://iopscience.iop.org/1538-3881/134/5/1843/fulltext/sourcecode.tar.gz
    """

    last_reported_progress = -1
    if frame is not None:
        frame.update_progress(0)

    # Speed of light in m/s
    c = 299792458.0

    # 1 shift = 1.0 km/s (or the specified value)
    shifts = np.arange(np.int32(np.floor(lower_velocity_limit)/velocity_step), np.int32(np.ceil(upper_velocity_limit)/velocity_step)+1)
    velocity = shifts * velocity_step

    waveobs = _sampling_uniform_in_velocity(np.min(spectrum['waveobs']), np.max(spectrum['waveobs']), velocity_step)
    flux = np.interp(waveobs, spectrum['waveobs'], spectrum['flux'], left=0.0, right=0.0)
    err = np.interp(waveobs, spectrum['waveobs'], spectrum['err'], left=0.0, right=0.0)


    if template:
        depth = np.abs(np.max(mask['flux']) - mask['flux'])
        resampled_mask = np.interp(waveobs, mask['waveobs'], depth, left=0.0, right=0.0)
    else:
        selected = __select_lines_for_mask(mask, minimum_depth=mask_depth, velocity_mask_size = mask_size, min_velocity_separation = 1.0)
        resampled_mask = __create_mask(waveobs, mask['wave_peak'][selected], mask['depth'][selected], velocity_mask_size=mask_size)

    if fourier:
        # Transformed flux and mask
        tflux = fft(flux)
        tresampled_mask = fft(resampled_mask)
        conj_tresampled_mask = np.conj(tresampled_mask)
        num = int(len(resampled_mask)/2+1)
        tmp = abs(ifft(tflux*conj_tresampled_mask))
        ccf = np.hstack((tmp[num:], tmp[:num]))

        # Transformed flux and mask powered by 2 (second)
        #ccf_err = np.zeros(len(ccf))
        # Conservative error propagation
        terr = fft(err)
        tmp = abs(ifft(terr*conj_tresampled_mask))
        ccf_err = np.hstack((tmp[num:], tmp[:num]))
        ## Error propagation
        #tflux_s = fft(np.power(flux, 2))
        #tresampled_mask_s = fft(np.power(resampled_mask, 2))
        #tflux_err_s = fft(np.power(err, 2))
        #tresampled_mask_err_s = fft(np.ones(len(err))*0.05) # Errors of 5% for masks

        #tmp = abs(ifft(tflux_s*np.conj(tresampled_mask_err_s)))
        #tmp += abs(ifft(tflux_err_s*np.conj(tresampled_mask_s)))
        #ccf_err = np.hstack((tmp[num:], tmp[:num]))
        #ccf_err = np.sqrt(ccf_err)

        # Velocities
        velocities = velocity_step * (np.arange(len(resampled_mask), dtype=float)+1 - num)

        # Filter to area of interest
        xfilter = np.logical_and(velocities >= lower_velocity_limit, velocities <= upper_velocity_limit)
        ccf = ccf[xfilter]
        ccf_err = ccf_err[xfilter]
        velocities = velocities[xfilter]
    else:
        num_shifts = len(shifts)
        # Cross-correlation function
        ccf = np.zeros(num_shifts)
        ccf_err = np.zeros(num_shifts)

        for shift, i in zip(shifts, np.arange(num_shifts)):
            #shifted_mask = resampled_mask
            if shift == 0:
                shifted_mask = resampled_mask
            elif shift > 0:
                #shifted_mask = np.hstack((shift*[0], resampled_mask[:-1*shift]))
                shifted_mask = np.hstack((resampled_mask[-1*shift:], resampled_mask[:-1*shift]))
            else:
                #shifted_mask = np.hstack((resampled_mask[-1*shift:], -1*shift*[0]))
                shifted_mask = np.hstack((resampled_mask[-1*shift:], resampled_mask[:-1*shift]))
            #ccf[i] = np.correlate(flux, shifted_mask)[0]
            #ccf_err[i] = np.correlate(err, shifted_mask)[0] # Propagate errors
            ccf[i] = np.average(flux*shifted_mask)
            ccf_err[i] = np.average(err*shifted_mask) # Propagate errors
            #ccf[i] = np.average(np.tanh(flux*shifted_mask))
            #ccf_err[i] = np.average(np.tanh(err*shifted_mask)) # Propagate errors

            current_work_progress = ((i*1.0)/num_shifts) * 100
            if report_progress(current_work_progress, last_reported_progress):
                last_reported_progress = current_work_progress
                logging.info("%.2f%%" % current_work_progress)
                if frame is not None:
                    frame.update_progress(current_work_progress)

    max_ccf = np.max(ccf)
    ccf = ccf/max_ccf # Normalize
    ccf_err = ccf_err/max_ccf # Propagate errors

    return velocity, ccf, ccf_err, len(flux)

def _sampling_uniform_in_velocity(wave_base, wave_top, velocity_step):
    """
    Create a uniformly spaced grid in terms of velocity:

    - An increment in position (i => i+1) supposes a constant velocity increment (velocity_step).
    - An increment in position (i => i+1) does not implies a constant wavelength increment.
    - It is uniform in log(wave) since:
          Wobs = Wrest * (1 + Vr/c)^[1,2,3..]
          log10(Wobs) = log10(Wrest) + [1,2,3..] * log10(1 + Vr/c)
      The last term is constant when dealing with wavelenght in log10.
    - Useful for building the cross correlate function used for determining the radial velocity of a star.
    """
    # Speed of light
    c = 299792.4580 # km/s
    #c = 299792458.0 # m/s

    ### Numpy optimized:
    # number of elements to go from wave_base to wave_top in increments of velocity_step
    i = int(np.ceil( (c * (wave_top - wave_base))/ (wave_base*velocity_step)))
    grid = wave_base * np.power((1 + (velocity_step/ c)), np.arange(i)+1)

    # Ensure wavelength limits since the "number of elements i" tends to be overestimated
    wfilter = grid <= wave_top
    grid = grid[wfilter]

    ### Non optimized:
    #grid = []
    #next_wave = wave_base
    #while next_wave <= wave_top:
        #grid.append(next_wave)
        ### Newtonian version:
        #next_wave = next_wave + next_wave * ((velocity_step) / c) # nm
        ### Relativistic version:
        ##next_wave = next_wave + next_wave * (1.-np.sqrt((1.-(velocity_step*1000.)/c)/(1.+(velocity_step*1000.)/c)))

    return np.asarray(grid)

def resample_spectrum():
    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Resampling  --------------------------------------------------------------
    logging.info("Resampling...")
    wavelengths = np.arange(480.0, 680.0, 0.001)
    resampled_star_spectrum = ispec.resample_spectrum(star_spectrum, wavelengths, method="linear", zero_edges=True)
    #resampled_star_spectrum = ispec.resample_spectrum(star_spectrum, wavelengths, method="bessel", zero_edges=True)
    
    
if __name__ == '__main__':

    custom_simbad = Simbad()
    custom_simbad.add_votable_fields('rv_value')
    
    files =sorted(glob.glob("/Users/daniel/Desktop/SALT_HRS_DATA/Red/2022/0716/reduced/?202207160016*product.fits"))
    mask_file = "./hrsreduce/ccf/Red_template.txt"
    mask_wave, mask_flux, mask_err = np.loadtxt(mask_file,usecols=(0,1,2),unpack=True,skiprows=2)
#    mask_file = './hrsreduce/ccf/G8_espresso.txt'
#    mask_wave, mask_flux = np.loadtxt(mask_file,usecols=(0,1),unpack=True,skiprows=0)
    template= np.recarray((len(mask_wave), ), dtype=[('waveobs', float),('flux', float),('err', float)])
    template['waveobs'] = mask_wave
    template['flux'] = mask_flux
    template['err'] = np.sqrt(mask_flux)
    
    mask_file = "./hrsreduce/ccf/Atlas_Sun.txt"
    mask_wave, mask_flux, mask_err = np.loadtxt(mask_file,usecols=(0,1,2),unpack=True,skiprows=1)
    ii=np.where(mask_flux > 0)[0]
    template= np.recarray((len(mask_wave[ii]), ), dtype=[('waveobs', float),('flux', float),('err', float)])
    template['waveobs'] = mask_wave[ii]*10.
    template['flux'] = mask_flux[ii]
    template['err'] = mask_err[ii]+1
    
    cont_wave, cont_flux =np.loadtxt('./hrsreduce/ccf/test_R_cont.dat',usecols=(0,1),unpack=True,skiprows=1)
    
    min_mask, max_mask = np.min(mask_wave),np.max(mask_wave)
    for file in files:
        with fits.open(file) as hdu:
            if hdu[0].header['PROPID'] == 'CAL_RVST':
                print(hdu[0].header['OBJECT'])
                bjd=float(hdu[0].header['BJD'])
                baryrv = float(hdu[0].header['BARYRV'])
                result = custom_simbad.query_object(hdu[0].header['OBJECT'])
                known_rv = (result["RV_VALUE"][0])
                
                print("ADD TO RV:",baryrv-known_rv)
                
                rv = []
                rv_err = []
                
                data = hdu[7].data
                hdr = hdu[7].header

                if data is None:
                    raise ValueError("FITS primary HDU has no data.")
                flux = np.asarray(data).copy()
                if flux.ndim > 1:
                    flux = flux.flatten()  # be conservative; user should supply 1D
                if 'CRVAL1' not in hdr or 'CDELT1' not in hdr:
                    raise ValueError("FITS header must contain CRVAL1 and CDELT1.")
                crval = float(hdr['CRVAL1'])
                cdelt = float(hdr['CDELT1'])
                n_pix = len(flux)
                wave = crval + cdelt * np.arange(n_pix)

#                np.savetxt('test_codes/R202207160016_product.fitsTEST_2D_6_6.dat',np.array([wave,flux,np.sqrt(flux)]).T)
#
                
                #    # --- Read stellar spectrum from FITS ---
                
                for ord in range(0,24):
                
                    data = hdu[2].data[ord][50:-50]
                    wave = hdu[4].data[ord][50:-50]
                    blaze = hdu[6].data[ord][50:-50]
                    flux = data / blaze
#                    data= hdu[10+ord*2].data
#                    wave = data['Wave']
#                    flux=data['Flux'] / data['Blaze']
                    
                    np.nan_to_num(flux,nan=1.)
                    
                    
                    ii=np.where(np.logical_and(cont_wave>np.min(wave), cont_wave< np.max(wave)))[0]
                    
                    cont = cont_flux[ii]
                    cont_rs = np.interp(wave, cont_wave[ii],cont)
                    
                    tell_region1 = [[5400,5500],[5650,5780],[5870,6050],[6250,6380],[6400,6650],[6850,7460],[7580,7750],[7850,8600]]

                    star_spectrum = np.recarray((len(wave), ), dtype=[('waveobs', float),('flux', float),('err', float)])
                    star_spectrum['waveobs'] = wave
                    tmp = (flux/cont_rs)
                    
                    tmp = np.nan_to_num(tmp,nan=1.,posinf=1., neginf=1.)
                    
                    jj=np.where(np.logical_and(template['waveobs']>np.min(star_spectrum['waveobs']),template['waveobs']<np.max(star_spectrum['waveobs'])))[0]
                    
                    offset = np.average(template['flux'])-np.average(tmp)
                    star_spectrum['flux'] = tmp + offset
                    star_spectrum['err'] = np.sqrt(tmp)
                    
#                    for reg in range(8):
#                        ii=np.where(np.logical_and(star_spectrum['waveobs']> tell_region1[reg][0], star_spectrum['waveobs']<tell_region1[reg][1]))[0]
#                        if len(ii)>0:
#                            star_spectrum['flux'][ii] = 1.
#
#                    plt.plot(template['waveobs'], template['flux'])
#                    plt.plot(star_spectrum['waveobs'],star_spectrum['flux'])
##
#                    plt.title(str(ord))
#                    plt.show()
                    if len(jj)>0:
                        rv_ord,rv_err_ord = determine_radial_velocity_with_template(star_spectrum,template)
                    
                        if np.abs((rv_ord+baryrv-known_rv)) < 10:
                            rv.append(rv_ord+baryrv-known_rv)
                            rv_err.append(rv_err_ord)
                            plt.errorbar(np.min(wave),rv_ord+baryrv-known_rv,rv_err_ord)
                            plt.plot(np.min(wave),rv_ord+baryrv-known_rv,'ko')
#                plt.ylim(-2,2)
#                plt.show()
                
                rv = np.array(rv)
                rv_err = np.array(rv_err)
                rv_overall = np.average(rv,weights = 1./rv_err)
                
                weights = 1./rv_err
                norm_weights = weights / np.sum(weights)
                rv_mn = np.sum(rv*norm_weights)

                var = 1./len(norm_weights) * sum((rv-rv_mn)**2)
                sigma = np.sqrt(var) * np.sqrt(sum(norm_weights**2))
            
                chunk_std = np.std(rv, ddof=1)
                chunk_err = chunk_std / np.sqrt(rv.size)  # standard error of the mean
                print(rv_overall, chunk_err, sigma)
                #print(rv+baryrv-known_rv,rv_err)

                data = hdu[7].data
                hdr = hdu[7].header
            
                if data is None:
                    raise ValueError("FITS primary HDU has no data.")
                flux = np.asarray(data).copy()
                if flux.ndim > 1:
                    flux = flux.flatten()  # be conservative; user should supply 1D
                if 'CRVAL1' not in hdr or 'CDELT1' not in hdr:
                    raise ValueError("FITS header must contain CRVAL1 and CDELT1.")
                crval = float(hdr['CRVAL1'])
                cdelt = float(hdr['CDELT1'])
                n_pix = len(flux)
                wave = crval + cdelt * np.arange(n_pix)
                
#                np.savetxt('test_codes/R202201130032_product.fitsTEST_2D_9_11.dat',np.array([wave,flux,np.sqrt(flux)]).T)

                star_spectrum = np.recarray((len(wave), ), dtype=[('waveobs', float),('flux', float),('err', float)])
                ii=np.where(np.logical_and(cont_wave>np.min(wave), cont_wave< np.max(wave)))[0]
                cont = cont_flux[ii]
                cont_rs = np.interp(wave, cont_wave[ii],cont)
                star_spectrum['waveobs'] = wave
                star_spectrum['flux'] = flux/cont_rs
                star_spectrum['err'] = np.sqrt(flux/cont_rs)
                
                plt.plot(star_spectrum['waveobs'] ,star_spectrum['flux'])
                plt.plot(template['waveobs'] ,template['flux'])
                plt.show()
                rv,rv_err = determine_radial_velocity_with_template(star_spectrum,template)
                
                print("WHOLE SPECTRUM=",rv+baryrv-known_rv,rv_err)
#                plt.plot(bjd,rv+baryrv-known_rv,'ko')
#                plt.errorbar(bjd,rv+baryrv-known_rv,rv_err)

#                tell_region1 = [[5870,6000],[6270,6330],[6436,6610],[6859,7060],[7154,7400],[7580,7725],[7984,8400]]
#
#                wave, flux = np.loadtxt('test_codes/R202201130032_product.fitsTEST_norm_2D_7_7.dat',usecols=(0,1),unpack=True,skiprows=2)
#                star_spectrum = np.recarray((len(wave), ), dtype=[('waveobs', float),('flux', float),('err', float)])
#                
#                for reg in range(7):
#                    ii=np.where(np.logical_and(wave> tell_region1[reg][0], wave<tell_region1[reg][1]))[0]
#                    flux[ii] = 1.
#                    star_spectrum['waveobs'] = wave
#                    star_spectrum['flux'] = flux - np.min(flux)
#                    star_spectrum['err'] = np.sqrt(flux- np.min(flux))
#
#                
#                rv,rv_err = determine_radial_velocity_with_template(star_spectrum,template)
#                
#                print(rv+baryrv-known_rv,rv_err)
#                plt.plot(bjd,rv+baryrv-known_rv,'ko')
#                plt.errorbar(bjd,rv+baryrv-known_rv,rv_err)
#                
#                wave, flux = np.loadtxt('test_codes/R202201130032_product.fitsTEST_norm_2D_6_6.dat',usecols=(0,1),unpack=True,skiprows=2)
#                star_spectrum = np.recarray((len(wave), ), dtype=[('waveobs', float),('flux', float),('err', float)])
#                for reg in range(8):
#                    ii=np.where(np.logical_and(wave> tell_region1[reg][0], wave<tell_region1[reg][1]))[0]
#                    flux[ii] = 1.
#                    star_spectrum['waveobs'] = wave
#                    star_spectrum['flux'] = flux - np.min(flux)
#                    star_spectrum['err'] = np.sqrt(flux- np.min(flux))
#                
#                rv,rv_err = determine_radial_velocity_with_template(star_spectrum,template)
#                
#                print(rv+baryrv-known_rv,rv_err)
#                plt.plot(bjd,rv+baryrv-known_rv,'bo')
#                plt.errorbar(bjd,rv+baryrv-known_rv,rv_err)
#                
#                wave, flux = np.loadtxt('test_codes/R202201130032_product.fitsTEST_norm_2D_5_5.dat',usecols=(0,1),unpack=True,skiprows=2)
#                star_spectrum = np.recarray((len(wave), ), dtype=[('waveobs', float),('flux', float),('err', float)])
#                for reg in range(7):
#                    ii=np.where(np.logical_and(wave> tell_region1[reg][0], wave<tell_region1[reg][1]))[0]
#                    flux[ii] = 1.
#                    star_spectrum['waveobs'] = wave
#                    star_spectrum['flux'] = flux - np.min(flux)
#                    star_spectrum['err'] = np.sqrt(flux- np.min(flux))
#                
#                rv,rv_err = determine_radial_velocity_with_template(star_spectrum,template)
#                
#                print(rv+baryrv-known_rv,rv_err)
#                plt.plot(bjd,rv+baryrv-known_rv,'go')
#                plt.errorbar(bjd,rv+baryrv-known_rv,rv_err)
#                
#                wave, flux = np.loadtxt('test_codes/R202201130032_product.fitsTEST_norm_2D_8_8.dat',usecols=(0,1),unpack=True,skiprows=2)
#                star_spectrum = np.recarray((len(wave), ), dtype=[('waveobs', float),('flux', float),('err', float)])
#                for reg in range(7):
#                    ii=np.where(np.logical_and(wave> tell_region1[reg][0], wave<tell_region1[reg][1]))[0]
#                    flux[ii] = 1.
#                    star_spectrum['waveobs'] = wave
#                    star_spectrum['flux'] = flux - np.min(flux)
#                    star_spectrum['err'] = np.sqrt(flux- np.min(flux))
#                
#                rv,rv_err = determine_radial_velocity_with_template(star_spectrum,template)
#                
#                print(rv+baryrv-known_rv,rv_err)
#                plt.plot(bjd,rv+baryrv-known_rv,'co')
#                plt.errorbar(bjd,rv+baryrv-known_rv,rv_err)
#                
#                
#                wave, flux = np.loadtxt('test_codes/R202201130032_product.fitsTEST_norm_2D_6_5.dat',usecols=(0,1),unpack=True,skiprows=2)
#                star_spectrum = np.recarray((len(wave), ), dtype=[('waveobs', float),('flux', float),('err', float)])
#                for reg in range(7):
#                    ii=np.where(np.logical_and(wave> tell_region1[reg][0], wave<tell_region1[reg][1]))[0]
#                    flux[ii] = 1.
#                    star_spectrum['waveobs'] = wave
#                    star_spectrum['flux'] = flux - np.min(flux)
#                    star_spectrum['err'] = np.sqrt(flux- np.min(flux))
#                
#                rv,rv_err = determine_radial_velocity_with_template(star_spectrum,template)
#                
#                print(rv+baryrv-known_rv,rv_err)
#                plt.plot(bjd,rv+baryrv-known_rv,'ro')
#                plt.errorbar(bjd,rv+baryrv-known_rv,rv_err)
#                
#                wave, flux = np.loadtxt('test_codes/R202201130032_product.fitsTEST_norm_2D_5_6.dat',usecols=(0,1),unpack=True,skiprows=2)
#                star_spectrum = np.recarray((len(wave), ), dtype=[('waveobs', float),('flux', float),('err', float)])
#                for reg in range(7):
#                    ii=np.where(np.logical_and(wave> tell_region1[reg][0], wave<tell_region1[reg][1]))[0]
#                    flux[ii] = 1.
#                    star_spectrum['waveobs'] = wave
#                    star_spectrum['flux'] = flux - np.min(flux)
#                    star_spectrum['err'] = np.sqrt(flux- np.min(flux))
#                
#                rv,rv_err = determine_radial_velocity_with_template(star_spectrum,template)
#                
#                print(rv+baryrv-known_rv,rv_err)
#                plt.plot(bjd,rv+baryrv-known_rv,'ko')
#                plt.errorbar(bjd,rv+baryrv-known_rv,rv_err)
#                
#                wave, flux = np.loadtxt('test_codes/R202201130032_product.fitsTEST_norm_2D_9_11.dat',usecols=(0,1),unpack=True,skiprows=2)
#                star_spectrum = np.recarray((len(wave), ), dtype=[('waveobs', float),('flux', float),('err', float)])
#                for reg in range(7):
#                    ii=np.where(np.logical_and(wave> tell_region1[reg][0], wave<tell_region1[reg][1]))[0]
#                    flux[ii] = 1.
#                    star_spectrum['waveobs'] = wave
#                    star_spectrum['flux'] = flux - np.min(flux)
#                    star_spectrum['err'] = np.sqrt(flux- np.min(flux))
#                
#                rv,rv_err = determine_radial_velocity_with_template(star_spectrum,template)
#                
#                print(rv+baryrv-known_rv,rv_err)
#                plt.plot(bjd,rv+baryrv-known_rv,'ko')
#                plt.errorbar(bjd,rv+baryrv-known_rv,rv_err)
                
#                tellurics_w,tellurics_i = np.loadtxt('./hrsreduce/ccf/tellurics.txt',usecols=(0,1),unpack=True,skiprows=2)
#                
#                ii=np.where(tellurics_i !=1)[0]
#                
#                for line in tellurics_w[ii]:
#                    jj=np.where(np.logical_and(star_spectrum['waveobs']>line-1,star_spectrum['waveobs']<line+1 ))[0]
#                    if len(jj)>0:
#                        star_spectrum['flux'][jj]=1.
                    
#
#                rv,rv_err = determine_radial_velocity_with_template(star_spectrum,template)
#                print(rv+baryrv-known_rv,rv_err)

    files =sorted(glob.glob("/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0716/reduced/?202207160016*product.fits"))
    mask_file = "./hrsreduce/ccf/Blu_template.txt"
    mask_wave, mask_flux, mask_err = np.loadtxt(mask_file,usecols=(0,1,2),unpack=True,skiprows=2)
    template= np.recarray((len(mask_wave), ), dtype=[('waveobs', float),('flux', float),('err', float)])
    template['waveobs'] = mask_wave
    template['flux'] = mask_flux
    template['err'] = mask_err
    
    mask_file = "./hrsreduce/ccf/spectrum_5400.0_4.0_0.0_0.0_1.07_R20000.txt"
    mask_wave, mask_flux, mask_err = np.loadtxt(mask_file,usecols=(0,1,2),unpack=True,skiprows=1)
    template= np.recarray((len(mask_wave), ), dtype=[('waveobs', float),('flux', float),('err', float)])
    template['waveobs'] = mask_wave*10.
    template['flux'] = mask_flux
    template['err'] = mask_err+1
    
#    mask_file = './hrsreduce/ccf/G8_espresso.txt'
#    mask_wave, mask_flux = np.loadtxt(mask_file,usecols=(0,1),unpack=True,skiprows=0)
#    template= np.recarray((len(mask_wave), ), dtype=[('waveobs', float),('flux', float),('err', float)])
#    template['waveobs'] = mask_wave
#    template['flux'] = 1-mask_flux
#    template['err'] = np.sqrt(mask_flux)
    
    
    cont_wave, cont_flux =np.loadtxt('./hrsreduce/ccf/test_H_cont.dat',usecols=(0,1),unpack=True,skiprows=1)
    
    min_mask, max_mask = np.min(mask_wave),np.max(mask_wave)
    for file in files:
        with fits.open(file) as hdu:
            if hdu[0].header['PROPID'] == 'CAL_RVST':
                print(hdu[0].header['OBJECT'])
                bjd=float(hdu[0].header['BJD'])
                baryrv = float(hdu[0].header['BARYRV'])
                result = custom_simbad.query_object(hdu[0].header['OBJECT'])
                known_rv = (result["RV_VALUE"][0])
                
                print("ADD TO RV:",baryrv-known_rv)
                
                data = hdu[7].data
                hdr = hdu[7].header

                if data is None:
                    raise ValueError("FITS primary HDU has no data.")
                flux = np.asarray(data).copy()
                if flux.ndim > 1:
                    flux = flux.flatten()  # be conservative; user should supply 1D
                if 'CRVAL1' not in hdr or 'CDELT1' not in hdr:
                    raise ValueError("FITS header must contain CRVAL1 and CDELT1.")
                crval = float(hdr['CRVAL1'])
                cdelt = float(hdr['CDELT1'])
                n_pix = len(flux)
                wave = crval + cdelt * np.arange(n_pix)

#                np.savetxt('test_codes/H202207160016_product.fitsTEST_2D_6_6.dat',np.array([wave,flux,np.sqrt(flux)]).T)
#
                
                rv = []
                rv_err = []
                
                #    # --- Read stellar spectrum from FITS ---
                
                for ord in range(41):
                
                    data= hdu[10+ord*2].data
                    wave = data['Wave']
                    flux=data['Flux'] / data['Blaze']
                    
                    np.nan_to_num(flux,nan=1.)
                    
                    
                    ii=np.where(np.logical_and(cont_wave>np.min(wave), cont_wave< np.max(wave)))[0]
                    
                    cont = cont_flux[ii]
                    cont_rs = np.interp(wave, cont_wave[ii],cont)
                    
                    tell_region1 = [[5400,5500],[5650,5780],[5870,6050],[6250,6380],[6400,6650],[6850,7460],[7580,7750],[7850,8600]]

                    star_spectrum = np.recarray((len(wave), ), dtype=[('waveobs', float),('flux', float),('err', float)])
                    star_spectrum['waveobs'] = wave
                    tmp = (flux/cont_rs)
                    
                    tmp = np.nan_to_num(tmp,nan=1.,posinf=1., neginf=1.)
                    
                    jj=np.where(np.logical_and(template['waveobs']>np.min(star_spectrum['waveobs']),template['waveobs']<np.max(star_spectrum['waveobs'])))[0]
                    
                    offset = np.average(template['flux'])-np.average(tmp)
                    
                    star_spectrum['flux'] = tmp/2.
                    star_spectrum['err'] = np.sqrt(tmp)
                    
#                    for reg in range(8):
#                        ii=np.where(np.logical_and(star_spectrum['waveobs']> tell_region1[reg][0], star_spectrum['waveobs']<tell_region1[reg][1]))[0]
#                        if len(ii)>0:
#                            star_spectrum['flux'][ii] = 1.
#
#                    plt.plot(template['waveobs'], template['flux'])
#                    plt.plot(star_spectrum['waveobs'],star_spectrum['flux'])
#                    plt.title(str(ord))
#                    plt.show()
                    
                    rv_ord,rv_err_ord = determine_radial_velocity_with_template(star_spectrum,template)
                    
                    if np.abs((rv_ord+baryrv-known_rv)) < 10:
                        rv.append(rv_ord+baryrv-known_rv)
                        rv_err.append(rv_err_ord)
                        plt.errorbar(np.min(wave),rv_ord+baryrv-known_rv,rv_err_ord)
                        plt.plot(np.min(wave),rv_ord+baryrv-known_rv,'ro')
#                plt.ylim(-2,2)

                
                rv = np.array(rv)
                rv_err = np.array(rv_err)
                rv_overall = np.average(rv,weights = 1./rv_err)
                
                weights = 1./rv_err
                norm_weights = weights / np.sum(weights)
                rv_mn = np.sum(rv*norm_weights)

                var = 1./len(norm_weights) * sum((rv-rv_mn)**2)
                sigma = np.sqrt(var) * np.sqrt(sum(norm_weights**2))
            
                chunk_std = np.std(rv, ddof=1)
                chunk_err = chunk_std / np.sqrt(rv.size)  # standard error of the mean
                print("BLUE HRSReduce", rv_overall, chunk_err, sigma)

                #plt.show()

    files =sorted(glob.glob("/Users/daniel/Documents/Work/SALT_Pipeline/HRSReduce/MIDAS_Reductions/20220716/mbgphH202207160016_1we.fits"))
    for file in files:
        print(file)
        with fits.open(file) as hdu:
            if hdu[0].header['PROPID'] == 'CAL_RVST':
                print(hdu[0].header['OBJECT'])
                #bjd=float(hdu[0].header['JD'])
                #baryrv = float(hdu[0].header['BARYRV'])
                #result = custom_simbad.query_object(hdu[0].header['OBJECT'])
                #known_rv = (result["RV_VALUE"][0])
                
                print("ADD TO RV:",baryrv-known_rv)
                rv = []
                rv_err = []
                for ord in range(37):
                    data = hdu[ord].data
                    hdr = hdu[ord].header

                    if data is None:
                        raise ValueError("FITS primary HDU has no data.")
                    flux = np.asarray(data).copy()
                    if flux.ndim > 1:
                        flux = flux.flatten()  # be conservative; user should supply 1D
                    if 'CRVAL1' not in hdr or 'CDELT1' not in hdr:
                        raise ValueError("FITS header must contain CRVAL1 and CDELT1.")
                    crval = float(hdr['CRVAL1'])
                    cdelt = float(hdr['CDELT1'])
                    n_pix = len(flux)
                    wave = crval + cdelt * np.arange(n_pix)

                    ii=np.where(np.logical_and(cont_wave>np.min(wave), cont_wave< np.max(wave)))[0]
                    
                    cont = cont_flux[ii]
                    cont_rs = np.interp(wave, cont_wave[ii],cont)
                    
                    star_spectrum = np.recarray((len(wave), ), dtype=[('waveobs', float),('flux', float),('err', float)])
                    star_spectrum['waveobs'] = wave
                    tmp = (flux/cont_rs)
                    tmp = np.nan_to_num(tmp,nan=1.,posinf=1., neginf=1.)
                    
                    offset = np.average(template['flux'])-np.average(tmp)
                    
                    star_spectrum['flux'] = tmp/2.
                    star_spectrum['err'] = np.sqrt(tmp)
                    
#                    for reg in range(8):
#                        ii=np.where(np.logical_and(star_spectrum['waveobs']> tell_region1[reg][0], star_spectrum['waveobs']<tell_region1[reg][1]))[0]
#                        if len(ii)>0:
#                            star_spectrum['flux'][ii] = 1.
#
#                    plt.plot(template['waveobs'], template['flux'])
#                    plt.plot(star_spectrum['waveobs'],star_spectrum['flux'])
#                    plt.title(str(ord))
#                    plt.show()
                    
                    rv_ord,rv_err_ord = determine_radial_velocity_with_template(star_spectrum,template)
                    
                    if np.abs((rv_ord+baryrv-known_rv)) < 10:
                        rv.append(rv_ord+baryrv-known_rv)
                        rv_err.append(rv_err_ord)
                        plt.errorbar(np.min(wave),rv_ord+baryrv-known_rv,rv_err_ord)
                        plt.plot(np.min(wave),rv_ord+baryrv-known_rv,'ko')
#                plt.ylim(-2,2)

                
                rv = np.array(rv)
                rv_err = np.array(rv_err)
                rv_overall = np.average(rv,weights = 1./rv_err)
                
                weights = 1./rv_err
                norm_weights = weights / np.sum(weights)
                rv_mn = np.sum(rv*norm_weights)

                var = 1./len(norm_weights) * sum((rv-rv_mn)**2)
                sigma = np.sqrt(var) * np.sqrt(sum(norm_weights**2))
            
                chunk_std = np.std(rv, ddof=1)
                chunk_err = chunk_std / np.sqrt(rv.size)  # standard error of the mean
                print("BLUE MIDAS",rv_overall, chunk_err, sigma)

                plt.show()
