import numpy as np
import numpy.ma as ma

class FrameStacker:
    """
    Combine a stack of image frames using robust pixel-by-pixel statistics.

    This class is used to create master calibration products from multiple
    input frames. It operates on a three-dimensional image stack and combines
    the frames on a pixel-by-pixel basis after rejecting outliers using
    sigma clipping. The output includes the stacked image, the corresponding
    variance, the number of contributing frames per pixel, and the estimated
    uncertainty.

    The clipping threshold is defined relative to a robust estimate of the
    pixel dispersion, computed from the 16th and 84th percentiles of the
    values in the stack. Because clipping artificially reduces the measured
    variance, the class also estimates and applies a correction factor using
    Monte Carlo simulations of normally distributed data.

    Parameters
    ----------
    frames_data : numpy.ndarray
        Three-dimensional image stack with shape
        `(n_frames, n_rows, n_cols)`.
    n_sigma : float, optional
        Sigma-clipping threshold applied about the per-pixel median.
    logger : logging.Logger, optional
        Logger used for status and debug messages.

    Attributes
    ----------
    frames_data : numpy.ndarray
        Input stack of image frames.
    n_sigma : float
        Sigma-clipping threshold.
    logger : logging.Logger or None
        Logger used for status and debug output.

    Notes
    -----
    Origin: KPF DRP
    """

    __version__ = '1.0.1'

    def __init__(self,frames_data,n_sigma=2.5,logger=None):
        self.frames_data = frames_data
        self.n_sigma = n_sigma
        if logger:
            self.logger = logger
        else:
            self.logger = None

        if self.logger:
            self.logger.info('Started {}'.format(self.__class__.__name__))

    def compute_clip_corr(self):

        """
        Estimate the variance correction factor introduced by sigma clipping.

        Sigma clipping suppresses outliers but also reduces the measured variance
        of the retained sample. This method uses Monte Carlo simulations of
        standard normal random values to estimate how much the variance is
        systematically reduced for the chosen clipping threshold, and returns the
        factor needed to restore the expected variance.

        Returns
        -------
        float
            Multiplicative correction factor applied to clipped variances.
        """

        n_sigma = self.n_sigma
        var_trials = []
        
        for x in range(0,10):
            a = np.random.normal(0.0, 1.0, 1000000)
            med = np.median(a, axis=0)
            p16 = np.percentile(a, 16, axis=0)
            p84 = np.percentile(a, 84, axis=0)
            sigma = 0.5 * (p84 - p16)
            mdmsg = med - n_sigma * sigma
            b = np.less(a,mdmsg)
            mdpsg = med + n_sigma * sigma
            c = np.greater(a,mdpsg)
            mask = np.any([b,c],axis=0)
            mx = ma.masked_array(a, mask)
            var = ma.getdata(mx.var(axis=0))
            var_trials.append(var)
        
        np_var_trials = np.array(var_trials)
        avg_var_trials = np.mean(np_var_trials)
        std_var_trials = np.std(np_var_trials)
        corr_fact = 1.0 / avg_var_trials

        if self.logger:
            self.logger.debug('{}.compute_clip_corr(): avg_var_trials,std_var_trials,corr_fact = {},{},{}'.\
                format(self.__class__.__name__,avg_var_trials,std_var_trials,corr_fact))
        else:
            print('---->{}.compute_clip_corr(): avg_var_trials,std_var_trials,corr_fact = {},{},{}'.\
                format(self.__class__.__name__,avg_var_trials,std_var_trials,corr_fact))

        return corr_fact

    def compute(self):

        """
        Compute a sigma-clipped mean stack and its associated statistics.

        This method performs pixel-by-pixel sigma clipping across the input frame
        stack, computes the mean of the retained values, and returns the stacked
        image together with a variance estimate, the number of contributing frames
        per pixel, and the propagated uncertainty.

        The variance is corrected using the factor returned by
        `compute_clip_corr()` to account for the bias introduced by clipping.

        Returns
        -------
        tuple
            Four-element tuple containing:
                - `avg` : sigma-clipped mean image
                - `var` : corrected variance image
                - `cnt` : number of contributing frames per pixel
                - `unc` : uncertainty image
        """

        cf = self.compute_clip_corr()

        a = self.frames_data
        n_sigma = self.n_sigma
        frames_data_shape = np.shape(a)

        if self.logger:
            self.logger.debug('{}.compute(): self.n_sigma,frames_data_shape = {},{}'.\
                format(self.__class__.__name__,self.n_sigma,frames_data_shape))
        else:
            print('---->{}.compute(): self.n_sigma,frames_data_shape = {},{}'.\
                format(self.__class__.__name__,self.n_sigma,frames_data_shape))

        med = np.median(a, axis=0)
        p16 = np.percentile(a, 16, axis=0)
        p84 = np.percentile(a, 84, axis=0)
        sigma = 0.5 * (p84 - p16)
        mdmsg = med - n_sigma * sigma
        b = np.less(a,mdmsg)
        mdpsg = med + n_sigma * sigma
        c = np.greater(a,mdpsg)
        mask = np.any([b,c],axis=0)
        mx = ma.masked_array(a, mask)
        avg = ma.getdata(mx.mean(axis=0))
        var = ma.getdata(mx.var(axis=0)) * cf
        cnt = ma.getdata(ma.count(mx,axis=0))
        unc = np.sqrt(var/cnt)

        if self.logger:
            self.logger.debug('{}.compute(): avg(stack_avg),avg(cnt),avg(unc) = {},{},{}'.\
                format(self.__class__.__name__,avg.mean(),cnt.mean(),unc.mean()))
        else:
            print('---->{}.compute(): avg(stack_avg),avg(cnt),avg(unc) = {},{},{}'.\
                format(self.__class__.__name__,avg.mean(),cnt.mean(),unc.mean()))

        return avg,var,cnt,unc

    def compute_noise_model(self, read_noise, exptimes=None, n_sigma=5.0, flux_tol=0.01, photon_noise=True):
        """
        Stack frames, rejecting outliers against the EXPECTED noise.

        Unlike compute(), the rejection threshold does not come from the scatter
        of the few frames in the stack but from the detector noise model
        (photon noise + read noise), so good frames are not rejected by chance
        and the method works with as few as two frames.

        Parameters
        ----------
        read_noise : float
            Single-frame read noise (e-), e.g. the RONOISE header value.
        exptimes : array-like, optional
            Exposure time of each frame if the stack was divided by EXPTIME
            (flats, arcs). None for frames in electrons (biases).
        n_sigma : float
            Rejection threshold in units of the expected single-frame noise.
        flux_tol : float
            Fractional frame-to-frame flux change tolerated without rejection
            (lamp drift); added in quadrature to the noise.

        photon_noise : bool
            Include photon noise from the signal level. Set False for bias
            frames, whose level is an electronic pedestal, not detected light.

        Returns
        -------
        avg, var, cnt, unc : as compute(). `var` is the expected variance of a
            single frame and `unc` the uncertainty of the stacked mean.
        """
        a = np.asarray(self.frames_data, dtype=float)
        n = a.shape[0]
        t = np.ones(n) if exptimes is None else np.asarray(exptimes, dtype=float)
        t = t.reshape((n,) + (1,) * (a.ndim - 1))

        # Reference level: median, or the lower frame when there are only two
        # (cosmic rays are positive, so the minimum is the clean one).
        ref = np.median(a, axis=0) if n >= 3 else np.min(a, axis=0)

        # Expected variance of each frame in the units of the stack.
        signal = np.clip(ref, 0.0, None) if photon_noise else np.zeros_like(ref)
        var_i = (signal * t + read_noise ** 2) / t ** 2
        tol2 = (flux_tol * signal) ** 2

        dev = a - ref
        if n >= 3:
            mask = np.abs(dev) > n_sigma * np.sqrt(var_i + tol2)
        elif n == 2:
            # reference is itself one noisy frame: compare with the noise of a difference
            mask = dev > n_sigma * np.sqrt(2.0 * var_i + tol2)
        else:
            mask = np.zeros(a.shape, dtype=bool)

        mx = ma.masked_array(a, mask)
        avg = ma.getdata(mx.mean(axis=0))
        cnt = ma.getdata(ma.count(mx, axis=0))
        kept_var = ma.getdata(ma.masked_array(np.broadcast_to(var_i, a.shape), mask).sum(axis=0))
        unc = np.sqrt(kept_var) / cnt          # uncertainty of the mean of the kept frames
        var = kept_var / cnt                   # mean single-frame variance
        return avg, var, cnt, unc

#
# Method similar to compute() to be called in case of insufficient number of frames in stack,
# in which case the estimator of the expected value will be the stack median.
# The calling program determines whether to call this method.
#

    def compute_stack_median(self):
        """
        Compute the median stack and associated statistics.

        This alternative stacking method is intended for situations where too few
        frames are available for reliable sigma-clipped averaging. It computes the
        pixel-by-pixel median of the stack and estimates the variance from the
        median absolute deviation.

        Returns
        -------
        tuple
            Four-element tuple containing:
                - `med` : median-stacked image
                - `var` : variance estimate from the median absolute deviation
                - `cnt` : number of input frames contributing to each pixel
                - `unc` : uncertainty image
        """

        a = self.frames_data
        frames_data_shape = np.shape(a)

        if self.logger:
            self.logger.debug('{}.compute(): frames_data_shape = {}'.\
                format(self.__class__.__name__,frames_data_shape))
        else:
            print('---->{}.compute(): frames_data_shape = {}'.\
                format(self.__class__.__name__,frames_data_shape))

        med = np.median(a, axis=0)
        mad = np.median(np.absolute(a - med),axis=0)

        var = mad * mad
        cnt = np.full((frames_data_shape[1],frames_data_shape[2]),frames_data_shape[0])
        unc = np.sqrt(var/cnt)

        if self.logger:
            self.logger.debug('{}.compute(): avg(stack_med),avg(cnt),avg(unc) = {},{},{}'.\
                format(self.__class__.__name__,med.mean(),cnt.mean(),unc.mean()))
        else:
            print('---->{}.compute(): avg(stack_med),avg(cnt),avg(unc) = {},{},{}'.\
                format(self.__class__.__name__,med.mean(),cnt.mean(),unc.mean()))

        return med,var,cnt,unc


def read_noise_from_headers(files, default=5.0, logger=None):
    """
    Median RONOISE (e-) from the headers of a list of FITS files.

    Falls back to `default` with a warning if no file carries the keyword,
    so that stacking never fails for want of a read-noise value.
    """
    from astropy.io import fits
    values = []
    for file in files:
        try:
            values.append(float(fits.getheader(file)['RONOISE']))
        except Exception:
            pass
    values = [v for v in values if np.isfinite(v) and v > 0]
    if values:
        return float(np.median(values))
    if logger:
        logger.warning('No RONOISE keyword in the frames to stack; assuming {} e-'.format(default))
    return default
