import numpy as np
import scipy
from scipy.ndimage.filters import median_filter
from astropy.io import fits
from hrsreduce.utils.header_comments import clean_comments
from scipy.linalg import solve_banded #lstsq, solve
from scipy.signal import savgol_filter
from itertools import chain
import matplotlib.pyplot as plt

import logging, os
from .. import __version__   # pipeline version, written to the product header

logger = logging.getLogger(__name__)

from scipy.sparse import coo_matrix

C_KMS = 299792.458
# velocity step (km/s) of the merged grids per mode: at least three bins per resolution element at the mode's best resolution
VSTEP_KMS = {'HR': 1.0, 'HS': 1.0, 'MR': 2.0, 'LR': 5.0}
# optional legacy outputs (off: the product holds the two MERGED tables and the native per-order arrays only)
LEGACY_MERGED_IMAGES = False      # also write FIBRE_L_SPEC_MERGED etc. as image extensions with a linear WCS (old readers)
WRITE_ORDER_TABLES = False        # also write ORD<m>_L/U binary tables (native pixels: Wave, Flux, Var, Blaze)
# quick-look S/N plot: Balmer lines marked on the panel and the reference S/N levels drawn (level, colour)
BALMER_LINES = {'H$\\alpha$': 6562.8, 'H$\\beta$': 4861.3, 'H$\\gamma$': 4340.5, 'H$\\delta$': 4101.7}
SNR_REFERENCE_LINES = ((10, 'green'), (50, 'blue'), (100, 'red'))


def _pixel_edges(w):
    """edges of the native pixels from their centre wavelengths (monotonic)"""
    e = np.empty(w.size + 1)
    e[1:-1] = 0.5 * (w[1:] + w[:-1]); e[0] = w[0] - (w[1] - w[0]) / 2; e[-1] = w[-1] + (w[-1] - w[-2]) / 2
    return e


def _overlap_matrix(e, E):
    """sparse (n_bins x n_pix) matrix: fraction of native pixel [e_i, e_i+1] that falls in output bin [E_k, E_k+1]"""
    n_pix = e.size - 1; n_bin = E.size - 1
    k0 = np.clip(np.searchsorted(E, e[:-1], side='right') - 1, 0, n_bin - 1)
    k1 = np.clip(np.searchsorted(E, e[1:], side='left') - 1, 0, n_bin - 1)
    n = k1 - k0 + 1
    rows_pix = np.repeat(np.arange(n_pix), n)
    cols = np.repeat(k0, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    lo = np.maximum(E[cols], e[rows_pix]); hi = np.minimum(E[cols + 1], e[rows_pix + 1])
    w = np.clip(hi - lo, 0, None) / (e[rows_pix + 1] - e[rows_pix])
    return coo_matrix((w, (cols, rows_pix)), shape=(n_bin, n_pix)).tocsr()


def _rebin_order(w, F, V, B, E):
    """Flux-conserving rebin of one order onto the bins with edges E: every native pixel's counts are distributed over
    the bins it overlaps (variance with the squared fractions, blaze like the counts).  Pixels without a finite flux,
    variance or blaze carry nothing and invalidate the bins they touch; bins not fully inside the order are invalid.
    Returns (counts, variance, blaze, valid) per bin, or None if the order is unusable."""
    okw = np.isfinite(w)
    if okw.sum() < 50:
        return None
    w, F, V, B = w[okw], F[okw], V[okw], B[okw]
    order = np.argsort(w); w, F, V, B = w[order], F[order], V[order], B[order]
    valid = np.isfinite(F) & np.isfinite(V) & np.isfinite(B) & (B > 0)
    M = _overlap_matrix(_pixel_edges(w), E); M2 = M.multiply(M)
    counts = M @ np.where(valid, F, 0.0); var = M2 @ np.where(valid, V, 0.0); blaze = M @ np.where(valid, B, 0.0)
    touched_bad = (M @ (~valid).astype(float)) > 0
    inside = (E[:-1] >= w[0]) & (E[1:] <= w[-1])
    return counts, var, blaze, inside & ~touched_bad & (blaze > 0)


# order ends: pixels whose raw blaze departs from its smoothed version by more than max(BLAZE_DEV_MIN, BLAZE_DEV_NSIG x the
# order's own scatter) are left out of the merge (detector edges, truncated apertures, defect columns), and each order's
# weight is faded to EDGE_FADE_MIN over EDGE_FADE_FRAC of its valid pixels at both ends
BLAZE_DEV_MIN = 0.03
BLAZE_DEV_NSIG = 5.0
EDGE_FADE_FRAC = 0.03
EDGE_FADE_MIN = 0.05


def _smooth_blaze(B):
    """Savitzky-Golay (61, 3) smoothing of the blaze, order by order, with NaN pixels bridged by linear interpolation."""
    out = np.full_like(B, np.nan)
    for o in range(B.shape[0]):
        b = B[o].copy(); nans = ~np.isfinite(b); x = np.arange(b.size)
        if (~nans).sum() < 100:
            continue
        b[nans] = np.interp(x[nans], x[~nans], b[~nans]); out[o] = savgol_filter(b, 61, 3)
    return out


def _clean_blaze(B):
    """Smoothed blaze and the mask of pixels where the raw blaze departs from it.

    The smoothing cannot follow the steep fall of the blaze over the last few columns at a detector edge, a truncated
    aperture or a defect column, and the counts there fall with it, so dividing those pixels by the smoothed blaze makes
    dips at the order ends.  Pixels departing by more than max(BLAZE_DEV_MIN, BLAZE_DEV_NSIG x the order's robust scatter)
    are flagged and the blaze is smoothed again without them (two passes); the threshold follows each order's own noise,
    so the faint blue-end blaze is not masked for its noise.  Returns (smoothed blaze with flagged pixels set to NaN, mask).
    """
    Bs = _smooth_blaze(B)
    bad = np.zeros(B.shape, dtype=bool)
    for _ in range(2):
        with np.errstate(invalid='ignore', divide='ignore'):
            dev = B / Bs - 1.0
        scatter = np.full(B.shape[0], np.nan)
        for o in range(B.shape[0]):
            d = dev[o][np.isfinite(dev[o])]
            if d.size > 50:
                scatter[o] = 1.4826 * np.median(np.abs(d - np.median(d)))
        thr = np.maximum(BLAZE_DEV_MIN, BLAZE_DEV_NSIG * np.nan_to_num(scatter, nan=BLAZE_DEV_MIN))[:, None]
        with np.errstate(invalid='ignore'):
            bad = ~np.isfinite(dev) | (np.abs(dev) > thr) | ~(B > 0)
        Bs = _smooth_blaze(np.where(bad, np.nan, B))
    return np.where(bad, np.nan, Bs), bad


def _edge_fade(B):
    """Weight of every pixel: a cosine fade from EDGE_FADE_MIN to 1 over EDGE_FADE_FRAC of each order's valid pixels at
    both ends (at least 10 pixels), 1 elsewhere.  Counts, blaze and variance of a pixel are scaled together (variance by
    the square), so a single order's counts/blaze is unchanged; only its share of an overlap falls off towards its end,
    which removes the step where an order drops out."""
    tau = np.ones(B.shape)
    for o in range(B.shape[0]):
        ok = np.where(np.isfinite(B[o]) & (B[o] > 0))[0]
        if ok.size < 50:
            continue
        n = max(int(EDGE_FADE_FRAC * ok.size), 10)
        ramp = EDGE_FADE_MIN + (1.0 - EDGE_FADE_MIN) * 0.5 * (1.0 - np.cos(np.pi * np.arange(n) / n))
        tau[o, ok[:n]] = ramp
        tau[o, ok[-n:]] = np.minimum(tau[o, ok[-n:]], ramp[::-1])
    return tau


def _merge_orders(W, F, V, B, E, have_var):
    """Normalised merged spectrum (counts / blaze) on the bins with edges E, combining overlapping orders with the
    blaze as weight (the photon-noise-optimal, data-independent weight), and its variance sum(w^2 v)/sum(w)^2."""
    nb = E.size - 1; num = np.zeros(nb); den = np.zeros(nb); vnum = np.zeros(nb)
    for o in range(W.shape[0]):
        r = _rebin_order(np.asarray(W[o], float), np.asarray(F[o], float), np.asarray(V[o], float), np.asarray(B[o], float), E)
        if r is None:
            continue
        c, v, b, good = r
        wgt = np.where(good, b, 0.0); bb = np.where(good, b, 1.0)
        num += wgt * np.where(good, c / bb, 0.0); den += wgt
        if have_var:
            vnum += wgt ** 2 * np.where(good, v / bb ** 2, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        S = np.where(den > 0, num / den, np.nan)
        VS = np.where(den > 0, vnum / den ** 2, np.nan) if have_var else np.full(nb, np.nan)
    return S, VS


class OrderMerge():
    """
    Merge extracted HRS echelle orders into continuous one-dimensional spectra.

    This class takes order-by-order extracted spectra and their corresponding
    blaze functions from a reduced science frame and master flat, combines
    overlapping orders, and produces merged spectra for both fibres as well as
    a sky-subtracted merged spectrum. It also writes uniformly sampled
    per-order products back into the FITS file for downstream analysis.

    The workflow includes:
        - preparing fibre, wavelength, blaze, and uncertainty arrays
        - masking known bad detector regions
        - interpolating and combining overlapping orders
        - constructing merged spectra on a common wavelength grid
        - generating a sky-subtracted merged spectrum from the two fibres
        - writing merged 1D products and per-order tables to the FITS file
        - creating a lightweight output product file

    Parameters
    ----------
    spec_file : str
        Path to the reduced science FITS file containing extracted orders.
    flat_file : str
        Path to the master flat FITS file containing blaze information.
    arm : str
        Spectrograph arm identifier. Only the first character is used.
    mode : str
        Observing mode, e.g. 'LR', 'MR', 'HR', or 'HS'.
    scaling : bool, optional
        If True, scale orders relative to their continuum before merging.
    plot : bool, optional
        If True, show diagnostic plots during the merge process.

    Attributes
    ----------
    spec : str
        Path to the input science FITS file.
    flat : str
        Path to the input master flat FITS file.
    arm : str
        Short arm identifier used internally.
    mode : str
        Observing mode.
    scaling : bool
        Flag controlling optional continuum scaling before splicing.
    plot : bool
        Flag controlling diagnostic plotting.
    logger : logging.Logger
        Logger used for status and debug messages.
    """
    
    def __init__(self,spec_file, flat_file,arm, mode, scaling=False, plot=False):
    
        self.spec = spec_file
        self.flat = flat_file
        self.arm = arm[0]
        self.mode = mode
        
        self.scaling = scaling
        self.plot = plot
        self.logger = logger
        
        # start a logger
        self.logger.info('Started OrderMerge')
        
    def middle(self,
        f,
        param,
        x=None,
        iterations=40,
        eps=0.001,
        poly=False,
        weight=1,
        lambda2=-1,
        mn=None,
        mx=None,
    ):
        """
        Fit a smooth central trend through a one-dimensional array.

        This method iteratively estimates a smooth curve that follows the middle
        of the input data rather than the upper or lower envelope. It can operate
        either with polynomial fitting or with the optimal filter implemented in
        `opt_filter`, and can optionally include weights and a second-derivative
        regularisation term.

        The routine rescales the input data, repeatedly smooths it, estimates the
        local scatter, and clips the solution toward the central trend until
        convergence or until the maximum number of iterations is reached.

        Parameters
        ----------
        f : array_like
            One-dimensional data array to be smoothed.
        param : float
            Smoothing strength for the filter, or polynomial degree if `poly` is
            True.
        x : array_like, optional
            Coordinate array associated with `f`. If omitted, a normalised
            coordinate grid is used.
        iterations : int, optional
            Maximum number of fitting iterations.
        eps : float, optional
            Convergence threshold for stopping the iterations.
        poly : bool, optional
            If True, use polynomial fitting instead of the optimal filter.
        weight : float or array_like, optional
            Weights applied during smoothing.
        lambda2 : float, optional
            Second-derivative regularisation strength.
        mn : float, optional
            Minimum allowed data value to include.
        mx : float, optional
            Maximum allowed data value to include.

        Returns
        -------
        numpy.ndarray
            Smoothed array tracing the middle of the input data.
        """
        mn = mn if mn is not None else np.min(f)
        mx = mx if mx is not None else np.max(f)

        f = np.asarray(f)

        if x is None:
            xx = np.linspace(-1, 1, num=f.size)
        else:
            xx = np.asarray(x)

        if poly:
            j = (f >= mn) & (f <= mx)
            n = np.count_nonzero(j)
            if n <= round(param):
                return f

            fmin = np.min(f[j]) - 1
            fmax = np.max(f[j]) + 1
            ff = (f[j] - fmin) / (fmax - fmin)
            ff_old = ff
        else:
            fmin = np.min(f) - 1
            fmax = np.max(f) + 1
            ff = (f - fmin) / (fmax - fmin)
            ff_old = ff
            n = len(f)

        for _ in range(iterations):
            if poly:
                param = round(param)
                if param > 0:
                    t = median_filter(np.polyval(np.polyfit(xx, ff, param), xx), 3)
                    tmp = np.polyval(np.polyfit(xx, (t - ff) ** 2, param), xx)
                else:
                    t = np.tile(np.polyfit(xx, ff, param), len(f))
                    tmp = np.tile(np.polyfit(xx, (t - ff) ** 2, param), len(f))
            else:
                t = median_filter(self.opt_filter(ff, param, weight=weight, lambda2=lambda2), 3)
                tmp = self.opt_filter(
                    weight * (t - ff) ** 2, param, weight=weight, lambda2=lambda2
                )

            dev = np.sqrt(np.clip(tmp, 0, None))
            ff = np.clip(t - dev, ff, t + dev)
            dev2 = np.max(weight * np.abs(ff - ff_old))
            ff_old = ff

            # print(dev2)
            if dev2 <= eps:
                break

        if poly:
            xx = np.linspace(-1, 1, len(f))
            if param > 0:
                t = median_filter(np.polyval(np.polyfit(xx, ff, param), xx), 3)
            else:
                t = np.tile(np.polyfit(xx, ff, param), len(f))

        return t * (fmax - fmin) + fmin
        
    @staticmethod
    def opt_filter(y, par, par1=None, weight=None, lambda2=-1, maxiter=100):
        """
        Optimal filtering of 1D and 2D arrays.
        Uses tridiag in 1D case and sprsin and linbcg in 2D case.
        Written by N.Piskunov 8-May-2000

        Parameters
        ----------
        f : array
            1d or 2d array
        xwidth : int
            filter width (for 2d array width in x direction (1st index)
        ywidth : int
            (for 2d array only) filter width in y direction (2nd index) if ywidth is missing for 2d array, it set equal to xwidth
        weight : array(float)
            an array of the same size(s) as f containing values between 0 and 1
        lambda1: float
            regularization parameter
        maxiter : int
            maximum number of iteration for filtering of 2d array
        """

        y = np.asarray(y)

        if y.ndim not in [1, 2]:
            raise ValueError("Input y must have 1 or 2 dimensions")

        if par < 1:
            par = 1

        # 1D case
        if y.ndim == 1 or (y.ndim == 2 and (y.shape[0] == 1 or y.shape[1] == 1)):
            y = y.ravel()
            n = y.size

            if weight is None:
                weight = np.ones(n)
            elif np.isscalar(weight):
                weight = np.full(n, weight)
            else:
                weight = weight[:n]

            if lambda2 > 0:
                # Apply regularization lambda
                aij = np.zeros((5, n))
                # 2nd lower subdiagonal
                aij[0, 2:] = lambda2
                # Lower subdiagonal
                aij[1, 1] = -par - 2 * lambda2
                aij[1, 2:-1] = -par - 4 * lambda2
                aij[1, -1] = -par - 2 * lambda2
                # Main diagonal
                aij[2, 0] = weight[0] + par + lambda2
                aij[2, 1] = weight[1] + 2 * par + 5 * lambda2
                aij[2, 2:-2] = weight[2:-2] + 2 * par + 6 * lambda2
                aij[2, -2] = weight[-2] + 2 * par + 5 * lambda2
                aij[2, -1] = weight[-1] + par + lambda2
                # Upper subdiagonal
                aij[3, 0] = -par - 2 * lambda2
                aij[3, 1:-2] = -par - 4 * lambda2
                aij[3, -2] = -par - 2 * lambda2
                # 2nd lower subdiagonal
                aij[4, 0:-2] = lambda2
                # RHS
                b = weight * y

                f = solve_banded((2, 2), aij, b)
            else:
                a = np.full(n, -abs(par))
                b = np.copy(weight) + abs(par)
                b[1:-1] += abs(par)
                aba = np.array([a, b, a])

                f = solve_banded((1, 1), aba, weight * y)

            return f
        else:
            # 2D case
            if par1 is None:
                par1 = par
            if par == 0 and par1 == 0:
                raise ValueError("xwidth and ywidth can't both be 0")
            n = y.size
            nx, ny = y.shape

            lam_x = abs(par)
            lam_y = abs(par1)

            n = nx * ny
            ndiag = 2 * nx + 1
            aij = np.zeros((n, ndiag))
            aij[nx, 0] = weight[0, 0] + lam_x + lam_y
            aij[nx, 1:nx] = weight[0, 1:nx] + 2 * lam_x + lam_y
            aij[nx, nx : n - nx] = weight[1 : ny - 1] + 2 * (lam_x + lam_y)
            aij[nx, n - nx : n - 1] = weight[ny - 1, 0 : nx - 1] + 2 * lam_x + lam_y
            aij[nx, n - 1] = weight[ny - 1, nx - 1] + lam_x + lam_y

            aij[nx - 1, 1:n] = -lam_x
            aij[nx + 1, 0 : n - 1] = -lam_x

            ind = np.arrange(ny - 1) * nx + nx + nx * n
            aij[ind - 1] = aij[ind - 1] - lam_x
            aij[ind] = aij[ind] - lam_x

            ind = np.arrange(ny - 1) * nx + nx
            aij[nx + 1, ind - 1] = 0
            aij[nx - 1, ind] = 0

            aij[0, nx:n] = -lam_y
            aij[ndiag - 1, 0 : n - nx] = -lam_y

            rhs = f * weight

            model = solve_banded((nx, nx), aij, rhs)
            model = np.reshape(model, (ny, nx))
            return model

        
    @staticmethod
    def bezier_interp(x_old, y_old, x_new):
        """
        Interpolate data onto a new grid using a spline-based Bezier-like scheme.

        This method sanitises the input by removing masked values and discarding
        duplicate x-coordinates before constructing a spline representation. The
        resulting spline is then evaluated on the requested output grid.

        Parameters
        ----------
        x_old : array_like
            Original x-coordinates.
        y_old : array_like
            Original y-values.
        x_new : array_like
            New x-coordinates at which to evaluate the interpolant.

        Returns
        -------
        numpy.ndarray
            Interpolated y-values on the new x-grid.
        """

        # Handle masked arrays
        if np.ma.is_masked(x_old):
            x_old = np.ma.compressed(x_old)
            y_old = np.ma.compressed(y_old)

        # avoid duplicate entries in x
        assert x_old.size == y_old.size
        x_old, index = np.unique(x_old, return_index=True)
        y_old = y_old[index]

        knots, coef, order = scipy.interpolate.splrep(x_old, y_old, s=0)
        y_new = scipy.interpolate.BSpline(knots, coef, order)(x_new)
        return y_new
    
    def splice_orders(self, spec, wave, cont, sigma, var=None, plot=False):
        """
        Combine overlapping spectral orders into a continuous merged spectrum.

        This method takes order-by-order spectra, wavelengths, continua/blaze
        functions, and uncertainties, identifies overlap regions between
        neighbouring orders, and replaces those regions with weighted averages.
        After overlap handling, all orders are resampled onto a single common
        wavelength grid and collapsed into a merged one-dimensional spectrum.

        If requested, the routine first scales each order by its continuum level
        to bring all orders onto a comparable relative scale. Diagnostic plots can
        also be generated before and after merging.
        From pyreduce.

        Parameters
        ----------
        spec : array_like
            Two-dimensional array of spectral fluxes with shape
            `(n_orders, n_pixels)`.
        wave : array_like
            Two-dimensional wavelength solution with the same shape as `spec`.
        cont : array_like
            Two-dimensional continuum or blaze array with the same shape as
            `spec`. If None, unity continuum is assumed.
        sigma : array_like
            Two-dimensional uncertainty array with the same shape as `spec`.
        plot : bool, optional
            If True, show diagnostic plots of the merge process.

        Returns
        -------
        tuple
            Two-element tuple containing:
                - `new_wave` : merged one-dimensional wavelength grid
                - `ssB` : merged one-dimensional spectrum
                - `vvB` : variance of the merged spectrum on the same grid
        """
        
        plot_title = 'Order Merging'
        nord, _ = spec.shape  # Number of sp. orders, Order length in pixels

        if cont is None:
            cont = np.ones_like(spec)

        # Just to be extra safe that they are all the same
        mask = (
            np.ma.getmaskarray(spec)
            | (np.ma.getdata(spec) == 0)
            | (np.ma.getdata(cont) == 0)
        )
        spec = np.ma.masked_array(spec, mask=mask)
        wave = np.ma.masked_array(np.ma.getdata(wave), mask=mask)
        cont = np.ma.masked_array(np.ma.getdata(cont), mask=mask)
        sigm = np.ma.masked_array(np.ma.getdata(sigma), mask=mask)
        # real variance of the flux (propagated by the extraction), carried alongside; the blaze-based sigma above
        # keeps defining the overlap weights, so the merged flux does not change because a variance is present
        varr = np.ma.masked_array(np.ma.getdata(var), mask=mask) if var is not None else None

        if self.scaling:
            # Scale everything to roughly the same size, around spec/blaze = 1
            scale = np.ma.median(spec / cont, axis=1)
            cont *= scale[:, None]

        if self.plot:  # pragma: no cover
            plt.subplot(411)
            if plot_title is not None:
                plt.suptitle(plot_title)
            plt.title("Before")
            for i in range(spec.shape[0]):
                plt.plot(wave[i], spec[i] / cont[i])
            plt.ylim([0, 2])

            plt.subplot(412)
            plt.title("Before Error")
            for i in range(spec.shape[0]):
                plt.plot(wave[i], sigm[i] / cont[i])
            plt.ylim((0, np.ma.median(sigm[i] / cont[i]) * 2))

        # Order with largest signal, everything is scaled relative to this order
        iord0 = np.argmax(np.ma.median(spec / cont, axis=1))

        # Loop from iord0 outwards, first to the top, then to the bottom
        tmp0 = chain(range(iord0, 0, -1), range(iord0, nord - 1))
        tmp1 = chain(range(iord0 - 1, -1, -1), range(iord0 + 1, nord))

        # Looping over order pairs
        for iord0, iord1 in zip(tmp0, tmp1):
            # Get data for current order
            # Note that those are just references to parts of the original data
            # any changes will also affect spec, wave, cont, and sigm
            s0, s1 = spec[iord0], spec[iord1]
            w0, w1 = wave[iord0], wave[iord1]
            c0, c1 = cont[iord0], cont[iord1]
            u0, u1 = sigm[iord0], sigm[iord1]
            if varr is not None:
                v0, v1 = varr[iord0], varr[iord1]

            # Calculate Overlap
            i0 = np.ma.where((w0 >= np.ma.min(w1)) & (w0 <= np.ma.max(w1)))
            i1 = np.ma.where((w1 >= np.ma.min(w0)) & (w1 <= np.ma.max(w0)))

            # Orders overlap
            if i0[0].size > 0 and i1[0].size > 0:
                # Interpolate the overlapping region onto the wavelength grid of the other order
                tmpS0 = self.bezier_interp(w1, s1, w0[i0])
                tmpB0 = self.bezier_interp(w1, c1, w0[i0])
                tmpU0 = self.bezier_interp(w1, u1, w0[i0])

                tmpS1 = self.bezier_interp(w0, s0, w1[i1])
                tmpB1 = self.bezier_interp(w0, c0, w1[i1])
                tmpU1 = self.bezier_interp(w0, u0, w1[i1])
                if varr is not None:
                    tmpV0 = self.bezier_interp(w1, v1, w0[i0])
                    tmpV1 = self.bezier_interp(w0, v0, w1[i1])

                # Combine the two orders weighted by the relative error
                wgt0 = np.ma.vstack([c0[i0].data / u0[i0].data, tmpB0 / tmpU0]) ** 2
                wgt1 = np.ma.vstack([c1[i1].data / u1[i1].data, tmpB1 / tmpU1]) ** 2

                s0[i0], utmp = np.ma.average(
                    np.ma.vstack([s0[i0], tmpS0]), axis=0, weights=wgt0, returned=True
                )
                c0[i0] = np.ma.average([c0[i0], tmpB0], axis=0, weights=wgt0)
                u0[i0] = c0[i0] * utmp ** -0.5
                if varr is not None:                               # variance of the weighted mean: sum(w^2 v) / sum(w)^2
                    v0[i0] = (wgt0[0] ** 2 * v0[i0] + wgt0[1] ** 2 * tmpV0) / np.ma.sum(wgt0, axis=0) ** 2

                s1[i1], utmp = np.ma.average(
                    np.ma.vstack([s1[i1], tmpS1]), axis=0, weights=wgt1, returned=True
                )
                c1[i1] = np.ma.average([c1[i1], tmpB1], axis=0, weights=wgt1)
                u1[i1] = c1[i1] * utmp ** -0.5
                if varr is not None:
                    v1[i1] = (wgt1[0] ** 2 * v1[i1] + wgt1[1] ** 2 * tmpV1) / np.ma.sum(wgt1, axis=0) ** 2
            else:  # pragma: no cover
                # TODO: Orders dont overlap
                continue

        if self.plot:  # pragma: no cover
            plt.subplot(413)
            plt.title("After")
            for i in range(nord):
                plt.plot(wave[i], spec[i] / cont[i], label="order=%i" % i)

            plt.subplot(414)
            plt.title("Error")
            for i in range(nord):
                plt.plot(wave[i], sigm[i] / cont[i], label="order=%i" % i)
            plt.ylim((0, np.ma.median(sigm[i] / cont[i]) * 2))
            plt.show()
            
        smooth_initial = 1e5
            
        # Create new equispaced wavelength grid
        tmp = np.asarray(wave.compressed(), dtype=np.float64)

        wmin = float(np.min(tmp))
        wmax = float(np.max(tmp))

        dwave = abs(tmp[tmp.size // 2] - tmp[tmp.size // 2 - 1]) * 0.5
        nwave = int(np.ceil((wmax - wmin) / dwave) + 1)

        new_wave = np.linspace(
            wmin,
            wmax,
            nwave,
            endpoint=True,
            dtype=np.float64,
        )

        # Combine all orders into one big spectrum, sorted by wavelength
        wsort, j, index = np.unique(tmp, return_index=True, return_inverse=True)
        sB = (spec / cont).compressed()[j]

        # Get initial weights for each point
        weight = self.middle(sB, 0.5, x=wsort - wmin)
        weight = weight / self.middle(weight, 3 * smooth_initial) + np.concatenate(
            ([0], 2 * weight[1:-1] - weight[0:-2] - weight[2:], [0])
        )
        weight = np.clip(weight, 0, None)
        # TODO for some reason the interpolation messes up, use linear instead for now
        # weight = self.safe_interpolation(wsort, weight, new_wave)
        weight = np.interp(new_wave, wsort, weight)
        weight /= np.max(weight)

        # Interpolate Spectrum onto the new grid
        # ssB = self.safe_interpolation(wsort, sB, new_wave)
        ssB = np.interp(new_wave, wsort, sB)
        # variance of the normalised spectrum (sigma / cont)^2, carried through the same ordering and the same
        # linear interpolation as the flux (neighbouring grid points are therefore correlated by the oversampling)
        if varr is not None:
            vB = (varr / np.ma.power(cont, 2)).compressed()[j]
            vvB = np.interp(new_wave, wsort, vB)
        else:
            vvB = np.full_like(new_wave, np.nan)
        # Keep the scale of the continuum
        bbb = self.middle(cont.compressed()[j], 1)
        
        if self.plot:
 
            for ord in range(nord):
                plt.plot(wave[ord],spec[ord])
            plt.show()

        return new_wave, ssB, vvB
        
    @staticmethod
    def add_wavelength_header(hdu, wave, spectrum):
        wave = np.asarray(wave, dtype=np.float64)

        if wave.size < 2:
            raise ValueError("At least two wavelength samples are required")

        wave_step = (wave[-1] - wave[0]) / (wave.size - 1)

        hdu.header["CRPIX1"] = (1.0, "Reference pixel")
        hdu.header["CRVAL1"] = (float(wave[0]), "Wavelength at reference pixel")
        hdu.header["CDELT1"] = (float(wave_step), "Wavelength increment per pixel")
        hdu.header["CTYPE1"] = ("WAVELENGTH", "Coordinate type")
        hdu.header["CUNIT1"] = ("Angstrom", "Wavelength unit")
        hdu.header["BUNIT"] = ("FLUX", "Units of data values")
        if np.isfinite(spectrum).any():                       # an all-NaN array (e.g. an empty variance) has no extrema to record
            hdu.header["DATAMAX"] = (float(np.nanmax(spectrum)), "Maximum data value")
            hdu.header["DATAMIN"] = (float(np.nanmin(spectrum)), "Minimum data value")

    @staticmethod
    def add_order_table_wavelength_header(hdu, wave):
        """Add numeric wavelength metadata to a resampled order table."""
        wave = np.asarray(wave, dtype=np.float64)

        if wave.size < 2:
            raise ValueError("At least two wavelength samples are required")

        wave_step = (wave[-1] - wave[0]) / (wave.size - 1)

        hdu.header["CRPIX1"] = (1.0, "Reference pixel")
        hdu.header["CRVAL1"] = (float(wave[0]), "Wavelength at reference pixel")
        hdu.header["CDELT1"] = (float(wave_step), "Wavelength increment per pixel")
        hdu.header["CTYPE1"] = ("WAVELENGTH", "Coordinate type")
        hdu.header["CUNIT1"] = ("Angstrom", "Wavelength unit")
        hdu.header["BUNIT"] = ("COUNTS", "Flux and blaze units")

    def snr_plot(self, wave, flux, var, star, starting_ord, header, out_file, per_order=True, dpi=300):
        """
        Write the quick-look signal-to-noise plot of the target fibre next to the data product.

        The plot shows the signal-to-noise per native pixel of the target fibre against wavelength,
        S/N = flux / sqrt(variance), with the variance propagated by the extraction (photon, read and
        calibration noise).  It needs that variance: the method raises when none is given, so it cannot
        be used on data reduced before the variance was carried through the extraction.

        By default every echelle order is drawn in its own colour, overplotted where orders overlap, with
        the order number written above the peak of each order.  Alternatively one black curve of the
        per-pixel S/N of the merged spectrum is drawn, the overlapping orders added in quadrature on a
        grid of one native pixel.  Reference lines mark S/N = 10, 50 and 100, the Balmer lines are
        marked, and the object, fibre, exposure, date, mode, arm and the median and 95th-percentile S/N
        are written on the panel.

        Parameters
        ----------
        wave : ndarray
            Wavelengths of the target fibre, one row per order (Angstrom).
        flux : ndarray
            Extracted counts of the target fibre, same shape as ``wave``.
        var : ndarray
            Variance of the extracted counts, same shape as ``wave`` (counts^2).
        star : str
            Fibre holding the target, 'L' or 'U'.
        starting_ord : int
            Echelle order of row 0; the order number decreases by one per row.
        header : astropy.io.fits.Header
            Primary header of the science file (OBJECT, EXPTIME and DATE-OBS are used).
        out_file : str
            Path of the data product; the PNG is written next to it as ``<frame>_SNR.png``.
        per_order : bool, optional
            If True (default), draw every order separately; if False, draw the merged-spectrum S/N.
        dpi : int, optional
            Resolution of the PNG (default 300, i.e. 3900 x 1950 pixels for the 13 x 6.5 in figure).

        Returns
        -------
        str
            Path of the PNG written.
        """
        wave = np.asarray(wave, dtype=float)
        flux = np.asarray(flux, dtype=float)
        if var is None:
            raise ValueError('the S/N plot needs the variance propagated by the extraction and none was given')
        var = np.asarray(var, dtype=float)
        if var.shape != flux.shape or not np.any(np.isfinite(var) & (var > 0)):
            raise ValueError('the S/N plot needs the variance propagated by the extraction and the array given holds none')

        with np.errstate(invalid='ignore', divide='ignore'):
            snr = np.where((var > 0) & np.isfinite(flux), flux / np.sqrt(var), np.nan)

        # the orders with enough valid pixels, each with its echelle order number
        orders = []
        for o in range(wave.shape[0]):
            good = np.isfinite(wave[o]) & np.isfinite(snr[o])
            if good.sum() > 20:
                orders.append((wave[o], snr[o], starting_ord - o))
        if len(orders) == 0:
            raise ValueError('no order of the target fibre has valid S/N values')

        if per_order:
            w_all = np.concatenate([w for w, _, _ in orders])
            s_all = np.concatenate([s for _, s, _ in orders])
        else:
            # per-pixel S/N of the merged spectrum: where orders overlap their S/N add in quadrature
            # (independent measurements of the same wavelength), evaluated on a grid of one native pixel
            step = float(np.nanmedian(np.abs(np.diff(wave, axis=1))))
            w_all = np.arange(np.nanmin(wave), np.nanmax(wave), step)
            total = np.zeros(w_all.size)
            covered = np.zeros(w_all.size, dtype=bool)
            for w, s, _ in orders:
                good = np.isfinite(w) & np.isfinite(s)
                idx = np.argsort(w[good])
                w_o = w[good][idx]
                s_o = s[good][idx]
                inside = (w_all >= w_o[0]) & (w_all <= w_o[-1])
                total[inside] += np.interp(w_all[inside], w_o, s_o) ** 2
                covered |= inside
            s_all = np.where(covered, np.sqrt(total), np.nan)

        # per order: the central wavelength (where its label goes) and the 99.5th percentile of its S/N (its peak
        # height, so that single-pixel spikes from cosmic rays or hot pixels do not set the height of the panel)
        centres = []
        peaks = []
        for w, s, m in orders:
            good = np.isfinite(s)
            centres.append(float(np.nanmedian(w[good])))
            peaks.append(float(np.nanpercentile(s[good], 99.5)))
        finite = np.isfinite(s_all)
        if per_order:
            top = max(peaks)
        else:
            top = float(np.nanpercentile(s_all[finite], 99.5))
        y_top = 1.16 * top
        median_snr = float(np.nanmedian(s_all[finite]))
        p95_snr = float(np.nanpercentile(s_all[finite], 95))
        wmin = float(np.nanmin(w_all))
        wmax = float(np.nanmax(w_all))

        obj = str(header.get('OBJECT', '')).strip()
        date = str(header.get('DATE-OBS', '')).replace('-', '')
        try:
            exptime = float(header.get('EXPTIME', np.nan))
        except (TypeError, ValueError):
            exptime = np.nan
        arm = 'blue' if self.arm == 'H' else 'red'
        fibre = 'lower fibre (L)' if star == 'L' else 'upper fibre (U)'

        fig, ax = plt.subplots(figsize=(13, 6.5))
        if per_order:
            cmap = plt.get_cmap('tab20')
            for i, (w, s, m) in enumerate(orders):
                colour = cmap(i % 20)
                ax.plot(w, s, color=colour, lw=0.5)
                # the order number just above the peak of its order (its 99.5th percentile plus a gap of 5 % of the
                # panel scale, 10 % for every second order so that the numbers of neighbouring narrow orders do not
                # overlap), above the centre of the order, on a white backing so that it stays readable where a spike
                # reaches up to it
                gap = (0.05 if i % 2 == 0 else 0.10) * top
                ax.text(centres[i], peaks[i] + gap, str(m), ha='center', va='center', fontsize=7, color=colour,
                        clip_on=True, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
        else:
            ax.plot(w_all, s_all, color='black', lw=0.5)
        for level, colour in SNR_REFERENCE_LINES:
            if level < y_top:
                ax.axhline(level, color=colour, ls='--', lw=1.0)
        label_height = 1.02 * y_top
        for name, lam in BALMER_LINES.items():
            if wmin < lam < wmax:
                ax.axvline(lam, color='0.3', ls='-.', lw=0.8)
                ax.text(lam, label_height, name, ha='center', va='bottom', fontsize=9, color='0.3')
        ax.set_xlim(wmin, wmax)
        ax.set_ylim(0, y_top)
        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel('Signal-to-noise ratio per pixel' + (' (per order)' if per_order else ' (merged spectrum)'))
        # the frame information as a title line above the panel (the Balmer labels sit between the two), so that
        # the top band of the panel is free for the order numbers
        ax.set_title(f'Object: {obj}    {fibre}    Exposure: {exptime:.1f} s    Date: {date}    Mode: {self.mode}    '
                     f'Arm: {arm}    median S/N {median_snr:.0f}, 95th percentile {p95_snr:.0f}',
                     loc='left', fontsize=10, color='blue', pad=24)
        ax.grid(alpha=0.2)
        fig.tight_layout()

        out_png = os.path.splitext(out_file)[0]
        if out_png.endswith('_product'):
            out_png = out_png[:-len('_product')]
        out_png = out_png + '_SNR.png'
        fig.savefig(out_png, dpi=dpi)
        plt.close(fig)
        self.logger.info('OrderMerge: S/N plot written to %s', out_png)
        return out_png

    def execute(self):
        """
        Merge the extracted orders of both fibres into one-dimensional spectra and write the data product.

        Both fibres are read from the extracted science file (counts, wavelengths and, when present, the variance
        propagated by the extraction) and the blaze from the master flat.  Each order is placed on the output grid by
        flux-conserving rebinning (every native pixel's counts are distributed over the output bins it overlaps), the
        overlapping orders are combined with the blaze as weight, and the variance is carried through the same
        operations.  Two grids are produced from the same engine, both with a constant velocity step per mode
        (``VSTEP_KMS``): a logarithmic grid (constant km/s per bin) and a linear grid (constant Angstrom per bin, the
        velocity step evaluated at the short-wavelength end of the arm, so never coarser than the log grid).

        Written into the science file and copied to ``<frame>_product.fits``:
            MERGED_LOG, MERGED_LIN : binary tables with columns Wave, Flux_L, Var_L, Flux_U, Var_U, Flux_Sub, Var_Sub
                                     (Flux = counts / blaze; Sub = star fibre minus sky fibre, normalised by the star
                                     blaze; Var the variance of each)
            FIBRE_L/U, FIBRE_L/U_VAR, WAVE_L/U, BLAZE_L/U : the native per-order arrays (row r = echelle order ORDER0 - r)
        Optional, for old readers (module flags): the linear spectra as image extensions and per-order ORD tables.
        Order ends: pixels whose raw blaze departs from its smoothed form (detector edges, truncated apertures, defect
        columns) are left out, and each order's weight is faded out over its last few per cent (see _clean_blaze and
        _edge_fade), so that orders join without dips or steps.
        Next to the product, when the extraction carried the variance: ``<frame>_SNR.png``, the quick-look S/N plot
        of the target fibre (see ``snr_plot``).

        Returns
        -------
        tuple
            (wavelengths of the log grid, merged normalised star-fibre spectrum on it)
        """
        with fits.open(self.spec) as hdu:
            FIBRE_L = np.array(hdu['FIBRE_L'].data, dtype=float); WAVE_L = np.array(hdu['WAVE_L'].data, dtype=float)
            FIBRE_U = np.array(hdu['FIBRE_U'].data, dtype=float); WAVE_U = np.array(hdu['WAVE_U'].data, dtype=float)
            VAR_L = np.array(hdu['FIBRE_L_VAR'].data, dtype=float) if 'FIBRE_L_VAR' in hdu else None
            VAR_U = np.array(hdu['FIBRE_U_VAR'].data, dtype=float) if 'FIBRE_U_VAR' in hdu else None
        with fits.open(self.flat) as hdu:
            BLAZE_L = np.array(hdu['BLAZE_L'].data, dtype=float); BLAZE_U = np.array(hdu['BLAZE_U'].data, dtype=float)
        have_var = VAR_L is not None and VAR_U is not None
        if not have_var:
            logger.warning('OrderMerge: no FIBRE_*_VAR in the extracted file; the merged variances are left empty')
            VAR_L = np.zeros(FIBRE_L.shape); VAR_U = np.zeros(FIBRE_U.shape)        # placeholders so that pixels stay valid; the Var columns are written as NaN
        # fibre names stay physical; the target sits in the upper fibre in LR, in the lower fibre otherwise
        star = 'U' if self.mode == 'LR' else 'L'
        # detector regions excluded from the merge (blue arm), as before
        if self.arm == 'H':
            for a in (FIBRE_L, FIBRE_U, BLAZE_L, BLAZE_U, VAR_L, VAR_U):
                a[20:, 482:500] = np.nan; a[:, 851:856] = np.nan; a[37:, 536:542] = np.nan; a[:, 0] = np.nan; a[:, -1] = np.nan
            starting_ord = 125
        else:
            starting_ord = 85
        # smooth blaze per order, NaN columns bridged (as before); the flux keeps its NaNs (they invalidate the bins they touch)
        # smoothed blaze, with the pixels whose raw blaze departs from it (detector edges, truncated apertures, defect
        # columns) left out of the merge in both fibres
        BLAZE_L_s, bad_L = _clean_blaze(BLAZE_L)
        BLAZE_U_s, bad_U = _clean_blaze(BLAZE_U)
        for a in (FIBRE_L, VAR_L):
            a[bad_L] = np.nan
        for a in (FIBRE_U, VAR_U):
            a[bad_U] = np.nan
        self.logger.info('OrderMerge: %d (L) and %d (U) pixels left out where the blaze departs from its smooth form',
                         int(bad_L.sum()), int(bad_U.sum()))
        FS, WS, VS_, BS = (FIBRE_L, WAVE_L, VAR_L, BLAZE_L_s) if star == 'L' else (FIBRE_U, WAVE_U, VAR_U, BLAZE_U_s)
        FK, WK, VK = (FIBRE_U, WAVE_U, VAR_U) if star == 'L' else (FIBRE_L, WAVE_L, VAR_L)
        # star minus sky on the native pixels of the star fibre (the sky fibre interpolated to them), variances added
        F_sub = np.full_like(FS, np.nan); V_sub = np.full_like(FS, np.nan)
        for o in range(FS.shape[0]):
            okk = np.isfinite(WK[o]) & np.isfinite(FK[o])
            if okk.sum() < 100 or not np.isfinite(WS[o]).any():
                continue
            sky = np.interp(WS[o], WK[o][okk], FK[o][okk], left=np.nan, right=np.nan)
            F_sub[o] = FS[o] - sky
            skyv = np.interp(WS[o], WK[o][okk], np.where(np.isfinite(VK[o][okk]), VK[o][okk], np.nan), left=np.nan, right=np.nan)
            V_sub[o] = VS_[o] + skyv                        # zeros when no variance is available (keeps the pixels valid)
        # the two grids
        vstep = VSTEP_KMS.get(self.mode, 1.0)
        wmin = float(np.nanmin([np.nanmin(WAVE_L), np.nanmin(WAVE_U)])); wmax = float(np.nanmax([np.nanmax(WAVE_L), np.nanmax(WAVE_U)]))
        q = 1.0 + vstep / C_KMS; nb = int(np.ceil(np.log(wmax / wmin) / np.log(q)))
        lam_log = wmin * q ** np.arange(nb); E_log = wmin * q ** (np.arange(nb + 1) - 0.5)
        dlam = wmin * vstep / C_KMS; nl = int(np.ceil((wmax - wmin) / dlam)) + 1
        lam_lin = wmin + dlam * np.arange(nl); E_lin = wmin + dlam * (np.arange(nl + 1) - 0.5)
        self.logger.info('OrderMerge: flux-conserving merge, %s mode, %.1f km/s bins: log grid %d bins, linear grid %d bins (%.4f A)', self.mode, vstep, nb, nl, dlam)
        # each order's weight faded out at its ends: counts, blaze and variance scaled together (see _edge_fade)
        tau_L = _edge_fade(BLAZE_L_s); tau_U = _edge_fade(BLAZE_U_s)
        tau_S = tau_L if star == 'L' else tau_U
        merge_in = {'L': (FIBRE_L * tau_L, VAR_L * tau_L ** 2, BLAZE_L_s * tau_L),
                    'U': (FIBRE_U * tau_U, VAR_U * tau_U ** 2, BLAZE_U_s * tau_U),
                    'Sub': (F_sub * tau_S, V_sub * tau_S ** 2, BS * tau_S)}
        tables = {}
        for grid, lam, E in (('MERGED_LOG', lam_log, E_log), ('MERGED_LIN', lam_lin, E_lin)):
            S_L, V_L = _merge_orders(WAVE_L, *merge_in['L'], E, have_var)
            S_U, V_U = _merge_orders(WAVE_U, *merge_in['U'], E, have_var)
            S_s, V_s = _merge_orders(WS, *merge_in['Sub'], E, have_var)
            cols = fits.ColDefs([fits.Column(name='Wave', format='D', unit='Angstrom', array=lam),
                                 fits.Column(name='Flux_L', format='E', unit='counts/blaze', array=S_L.astype(np.float32)),
                                 fits.Column(name='Var_L', format='E', unit='(counts/blaze)^2', array=V_L.astype(np.float32)),
                                 fits.Column(name='Flux_U', format='E', unit='counts/blaze', array=S_U.astype(np.float32)),
                                 fits.Column(name='Var_U', format='E', unit='(counts/blaze)^2', array=V_U.astype(np.float32)),
                                 fits.Column(name='Flux_Sub', format='E', unit='counts/blaze', array=S_s.astype(np.float32)),
                                 fits.Column(name='Var_Sub', format='E', unit='(counts/blaze)^2', array=V_s.astype(np.float32))])
            t = fits.BinTableHDU.from_columns(cols, name=grid); h = t.header
            h['WGRID'] = ('log' if grid == 'MERGED_LOG' else 'linear', 'wavelength grid of the Wave column')
            h['WAVE0'] = (float(lam[0]), 'first wavelength (Angstrom, air)')
            h['WAVEN'] = (float(lam[-1]), 'last wavelength (Angstrom, air)')
            h['VSTEP'] = (vstep, 'velocity step of the bins (km s^-1)')
            if grid == 'MERGED_LOG':
                h['WSTEP'] = (float(np.log(q)), 'ln step: Wave[i] = WAVE0 * exp(i * WSTEP)')
            else:
                h['WSTEP'] = (dlam, 'step (Angstrom): Wave[i] = WAVE0 + i * WSTEP')
            h['STARFIB'] = (star, 'fibre holding the target; Sub = star - sky')
            h['MRGMETH'] = ('flux-conserving rebin', 'orders combined with blaze weights')
            h['VARNOTE'] = ('bins < 1 pixel are correlated', 'Var columns; see VSTEP')
            h['BLZDEVMX'] = (BLAZE_DEV_MIN, 'blaze departure limit, at least this fraction')
            h['BLZDEVNS'] = (BLAZE_DEV_NSIG, 'blaze departure limit, sigma of the order')
            h['EDGEFADE'] = (EDGE_FADE_FRAC, 'order weight faded over this end fraction')
            tables[grid] = (t, lam, S_L, V_L, S_U, V_U, S_s, V_s)
        # ---- write into the science file ----
        with fits.open(self.spec) as HDU:
            drop = [x.name for x in HDU if x.name in ('MERGED_LOG', 'MERGED_LIN', 'WAVE_MERGED', 'FIBRE_L_SPEC_MERGED', 'FIBRE_U_SPEC_MERGED', 'SKY_SUBTRACTED_SPEC_MERGED',
                                                       'FIBRE_L_SPEC_MERGED_VAR', 'FIBRE_U_SPEC_MERGED_VAR', 'SKY_SUBTRACTED_SPEC_MERGED_VAR') or x.name.startswith('ORD')]
            for name in drop:
                HDU.pop(name)
            H0 = HDU[0].header
            H0['MSTRFLAT'] = (str(os.path.basename(self.flat)), "Master Flat File")
            H0['PIPEVERS'] = (str(__version__), "HRSReduce pipeline version")
            H0['VARSRC'] = ('extracted FIBRE_*_VAR' if have_var else 'none', 'variance source of the Var columns')
            H0['STARFIB'] = (star, 'fibre holding the target')
            H0['ORDER0'] = (starting_ord, 'echelle order of row 0 of the per-order arrays')
            H0['ORDSTEP'] = (-1, 'echelle order changes by this per row')
            H0['MRGVSTEP'] = (vstep, 'velocity step of the merged grids (km s^-1)')
            for name in ('FIBRE_L', 'FIBRE_U'):
                if name in HDU:
                    HDU[name].header['ORDER0'] = (starting_ord, 'echelle order of row 0'); HDU[name].header['ORDSTEP'] = (-1, 'echelle order changes by this per row')
            for grid in ('MERGED_LOG', 'MERGED_LIN'):
                HDU.append(tables[grid][0])
            if LEGACY_MERGED_IMAGES:
                _, lam, S_L, V_L, S_U, V_U, S_s, V_s = tables['MERGED_LIN']
                for name, arr in (('FIBRE_L_SPEC_MERGED', S_L), ('FIBRE_L_SPEC_MERGED_VAR', V_L), ('FIBRE_U_SPEC_MERGED', S_U), ('FIBRE_U_SPEC_MERGED_VAR', V_U),
                                  ('SKY_SUBTRACTED_SPEC_MERGED', S_s), ('SKY_SUBTRACTED_SPEC_MERGED_VAR', V_s)):
                    im = fits.ImageHDU(data=arr.astype(np.float32), name=name); self.add_wavelength_header(im, lam, arr); HDU.append(im)
            if WRITE_ORDER_TABLES:
                for fib, F, W, V, B in (('L', FIBRE_L, WAVE_L, VAR_L, BLAZE_L_s), ('U', FIBRE_U, WAVE_U, VAR_U, BLAZE_U_s)):
                    for o in range(F.shape[0]):
                        cols = fits.ColDefs([fits.Column(name='Wave', format='D', unit='Angstrom', array=W[o]), fits.Column(name='Flux', format='E', unit='count', array=F[o]),
                                             fits.Column(name='Var', format='E', unit='count^2', array=V[o]), fits.Column(name='Blaze', format='E', unit='count', array=B[o])])
                        t = fits.BinTableHDU.from_columns(cols, name=f'ORD{starting_ord - o}_{fib}'); t.header['SAMPLING'] = ('native pixels', 'no resampling'); HDU.append(t)
            clean_comments(HDU)      # no slash in header comments (some FITS viewers cut the card there)
            HDU.writeto(self.spec, overwrite=True)
            # ---- the data product: the native arrays (float32, wavelengths in double) and the merged tables ----
            short_HDU = HDU
            for ext in ('VAR', 'RECT', 'STRAIGHT', 'RECT_VAR', 'RECT_MASK'):
                if ext in short_HDU:
                    short_HDU.pop(ext)
            for name in ('FIBRE_L', 'FIBRE_U', 'FIBRE_L_VAR', 'FIBRE_U_VAR', 'BLAZE_L', 'BLAZE_U'):
                if name in short_HDU and short_HDU[name].data is not None:
                    short_HDU[name].data = np.asarray(short_HDU[name].data, dtype=np.float32)
            short_HDU[0].data = [0, 0]
            out_file_name = os.path.splitext(os.path.basename(self.spec))[0][3:] + "_product.fits"
            out_file = os.path.join(os.path.dirname(self.spec), out_file_name)
            short_HDU.writeto(out_file, overwrite=True)
        # quick-look S/N plot of the target fibre next to the product; needs the variance propagated by the
        # extraction, so data reduced before it was carried get no plot (a plotting failure never stops the reduction)
        if have_var:
            try:
                self.snr_plot(WS, FS, VS_, star, starting_ord, H0, out_file)
            except Exception as e:
                self.logger.warning('OrderMerge: S/N plot not written (%s)', e)
        else:
            self.logger.warning('OrderMerge: no S/N plot, the extracted file carries no variance')
        _, lam, S_L, V_L, S_U, V_U, S_s, V_s = tables['MERGED_LOG']
        return lam, (S_L if star == 'L' else S_U)
