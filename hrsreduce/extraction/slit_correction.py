"""
Slit-tilt and curvature correction for rectified spectral orders.


    1. Line selection (``find_peaks`` / ``determine_curvature_all_lines``):
       bright, unsaturated lines above a local S/N threshold.
    2. Tilt per line (``determine_curvature_single_line``): per-row centroids
       and a straight-line (optionally quadratic) fit of x against row,
       iterated with the window of each row re-centred on the predicted line
       position (a window fixed on the peak column biases the tilt low by
       ~15 %).
    3. Smooth model (``fit`` / ``eval``): a robust surface in (x, y) per
       fibre -- the lower ('L') and upper ('U') trace of each order, paired
       by position in the order table (``_fibre_labels``) -- across all orders,
       quadratic by default (``fit_degree = 2``; 1 gives a plane); the per-row mode is kept as
       ``fit_mode = "1D"``.
    4. Validation: after straightening, the residual tilt is re-measured on
       the corrected order and logged; a warning is raised above
       ``validate_tol`` px/px.
    5. Robustness: the frame is not re-read from disk for every order,
       column_range is not mutated in place, exceptions are logged instead
       of swallowed, ``arm_col`` is always defined.
    6. Provenance: ``*_Curvature.npz`` carries a version tag and the order-table
       row centres it was built for; a file from another version or another
       order table is recomputed instead of reused.

New constructor keywords (all optional, defaults reproduce the improved
behaviour): ``fit_mode`` ("2D" | "1D"), ``curv_degree`` (1 | 2),
``saturation`` (counts, in the units of the frame), ``snr_min``,
``validate_tol``.

This module defines the ``SlitCorrection`` class, which measures the geometric
curvature of spectral lines within each order and applies a row-by-row
correction to straighten the slit profile in a rectified 2D spectrum. Orders
are corrected serially in the parent process so that the detector arrays remain
local and are not serialised for multiprocessing tasks.

Overview
--------
The class is responsible for:

    1. Loading a science FITS frame and the selected image extension.
    2. Reading an order-definition table describing the rectified orders.
    3. Detecting arc or spectral peaks within each order.
    4. Measuring line tilt and shear for individual peaks.
    5. Fitting smooth curvature models order by order.
    6. Applying the derived correction serially to every row in each order.
    7. Filling vertically only when the input contains genuinely masked pixels.
    8. Writing the corrected 2D image into a ``STRAIGHT`` FITS extension.

Concept
-------
For each order, the code treats spectral-line curvature as a combination of:

    - ``tilt``: a first-order displacement with row position
    - ``shear``: a second-order displacement with row position

These terms are estimated from the observed shape of bright lines across the
cross-dispersion direction and then used to interpolate each order onto a
straighter geometry. Every row from ``-BottomEdge`` through ``TopEdge`` is
included. Ordinary unmasked FITS data requires only the spectral-direction
interpolation; vertical interpolation is reserved for masked input.

Main class
----------
``SlitCorrection``

Constructor
-----------
``__init__(sci_frame, ext_name, order_file, arm, mode, base_dir, yyyymmdd,
plot=False, super_arc=None)``

Parameters
----------
sci_frame : str
    Path to the science FITS frame to be corrected.
ext_name : str
    Name of the FITS extension containing the rectified 2D image to analyse.
order_file : str
    Path to the CSV file containing the rectified order definitions.
arm : str
    Spectrograph arm identifier, typically ``'H'`` or ``'R'``.
mode : str
    Instrument mode, used when constructing associated calibration filenames.
base_dir : str
    Base reduction directory.
yyyymmdd : str
    Observation date string used in output paths.
plot : bool, optional
    If ``True``, diagnostic plots are generated.
super_arc : str, optional
    Reference arc filename used to define the curvature solution output path.

Primary attributes set during initialisation
--------------------------------------------
sci_frame : str
    Input science FITS filename.
ext_name : str
    Name of the working FITS extension.
order_file : str
    CSV file containing order definitions.
sarm : str
    Short arm label, e.g. ``'H'`` or ``'R'``.
arm_col : str
    Expanded arm label used in directory names, e.g. ``'Blu'`` or ``'Red'``.
super_arc : str or None
    Reference arc file used for curvature-solution naming.
mode : str
    Instrument mode.
base_dir : str
    Base reduction directory.
logger : logging.Logger
    Module logger.
plot : bool
    Flag controlling diagnostic plot generation.
yyyymmdd : str
    Observation date string.
spec_header : astropy.io.fits.Header
    Primary FITS header from the input science frame.
flux : numpy.ndarray
    Input 2D flux image from the selected extension.
straight : numpy.ndarray
    Working copy of the flux image that will receive the slit correction.
order_trace_data : pandas.DataFrame
    Order-definition table read from the CSV file.

Main workflow
-------------
The main processing sequence is:

    - ``calculate()``
        Measures or reloads the curvature solution (tilt and shear) for all
        orders and stores it in an ``.npz`` file for reuse.

    - ``correct_for_curvature(img_order, tilt, shear, xwd)``
        Applies the spectral-direction correction to every row in one order.
        If the input contains masked pixels, it also applies the vertical
        interpolation fallback.

    - ``run_correction(i)``
        Applies the curvature correction to a single order and returns the
        corrected order image plus its insertion index.

    - ``_apply_correction_serial()``
        Corrects and inserts orders one at a time in the parent process.

    - ``correct()``
        Runs the full serial slit-correction workflow and writes the corrected
        2D image to a ``STRAIGHT`` FITS extension.

Supporting methods
------------------
The class also provides helper methods to:

    - build Gaussian and Lorentzian peak models;
    - find spectral peaks in extracted order profiles;
    - construct image indices for order cutouts;
    - determine curvature for one line or all lines in an order;
    - fit smooth order-by-order curvature models; and
    - generate diagnostic comparison plots.

Outputs
-------
The code may create the following outputs:

    - ``*_Curvature.npz``
        Saved tilt and shear arrays for reuse.

    - ``Curvature_fits.png``
        Diagnostic plot showing the fitted curvature tracks, if plotting is
        enabled.

    - ``STRAIGHT`` FITS extension
        The corrected 2D image appended to the science FITS file.

Typical usage
-------------
A typical workflow is:

    slit = SlitCorrection(
        sci_frame,
        ext_name,
        order_file,
        arm,
        mode,
        base_dir,
        yyyymmdd,
        plot=False,
        super_arc=super_arc,
    )
    slit.correct()

Notes
-----
- This class assumes that the order-definition CSV contains at least the
  columns ``Coeff0``, ``BottomEdge``, and ``TopEdge``.
- The curvature model is currently fit in ``1D`` mode within the code.
- Existing curvature or straightened products are reused when present, so the
  class does not necessarily recompute everything on every run.
- Orders are processed serially to avoid repeatedly serialising detector-sized
  arrays for per-order multiprocessing tasks.
- The expected order height is checked against
  ``BottomEdge + TopEdge + 1`` before interpolation.
- Vertical interpolation is skipped for unmasked data and retained as a
  fallback for genuinely masked input.
"""

import numpy as np
import os.path
from astropy.io import fits
import logging
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
from scipy.optimize import least_squares
import glob
logger = logging.getLogger(__name__)

class SlitCorrection():

    CURVATURE_VERSION = 5

    def __init__(self, sci_frame, ext_name,order_file,arm,mode,base_dir,yyyymmdd,plot=False,super_arc=None,
                 fit_mode="2D", curv_degree=1, saturation=None, snr_min=8.0, validate_tol=0.015, fit_degree=2,
                 validate=None):
        
        self.sci_frame = sci_frame
        self.ext_name = ext_name
        self.order_file = order_file
        self.sarm = arm
        self.super_arc = super_arc
        self.arm_col = {'H': 'Blu', 'R': 'Red'}.get(arm, str(arm))
        self.fit_mode = fit_mode          # "2D": plane per fibre across orders; "1D": line per row
        self.curv_degree = curv_degree    # 1: tilt only; 2: tilt + shear (quadratic)
        self.saturation = saturation      # lines with any pixel above this are not used (None: 0.9 x data max)
        self.snr_min = snr_min            # minimum line S/N above the local background
        self.validate_tol = validate_tol  # residual tilt (px/px) above which a warning is logged
        self.fit_degree = fit_degree      # degree of the per-fibre tilt surface in (x, y): 1 = plane, 2 = quadratic
        self.validate = validate          # re-measure the tilt after correction: True/False, or None = only on arc/comb frames
        self.mode = mode
        self.base_dir = base_dir
        self.logger = logger
        self.plot = plot
        self.yyyymmdd = yyyymmdd
        
        # start a logger
        self.logger.info('Started SlitCorrection')
        
        # Open the data files
        with fits.open(self.sci_frame) as hdul:
            self.spec_header = hdul[0].header
            self.flux = np.ascontiguousarray(hdul[self.ext_name].data, dtype=np.float64)
        if self.validate is None:
            self.validate = str(self.spec_header.get('OBSTYPE', '')).strip().lower() == 'arc'
        self.straight = self.flux.copy()
        self._saturation_given = self.saturation is not None
        if self.saturation is None:
            # frames here are overscan/gain corrected: use 90 % of the frame maximum as the
            # saturation level unless given explicitly
            self.saturation = 0.9 * float(np.nanmax(self.flux))

        if self.order_file:
            self.order_trace_data = pd.read_csv(self.order_file, header=0, index_col=0)
            self.fibre_labels = self._fibre_labels()
      
    def gaussian(self,x, A, mu, sig):
        """
        A: height
        mu: offset from central line
        sig: standard deviation
        """
        return A * np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))
        
    def lorentzian(self,x, A, x0, mu):
        """
        A: height
        x0: offset from central line
        mu: width of lorentzian
        """
        return A * mu / ((x - x0) ** 2 + 0.25 * mu ** 2)
        
    def fit_curvature_single_order(self,peaks, tilt, shear):
        try:
            middle = np.median(tilt)
            sigma = np.percentile(tilt, (32, 68))
            sigma = middle - sigma[0], sigma[1] - middle
            mask = (tilt >= middle - 5 * sigma[0]) & (tilt <= middle + 5 * sigma[1])
            peaks, tilt, shear = peaks[mask], tilt[mask], shear[mask]

            coef_tilt = np.zeros(1 + 1)
            res = least_squares(
                lambda coef: np.polyval(coef, peaks) - tilt,
                x0=coef_tilt,
                loss="arctan",
            )
            coef_tilt = res.x

            coef_shear = np.zeros(1 + 1)
            res = least_squares(
                lambda coef: np.polyval(coef, peaks) - shear,
                x0=coef_shear,
                loss="arctan",
            )
            coef_shear = res.x

        except Exception as e:
            self.logger.warning(
                "Could not fit the curvature of this order (%s). Using no curvature instead" % e
            )
            coef_tilt = np.zeros(1 + 1)
            coef_shear = np.zeros(1 + 1)

        return coef_tilt, coef_shear, peaks
        
    def _fibre_labels(self):
        """Label every row of the order table as the lower ('L') or upper
        ('U') trace of its order, the same convention as FIBRE_L / FIBRE_U.

        Rows are sorted by their rectified centre (Coeff0).  The gaps between
        consecutive rows alternate between the fibre gap (the two images of
        one order, 25-35 px, changing by ~1 px from order to order) and the
        order gap (29-120 px, changing quickly), so neither 'nearest rows' nor
        a constant gap identifies the pairs reliably at the dense end.  The
        fibre class is the smoother of the two alternating gap sequences,
        decided locally (running roughness over +-4 gaps), which also
        localises a missing trace: the alternation flips there and the
        unpaired row is reported.  Within a pair the smaller centre is 'L'.
        """
        y = np.asarray(self.order_trace_data['Coeff0'], float)
        n = y.size
        if n < 2:
            return ['L'] * n
        order = np.argsort(y)
        ys = y[order]
        gaps = np.diff(ys)
        m = gaps.size
        # roughness of each gap within its alternation class: |g_k - g_{k-2}| + |g_k - g_{k+2}|,
        # relative to the gap, then smoothed over the neighbouring gaps of the same class
        rough = np.full(m, np.nan)
        for k in range(m):
            nb = [gaps[j] for j in (k - 2, k + 2) if 0 <= j < m]
            rough[k] = np.mean([abs(gaps[k] - v) for v in nb]) / gaps[k] if nb else 0.0
        is_fibre = np.zeros(m, bool)
        half = 4
        for k in range(m):
            lo, hi = max(0, k - 2 * half), min(m, k + 2 * half + 1)
            same = [rough[j] for j in range(lo, hi) if (j - k) % 2 == 0]
            other = [rough[j] for j in range(lo, hi) if (j - k) % 2 == 1]
            is_fibre[k] = (np.nanmedian(same) if same else np.inf) <= (np.nanmedian(other) if other else np.inf)
        labels = [None] * n
        k = 0
        while k < m:
            if is_fibre[k] and labels[order[k]] is None and labels[order[k + 1]] is None:
                labels[order[k]] = 'L'; labels[order[k + 1]] = 'U'; k += 2
            else:
                k += 1
        unpaired = [i for i in range(n) if labels[i] is None]
        for i in unpaired:
            paired = [j for j in range(n) if labels[j] is not None]
            if not paired:
                labels[i] = 'L'; continue
            j = min(paired, key=lambda j: abs(y[j] - y[i]))
            labels[i] = 'U' if y[i] > y[j] else 'L'
        if unpaired:
            self.logger.warning("SlitCorrection -- %d order-table row(s) have no partner fibre (labelled by position): %s"
                                % (len(unpaired), unpaired))
        fg = gaps[is_fibre]
        self.logger.info("SlitCorrection -- order table: %d rows -> %d lower (L) + %d upper (U) traces, fibre gap %.0f-%.0f px"
                         % (n, labels.count('L'), labels.count('U'), fg.min() if fg.size else 0, fg.max() if fg.size else 0))
        return labels

    def _surface_design(self, xn, yn):
        """Design matrix of the per-fibre tilt surface in normalised x, y (fit_degree 1 or 2)."""
        cols = [np.ones_like(xn), xn, yn]
        if self.fit_degree >= 2:
            cols += [xn * xn, xn * yn, yn * yn]
        return np.column_stack(cols)

    def _robust_plane(self, X, y):
        """Least-squares plane with three passes of 3-sigma (MAD) clipping."""
        good = np.ones(y.size, bool)
        coef = np.zeros(X.shape[1]); sd = 0.0
        for _ in range(3):
            coef, *_ = np.linalg.lstsq(X[good], y[good], rcond=None)
            r = y - X @ coef
            sd = 1.4826 * np.median(np.abs(r[good])) + 1e-4
            g2 = np.abs(r) < 3 * sd
            if g2.sum() == good.sum():
                break
            good = g2
        return coef, good, sd

    def fit(self, peaks, tilt, shear, nord, ycen_rows=None):
        """Smooth model of the tilt (and shear) over the detector.

        fit_mode "2D": one robust surface in (x', y') per
        fibre, 'L' (lower trace) and 'U' (upper trace) as labelled by
        ``_fibre_labels`` from the order table, x' and y' being the column
        and the rectified row centre normalised to [-1, 1].  Rows with few
        lines are still covered because the surface is global.
        fit_mode "1D": the original independent linear fit per row.
        Returns coefficient containers understood by ``eval``.
        """
        n_col = self.flux.shape[1]; n_row = self.flux.shape[0]
        if self.fit_mode == "1D" or ycen_rows is None:
            coef_tilt = np.zeros((nord, 1 + 1))
            coef_shear = np.zeros((nord, 1 + 1))
            for j in range(nord):
                coef_tilt[j], coef_shear[j], _ = self.fit_curvature_single_order(
                    peaks[j], tilt[j], shear[j]
                )
            return {"mode": "1D", "tilt": coef_tilt, "shear": coef_shear}
        surfaces = {}
        labels = self.fibre_labels
        for fib in ('L', 'U'):
            xs, ys, ts, ss = [], [], [], []
            for j in range(nord):
                if labels[j] != fib or len(peaks[j]) == 0:
                    continue
                xs.append(np.asarray(peaks[j], float)); ys.append(np.full(len(peaks[j]), float(ycen_rows[j])))
                ts.append(np.asarray(tilt[j], float)); ss.append(np.asarray(shear[j], float))
            if not xs:
                continue
            x = np.concatenate(xs); y = np.concatenate(ys); t = np.concatenate(ts); sh = np.concatenate(ss)
            if x.size < 10:
                # a surface needs a few lines; eval falls back to the other fibre's surface
                self.logger.warning("SlitCorrection -- fibre %s: only %d tilt measurements, no surface fitted" % (fib, x.size))
                continue
            X = self._surface_design(2 * x / (n_col - 1) - 1, 2 * y / (n_row - 1) - 1)
            ct, good, sd = self._robust_plane(X, t)
            cs, _, _ = self._robust_plane(X, sh) if np.any(sh != 0) else (np.zeros(X.shape[1]), None, 0.0)
            surfaces[fib] = (ct, cs)
            self.logger.info("SlitCorrection -- fibre %s (%s trace): tilt surface (degree %d) %s from %d lines (%d kept), scatter %.4f px/px"
                             % (fib, "lower" if fib == 'L' else "upper", self.fit_degree, " ".join("%+.4f" % c for c in ct), x.size, int(good.sum()), sd))
        return {"mode": "2D", "surfaces": surfaces, "labels": list(labels), "n_col": n_col, "n_row": n_row, "ycen_rows": np.asarray(ycen_rows, float)}
        
    def eval(self, peaks, order, coef):
        """Evaluate the tilt and shear model on index arrays (peaks = column, order = row)."""
        tilt = np.zeros(peaks.shape)
        shear = np.zeros(peaks.shape)
        if coef["mode"] == "1D":
            for i in np.unique(order):
                idx = order == i
                tilt[idx] = np.polyval(coef["tilt"][i], peaks[idx])
                shear[idx] = np.polyval(coef["shear"][i], peaks[idx])
            return tilt, shear
        n_col, n_row, ycen_rows = coef["n_col"], coef["n_row"], coef["ycen_rows"]
        if not coef["surfaces"]:
            self.logger.error("SlitCorrection -- no tilt measurements in any order: the correction is a no-op (zero tilt). "
                              "Check the detection threshold (snr_min) and the frame units.")
            return tilt, shear
        for i in np.unique(order):
            idx = order == i
            fib = coef["labels"][int(i)]
            if fib not in coef["surfaces"]:
                fib = 'U' if fib == 'L' else 'L'
            ct, cs = coef["surfaces"][fib]
            xn = 2 * peaks[idx] / (n_col - 1) - 1
            yn = np.full(xn.shape, 2 * ycen_rows[int(i)] / (n_row - 1) - 1)
            X = self._surface_design(xn, yn)
            tilt[idx] = X @ ct
            shear[idx] = X @ cs
        return tilt, shear



    def make_index(self,ymin, ymax, xmin, xmax, zero=0):
        """Create an index (numpy style) that will select part of an image with changing position but fixed height

        The user is responsible for making sure the height is constant, otherwise it will still work, but the subsection will not have the desired format

        Parameters
        ----------
        ymin : array[ncol](int)
            lower y border
        ymax : array[ncol](int)
            upper y border
        xmin : int
            leftmost column
        xmax : int
            rightmost colum
        zero : bool, optional
            if True count y array from 0 instead of xmin (default: False)

        Returns
        -------
        index : tuple(array[height, width], array[height, width])
            numpy index for the selection of a subsection of an image
        """

        # in x: the rows between ymin and ymax
        # in y: the column, but n times to match the x index
        ymin = np.asarray(ymin, dtype=int)
        ymax = np.asarray(ymax, dtype=int)
        xmin = int(xmin)
        xmax = int(xmax)

        if zero:
            zero = xmin

        index_x = np.array(
            [np.arange(ymin[col], ymax[col] + 1) for col in range(xmin - zero, xmax - zero)]
        )
        index_y = np.array(
            [
                np.full(ymax[col] - ymin[col] + 1, col)
                for col in range(xmin - zero, xmax - zero)
            ]
        )
        index = index_x.T, index_y.T + zero

        return index
        
    @staticmethod
    def _window_sizes(fwhm):
        """Centroid windows scaled to the line width (px FWHM in the summed profile):
        pass-1 half width, pass-2/3 half width, and the gap between the window edge and the
        background samples.  HR lines (~3 px) give the original 7 / 4 / 1; LR lines (~12 px)
        need 22 / 16 / 5, otherwise the window is narrower than the line and the background
        is sampled on the line itself."""
        f = 3.0 if fwhm is None or not np.isfinite(fwhm) else float(np.clip(fwhm, 1.5, 30.0))
        return max(7, int(round(1.8 * f))), max(4, int(round(1.3 * f))), max(1, int(round(0.4 * f)))

    @staticmethod
    def _centroid_rows(band, pred, w_half, pad, row_noise, dy):
        """Background-subtracted centroid of each band row in a window of half width
        ``w_half`` around the predicted position ``pred[j]``; the background is the mean of two
        3-px medians taken ``pad`` px outside the window.  Returns an (n, 2) array of (dy, x)."""
        nrow, ncol = band.shape
        cen = []
        for j in range(nrow):
            c0 = int(round(pred[j])); lo, hi = c0 - w_half, c0 + w_half + 1
            if lo - pad - 3 < 0 or hi + pad + 3 > ncol:
                continue
            row = band[j, lo:hi]
            if not np.all(np.isfinite(row)):
                continue
            bg = 0.5 * (np.nanmedian(band[j, lo - pad - 3:lo - pad]) + np.nanmedian(band[j, hi + pad:hi + pad + 3]))
            w = row - bg
            noise = row_noise[j] if (row_noise is not None and j < len(row_noise)) else 1.4826 * np.median(np.abs(row - np.median(row))) + 1e-12
            if not np.isfinite(bg) or w.max() < 4 * noise or w.sum() <= 0:
                continue
            w = np.clip(w, 0, None)
            cen.append((dy[j], np.sum(w * np.arange(lo, hi, dtype=float)) / w.sum()))
        return np.array(cen).reshape(-1, 2)

    def determine_curvature_single_line(self,original, peak, ycen, ycen_int, xwd, row_noise=None, fwhm=None):
        """
        Tilt of a single line from per-row centroids, three passes.

        Pass 1 centroids every row of the order band in a window centred on
        the line's peak column and fits x(dy) with a polynomial of degree
        ``curv_degree``.  Passes 2 and 3 repeat with the window of each row
        centred on the position predicted by the previous fit: a window
        centred on the peak column pulls the outer rows' centroids back
        toward it, which underestimates the tilt by ~15 % on faint lines.
        The window widths scale with ``fwhm`` (line width in the summed
        profile): see ``_window_sizes``.  dy is the row offset from the
        order centre; the linear coefficient is the tilt and the quadratic
        one the shear, in the convention of ``correct_for_curvature``.
        Raises RuntimeError when too few rows are usable or the fit is poor.
        """
        _, ncol = original.shape
        r0, r1 = int(ycen_int[0] - xwd[0]), int(ycen_int[0] + xwd[1])
        band = original[r0:r1]
        nrow = band.shape[0]
        dy = np.arange(-xwd[0], xwd[1]) - float(np.mean(ycen))
        half, half_fine, pad = self._window_sizes(fwhm)
        if self.curv_degree != 1:
            half += 1
        f = 3.0 if fwhm is None or not np.isfinite(fwhm) else float(fwhm)
        max_resid = max(0.6, 0.15 * f)                   # centroid scatter allowed about the fit (px)
        pred = np.full(nrow, float(peak))
        coef = None
        for it in range(3):
            w_half = half if it == 0 else half_fine
            cen = self._centroid_rows(band, pred, w_half, pad, row_noise, dy)
            if len(cen) < max(6, int(0.35 * nrow)):
                raise RuntimeError("too few usable rows")
            coef = np.polyfit(cen[:, 0], cen[:, 1], self.curv_degree)
            resid = cen[:, 1] - np.polyval(coef, cen[:, 0])
            if np.std(resid) > max_resid:
                raise RuntimeError("poor tilt fit")
            pred = np.polyval(coef, dy)
        if self.curv_degree == 1:
            tilt, shear = float(coef[0]), 0.0
        else:
            tilt, shear = float(coef[1]), float(coef[0])
        if abs(tilt) > 0.6:
            raise RuntimeError("implausible tilt")
        return tilt, shear
        
    @staticmethod
    def _line_fwhm(vec, idx, default=3.0):
        """Median FWHM (px) of the detected lines in the background-subtracted summed
        profile ``vec`` (``idx``: peak indices into ``vec``); ``default`` when nothing usable."""
        idx = np.asarray(idx, int); idx = idx[(idx > 0) & (idx < len(vec) - 1)]
        if len(idx) < 3:
            return default
        try:
            w = signal.peak_widths(vec, idx, rel_height=0.5)[0]
        except ValueError:
            return default
        w = w[np.isfinite(w) & (w >= 1.5) & (w <= 30.0)]
        return float(np.median(w)) if len(w) >= 3 else default

    def find_peaks(self,vec, cr,pk_height, noise=None, sat_mask=None):
        """
        Emission lines in the summed order profile.
        ``vec`` is background-subtracted; ``pk_height`` is the minimum height
        in the same units (snr_min x local noise); ``sat_mask`` marks columns
        where any pixel of the band is saturated.  No upper height bound:
        bright lines are the ones that pin the tilt.
        """
        peak_width = 2.
        if self.sarm == "R" and self.mode == "HS":
            window_width = 4.
        else:
            window_width = 15.
        prom = pk_height if noise is None else np.maximum(pk_height, 3.0 * np.nanmedian(noise))   # arrays: per-column threshold
        peaks, _ = signal.find_peaks(
            vec, height=pk_height, width=peak_width, distance=window_width, prominence=prom
        )
        # Remove peaks at the edge
        peaks = peaks[
            (peaks >= window_width + 1)
            & (peaks < len(vec) - window_width - 1)
        ]
        # Remove saturated lines (any pixel of the band above the saturation level within +-3 columns)
        if sat_mask is not None and len(peaks):
            keep = np.array([not sat_mask[max(0, p - 3):p + 4].any() for p in peaks])
            peaks = peaks[keep]
        # Remove the offset, due to vec being a subset of extracted
        peaks += cr[0]
        return vec, peaks

        
    def determine_curvature_all_lines(self,original, extracted,column_range,extraction_width):
        n_col = original.shape[1]
        n_ord =extracted.shape[0]
        # Store data from all orders
        all_peaks = []
        all_tilt = []
        all_shear = []
        plot_vec = []
        
        original = np.ma.filled(original, 0.0) if np.ma.isMaskedArray(original) else original
        original = np.ascontiguousarray(original, dtype=np.float64)

        for j in tqdm(range(n_ord), desc="Order",leave=False):
            cr = [int(column_range[j][0]) + 1, int(column_range[j][1]) - 1]
            xwd = extraction_width[j]
            ycen = np.zeros(n_col)+self.order_trace_data['Coeff0'][j]
            ycen_int = ycen.astype(int)
            ycen -= ycen_int

            # Find peaks: local background (running 20th percentile) and local noise
            vec = np.ascontiguousarray(np.ma.filled(extracted[j,cr[0] : cr[1]].astype('float64'), 0.0), dtype=np.float64)
            vec=vec[1:-1]
            from scipy.ndimage import percentile_filter, median_filter
            base = percentile_filter(vec, 20, size=201, mode="nearest")
            vec_sub = vec - base
            # noise of the summed band, estimated from the data (frames may be normalised
            # or scaled, so no Poisson assumption): robust scatter of the baseline-subtracted
            # profile in a running 201-px window, and a per-row noise for the line fits
            band = np.ascontiguousarray(original[ycen_int[0] - int(xwd[0]):ycen_int[0] + int(xwd[1]), cr[0] + 1:cr[1] - 1], dtype=np.float64)
            dev = np.abs(vec_sub - median_filter(vec_sub, size=201, mode="nearest"))
            noise = 1.4826 * median_filter(dev, size=201, mode="nearest") + 1e-9 * max(np.nanmax(vec), 1.0)
            row_noise = np.array([1.4826 * np.nanmedian(np.abs(band[r] - median_filter(band[r], size=101, mode="nearest"))) for r in range(band.shape[0])]) + 1e-12
            sat_mask = (band >= self.saturation).any(axis=0)
            pk_height = self.snr_min * noise
            vec_sub, peaks = self.find_peaks(vec_sub, cr, pk_height, noise=noise, sat_mask=sat_mask)
            npeaks = len(peaks)
            fwhm = self._line_fwhm(vec_sub, peaks - cr[0])
            
            # Determine curvature for each line seperately
            tilt = np.zeros(npeaks)
            shear = np.zeros(npeaks)
            mask = np.full(npeaks, True)
            for ipeak, peak in tqdm(
                enumerate(peaks), total=len(peaks), desc="Peak", leave=False
            ):
                try:
                    tilt[ipeak], shear[ipeak] = self.determine_curvature_single_line(
                        original, peak, ycen, ycen_int, xwd, row_noise=row_noise, fwhm=fwhm
                    )
                except (RuntimeError, ValueError, np.linalg.LinAlgError, IndexError):  # pragma: no cover
                    mask[ipeak] = False
            msg = "SlitCorrection -- row %d: %d lines detected (FWHM %.1f px), %d tilt measurements, median tilt %+.4f px/px" % (
                j, npeaks, fwhm, int(mask.sum()), np.median(tilt[mask]) if mask.any() else float("nan"))
            (self.logger.info if mask.any() else self.logger.debug)(msg)
            # Store results
            all_peaks += [peaks[mask]]
            all_tilt += [tilt[mask]]
            all_shear += [shear[mask]]
            plot_vec += [vec]
        return all_peaks, all_tilt, all_shear, plot_vec
        
    def plot_comparison(self,original, tilt, shear, peaks,extraction_width,nord,column_range,savefile):  # pragma: no cover
        
        original = np.asarray(original)
            
        _, ncol = original.shape
        output = np.zeros((np.sum(extraction_width) + nord, ncol),dtype=np.float32)
        pos = [0]
        x = np.arange(ncol)
        for i in range(nord):
            ycen = np.zeros([ncol])+self.order_trace_data['Coeff0'][i]
            yb = ycen - extraction_width[i, 0]
            yt = ycen + extraction_width[i, 1]
            xl, xr = column_range[i]
            index = self.make_index(yb, yt, xl, xr)
            yl = pos[i]
            yr = pos[i] + index[0].shape[0]
            output[yl:yr, xl:xr] = original[index]
            pos += [yr]

        vmin, vmax = np.percentile(output[output != 0], (5, 95))
        plt.imshow(output, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")

        for i in range(nord):
            for p in peaks[i]:
                ew = extraction_width[i]
                x = np.zeros(ew[0] + ew[1] + 1)
                y = np.arange(-ew[0], ew[1] + 1)
                for j, yt in enumerate(y):
                    x[j] = p + yt * tilt[i, p] + yt ** 2 * shear[i, p]
                y += pos[i] + ew[0]
                plt.plot(x, y, "r",lw=0.5)

        locs = np.sum(extraction_width, axis=1) + 1
        locs = np.array([0, *np.cumsum(locs)[:-1]])
        locs[:-1] += (np.diff(locs) * 0.5).astype(int)
        locs[-1] += ((output.shape[0] - locs[-1]) * 0.5).astype(int)

        plt.yticks(locs, range(len(locs)))

        plt.xlabel("x [pixel]")
        plt.ylabel("order")
        plt.tight_layout()
        plt.savefig(savefile,dpi=600)
        plt.close()
        
    def correct_for_curvature(self, img_order, tilt, shear, xwd):
        """Straighten one order and fill vertically only when it is masked."""

        valid = ~np.ma.getmaskarray(img_order)
        spectral_pixels = np.arange(img_order.shape[1])
        bottom = int(xwd[0])
        top = int(xwd[1])
        row_offsets = range(-bottom, top + 1)

        expected_rows = bottom + top + 1
        if img_order.shape[0] != expected_rows:
            raise ValueError(
                "Order image height does not match BottomEdge + TopEdge + 1: "
                f"{img_order.shape[0]} != {expected_rows}"
            )

        for row, offset in enumerate(row_offsets):
            row_valid = valid[row]
            if not np.any(row_valid):
                img_order[row] = 0
                continue

            source_pixels = (
                spectral_pixels
                + offset * tilt
                + offset**2 * shear
            )
            img_order[row] = np.interp(
                source_pixels,
                spectral_pixels[row_valid],
                img_order[row][row_valid],
                left=0,
                right=0,
            )

        if valid.all():
            return img_order

        spatial_pixels = np.arange(img_order.shape[0])
        for column in range(img_order.shape[1]):
            column_valid = valid[:, column]
            if not np.any(column_valid):
                img_order[:, column] = 0
                continue

            img_order[:, column] = np.interp(
                spatial_pixels,
                spatial_pixels[column_valid],
                img_order[:, column][column_valid],
                left=0,
                right=0,
            )

        return img_order

    def _curvature_source(self):
        """The frame the tilt is measured on.  The stored solution is named after the
        super arc, so it is measured on the super arc's own rectified image whenever that file
        exists -- never on the frame being corrected, which may be a flat or a science frame
        with no emission lines."""
        sa = self.super_arc
        if sa and os.path.isfile(sa) and os.path.abspath(sa) != os.path.abspath(self.sci_frame):
            with fits.open(sa, memmap=True) as h:
                ext = self.ext_name if self.ext_name in h else ('RECT' if 'RECT' in h else None)
                if ext is not None and h[ext].data is not None and h[ext].data.shape == self.flux.shape:
                    data = np.ascontiguousarray(h[ext].data, dtype=np.float64)
                    self.logger.info("SlitCorrection -- measuring the curvature on the super arc %s [%s]" % (sa, ext))
                    return data, os.path.basename(sa)
                self.logger.warning("SlitCorrection -- super arc %s has no usable %s extension; measuring on %s instead"
                                    % (sa, self.ext_name, os.path.basename(self.sci_frame)))
        return self.flux, os.path.basename(self.sci_frame)

    def calculate(self):
    
        #Initially test to see if the super arc already has a correction file
        curve_file = os.path.dirname(self.super_arc)+"/"+os.path.splitext(os.path.basename(self.super_arc))[0]+"_Curvature.npz"
        tiltfile = glob.glob(curve_file)

        ycen_now = np.asarray(self.order_trace_data['Coeff0'], float)
        if len(tiltfile):
            try:
                prev = np.load(tiltfile[0], allow_pickle=True)
                ok = ("version" in prev.files and int(prev["version"]) == self.CURVATURE_VERSION
                      and prev["tilt"].shape == (ycen_now.size, self.flux.shape[1])
                      and np.allclose(prev["ycen_rows"], ycen_now, atol=1.0))
            except Exception as e:
                ok = False
            if not ok:
                self.logger.warning("SlitCorrection -- stored curvature %s was written by another version or for a different order table; recomputing" % tiltfile[0])
                tiltfile = []
        if len(tiltfile) == 0:
            self.logger.info("SlitCorrection -- Calculating Correction")
            src, src_name = self._curvature_source()
            if not self._saturation_given:
                self.saturation = 0.9 * float(np.nanmax(src))
            if str(fits.getheader(self.super_arc if src is not self.flux else self.sci_frame).get('OBSTYPE', '')).strip().lower() != 'arc':
                self.logger.warning("SlitCorrection -- curvature source %s is not an arc frame (OBSTYPE); the measurement may fail" % src_name)
        
            n_ord = len(self.order_trace_data)
            n_col = src.shape[1]
            
            extracted = np.zeros((n_ord,n_col))
            x = np.arange(n_col)
            mask = np.full(src.shape, True)
            
            # Add mask as defined by column ranges
            mask1 = np.full((n_ord, n_col), True)
            for i in range(n_ord):
                mask1[i, 0 : n_col] = False
            extracted = np.ma.array(extracted, mask=mask1)
            
            column_range = np.zeros((n_ord,2),dtype=np.int16)
            extraction_width = np.zeros((n_ord,2),dtype=np.int16)
            
            for i in range(n_ord):
                x_left_lim = 0
                x_right_lim = src.shape[1]
                column_range[i][0] = int(x_left_lim)
                column_range[i][1] = int(x_right_lim)
                
                extraction_width[i][0] =self.order_trace_data['BottomEdge'][i]
                extraction_width[i][1] =self.order_trace_data['TopEdge'][i]
                
                ycen = np.zeros([x_right_lim])+self.order_trace_data['Coeff0'][i]
                upper,lower =self.order_trace_data['TopEdge'][i],self.order_trace_data['BottomEdge'][i]
                yt,yb = ycen + upper, ycen-lower
                
                index = self.make_index(yb, yt, x_left_lim, x_right_lim)
                
                mask[index] = False
                
                extracted_ord = np.sum(src[index],axis=0)
                
                extracted[i,0 : n_col] = extracted_ord

            peaks, tilt, shear, vec = self.determine_curvature_all_lines(
                    src, extracted, column_range,extraction_width)
                    
            ycen_rows = [float(self.order_trace_data['Coeff0'][i]) for i in range(n_ord)]
            coef = self.fit(peaks, tilt, shear, n_ord, ycen_rows=ycen_rows)

            iorder,ipeaks = np.indices(extracted.shape)
            
            tilt, shear = self.eval(ipeaks, iorder, coef)
            if coef.get("mode") == "2D" and not coef.get("surfaces"):

                self.logger.error("SlitCorrection -- no solution measured on %s; nothing saved to %s" % (src_name, curve_file))
                return tilt, shear

            if self.plot:
                savefile = (self.base_dir+self.arm_col+"/"+self.yyyymmdd[0:4]+"/"+self.yyyymmdd[4:8]+"/reduced/Curvature_fits.png")
                self.plot_comparison(src, tilt, shear, peaks,extraction_width,n_ord,column_range,savefile)
                
            np.savez(curve_file, tilt=tilt, shear=shear, version=self.CURVATURE_VERSION, ycen_rows=ycen_now,
                     fibre_labels=np.array(self.fibre_labels), fit_mode=self.fit_mode, fit_degree=self.fit_degree, curv_degree=self.curv_degree)
        else:
            self.logger.info("SlitCorrection -- Using predetermined correction from "+tiltfile[0])
            previous_data = np.load(tiltfile[0],allow_pickle=True)
            tilt = previous_data['tilt']
            shear = previous_data['shear']
            
        return tilt,shear
        
    
    def run_correction(self,i):
        x_right_lim = self.flux.shape[1]
        x_left_lim = 0

        ycen = np.zeros([x_right_lim],dtype=np.int16)+self.order_trace_data['Coeff0'][i]
        yb, yt = ycen - self.order_trace_data['BottomEdge'][i], ycen + self.order_trace_data['TopEdge'][i]
        index = self.make_index(yb, yt, x_left_lim, x_right_lim)
        #Do the actual correction
        img_order = self.correct_for_curvature(
                        self.straight[index],
                        self.tilt[i, x_left_lim:x_right_lim],
                        self.shear[i, x_left_lim:x_right_lim],
                        [self.order_trace_data['BottomEdge'][i],self.order_trace_data['TopEdge'][i]])

        if self.plot:

            f, axarr = plt.subplots(2,figsize=(15,7),sharex=True)
            f.suptitle("Order "+str(i))
            axarr[0].imshow(self.flux[index],origin='lower',aspect='auto')
            axarr[1].imshow(img_order,origin='lower',aspect='auto')
            axarr[0].set_title("Input order")
            axarr[1].set_title("Corrected order")
            plt.tight_layout()
            plt.xlabel("Pixel")
            base=os.path.basename(self.sci_frame)
            save_dir = os.path.dirname(self.sci_frame)+"/"+os.path.splitext(base)[0]+"_tilt_correction/"
            try:
                os.mkdir(save_dir)
            except:
                pass
            plt.savefig(
            '{}/Order_{}.png'.format(save_dir,i),
            dpi=500
        )
            plt.close()

        #self.straight[index] = img_order
        
        return index,img_order
            
    
        
    def residual_tilt(self, img_order, xwd, n_lines=40, min_lines=5):
        """Re-measure the tilt on a straightened order with the same
        width-scaled three-pass per-row centroiding as the measurement (brightest
        lines); returns the median residual tilt in px/px, or NaN when fewer
        than ``min_lines`` lines could be measured."""
        bottom, top = int(xwd[0]), int(xwd[1])
        img_order = np.ascontiguousarray(np.ma.filled(img_order, np.nan) if np.ma.isMaskedArray(img_order) else img_order, dtype=np.float64)
        ncol = img_order.shape[1]
        prof = np.nansum(img_order, axis=0)
        base = np.median(prof)
        pk, _ = signal.find_peaks(prof - base, height=np.percentile(prof - base, 96), distance=12)
        pk = pk[(pk > 10) & (pk < ncol - 10)]
        pk = pk[np.argsort(-(prof - base)[pk])][:n_lines]
        fwhm = self._line_fwhm(prof - base, pk)
        self._last_fwhm = fwhm
        half, half_fine, pad = self._window_sizes(fwhm)
        max_resid = max(0.6, 0.15 * fwhm)
        dy = np.arange(-bottom, top + 1, dtype=float)[:img_order.shape[0]]
        row_noise = np.array([1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r))) + 1e-12 for r in img_order])
        slopes = []
        for p in pk:
            pred = np.full(img_order.shape[0], float(p)); slope = np.nan
            for it in range(3):
                cen = self._centroid_rows(img_order, pred, half if it == 0 else half_fine, pad, row_noise, dy)
                if len(cen) < max(6, int(0.35 * img_order.shape[0])):
                    slope = np.nan; break
                c = np.polyfit(cen[:, 0], cen[:, 1], 1); slope = c[0]; pred = np.polyval(c, dy)
                if np.std(cen[:, 1] - np.polyval(c, cen[:, 0])) > max_resid:
                    slope = np.nan; break
            if np.isfinite(slope):
                slopes.append(slope)
        return float(np.median(slopes)) if len(slopes) >= min_lines else float("nan")

    def _apply_correction_serial(self) -> None:
        """Correct and insert each order without a multiprocessing pool; validate each order."""

        residuals = []; fwhms = []
        if not self.validate:
            self.logger.info("SlitCorrection -- validation skipped (OBSTYPE %r is not an arc frame)"
                             % self.spec_header.get('OBSTYPE', ''))
        for order_index in range(len(self.order_trace_data)):
            index, corrected_order = self.run_correction(order_index)
            self.straight[index] = corrected_order
            if not self.validate:
                continue
            xwd = [self.order_trace_data['BottomEdge'][order_index], self.order_trace_data['TopEdge'][order_index]]
            res = self.residual_tilt(np.asarray(corrected_order, float), xwd)
            residuals.append(res); fwhms.append(getattr(self, "_last_fwhm", 3.0))
            # Tolerance scaled to the line width (HR ~3 px: validate_tol; LR ~12 px: 4x),
            # i.e. a fixed fraction of the line width across the fibre
            tol = self.validate_tol * max(1.0, fwhms[-1] / 3.0)
            if np.isfinite(res) and abs(res) > tol:
                self.logger.warning("SlitCorrection -- row %d: residual tilt after correction %+.4f px/px exceeds %.3f (line FWHM %.1f px)"
                                    % (order_index, res, tol, fwhms[-1]))
        residuals = np.array(residuals)
        ok = np.isfinite(residuals)
        if ok.any():
            self.logger.info("SlitCorrection -- validation: residual tilt after correction, median %+.4f px/px, max |%.4f| over %d rows (line FWHM %.1f px)"
                             % (np.median(residuals[ok]), np.abs(residuals[ok]).max(), int(ok.sum()), float(np.median(fwhms))))

    def correct(self) -> None:
        """Apply slit correction and append the ``STRAIGHT`` FITS extension."""

        with fits.open(self.sci_frame, memmap=True) as hdul:
            corrected = "STRAIGHT" in hdul

        if corrected:
            return

        self.logger.info("SlitCorrection -- Applying Correction")
        self.tilt, self.shear = self.calculate()

        if self.tilt is None or self.shear is None:
            return

        self._apply_correction_serial()

        with fits.open(self.sci_frame) as hdul:
            straight_image = fits.ImageHDU(data=self.straight, name="STRAIGHT")
            hdul.append(straight_image)
            hdul.writeto(self.sci_frame, overwrite=True)


__all__ = ["SlitCorrection"]

