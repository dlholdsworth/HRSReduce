import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from numpy.polynomial.legendre import Legendre
from scipy.interpolate import interp1d
import os
import pandas as pd
import scipy
import warnings
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.linalg import lstsq

from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import least_squares
from scipy.signal import find_peaks

class WaveCalAlg:
    """
    Provide wavelength-calibration utilities for HRS arc, LFC, and related calibrations.

    This class groups together the algorithms used to identify calibration
    lines, fit Gaussian peak positions, build 1D and 2D wavelength solutions,
    reject outliers, evaluate calibration precision, and generate diagnostic
    plots. It supports several calibration workflows, including ThAr arc
    lamps, laser-frequency-comb frames, and iterative global wavelength
    solutions across many orders.

    The class contains both lower-level fitting helpers and higher-level
    orchestration routines that operate on extracted order spectra. It also
    includes tools for loading saved line lists, fitting polynomial surfaces
    in pixel and order space, estimating residual statistics, and exporting
    wavelength images.

    Parameters
    ----------
    cal_type : str
        Calibration type, e.g. "ThAr", "LFC", or "Etalon".
    logger : logging.Logger
        Logger instance used for status and diagnostic messages.
    save_diagnostics : str, optional
        Directory in which to save diagnostic plots and calibration outputs.
    config : object, optional
        Optional configuration context.
    plot : bool, optional
        If True, enable diagnostic plotting where supported.

    Attributes
    ----------
    cal_type : str
        Calibration type being processed.
    save_diagnostics_dir : str or None
        Output directory for diagnostic products.
    red_skip_orders : object
        Optional configuration for red orders to skip.
    green_skip_orders : object
        Optional configuration for green orders to skip.
    chi_2_threshold : float
        Maximum allowed chi-squared threshold for Gaussian line fits.
    skip_orders : object
        Optional set of orders to skip.
    quicklook_steps : int
        Step size used in quick-look calculations.
    min_wave : float
        Minimum wavelength for calibration use.
    max_wave : float
        Maximum wavelength for calibration use.
    fit_order : int
        Polynomial order used in 1D wavelength fitting.
    fit_type : str
        Functional form used for 1D wavelength fitting.
    n_sections : int
        Number of spectrum sections used in peak-finding.
    clip_peaks_toggle : bool
        Flag controlling optional clipping of detected peaks.
    clip_below_median : bool
        Flag controlling optional removal of peaks below the median.
    peak_height_threshold : float
        Threshold used to detect significant peaks.
    sigma_clip : float
        Sigma-clipping threshold used in polynomial fitting.
    fit_iterations : int
        Number of fitting iterations for iterative polynomial solutions.
    logger : logging.Logger
        Logger instance.
    etalon_mask_in : object
        Optional etalon mask input.
    plot : bool
        Flag controlling diagnostic plotting.
    """

    def __init__(
        self, cal_type, logger, save_diagnostics=None, config=None,plot=False):
        """Initializes WaveCalibration class.
        Args:
            clip_peaks_toggle (bool): Whether or not to clip any peaks. True to clip, false to not clip.
            min_order (int): minimum order to fit
            max_order (int): maximum order to fit
            save_diagnostics (str) : Directory in which to save diagnostic plots and information. Defaults to None, which results
                in no saved diagnostics info.
            config (configparser.ConfigParser, optional): Config context.
                Defaults to None.
            logger (logging.Logger, optional): Instance of logging.Logger.
                Defaults to None.
        """
        self.cal_type = cal_type
        self.save_diagnostics_dir = save_diagnostics
        self.red_skip_orders = None #configpull.get_config_value('red_skip_orders')
        self.green_skip_orders = None #configpull.get_config_value('green_skip_orders')
        self.chi_2_threshold = 2.1 #configpull.get_config_value('chi_2_threshold')
        self.skip_orders = None #configpull.get_config_value('skip_orders',None)
        self.quicklook_steps = 1 #configpull.get_config_value('quicklook_steps',10)
        self.min_wave = 3600 #configpull.get_config_value('min_wave',3800)
        self.max_wave = 9500 #configpull.get_config_value('max_wave',9300)
        self.fit_order = 9 #configpull.get_config_value('fit_order',9)
        self.fit_type = 'Legendre' #configpull.get_config_value('fit_type', 'Legendre')
        self.n_sections = 1 #configpull.get_config_value('n_sections',1)
        self.clip_peaks_toggle = False #configpull.get_config_value('clip_peaks',False)
        self.clip_below_median  = True #configpull.get_config_value('clip_below_median',True)
        self.peak_height_threshold = 1.5 #configpull.get_config_value('peak_height_threshold',1.5)
        self.sigma_clip = 2.1 #configpull.get_config_value('sigma_clip',2.1)
        self.fit_iterations = 5 #configpull.get_config_value('fit_iterations',5)
        self.logger = logger
        self.etalon_mask_in = None #configpull.get_config_value('master_etalon_file',None)
        self.plot = plot
        
    

    # --------- 2D polynomial surface fit (pix, order) -> wavelength ----------
    def _poly_design_matrix(self,pix_s, ord_s, deg_pix=6, deg_ord=3, cross=True):
        pix_s = np.asarray(pix_s, float).ravel()
        ord_s = np.asarray(ord_s, float).ravel()
        if pix_s.shape != ord_s.shape:
            raise ValueError("pix and order must have the same shape")

        cols = []
        exps = []
        if cross:
            for i in range(deg_pix + 1):
                for j in range(deg_ord + 1):
                    cols.append((pix_s**i) * (ord_s**j))
                    exps.append((i, j))
        else:
            for i in range(deg_pix + 1):
                cols.append(pix_s**i)
                exps.append((i, 0))
            for j in range(1, deg_ord + 1):
                cols.append(ord_s**j)
                exps.append((0, j))

        A = np.vstack(cols).T
        return A, exps
        
    def _compute_used_mask_for_plotting(self,pix, order, wave, model, n_iter=7, clip_sigma=4.0, ridge=1e-10):
        """
        Recompute the robust weights/mask for diagnostics ONLY.
        This leaves the wavelength-fitting code untouched.
        """
        pix = np.asarray(pix, float).ravel()
        order = np.asarray(order, float).ravel()
        wave = np.asarray(wave, float).ravel()

        good = np.isfinite(pix) & np.isfinite(order) & np.isfinite(wave)
        pix_g, ord_g, wav_g = pix[good], order[good], wave[good]

        deg_pix = model["deg_pix"]
        deg_ord = model["deg_ord"]
        cross = model["cross"]

        pix_mu = model["pix_mu"]
        pix_span = model["pix_span"]
        ord_mu = model["ord_mu"]
        ord_span = model["ord_span"]

        pix_s = (pix_g - pix_mu) / (0.5 * pix_span)
        ord_s = (ord_g - ord_mu) / (0.5 * ord_span)

        A, _ = self._poly_design_matrix(pix_s, ord_s, deg_pix=deg_pix, deg_ord=deg_ord, cross=cross)

        # Start from the model coefficients
        coeff = model["coeff"]

        def solve(A, y, w):
            W = w[:, None]
            ATA = A.T @ (W * A)
            ATy = A.T @ (w * y)
            if ridge and ridge > 0:
                ATA = ATA + ridge * np.eye(ATA.shape[0])
            return np.linalg.solve(ATA, ATy)

        # Iterate weights (and allow coeff to update, matching the original algorithm behaviour)
        w = np.ones_like(wav_g)
        coeff = solve(A, wav_g, w)

        for _ in range(n_iter):
            resid = wav_g - (A @ coeff)
            med = np.median(resid)
            mad = np.median(np.abs(resid - med))
            sigma = 1.4826 * mad if mad > 0 else (np.std(resid) + 1e-12)

            u = resid / (clip_sigma * sigma)
            w = np.zeros_like(wav_g)
            m = np.abs(u) < 1
            w[m] = (1 - u[m]**2)**2

            if np.sum(w > 0) < A.shape[1]:
                break
            coeff = solve(A, wav_g, w)

        used_good = w > 0
        used_full = np.zeros_like(good, dtype=bool)
        used_full[np.where(good)[0]] = used_good
        return used_full
        
    def plot_diagnostics(self,pix, order, wave, model,diag_plot,fibre,
                                      n_iter=7, clip_sigma=4.0,
                                      order_min=84, order_max=125,
                                      n_pix_plot=None):
        """
        Make a diagnostic plot matching the example layout:
          - Left big: wavelength curves per order + points (used filled, rejected open)
          - Top-right: residuals vs pixel with RMS and N used/total
          - Bottom-right: residuals vs aperture index with fit + clipping + iter annotation
        """
        pix = np.asarray(pix, float).ravel()
        order = np.asarray(order, float).ravel()
        wave = np.asarray(wave, float).ravel()

        used_mask = self._compute_used_mask_for_plotting(
            pix, order, wave, model, n_iter=n_iter, clip_sigma=clip_sigma
        )
        good = np.isfinite(pix) & np.isfinite(order) & np.isfinite(wave)
        used = used_mask & good
        rej = (~used_mask) & good

        # Residuals (at measured line points)
        wave_fit = model["predict"](pix[good], order[good])
        resid = wave[good] - wave_fit

        # Used/rejected counts
        n_used = int(np.sum(used))
        n_tot = int(np.sum(good))

        # RMS on used points only
        if np.any(used[good]):
            rms = float(np.sqrt(np.mean((wave[used] - model["predict"](pix[used], order[used]))**2)))
        else:
            rms = np.nan

        # Define "aperture" indices 0..Norders-1 (like the example plot)
        orders_unique = np.unique(order[good]).astype(int)
        orders_unique = np.sort(orders_unique)  # ascending to make index stable
        order_to_ap = {o: i for i, o in enumerate(orders_unique)}
        aperture = np.array([order_to_ap[int(o)] for o in order[good]], dtype=int)

        # Pixel range for curves in the left panel
        if n_pix_plot is None:
            x_min = float(np.nanmin(pix[good]))
            x_max = float(np.nanmax(pix[good]))
            x = np.linspace(x_min, x_max, 400)
        else:
            x = np.linspace(0, n_pix_plot - 1, 400)

        # Orders shown on the left panel (descending like your example)
        orders_desc = np.arange(order_min, order_max + 1, dtype=float)[::-1]

        # ---- Layout: big left + 2 right panels ----
        fig = plt.figure(figsize=(11, 6), dpi=150)
        gs = gridspec.GridSpec(2, 2, width_ratios=[2.25, 1.0], height_ratios=[1, 1], wspace=0.25, hspace=0.25)

        axL = fig.add_subplot(gs[:, 0])
        axTR = fig.add_subplot(gs[0, 1])
        axBR = fig.add_subplot(gs[1, 1])

        # ---- Left: wavelength curves per order + points ----
        for o in orders_desc:
            w_curve = model["predict"](x, np.full_like(x, o))
            axL.plot(x, w_curve, linewidth=0.8, alpha=0.9)

        # Points: used filled, rejected open (use same color mapping as order)
        sc_used = axL.scatter(pix[used], wave[used], c=order[used], s=10, edgecolors="none")
        axL.scatter(pix[rej], wave[rej], facecolors="none", edgecolors="0.3", s=10, linewidths=0.7)

        axL.set_xlabel("Pixel")
        axL.set_ylabel("λ (Å)")
        axL.grid(True, alpha=0.25)
        cbar = fig.colorbar(sc_used, ax=axL, pad=0.01)
        cbar.set_label("Order")

        # ---- Top-right: residuals vs pixel ----
        axTR.scatter(pix[good], resid, c=order[good], s=8, alpha=0.9, edgecolors="none")
        axTR.set_xlabel("Pixel")
        axTR.set_ylabel("Residual on λ (Å)")
        axTR.set_title(f"R.M.S. = {rms:.5f}, N = {n_used}/{n_tot}", fontsize=9)
        axTR.grid(True, axis="y", linestyle="--", alpha=0.35)
        axTR.axhline(0, linewidth=0.8, alpha=0.7)

        # ---- Bottom-right: residuals vs aperture ----
        axBR.scatter(aperture, resid, c=order[good], s=8, alpha=0.9, edgecolors="none")
        axBR.set_xlabel("Aperture")
        axBR.set_ylabel("Residual on λ (Å)")
        axBR.set_title(
            f"Xorder = {model['deg_pix']}, Yorder = {model['deg_ord']}, clipping = ±{clip_sigma:g}, Niter = {n_iter}",
            fontsize=9
        )
        axBR.grid(True, axis="y", linestyle="--", alpha=0.35)
        axBR.axhline(0, linewidth=0.8, alpha=0.7)

        plt.savefig(str(diag_plot+fibre),dpi=600)
        return (rms)
        
    C_KMS = 299792.458


    # ============================================================
    # Peak detection
    # ============================================================

    def detect_well_defined_peaks(self,
        pix,
        flux,
        cut=10,
        prominence=None,
        distance=2,
        min_snr=2.0,
    ):
        pix = np.asarray(pix, dtype=float)
        flux = np.asarray(flux, dtype=float)

        good = np.isfinite(pix) & np.isfinite(flux)
        pix = pix[good]
        flux = flux[good]

        if len(pix) < 5:
            return []

        flux0 = flux - np.median(flux)

        noise = 1.4826 * np.median(np.abs(flux0))
        noise = max(noise, 1e-10)

        if prominence is None:
            prominence = 3.0 * noise

        peaks, _ = find_peaks(flux0, prominence=prominence, distance=distance)

        out = []
        for pk in peaks:
            
            if pk - cut > 0 and pk+cut < len(flux0):
                cut_line = flux0[int(pk)-cut:int(pk)+cut]
                cut_pix = np.arange(len(cut_line))+int(pk)-cut
            
                coef,_=self.fit_gaussian_integral(cut_pix,cut_line,x0=pk)
            
            else:
                coef = None
            if coef is None:
                continue

            #if mu is None:
            #    continue

            amp = flux0[pk]
            snr = amp / noise
            if snr < min_snr:
                continue

            out.append({
                "pixel": coef[1],
                "snr": snr,
                "peak_index": int(pk),
                "amp": amp,
            })

        return out


    # ============================================================
    # Global wavelength fit
    # ============================================================

    class GlobalEchelleWavelengthFit:
        def __init__(self, deg_m=3, deg_x=5):
            self.deg_m = deg_m
            self.deg_x = deg_x

            self.coeff = None
            self.dv_inst = 0.0

            self.x_min = None
            self.x_max = None
            self.m_min = None
            self.m_max = None

            self.available_orders = None
            
            self.C_KMS = 299792.458

        def set_domain(self, x_min, x_max, m_min, m_max, orders):
            self.x_min = float(x_min)
            self.x_max = float(x_max)
            self.m_min = float(m_min)
            self.m_max = float(m_max)
            self.available_orders = np.array(sorted(np.unique(orders)), dtype=int)

        @staticmethod
        def scale(x, xmin, xmax):
            x = np.asarray(x, dtype=float)
            if xmax == xmin:
                return np.zeros_like(x)
            return 2.0 * (x - xmin) / (xmax - xmin) - 1.0

        def design_matrix(self, x, m):
            x = np.asarray(x, dtype=float).ravel()
            m = np.asarray(m, dtype=float).ravel()

            xhat = self.scale(x, self.x_min, self.x_max)
            mhat = self.scale(m, self.m_min, self.m_max)

            Vx = chebvander(xhat, self.deg_x)
            Vm = chebvander(mhat, self.deg_m)

            cols = []
            for i in range(self.deg_m + 1):
                for j in range(self.deg_x + 1):
                    cols.append(Vm[:, i] * Vx[:, j])

            return np.column_stack(cols)

        def model(self, params, x, m):
            dv = params[-1]
            a = params[:-1]

            A = self.design_matrix(x, m)
            poly = A @ a

            return (1.0 / m) * (1.0 + dv / self.C_KMS) * poly

        def fit(self, x, m, wave):
            x = np.asarray(x, dtype=float).ravel()
            m = np.asarray(m, dtype=float).ravel()
            wave = np.asarray(wave, dtype=float).ravel()

            good = np.isfinite(x) & np.isfinite(m) & np.isfinite(wave) & (m != 0)
            x = x[good]
            m = m[good]
            wave = wave[good]

            if len(x) == 0:
                raise ValueError("No valid lines to fit.")

            A = self.design_matrix(x, m)
            y = wave * m

            a0, *_ = np.linalg.lstsq(A, y, rcond=None)
            p0 = np.concatenate([a0, [0.0]])

            def residuals(p):
                return self.model(p, x, m) - wave

            res = least_squares(
                residuals,
                p0,
                loss="soft_l1",
                f_scale=0.01,
                max_nfev=20000,
            )

            self.dv_inst = res.x[-1]
            self.coeff = res.x[:-1]

            resid = self.predict(x, m) - wave
            resid_ms = (resid / wave) * self.C_KMS * 1000.0
            rms_ms = np.sqrt(np.mean(resid_ms**2))

            return {
                "rms_ms": rms_ms,
                "resid_ms": resid_ms,
                "x_used": x,
                "m_used": m,
                "w_used": wave,
            }

        def predict(self, x, m):
            x = np.asarray(x, dtype=float).ravel()
            m = np.asarray(m, dtype=float).ravel()

            A = self.design_matrix(x, m)
            poly = A @ self.coeff

            return (1.0 / m) * (1.0 + self.dv_inst / self.C_KMS) * poly

        def predict_order(self, x, order):
            x = np.asarray(x, dtype=float)
            m = np.full_like(x, order, dtype=float)
            return self.predict(x, m)


    # ============================================================
    # Helpers
    # ============================================================

    def flatten_seed_lines(self,seed_lines):
        x_all, m_all, w_all = [], [], []
        for order in sorted(seed_lines):
            x = np.asarray(seed_lines[order]["pixel"], dtype=float).ravel()
            w = np.asarray(seed_lines[order]["wave"], dtype=float).ravel()

            if len(x) != len(w):
                raise ValueError(f"Order {order}: pixel/wave length mismatch")

            good = np.isfinite(x) & np.isfinite(w)
            x = x[good]
            w = w[good]

            x_all.append(x)
            m_all.append(np.full(len(x), float(order)))
            w_all.append(w)

        if not x_all:
            raise ValueError("No seed lines supplied.")

        return np.concatenate(x_all), np.concatenate(m_all), np.concatenate(w_all)


    def wavelength_margin_from_velocity(self,wave, dv_ms):
        return wave * (dv_ms / 1000.0) / self.C_KMS


    def get_order_predicted_wave_span(self,fitter, order_number, pixel_array, extra_margin_ms=10000.0):
        pixel_array = np.asarray(pixel_array, dtype=float)
        if pixel_array.size == 0:
            return None

        x0 = np.nanmin(pixel_array)
        x1 = np.nanmax(pixel_array)

        lam0 = fitter.predict_order(np.array([x0]), order_number)[0]
        lam1 = fitter.predict_order(np.array([x1]), order_number)[0]

        lam_min = min(lam0, lam1)
        lam_max = max(lam0, lam1)

        lam_mid = 0.5 * (lam_min + lam_max)
        dlam = self.wavelength_margin_from_velocity(lam_mid, extra_margin_ms)

        return lam_min - dlam, lam_max + dlam


    def select_master_lines_in_order_span(self,master_linelist, lam_min, lam_max):
        master_linelist = np.asarray(master_linelist, dtype=float).ravel()
        s = (master_linelist >= lam_min) & (master_linelist <= lam_max)
        return master_linelist[s]


    def robust_match_lines(self,
        peaks,
        order,
        fitter,
        master_lines,
        pixel_array,
        tol_ms,
        span_margin_ms=10000.0,
        ambiguity_ratio=0.7,
        used_waves=None,
    ):
        """
        Match only if the best candidate is clearly better than the second-best
        and not already used.
        """
        if used_waves is None:
            used_waves = np.array([], dtype=float)
        else:
            used_waves = np.asarray(used_waves, dtype=float).ravel()

        peak_pix = np.array([p["pixel"] for p in peaks], dtype=float)
        if len(peak_pix) == 0:
            return []

        span = self.get_order_predicted_wave_span(
            fitter, order, pixel_array, extra_margin_ms=span_margin_ms
        )
        if span is None:
            return []

        lam_min, lam_max = span
        candidate_lines = self.select_master_lines_in_order_span(master_lines, lam_min, lam_max)
        if len(candidate_lines) == 0:
            return []

        lam_pred = fitter.predict_order(peak_pix, order)
        matches = []

        for px, lp in zip(peak_pix, lam_pred):
            dv = (candidate_lines - lp) / candidate_lines * self.C_KMS * 1000.0
            good = np.where(np.abs(dv) < tol_ms)[0]

            if len(good) == 0:
                continue

            abs_dv = np.abs(dv[good])
            sort_idx = np.argsort(abs_dv)

            best_i = good[sort_idx[0]]
            best_abs = abs_dv[sort_idx[0]]

            # Ambiguity rejection: if 2nd-best is nearly as good, skip
            if len(sort_idx) > 1:
                second_abs = abs_dv[sort_idx[1]]
                if best_abs / second_abs > ambiguity_ratio:
                    # Example: 0.92 means best and second-best are too similar
                    continue

            lam_match = candidate_lines[best_i]

            # Prevent repeated use of nearly identical wavelength
            if used_waves.size > 0:
                sep_ms = np.abs((used_waves - lam_match) / lam_match * self.C_KMS * 1000.0)
                if np.any(sep_ms < 20.0):
                    continue

            matches.append((px, order, lam_match, dv[best_i]))

        return matches


    # ============================================================
    # Iterative global solution
    # ============================================================

    def build_global_solution(self,
        order_spectra,
        master_lines,
        seed_lines,
        deg_m=4,
        deg_x=4,
        cut = 10,
        max_iter=10,
        peak_prominence=None,
        peak_distance=2,
        peak_min_snr=2.0,
        initial_match_tol_ms=3000.0,
        min_match_tol_ms=500.0,
        span_margin_ms=10000.0,
        new_line_clip_ms=1500.0,
        global_clip_ms=200.0,
        max_new_lines_per_order=15,
    ):
        x_all, m_all, w_all = self.flatten_seed_lines(seed_lines)

        orders = np.array(sorted(order_spectra.keys()), dtype=int)

        pix_min = min(np.nanmin(np.asarray(order_spectra[o]["pixel"], dtype=float)) for o in orders)
        pix_max = max(np.nanmax(np.asarray(order_spectra[o]["pixel"], dtype=float)) for o in orders)

        fitter = self.GlobalEchelleWavelengthFit(deg_m=deg_m, deg_x=deg_x)
        fitter.set_domain(pix_min, pix_max, orders.min(), orders.max(), orders)

        for it in range(max_iter):
            fitres = fitter.fit(x_all, m_all, w_all)
            rms = fitres["rms_ms"]
            #print(f"Iter {it} RMS {rms:.3f} m/s")

            # Global clipping of currently used lines
            keep = np.abs(fitres["resid_ms"]) < global_clip_ms
            x_all = fitres["x_used"][keep]
            m_all = fitres["m_used"][keep]
            w_all = fitres["w_used"][keep]

            # Refit after clipping
            fitres = fitter.fit(x_all, m_all, w_all)
            rms = fitres["rms_ms"]

            # Tolerance starts loose, then tightens; never let it blow up
            tol_ms = max(min_match_tol_ms, min(initial_match_tol_ms, 3.0 * rms))

            used_waves = w_all.copy()
            new_lines = []

            for order in orders:
                pix = np.asarray(order_spectra[order]["pixel"], dtype=float)
                flux = np.asarray(order_spectra[order]["flux"], dtype=float)

                peaks = self.detect_well_defined_peaks(
                    pix,
                    flux,
                    cut = cut,
                    prominence=peak_prominence,
                    distance=peak_distance,
                    min_snr=peak_min_snr,
                )

                span = self.get_order_predicted_wave_span(
                    fitter, order, pix, extra_margin_ms=span_margin_ms
                )
                if span is None:
                    #print(f"Order {order}: 0 peaks, 0 candidate lines in span [nan, nan], 0 raw matches")
                    continue

                lam_min, lam_max = span
                candidate_lines = self.select_master_lines_in_order_span(master_lines, lam_min, lam_max)

                matches = self.robust_match_lines(
                    peaks=peaks,
                    order=order,
                    fitter=fitter,
                    master_lines=master_lines,
                    pixel_array=pix,
                    tol_ms=tol_ms,
                    span_margin_ms=span_margin_ms,
                    ambiguity_ratio=0.7,
                    used_waves=used_waves,
                )

    #            print(
    #                f"Order {order}: {len(peaks)} peaks, "
    #                f"{len(candidate_lines)} candidate lines in span "
    #                f"[{lam_min:.3f}, {lam_max:.3f}], {len(matches)} raw matches"
    #            )

                if len(matches) == 0:
                    continue

                # Sort by absolute match residual and keep only best few per order
                matches = sorted(matches, key=lambda t: abs(t[3]))
                matches = matches[:max_new_lines_per_order]

                x_exist = x_all[m_all == order]

                for px, o, lam, dv in matches:
                    # avoid duplicate pixel use in same order
                    if np.any(np.abs(x_exist - px) < 0.2):
                        continue

                    # avoid duplicate within this iteration for same order
                    this_order_new = [d[0] for d in new_lines if d[1] == o]
                    if len(this_order_new) > 0 and np.any(np.abs(np.array(this_order_new) - px) < 0.2):
                        continue

                    new_lines.append((px, o, lam))
                    used_waves = np.append(used_waves, lam)

            if len(new_lines) == 0:
                #print("No new lines found")
                break

            # Tentatively add new lines
            x_trial = np.concatenate([x_all, np.array([t[0] for t in new_lines])])
            m_trial = np.concatenate([m_all, np.array([t[1] for t in new_lines])])
            w_trial = np.concatenate([w_all, np.array([t[2] for t in new_lines])])

            # Refit with new lines
            trial_fit = fitter.fit(x_trial, m_trial, w_trial)
            trial_pred = fitter.predict(x_trial, m_trial)
            trial_resid_ms = (trial_pred - w_trial) / w_trial * self.C_KMS * 1000.0

            # Keep old lines always; keep new lines only if they survive a stricter clip
            old_n = len(x_all)
            keep_trial = np.ones(len(x_trial), dtype=bool)
            keep_trial[old_n:] = np.abs(trial_resid_ms[old_n:]) < new_line_clip_ms

            n_kept_new = np.sum(keep_trial[old_n:])

            x_all = x_trial[keep_trial]
            m_all = m_trial[keep_trial]
            w_all = w_trial[keep_trial]

            #print(f"  added {len(new_lines)} tentative lines, kept {n_kept_new}")

            # Stop if nothing survived
            if n_kept_new == 0:
                #print("No new lines survived residual clipping")
                break

        return fitter, x_all, m_all, w_all


    # ============================================================
    # Diagnostic plot
    # ============================================================


    def plot_solution_diagnostics(self,
        fitter,
        x,
        m,
        w,
        diag_plot,
        fibre,
        pixel_range=None,
        figsize=(11, 10),
        title="Global wavelength solution diagnostics",
    ):
        """
        4-panel diagnostic plot:

        1. Pixel vs wavelength, with full solution for every order
        2. Residual vs pixel [m/s], with right axis in Delta-lambda
        3. Residual vs order [m/s], with right axis in Delta-lambda
        4. Residual vs wavelength [m/s], with right axis in Delta-lambda
           and a binned mean residual curve
        """

        x = np.asarray(x, dtype=float).ravel()
        m = np.asarray(m, dtype=float).ravel()
        w = np.asarray(w, dtype=float).ravel()

        good = np.isfinite(x) & np.isfinite(m) & np.isfinite(w)
        x = x[good]
        m = m[good]
        w = w[good]

        if x.size == 0:
            raise ValueError("No valid points to plot.")

        w_fit = fitter.predict(x, m)
        resid_wave = w_fit - w
        resid_ms = (resid_wave / w) * self.C_KMS * 1000.0

        if getattr(fitter, "available_orders", None) is not None:
            orders = np.asarray(fitter.available_orders, dtype=int)
        else:
            orders = np.unique(m.astype(int))

        if pixel_range is None:
            if getattr(fitter, "x_min", None) is not None and getattr(fitter, "x_max", None) is not None:
                pixel_range = (fitter.x_min, fitter.x_max)
            else:
                pixel_range = (np.min(x), np.max(x))

        cmap = plt.get_cmap("turbo", len(orders))

        fig, axes = plt.subplots(
            4,
            1,
            figsize=figsize,
            gridspec_kw={"height_ratios": [2.5, 1.2, 1.2, 1.4]},
        )

        ax0, ax1, ax2, ax3 = axes

        # -----------------------------------------------------
        # Panel 1: full wavelength solution for every order
        # -----------------------------------------------------
        xgrid = np.linspace(pixel_range[0], pixel_range[1], 1200)

        for i, order in enumerate(orders):
            color = cmap(len(orders)+(i*-1))
            s = m.astype(int) == order

            if np.any(s):
                ax0.plot(
                    x[s],
                    w[s],
                    "o",
                    ms=4,
                    color=color,
                    alpha=0.9,
                )

            w_model = fitter.predict_order(xgrid, order)
            ax0.plot(
                xgrid,
                w_model,
                "-",
                lw=1.4,
                color=color,
                alpha=0.95,
            )

        ax0.set_ylabel("Wavelength")
        ax0.set_xlabel("Pixel")
        ax0.set_title(title)
        ax0.grid(alpha=0.25)

        # -----------------------------------------------------
        # Panel 2: residual vs pixel
        # -----------------------------------------------------
        for i, order in enumerate(orders):
            s = m.astype(int) == order
            if np.any(s):
                ax1.plot(
                    x[s],
                    resid_ms[s],
                    "o",
                    ms=4,
                    color=cmap(len(orders)+(i*-1)),
                    alpha=0.9,
                )

        ax1.axhline(0.0, color="k", ls="--", lw=1)
        ax1.set_ylabel("Residual [m/s]")
        ax1.set_xlabel("Pixel")
        ax1.grid(alpha=0.25)

        ax1r = ax1.twinx()
        y1_ms, y2_ms = ax1.get_ylim()
        w_med = np.median(w)
        y1_dl = (y1_ms / 1000.0) / self.C_KMS * w_med
        y2_dl = (y2_ms / 1000.0) / self.C_KMS * w_med
        ax1r.set_ylim(y1_dl, y2_dl)
        ax1r.set_ylabel(r"$\Delta \lambda$")

        # -----------------------------------------------------
        # Panel 3: residual vs order
        # -----------------------------------------------------
        order_rms = []
        order_mean = []
        order_n = []

        for i, order in enumerate(orders):

            s = m.astype(int) == order

            if np.any(s):

                r = resid_ms[s]

                mean = np.mean(r)
                rms = np.sqrt(np.mean((r - mean)**2))

                order_rms.append(rms)
                order_mean.append(mean)
                order_n.append(len(r))

                ax2.plot(
                    np.full(np.sum(s), order),
                    r,
                    "o",
                    ms=4,
                    color=cmap(len(orders)+(i*-1)),
                    alpha=0.9,
                )

            else:
                order_rms.append(np.nan)
                order_mean.append(np.nan)
                order_n.append(0)

        order_rms = np.asarray(order_rms, dtype=float)
        order_mean = np.asarray(order_mean, dtype=float)
        order_n = np.asarray(order_n, dtype=int)

        # Mean residual curve on the residual axis
        ax2.plot(
            orders,
            order_mean,
            "-k",
            lw=2,
            marker="o",
            ms=5,
            label="Order mean residual",
        )

        ax2.axhline(0.0, color="k", ls="--", lw=1)

        ax2.set_ylabel("Residual [m/s]")
        ax2.set_xlabel("Order")
        ax2.invert_xaxis()
        ax2.grid(alpha=0.25)
        ax2.legend(loc="upper left")

        # Annotate each order
        for order, mean, rms, n in zip(orders, order_mean, order_rms, order_n):

            if n == 0:
                label = "N=0"
                ytxt = 0.0
            else:
                label = f"μ={mean:.0f}\nRMS={rms:.0f}\nN={n}"
                ytxt = mean

            ax2.text(
                order,
                ytxt,
                label,
                fontsize=8,
                ha="center",
                va="bottom",
                rotation=90,
            )

    #    # Separate axis for RMS
    #    ax2_rms = ax2.twinx()
    #    ax2_rms.plot(
    #        orders,
    #        order_rms,
    #        color="0.35",
    #        lw=1.5,
    #        marker="s",
    #        ms=4,
    #        label="Order RMS",
    #    )
    #    ax2_rms.set_ylabel("Order RMS [m/s]")
    #    ax2_rms.invert_xaxis()
        
        
        # -----------------------------------------------------
        # Panel 4: residual vs wavelength + binned mean
        # -----------------------------------------------------
        for i, order in enumerate(orders):
            s = m.astype(int) == order
            if np.any(s):
                ax3.plot(
                    w[s],
                    resid_ms[s],
                    "o",
                    ms=4,
                    color=cmap(len(orders)+(i*-1)),
                    alpha=0.85,
                )

        ax3.axhline(0.0, color="k", ls="--", lw=1)
        ax3.set_xlabel("Wavelength")
        ax3.set_ylabel("Residual [m/s]")
        ax3.grid(alpha=0.25)

        ax3r = ax3.twinx()
        y1_ms, y2_ms = ax3.get_ylim()
        y1_dl = (y1_ms / 1000.0) / self.C_KMS * w_med
        y2_dl = (y2_ms / 1000.0) / self.C_KMS * w_med
        ax3r.set_ylim(y1_dl, y2_dl)
        ax3r.set_ylabel(r"$\Delta \lambda$")

        # -----------------------------------------------------
        # Diagnostic box
        # -----------------------------------------------------
        global_rms = np.sqrt(np.mean(resid_ms**2))
        n_lines = len(x)

        orders_with_lines = np.unique(m.astype(int))
        n_orders_with_lines = len(orders_with_lines)

        text = (
            f"N lines = {n_lines}\n"
            f"Global RMS = {global_rms:.1f} m/s\n"
            #f"$\\delta v_{{inst}}$ = {fitter.dv_inst:.4f} km/s\n"
            f"deg_x = {fitter.deg_x}\n"
            f"deg_m = {fitter.deg_m}\n"
            f"orders = {orders.min()}–{orders.max()}\n"
            f"orders with lines = {n_orders_with_lines}"
        )

        ax0.text(
            0.01,
            0.99,
            text,
            transform=ax0.transAxes,
            fontsize=10,
            va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(str(diag_plot+fibre),dpi=600)
        return order_rms, global_rms
        
    def apply_velocity_correction(self,waves, rv_kms,relativistic=True):
        """
        Apply a velocity correction to wavelengths.

        Parameters
        ----------
        waves : array_like
            Wavelengths (any unit; e.g. Angstrom). Output keeps same unit.
        rv_kms : float
            Correction velocity in km/s.
            Convention here:
              - If rv_kms > 0 (observer moving away from target), observed wavelengths are redder.
              - To correct to the frame (remove observer motion), wavelengths should be shifted by
                the *inverse* Doppler factor (i.e., blueshift).
            This function applies:
              - forward (default): lambda_bary = lambda_obs / D
              - inverse=True:      lambda_obs  = lambda_bary * D
            where D is the Doppler factor for +bcv_kms.
        relativistic : bool
            If True (default): use relativistic Doppler factor.
            If False: use classical approximation (1 + v/c).

        Returns
        -------
        waves_out : ndarray
            Shifted wavelengths, same shape as input.
        """
        w = np.asarray(waves, dtype=float)

        beta = float(rv_kms) / self.C_KMS

        if relativistic:
            # Doppler factor for wavelength: lambda_obs = lambda_rest * sqrt((1+beta)/(1-beta))
            D = np.sqrt((1.0 + beta) / (1.0 - beta))
        else:
            # Classical: lambda_obs ~ lambda_rest * (1 + v/c)
            D = 1.0 + beta

        return w / D          # obs -> corr


    def run_wavelength_cal_nonHS(self,all_obs,all_super,all_ref,linelist_path, nord, arm,mode,diag_plot,fibre,obs_date):
                
        line_list = np.load(linelist_path,allow_pickle=True).item()
        
        pixels = np.arange(all_obs.shape[1])
        
        def gaussian(x, amp, cen, wid, offset):
            """1-d gaussian: gaussian(x, amp, cen, wid)"""
            #return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))
            #return (1./(wid*np.sqrt(2*np.pi))) * np.exp(-(x-cen)**2 / (2*wid**2))
            return amp*np.exp(-(x-cen)**2/(2*wid**2)) + offset
        
        wls = []
        new_line_list = {}
        
        #Do a 2D CCF with the extracted arc and the reference arc (which the line list is based on). Then we get the rough pixel
        #offset between the two to align the reference list pixel locations to the observed arc.
        dx2 = []
        shift_ord = []
        for ord in range(10,nord-6,2):
            dx1, dy1 = self.compute_offset_fft_subpixel(all_ref[ord:ord+1,10:-10], all_obs[ord:ord+1,10:-10])
            dx2.append(dx1)
            shift_ord.append(ord)
        dx2 = np.array(dx2)
        shift_ord = np.array(shift_ord)
        dx_std = np.std(dx2)
        if dx_std == 0.0 and np.mean(dx2) == 0.0:
            coef = np.polyfit(shift_ord,dx2,1)
            dx = np.polyval(coef,np.arange(nord))
        else:
            med = np.median(dx2)
            mad = np.median(np.abs(dx2 - med)) + 1e-3
            ii = np.where(np.abs(0.6745*(dx2 - med)/mad) < 3.5)[0]
            dx2 = dx2[ii]
            shift_ord = shift_ord[ii]
            coef = np.polyfit(shift_ord,dx2,1)
            dx = np.polyval(coef,np.arange(nord))
        

        for ord in range(nord):
            new_line_list[ord] = {}
            new_pix = []
            new_wav = []
            #Remove nan values and normalsie the two spectra
            obs = all_obs
            super = all_super
            ref = all_ref
            
            #Trim the data a bit to remove the edge effects.
            if arm =='H' and ord ==41:
                obs = obs[ord][:1650]-np.nanmedian(obs[ord][4:1650])
                obs /= np.nanmax(obs)
                super = super[ord][:1650]-np.nanmedian(super[ord][4:1650])
                super /= np.nanmax(super)
                ref = ref[ord][:1650]-np.nanmedian(ref[ord][4:1650])
                ref /= np.nanmax(ref)
            elif arm =='R' and ord == 32:
                obs = obs[ord][:2000]-np.nanmedian(obs[ord][4:2000])
                obs /= np.nanmax(obs)
                super = super[ord][:2000]-np.nanmedian(super[ord][4:2000])
                super /= np.nanmax(super)
                ref = ref[ord][:2000]-np.nanmedian(ref[ord][4:2000])
                ref /= np.nanmax(ref)
            else:
                obs = obs[ord][:-2]-np.nanmedian(obs[ord][4:-2])
                obs /= np.nanmax(obs)
                super = super[ord][:-2]-np.nanmedian(super[ord][4:-2])
                super /= np.nanmax(super)
                ref = ref[ord][:-2]-np.nanmedian(ref[ord][4:-2])
                ref /= np.nanmax(ref)
            if arm == 'H':
                min_ord = 84
                max_ord = 125
                n_pix = 2048
                cut = 10

            if arm == 'R':
                cut = 10
                n_pix=4096
                min_ord = 53
                max_ord = 85
                cut = 20

            if mode =='MR':
                HRS_lines = np.loadtxt("./hrsreduce/wave_cal/HRS_MR_linelist_2026.txt", usecols=0, unpack=True)
            if mode == 'HR':
                HRS_lines = np.loadtxt("./hrsreduce/wave_cal/HRS_HR_linelist_2026.txt", usecols=0, unpack=True)
            if mode =='LR':
                HRS_lines = np.loadtxt("./hrsreduce/wave_cal/HRS_LR_linelist_2026.txt", usecols=0, unpack=True)
            

            
            obs[np.isnan(obs)] = 0
            super[np.isnan(super)] = 0
            
            pix=np.arange(n_pix)
            
            chi_plus=0
            while len(new_pix) <1:
                line_count = 0
                #Update the line positions by fitting gaussians.
                for old_pix in line_list[ord]['line_positions']:
                    old_pix -=dx[ord]
                    if np.logical_and(old_pix-cut > 0, old_pix+cut < len(obs)):
                    
                        cut_line = obs[int(old_pix)-cut:int(old_pix)+cut]
                        cut_pix = np.arange(len(cut_line))+int(old_pix)-cut
                        coef,gauss_out=self.fit_gaussian_integral(cut_pix,cut_line,x0=old_pix,chi_plus=chi_plus)
                        
                        if coef is not None:
                            new_pix.append(coef[1])
                            new_wav.append(line_list[ord]['known_wavelengths_air'][line_count])
                            
                    line_count += 1
                chi_plus += 0.1
 
                new_line_list[ord]['line_positions'] = new_pix
                new_line_list[ord]['known_wavelengths_air'] = new_wav

            new_pix = []
            new_wav = []
        
        #Save a temporary file
        np.save("./line_list_tmp"+str(obs_date)+".npy",new_line_list)

        seed_lines_rel = np.load("./line_list_tmp"+str(obs_date)+".npy",allow_pickle=True).item()
        master_lines = HRS_lines

        order_spectra = {}
        seed_lines = {}
        count = 0

        for ord in range(len(all_obs)):
            abs_ord = max_ord + (ord*-1)
            order_spectra[abs_ord] = {}
            pixels = np.arange(len(all_obs[ord]))
            order_spectra[abs_ord]['pixel'] = pixels
            order_spectra[abs_ord]['flux'] = np.nan_to_num(all_obs[ord])
            seed_lines[abs_ord] = {}
            seed_lines[abs_ord]['pixel'] = seed_lines_rel[ord]['line_positions']
            seed_lines[abs_ord]['wave'] = seed_lines_rel[ord]['known_wavelengths_air']

        fitter, x_lines, m_lines, w_lines = self.build_global_solution(
            order_spectra=order_spectra,
            master_lines=master_lines,
            seed_lines=seed_lines,
            deg_m=5,
            deg_x=5,
            cut = cut,
            max_iter=10,
            peak_distance=2,
            peak_min_snr=2.0,
            initial_match_tol_ms=3000.0,
            min_match_tol_ms=500.0,
            span_margin_ms=10000.0,
            new_line_clip_ms=500.0,
            global_clip_ms=500.0,
            max_new_lines_per_order=50,
        )

        rms_vals, overall_rms = self.plot_solution_diagnostics(
            fitter,
            x_lines,
            m_lines,
            w_lines,
            diag_plot,
            fibre,
            title="Global wavelength solution diagnostics",
        )
        
        wave_img = []
        
        inst_offset = fitter.dv_inst*-1.
        
        for ord in range(len(all_obs)):

            abs_ord = max_ord + (ord*-1)
            xgrid = np.arange(all_obs.shape[1])
            w_model = fitter.predict_order(xgrid, abs_ord)
            #correcto for offset
            #wave_cor = self.apply_velocity_correction(w_model,inst_offset)
            
            wave_img.append(w_model)
        wave_img = np.array(wave_img)

        os.remove("./line_list_tmp"+str(obs_date)+".npy")

        return wave_img,rms_vals,overall_rms
            
    

    def run_wavelength_cal(
        self, calflux, rough_wls=None, our_wavelength_solution_for_order=None,
        peak_wavelengths_ang=None, lfc_allowed_wls=None,input_filename=None,fibre=None,plot=False):
        """ Runs all wavelength calibration algorithm steps in order.
        Args:
            calflux (np.array): (N_orders x N_pixels) array of L1 flux data of a
                calibration source
            rough_wls (np.array): (N_orders x N_pixels) array of wavelength
                values describing a "rough" wavelength solution. Always None for
                lamps. For LFC, this is generally a lamp-derived solution.
                For Etalon, this is generally an LFC-derived solution. Default None.
            peak_wavelengths_ang (dict of dicts): dictionary of order number-dict
                pairs. Each order number corresponds to a dict containing
                an array of expected line positions (pixel values) under the key
                "line_positions" and an array of corresponding wavelengths in
                Angstroms under the key "known_wavelengths_vac". This value must be
                set for lamps. Can be set or not set for LFC and Etalon. If set to None,
                then peak finding is not run. Defaults to None. Ex:
                    {51: {
                            "line_positions" : array([500.2, ... 8000.3]),
                            "known_wavelengths_vac" : array([3633.1, ... 3570.1])
                        }
                    }
        
            lfc_allowed_wls (np.array): array of all allowed wavelengths for the
                LFC, computed using the order_flux equation. Should be None unless we
                are calibrating an LFC frame. Defaults to None.
        Examples:
            1: Calibrating an LFC frame using a rough ThAr solution,
               with no previous LFC frames to inform this one:
                rough_wls -> ThAr-derived wavelength solution
                lfc_allowed_wls -> wavelengths computed from comb eq
            2: Calibrating an LFC frame using a rough ThAr solution,
               given information about expected mode position:
                rough_wls -> ThAr-derived wavelength solution
                lfc_allowed_wls -> wavelengths computed from comb eq
                peak_wavelengths_ang -> LFC mode wavelengths and their
                    expected pixel locations
            3: Calibrating a lamp frame:
                peak_wavelengths_ang -> lamp line wavelengths in vacuum and their
                    expected rough pixel locations
            4: Calibrating an Etalon frame using an LFC-derived solution, with
               no previous Etalon frames to inform this one:
                rough_wls -> LFC-derived wavelength solution
            5: Calibrating an Etalon frame using an LFC-derived solution and
               at least one other Etalon frame to inform this one:
                rough_wls -> LFC-derived wavelength solution
                peak_wavelengths_ang -> Etalon peak wavelengths and their
                    expected pixel locations
        Returns:
            tuple of:
                np.array: Calculated polynomial solution
                np.array: (N_orders x N_pixels) array of the computed wavelength
                    for each pixel.
                dictionary: information about the fits for each line and order (orderlet_dict)
        """
        self.filename=input_filename
        self.fibre = fibre
        # create directories for diagnostic plots
#        if type(self.save_diagnostics_dir) == str:
#            if not os.path.isdir(self.save_diagnostics_dir):
#                os.makedirs(self.save_diagnostics_dir)
#            if not os.path.isdir(self.save_diagnostics_dir + '/order_diagnostics'):
#                os.makedirs(self.save_diagnostics_dir + '/order_diagnostics')


        n_orders = calflux.shape[0]
        order_list = np.arange(n_orders)

        # masked_calflux = self.mask_array_neid(calflux, n_orders)
        masked_calflux = calflux # TODO: fix

        # perform wavelength calibration
        poly_soln, wls_and_pixels, orderlet_dict, absolute_precision, order_precisions = self.fit_many_orders(
            masked_calflux, order_list, rough_wls=rough_wls,
            comb_lines_angstrom=lfc_allowed_wls,
            expected_peak_locs=peak_wavelengths_ang, peak_wavelengths_ang=peak_wavelengths_ang,
            our_wavelength_solution_for_order=our_wavelength_solution_for_order, print_update=True, plt_path=self.save_diagnostics_dir
        )

        # make a plot of all of the precise new wls minus the rough input  wls
        if self.save_diagnostics_dir is not None and rough_wls is not None:
            if plot:
                # don't do this for etalon exposures, where we're either not
                # deriving a new wls or using drift to do so
                if self.cal_type != 'Etalon':
                    fig, ax = plt.subplots(2,1, figsize=(12,5))
                    for i in order_list:
                        wls_i = poly_soln[i, :]
                        rough_wls_i = rough_wls[i,:]
                        ax[0].plot(wls_i - rough_wls_i, color='grey', alpha=0.5)

                        pixel_sizes = rough_wls_i[1:] - rough_wls_i[:-1]
                        ax[1].plot(
                            (wls_i[:-1] - rough_wls_i[:-1]) / pixel_sizes,
                            color='grey', alpha=0.5
                        )

                    ax[0].set_title('Derived WLS - Approx WLS')
                    ax[0].set_xlabel('Pixel')
                    ax[0].set_ylabel('[$\\rm \AA$]')
                    ax[1].set_xlabel('Pixel')
                    ax[1].set_ylabel('[Pixel]')
                    plt.tight_layout()
                    plt.savefig(
                        '{}/all_wls_{}.png'.format(self.save_diagnostics_dir,self.fibre),
                        dpi=500
                    )
                    plt.close()


        return poly_soln, wls_and_pixels, orderlet_dict, absolute_precision, order_precisions

    def fit_many_orders(
        self, cal_flux, order_list, rough_wls=None, comb_lines_angstrom=None,
        expected_peak_locs=None,peak_wavelengths_ang=None, our_wavelength_solution_for_order=None, plt_path=None, print_update=False):
        """
        Iteratively performs wavelength calibration for all orders.
        Args:
            cal_flux (np.array): (n_orders x n_pixels) array of calibrator fluxes
                for which to derive a wavelength solution
            order_list (list of int): list order to compute wls for
            rough_wls (np.array): (N_orders x N_pixels) array of wavelength
                values describing a "rough" wavelength solution. Always None for
                lamps. For LFC, this is generally a lamp-derived solution.
                For Etalon, this is generally an LFC-derived solution. Default None.
            comb_lines_angstrom (np.array): array of all allowed wavelengths for the
                LFC, computed using the order_flux equation. Should be None unless we
                are calibrating an LFC frame. Default None.
            expected_peak_locs (dict): dictionary of order number-dict
                pairs. See description in run_wavelength_cal().
            plt_path (str): if set, all diagnostic plots will be saved in this
                directory. If None, no plots will be made.
            print_update (bool): whether subfunctions should print updates.
        Returns:
            tuple of:
                np.array of float: (N_orders x N_pixels) derived wavelength
                    solution for each pixel
                dict: the peaks and wavelengths used for wavelength cal. Keys
                    are ints representing order numbers, values are 2-tuples of:
                        - lists of wavelengths corresponding to peaks
                        - the corresponding (fractional) pixels on which the
                          peaks fall
                dict: the orderlet dictionary, that is folded into wls_dict at a higher level
        """
        
        # Construct dictionary for each order in wlsdict
        orderlet_dict = {}
        for order_num in order_list:
            orderlet_dict[order_num] = {"ordernum" : order_num}

        # Plot 2D extracted spectra
        if plt_path is not None and self.plot:
            plt.figure(figsize=(20,10), tight_layout=True)
            im = plt.imshow(cal_flux, aspect='auto',origin='lower')
            im.set_clim(0, 20000)
            plt.xlabel('Pixel')
            plt.ylabel('Order Number')
            plt.savefig('{}/extracted_spectra_{}.png'.format(plt_path,self.fibre), dpi=600)
            plt.close()

        # Define variables to be used later
        order_precisions = []
        num_detected_peaks = []
        wavelengths_and_pixels = {}
        poly_soln_final_array = np.zeros(np.shape(cal_flux))

        # Iterate over orders
        for order_num in order_list:
            if print_update:
                print('\nRunning order # {}'.format(order_num))

            if plt_path is not None:
                order_plt_path = '{}/order_diagnostics/order{}'.format(plt_path, order_num)
                if not os.path.isdir(order_plt_path):
                    os.makedirs(order_plt_path)

                plt.figure(figsize=(20,10), tight_layout=True)
                #plt.plot(cal_flux[order_num,:], color='k', alpha=0.5)
                plt.plot(cal_flux[order_num,:], color='k', linewidth = 0.5)
                plt.title('Order # {}'.format(order_num), fontsize=36)
                plt.xlabel('Pixel', fontsize=28)
                plt.ylabel('Flux', fontsize=28)
                plt.yscale('symlog')
                plt.tick_params(axis='both', direction='inout', length=6, width=3, colors='k', labelsize=24)
                plt.savefig('{}/order_spectrum_{}.png'.format(order_plt_path,self.fibre), dpi=500)
                plt.close()
            else:
                order_plt_path = None

            order_flux = cal_flux[order_num,:]
            order_flux -= np.nanmedian(order_flux)
            rough_wls_order = rough_wls[order_num,:]
            n_pixels = len(order_flux)
            
            # Add information for this order to the orderlet dictionary
            orderlet_dict[order_num]['flux'] = order_flux
            orderlet_dict[order_num]['initial_wls'] = rough_wls_order
 #           orderlet_dict[order_num]['echelle_order'] = echelle_ord[order_num]
            orderlet_dict[order_num]['n_pixels'] = n_pixels
            orderlet_dict[order_num]['lines'] = {}

            # check if there's flux in the orderlet (e.g., SKY order 0 is off of the GREEN CCD)
            npixels_wflux = len([x for x in order_flux if x != 0])
            if npixels_wflux == 0:
                self.logger.warn('This order has no flux, defaulting to rough WLS')
                continue

            if self.cal_type == 'Etalon':  # For etalon
                etalon_mask = pd.read_csv(self.etalon_mask_in, names=['wave','weight'], sep='\s+')
                wls, fitted_peak_pixels = self.find_etalon_peaks(order_flux,rough_wls_order,etalon_mask) # returns original mask and new mask positions for one order.
                wls=wls.tolist()

            # find, clip, and compute precise wavelengths for peaks.
            # this code snippet will only execute for Etalon and LFC frames.
            elif expected_peak_locs is None:
                skip_orders_wls = None
                if self.red_skip_orders and max(order_list) == 31:  # KPF max order for red chip (update if changed in KPF.cfg)
                    skip_orders_wls = np.fromstring(self.red_skip_orders, dtype=int, sep=',')
                elif self.green_skip_orders and max(order_list) == 34:  # KPF max order for green chip (update if changed in KPF.cfg)
                    skip_orders_wls = np.fromstring(self.green_skip_orders, dtype=int, sep=',')

                if skip_orders_wls is not None:
                    try:
                        if order_num in skip_orders_wls:
                            raise Exception(f'Order {order_num} is skipped in the config, defaulting to rough WLS')
                    except Exception as e:
                        print(e)
                        poly_soln_final_array[order_num, :] = rough_wls_order
                        wavelengths_and_pixels[order_num] = {
                            'known_wavelengths_air': rough_wls_order,
                            'line_positions': []
                        }
                        continue

                try:
                    fitted_peak_pixels, detected_peak_pixels, \
                        detected_peak_heights, gauss_coeffs, lines_dict = self.find_peaks_in_order(
                        order_flux, plot_path=order_plt_path
                    )
                    orderlet_dict[order_num]['lines'] = lines_dict
                    
                except TypeError as e:
                    self.logger.warn('Not enough peaks found in order, defaulting to rough WLS')
                    self.logger.warn('TypeError = ' + str(e))
                    poly_soln_final_array[order_num,:] = rough_wls_order
                    wavelengths_and_pixels[order_num] = {
                        'known_wavelengths_air': rough_wls_order,
                        'line_positions':[]
                    }
                    order_dict = {}
                    continue

                if self.clip_peaks_toggle:
                    good_peak_idx = self.clip_peaks(
                        order_flux, fitted_peak_pixels, detected_peak_pixels,
                        gauss_coeffs, detected_peak_heights,
                        clip_below_median=self.clip_below_median,
                        plot_path=order_plt_path, print_update=print_update
                    )
                else:
                    good_peak_idx = np.arange(len(detected_peak_pixels))

                if self.cal_type == 'LFC':
                    try:
                        wls, _, good_peak_idx = self.mode_match(
                            order_flux, fitted_peak_pixels, good_peak_idx,
                            rough_wls_order, comb_lines_angstrom,
                            print_update=print_update, plot_path=order_plt_path
                        )
                    except:
                        poly_soln_final_array[order_num,:] = rough_wls_order
                        wavelengths_and_pixels[order_num] = {
                            'known_wavelengths_air': rough_wls_order,
                            'line_positions':[]
                        }
                        order_dict = {}
                        continue
                elif self.cal_type == 'Etalon':

                    assert comb_lines_angstrom is None, '`comb_lines_angstrom` \
                        should not be set for Etalon frames.'

                    wls = np.interp(
                        fitted_peak_pixels[good_peak_idx], np.arange(n_pixels)[rough_wls_order>0],
                        rough_wls_order[rough_wls_order>0]
                    )

                fitted_peak_pixels = fitted_peak_pixels[good_peak_idx]

                # Mark lines with bad fits and lambda_fit for each line in dictionary:
                '''
                good_line_ind = 0
                for l in np.arange(len(lines_dict)):
                    if l not in good_peak_idx:
                        orderlet_dict[order_num]['lines'][l]['quality'] = 'bad' #TODO: add this functionality to ThAr dictionaries
                    else:
                        orderlet_dict[order_num]['lines'][l]['lambda_fit'] = wls[good_line_ind]
                        good_line_ind += 1
                '''
            # use expected peak locations to compute updated precise wavelengths for each pixel
            # (only ThAr)
            else:
                if order_plt_path is not None:
                    plot_toggle = True
                else:
                    plot_toggle = False

                min_order_wave = np.min(rough_wls_order)
                max_order_wave = np.max(rough_wls_order)
#                line_wavelengths = expected_peak_locs.query(f'{min_order_wave} < wave < {max_order_wave}')['wave'].values
                line_wavelengths = expected_peak_locs[order_num]['known_wavelengths_air']
                ii = np.where(np.logical_and(line_wavelengths > min_order_wave, line_wavelengths < max_order_wave))[0]
                
                line_wavelengths = line_wavelengths[ii]
                
                pixels_order = np.arange(0, len(rough_wls_order))
                wave_to_pix = interp1d(rough_wls_order, pixels_order,
                                       assume_sorted=False)
                line_pixels_expected = wave_to_pix(line_wavelengths)

                sorted_indices = np.argsort(line_pixels_expected)
                line_wavelengths = line_wavelengths[sorted_indices]
                
                line_pixels_expected = line_pixels_expected[sorted_indices]

                line_wavelengths = np.array([
                    line_wavelengths[i] for i in
                    np.arange(1, len(line_pixels_expected)) if
                    line_pixels_expected[i] != line_pixels_expected[i-1]
                ])
                line_pixels_expected = np.array([
                    line_pixels_expected[i] for i in
                    np.arange(1, len(line_pixels_expected)) if
                    line_pixels_expected[i] != line_pixels_expected[i-1]
                ])
                wls, gauss_coeffs, lines_dict = self.line_match(
                    order_flux, line_wavelengths, line_pixels_expected,
                    plot_toggle, order_plt_path
                )
                
                orderlet_dict[order_num]['lines'] = lines_dict
                
                fitted_peak_pixels = gauss_coeffs[1,:]

            # if we don't have an etalon frame, we won't use drift to calculate the wls
            # To-do for Etalon: add line_dicts
            if self.cal_type != 'Etalon':
                if expected_peak_locs is None:
                    peak_heights = detected_peak_heights[good_peak_idx]
                else:
                    peak_heights = fitted_peak_pixels

                # calculate the wavelength solution for the order
                polynomial_wls, leg_out = self.fit_polynomial(
                    wls, rough_wls_order, peak_wavelengths_ang, order_list, n_pixels, fitted_peak_pixels, peak_heights=peak_heights,
                    plot_path=order_plt_path, fit_iterations=self.fit_iterations,
                    sigma_clip=self.sigma_clip)
                poly_soln_final_array[order_num,:] = polynomial_wls

                if plt_path is not None:
                    fig, ax = plt.subplots(2, 1, figsize=(12,5))
                    ax[0].set_title('Precise WLS - Rough WLS')
                    ax[0].plot(np.arange(n_pixels), leg_out(np.arange(n_pixels)) - rough_wls_order, color='k')
                    ax[0].set_ylabel('[$\\rm \AA$]')
                    pixel_sizes = rough_wls_order[1:] - rough_wls_order[:-1]
                    ax[1].plot(np.arange(n_pixels - 1),
                              (leg_out(np.arange(n_pixels - 1)) - rough_wls_order[:-1]) / pixel_sizes, color='k')
                    ax[1].set_ylabel('[Pixels]')
                    ax[1].set_xlabel('Pixel')
                    plt.tight_layout()
                    plt.savefig('{}/precise_vs_rough_{}.png'.format(order_plt_path,self.fibre), dpi=500)
                    plt.close()

                # compute various RV precision values for order
                rel_precision, abs_precision = self.calculate_rv_precision(
                    fitted_peak_pixels, wls, leg_out, rough_wls_order, our_wavelength_solution_for_order, rough_wls_order, plot_path=order_plt_path,
                    print_update=print_update
                )
                order_precisions.append(abs_precision)
                num_detected_peaks.append(len(fitted_peak_pixels))

                # Add to dictionary for this order
                orderlet_dict[order_num]['fitted_wls'] = polynomial_wls
                orderlet_dict[order_num]['rel_precision_cms'] = rel_precision
                orderlet_dict[order_num]['abs_precision_cms'] = abs_precision
                orderlet_dict[order_num]['num_detected_peaks'] = len(fitted_peak_pixels)
                orderlet_dict[order_num]['known_wavelengths_air'] = wls
                orderlet_dict[order_num]['line_positions'] = fitted_peak_pixels

            # compute drift, and use this to update the wavelength solution
            else:
                pass
                
            wavelengths_and_pixels[order_num] = {
                'known_wavelengths_air':wls,
                'line_positions':fitted_peak_pixels
            }

        # for lamps and LFC, we can compute absolute precision across all orders
        if self.cal_type != 'Etalon':
            squared_resids = (np.array(order_precisions) * num_detected_peaks)**2
            sum_of_squared_resids = np.sum(squared_resids)
            overall_std_error = (np.sqrt(sum_of_squared_resids) / np.sum(num_detected_peaks))
            #orderlet_dict['overall_std_error_cms'] = overall_std_error
            print('\n\n\nOverall absolute precision (all orders): {:2.2f} cm/s\n\n\n'.format(overall_std_error))

        return poly_soln_final_array, wavelengths_and_pixels, orderlet_dict, overall_std_error, order_precisions

    def line_match(self, flux, linelist, line_pixels_expected, plot_toggle, savefig, gaussian_fit_width=10):
        """
        Given a linelist of known wavelengths of peaks and expected pixel locations
        (from a previous wavelength solution), returns precise, updated pixel locations
        for each known peak wavelength.
        Args:
            flux (np.array): flux of order
            linelist (np.array of float): wavelengths of lines to be fit (Angstroms)
            line_pixels_expected (np.array of float): expected pixels for each wavelength
                (Angstroms); must be same length as `linelist`
            plot_toggle (bool): if True, make and save plots.
            savefig (str): path to directory where plots will be saved
            gaussian_fit_width (int): pixel +/- range to use for Gaussian fitting
        Retuns:
            tuple of:
                np.array: same input linelist, with unfit lines removed
                np.array: array of size (4, n_peaks) containing best-fit
                    Gaussian parameters [a, mu, sigma**2, const] for each detected peak
                dictionary: a dictionary of information about the lines fit within this order
        """
        if self.cal_type == 'ThAr':
            gaussian_fit_width = 5
        num_input_lines = len(linelist)
        num_pixels = len(flux)
        successful_fits = []
        lines_dict = {}

        missed_lines = 0
        coefs = np.zeros((4,num_input_lines))
        for i in np.arange(num_input_lines):
            line_location = line_pixels_expected[i]
            peak_pixel = np.floor(line_location).astype(int)
            # don't fit saturated lines
            if peak_pixel < len(flux) and flux[peak_pixel] <= 1e6:
                if peak_pixel < gaussian_fit_width:
                    first_fit_pixel = 0
                else:
                    first_fit_pixel = peak_pixel - gaussian_fit_width
                
                if peak_pixel + gaussian_fit_width > num_pixels:
                    last_fit_pixel = num_pixels
                else:
                    last_fit_pixel = peak_pixel + gaussian_fit_width

                # fit gaussian to matched peak location
                result, line_dict = self.fit_gaussian_integral(
                    np.arange(first_fit_pixel,last_fit_pixel),
                    flux[first_fit_pixel:last_fit_pixel]
                )

                #add_to_line_dict = False
                if result is not None:
                    coefs[:, i] = result
                    successful_fits.append(i)  # Append index of successful fit
                    line_dict['lambda_fit'] = linelist[i]
                    lines_dict[str(i)] = line_dict  # Add line dictionary to lines dictionary
                else:
                    missed_lines += 1

                amp = coefs[0,i]
                if amp < 0:
                    missed_lines += 1
                    coefs[:,i] = np.nan

            else:
                coefs[:,i] = np.nan
                missed_lines += 1

        linelist = linelist[successful_fits]
        coefs = coefs[:, successful_fits]
        linelist = linelist[np.isfinite(coefs[0,:])]
        coefs = coefs[:, np.isfinite(coefs[0,:])]
        
        if plot_toggle:

            n_zoom_sections = 10
            zoom_section_pixels = num_pixels // n_zoom_sections

            zoom_section_pixels = (num_pixels // n_zoom_sections)
            _, ax_list = plt.subplots(n_zoom_sections,1,figsize=(10, 20))
            ax_list[0].set_title('({} missed lines)'.format(missed_lines))
            for i, ax in enumerate(ax_list):

                # plot the flux
                ax.plot(
                    np.arange(num_pixels)[i*zoom_section_pixels:(i+1)*zoom_section_pixels],
                    flux[i*zoom_section_pixels:(i+1)*zoom_section_pixels],color='k'
                )

                # #  plot the fitted peak maxima as points
                # ax.scatter(
                #     coefs[1,:][
                #         (coefs[1,:] > i * zoom_section_pixels) &
                #         (coefs[1,:] < (i+1) * zoom_section_pixels)
                #     ],
                #     coefs[0,:][
                #         (coefs[1,:] > i * zoom_section_pixels) &
                #         (coefs[1,:] < (i+1) * zoom_section_pixels)
                #     ] +
                #     coefs[3,:][
                #         (coefs[1,:] > i * zoom_section_pixels) &
                #         (coefs[1,:] < (i+1) * zoom_section_pixels)
                #     ],
                #     color='red'
                # )

                # overplot the Gaussian fits
                for j in np.arange(num_input_lines-missed_lines):

                    # if peak in range:
                    if (
                        (coefs[1,j] > i * zoom_section_pixels) &
                        (coefs[1,j] < (i+1) * zoom_section_pixels)
                    ):

                        xs = np.floor(coefs[1,j]) - gaussian_fit_width + \
                            np.linspace(
                                0,
                                2 * gaussian_fit_width,
                                2 * gaussian_fit_width
                            )
                        gaussian_fit = self.integrate_gaussian(
                            xs, coefs[0,j], coefs[1,j], coefs[2,j], coefs[3,j]
                        )

                        ax.plot(xs, gaussian_fit, alpha=0.5, color='red')

            plt.tight_layout()
            plt.savefig('{}/spectrum_and_gaussian_fits_{}.png'.format(savefig,self.fibre), dpi=500)
            plt.close()

        return linelist, coefs, lines_dict

    def fit_gaussian_integral(self, x, y,x0=None,do_test=True,Silent=True,chi_plus = 0):
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(self.integrate_gaussian, x, y, p0=p0, maxfev=1000000)
            pcov[np.isinf(pcov)] = 0 # convert inf to zero
            pcov[np.isnan(pcov)] = 0 # convert nan to zero
            line_dict['amp']   = popt[0] # optimized parameters
            line_dict['mu']    = popt[1] # "
            line_dict['sig']   = popt[2] # "
            line_dict['const'] = popt[3] # ""
            line_dict['covar'] = pcov    # covariance
            line_dict['data']  = y
            line_dict['model'] = self.integrate_gaussian(x, *popt)
            line_dict['quality'] = 'good' # fits are assumed good until marked bad elsewhere
            

        if self.cal_type == 'ThAr' and do_test:
            # Quality Checks for Gaussian Fits
            
            if max(y) == 0:
                if not Silent:
                    print('Amplitude is 0')
                return(None, line_dict)
            
            chi_squared_threshold = int(self.chi_2_threshold) + chi_plus

            # Calculate chi^2
            predicted_y = self.integrate_gaussian(x, *popt)
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
                if not Silent:
                    print("Chi squared exceeded the threshold for this line. Line skipped")
                return None, line_dict

            # Check if the Gaussian amplitude is positive, the peak is higher than the wings, or the peak is too high
            if popt[0] <= 0 or popt[0] <= popt[3] or popt[0] >= 500*max(y):
                line_dict['quality'] = 'bad_amplitude'  # Mark the fit as bad due to bad amplitude or U shaped gaussian
                if not Silent:
                    print('Bad amplitude detected')
                return None, line_dict

        return (popt, line_dict)
    
    def integrate_gaussian(self, x, a, mu, sig, const, int_width=0.5):
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

    def fit_polynomial(self, wls, rough_wls_order, peak_wavelengths_ang, order_list, n_pixels, fitted_peak_pixels, fit_iterations=5, sigma_clip=2.1, peak_heights=None, plot_path=None):
        """
        Given precise wavelengths of detected LFC order_flux lines, fits a
        polynomial wavelength solution.
        Args:
            wls (np.array): the known, precise wavelengths of the detected peaks,
                either from fundamental physics or a previous wavelength solution.
            n_pixels (int): number of pixels in the order
            fitted_peak_pixels (np.array): array of true detected peak locations as
                determined by Gaussian fitting.
            fit_iterations (int): number of sigma-clipping iterations in the polynomial fit
            sigma_clip (float): clip outliers in fit with residuals greater than sigma_clip away from fit
            peak_heights (np.array): heights of peaks (either detected heights or
                fitted heights). We use this to weight the peaks in the polynomial
                fit, assuming Poisson errors.
            plot_path (str): if defined, the path to the output directory for
                diagnostic plots. If None, plots are not made.
        Returns:
            tuple of:
                np.array: calculated wavelength solution for the order (i.e.
                    wavelength value for each pixel in the order)
                func: a Python function that, given an array of pixel locations,
                    returns the Legendre polynomial wavelength solutions
        """
        weights = 1 / np.sqrt(peak_heights)
        if self.fit_type.lower() not in ['legendre', 'spline']:
            raise NotImplementedError("Fit type must be either legendre or spline")
        
        if self.fit_type.lower() == 'legendre' or self.fit_type.lower() == 'spline':

            _, unique_idx, count = np.unique(fitted_peak_pixels, return_index=True, return_counts=True)
            unclipped_idx = np.where(
                (fitted_peak_pixels > 0)
            )[0]
            unclipped_idx = np.intersect1d(unclipped_idx, unique_idx[count < 2])
            
            sorted_idx = np.argsort(fitted_peak_pixels[unclipped_idx])
            x, y, w = fitted_peak_pixels[unclipped_idx][sorted_idx], wls[unclipped_idx][sorted_idx], weights[unclipped_idx][sorted_idx]

            for i in range(fit_iterations):
                if self.fit_type.lower() == 'legendre':
                    if self.cal_type == 'ThAr':
                        # fit ThAr based on 4/30 WLS
                        rough_wls_int = interp1d(np.arange(n_pixels), rough_wls_order, kind='linear', fill_value="extrapolate")
                        
                        def polynomial_func_6(x, c0, c1, c2, c3, c4, c5):
                            """
                            Polynomial function to fit.
                            Args:
                                x (np.array): Pixel values.
                                c0, c1, c2, c3 (float): Coefficients of the polynomial.
                            Returns:
                                np.array: Evaluated polynomial.
                            """
                            return rough_wls_int(x) + c0 + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4 + c5 * x**5
                            
                        def polynomial_func_3(x, c0, c1, c2, c3):
                            """
                            Polynomial function to fit.
                            Args:
                                x (np.array): Pixel values.
                                c0, c1, c2, c3 (float): Coefficients of the polynomial.
                            Returns:
                                np.array: Evaluated polynomial.
                            """
                            return rough_wls_int(x) + c0 + c1 * x + c2 * x**2 + c3 * x**3
                        
                        # Using curve_fit to find the best-fit values of {c0, c1}
                        if len(x) < 6:
                            try:
                                popt, _ = curve_fit(polynomial_func_3, x, y)
                                # Create the wavelength solution for the order
                                our_wavelength_solution_for_order = polynomial_func_3(np.arange(len(rough_wls_order)), *popt)
                            except:
                                our_wavelength_solution_for_order = np.zeros(len(rough_wls_order))
                                leg_out = our_wavelength_solution_for_order
                        else:
                            popt, _ = curve_fit(polynomial_func_6, x, y)
                            # Create the wavelength solution for the order
                            our_wavelength_solution_for_order = polynomial_func_6(np.arange(len(rough_wls_order)), *popt)

                        
                        leg_out = Legendre.fit(np.arange(n_pixels), our_wavelength_solution_for_order, 9)
                    
                    if self.cal_type == 'LFC':
                        leg_out = Legendre.fit(x, y, self.fit_order, w=w)
                        our_wavelength_solution_for_order = leg_out(np.arange(n_pixels))
                if self.fit_type == 'spline':
                    leg_out = UnivariateSpline(x, y, w, k=5)
                    our_wavelength_solution_for_order = leg_out(np.arange(n_pixels))
                
                res = y - leg_out(x)
                good = np.where(np.abs(res) <= sigma_clip*np.std(res))
                x = x[good]
                y = y[good]
                w = w[good]
                res = res[good]
            
#            plt.plot(x, res, 'k.')
#            plt.axhline(0, color='b', lw=2)
#            plt.xlabel('Pixel')
#            plt.ylabel('Fit residuals [$\AA$]')
#            plt.tight_layout()
#            plt.savefig('{}/polyfit.png'.format(plot_path))
#            plt.close()
            
            if plot_path is not None and self.cal_type =='ThAr':
                approx_dispersion = (our_wavelength_solution_for_order[int(len(rough_wls_order)/2)] - our_wavelength_solution_for_order[int(len(rough_wls_order)/2)+100])/100
                #fig, ax1 = plt.subplots(tight_layout=True, figsize=(8, 4))
                
                # Range of interest b/c CCF chops off first/last 500 pixels
                pixel_range = np.arange(500, int(len(rough_wls_order))-500)
                rough_wls_int_range = rough_wls_int(pixel_range)
                wavelength_solution_range = our_wavelength_solution_for_order[500:int(len(rough_wls_order))-500]

                # Create the plot
                fig, ax1 = plt.subplots(tight_layout=True, figsize=(8, 4))
                ax1.plot(
                    pixel_range,
                    rough_wls_int_range - wavelength_solution_range,
                    color='k'
                )
                ax1.set_xlabel('Pixel')
                ax1.set_ylabel(r'Wavelength Difference ($\AA$)')
                ax2 = ax1.twinx()
                ax2.set_ylabel("Difference (pixels) \nusing dispersion " + r'$\approx$' + '{0:.2}'.format(approx_dispersion) + r' $\AA$/pixel')
                ax2.set_ylim(ax1.get_ylim())
                ax1_ticks = ax1.get_yticks()
                ax2.set_yticklabels([str(round(tick / approx_dispersion, 2)) for tick in ax1_ticks])
                plt.savefig('{}/interp_vs_our_wls_{}.png'.format(plot_path,self.fibre), dpi=500)
                plt.close()
        else:
            raise ValueError('Only set up to perform Legendre fits currently! Please set fit_type to "Legendre"')

        return our_wavelength_solution_for_order, leg_out

    def calculate_rv_precision(
        self, fitted_peak_pixels, wls, leg_out, rough_wls, our_wavelength_solution_for_order, rough_wls_order,
        print_update=True, plot_path=None
    ):
        """
        Calculates 1) RV precision from the difference between the known (from
        physics) wavelengths of pixels containing peak flux values and the
        fitted wavelengths of the same pixels, generated using a polynomial
        wavelength solution ("absolute RV precision") and 2) RV precision from
        the difference between the "master" wavelength solution and our
        fitted wavelength solution ("relative RV precision")
        Args:
            fitted_peak_pixels (np.array of float): array of true detected peak locations as
                determined by Gaussian fitting (already clipped)
            wls (np.array of float): precise wavelengths of `fitted_peak_pixels`,
                from fundamental physics or another wavelength solution.
            leg_out (func): a Python function that, given an array of pixel
                locations, returns the Legendre polynomial wavelength solutions
            rough_wls (np.array of float): rough wavelength values for each
                pixel in the order [Angstroms]
            print_update (bool): If true, prints standard error per order.
            plot_path (str): if defined, the path to the output directory for
                diagnostic plots. If None, plots are not made.
        Returns:
            tuple of:
                float: absolute RV precision in cm/s
                float: relative RV precision in cm/s
        """
        our_wls_peak_pos = leg_out(fitted_peak_pixels)
        # absolute/polynomial precision of order = difference between fundemental wavelengths
        # and our wavelength solution wavelengths for (fractional) peak pixels
        abs_residual = ((our_wls_peak_pos - wls) * scipy.constants.c) / wls
        abs_precision_cm_s = 100 * np.nanstd(abs_residual)/np.sqrt(len(fitted_peak_pixels))
        # the above line should use RMS not STD

        # relative RV precision of order = difference between rough wls wavelengths
        # and our wavelength solution wavelengths for all pixels
        n_pixels = len(rough_wls)
        our_wavelength_solution_for_order = leg_out(np.arange(n_pixels))
        rel_residual = (our_wavelength_solution_for_order[rough_wls>0] -  rough_wls[rough_wls>0]) * scipy.constants.c /rough_wls[rough_wls>0]
        rel_precision_cm_s = 100 * np.std(rel_residual)/np.sqrt(len(rough_wls[rough_wls>0]))
        if print_update:
            print('Absolute standard error (this order): {:.2f} cm/s'.format(abs_precision_cm_s))
            print('Relative standard error (this order): {:.2f} cm/s'.format(rel_precision_cm_s))
        
        if plot_path is not None:
            fig, ax = plt.subplots(2,1) #figsize=(20,16), tight_layout=True
            ax[0].plot(abs_residual)
            ax[0].set_xlabel('Pixel')
            ax[0].set_ylabel('Absolute Error [m/s]')
            ax[1].plot(rel_residual)
            ax[1].set_xlabel('Pixel')
            ax[1].set_ylabel('Relative Error [m/s]')
            plt.savefig('{}/rv_precision_{}.png'.format(plot_path,self.fibre), dpi=500)
            plt.close()

        return rel_precision_cm_s, abs_precision_cm_s
        
    def find_peaks_in_order(self, order_flux, plot_path=None):
        """
        Runs find_peaks on successive subsections of the order_flux lines and concatenates
        the output. The difference between adjacent peaks changes as a function
        of position on the detector, so this results in more accurate peak-finding.
        Based on pyreduce.
        Args:
            order_flux (np.array): flux values. Their indices correspond to
                their pixel numbers. Generally the entire order.
            plot_path (str): Path for diagnostic plots. If None, plots are not made.
        Returns:
            tuple of:
                np.array: array of true peak locations as determined by Gaussian fitting
                np.array: array of detected peak locations (pre-Gaussian fitting)
                np.array: array of detected peak heights (pre-Gaussian fitting)
                np.array: array of size (4, n_peaks) 
                    containing best-fit Gaussian parameters [a, mu, sigma**2, const]
                    for each detected peak
                dict: dictionary of information about each line in the order
        """

        lines_dict = {}
    
        n_pixels = len(order_flux)
        fitted_peak_pixels = np.array([])
        detected_peak_pixels = np.array([])
        detected_peak_heights = np.array([])
        gauss_coeffs = np.zeros((4,0))
        ind_dict = 0

        try:
            for i in np.arange(self.n_sections):
    
                if i == self.n_sections - 1:
                    indices = np.arange(i * n_pixels // self.n_sections, n_pixels)
                else:
                    indices = np.arange(i * n_pixels // self.n_sections, (i+1) * n_pixels // self.n_sections)
                    
                fitted_peaks_section, detected_peaks_section, peak_heights_section, \
                    gauss_coeffs_section, this_lines_dict = self.find_peaks(order_flux[indices], peak_height_threshold=self.peak_height_threshold)
    
                for ii, row in enumerate(this_lines_dict):
                    lines_dict[ind_dict] = this_lines_dict[ii]
                    ind_dict += 1
                
                detected_peak_heights = np.append(detected_peak_heights, peak_heights_section)
                gauss_coeffs = np.append(gauss_coeffs, gauss_coeffs_section, axis=1)
                if i == 0:
                    fitted_peak_pixels = np.append(fitted_peak_pixels, fitted_peaks_section)
                    detected_peak_pixels = np.append(detected_peak_pixels, detected_peaks_section)
    
                else:
                    fitted_peak_pixels = np.append(
                        fitted_peak_pixels,
                        fitted_peaks_section + i * n_pixels // self.n_sections
                    )
                    detected_peak_pixels = np.append(
                        detected_peak_pixels,
                        detected_peaks_section + i * n_pixels // self.n_sections
                    )
        
        except Exception as e:
            print('Exception: ' + str(e))
            print('self.n_sections = ', str(self.n_sections))
        
        if plot_path is not None:
            plt.figure(figsize=(20,10), tight_layout=True)
            #plt.plot(order_flux, color='k', lw=0.1)
            plt.plot(order_flux, color='k', lw=0.5)
            plt.scatter(detected_peak_pixels, detected_peak_heights, s=2, color='r')
            plt.xlabel('Pixel', fontsize=28)
            plt.ylabel('Flux', fontsize=28)
            plt.yscale('symlog')
            plt.tick_params(axis='both', direction='inout', length=6, width=3, colors='k', labelsize=24)
            plt.savefig('{}/detected_peaks.png'.format(plot_path), dpi=250)
            plt.close()

            n_zoom_sections = 5
            zoom_section_pixels = n_pixels // n_zoom_sections

            _, ax_list = plt.subplots(n_zoom_sections, 1, figsize=(12,6))
            for i, ax in enumerate(ax_list):
                ax.plot(order_flux,color='k', lw=0.5)
                ax.scatter(detected_peak_pixels,detected_peak_heights,s=1,color='r')
                ax.set_xlim(zoom_section_pixels * i, zoom_section_pixels * (i+1))
                ax.set_ylim(
                    0,
                    np.max(
                        order_flux[zoom_section_pixels * i : zoom_section_pixels * (i+1)]
                    )
                )
                ax.set_ylabel('Flux', fontsize=14)
                if i == n_zoom_sections-1:
                    ax.set_xlabel('Pixel', fontsize=14)

            plt.tight_layout()
            plt.savefig('{}/detected_peaks_zoom.png'.format(plot_path),dpi=250)
            plt.close()
                  
        return fitted_peak_pixels, detected_peak_pixels, detected_peak_heights, gauss_coeffs, lines_dict


    def compute_offset_fft_subpixel(self,ref, target):
        """
        Compute (x, y) offset between two 2D arrays using FFT phase correlation
        with subpixel accuracy (via parabolic peak fitting).
        """

        # Ensure floating-point data
        ref = ref.astype(float)
        target = target.astype(float)
        
        # Compute cross power spectrum
        F_ref = fft2(ref)
        F_target = fft2(target)
        R = F_ref * F_target.conj()
        R /= np.abs(R) + 1e-15  # normalize
        
        # Inverse FFT to get correlation
        corr = fftshift(ifft2(R).real)
        
        # Find integer location of maximum
        max_y, max_x = np.unravel_index(np.argmax(corr), corr.shape)
        center_y, center_x = np.array(corr.shape) // 2
        offset_y = max_y - center_y
        offset_x = max_x - center_x

        # --- Subpixel refinement using quadratic fit around the peak ---
        def quadratic_subpixel_peak(zm1, z0, zp1):
            """Estimate subpixel shift of peak using 3-point quadratic fit."""
            denom = 2 * (zm1 - 2*z0 + zp1)
            if abs(denom) < 1e-10:
                return 0.0
            return (zm1 - zp1) / denom

        # Get 3x3 neighborhood around peak (handle edges safely)
        y0, x0 = max_y, max_x
        if 1 <= y0 < corr.shape[0]-1 and 1 <= x0 < corr.shape[1]-1:
            dy = quadratic_subpixel_peak(corr[y0-1, x0], corr[y0, x0], corr[y0+1, x0])
            dx = quadratic_subpixel_peak(corr[y0, x0-1], corr[y0, x0], corr[y0, x0+1])
        else:
            dx = dy = 0.0

        # Combine integer and fractional parts
        offset_x += dx
        offset_y += dy

        return offset_x, offset_y
