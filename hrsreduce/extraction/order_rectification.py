"""
Order rectification in ONE resampling.

This maps every output pixel straight back to the raw frame:

    y_src(x, dy) = y_trace(x) + dy
    x_src(x, dy) = x + dy * tilt(x) + dy**2 * shear(x)

with dy the row offset from the order centre, and interpolates the raw frame
ONCE at that position (bilinear by default, Lanczos optional).  When no tilt
is given the result is a plain vertical rectification (x untouched, y shifted
by the trace), which is what the slit-tilt measurement should be run on.
When a tilt solution is given (the *_Curvature.npz written by
slit_correction.py) the rectified image comes out straight, and it is
written as both RECT and STRAIGHT so that the rest of the pipeline runs
unchanged (SlitCorrection sees STRAIGHT and does nothing).

The variance (VAR extension, if present) and the bad-pixel mask are mapped
with the same weights and written as RECT_VAR and RECT_MASK.

Constructor:
    OrderRectification(sci_frame, flat_frame, order_trace_file, sarm, mode,
                       base_dir, super_arc=None, processes=None,
                       mapping_cache_dir=None,
                       tilt_file=None, kernel="linear", overwrite=False)

    tilt_file : *_Curvature.npz from slit_correction (tilt[n_orderlet,
                n_col], shear[...]); None -> vertical rectification only.
    kernel    : "linear" (bilinear, no overshoot) or "lanczos2" / "lanczos3"
                (sharper, small ringing beside bright lines).
    overwrite : replace an existing RECT/STRAIGHT instead of returning early
                (needed to re-rectify the arc once its tilt is known).

Typical use (per mode, per night, no super arc):
    rect = OrderRectification(arc, flat, trace, arm, mode, base_dir, super_arc=arc).perform()
    SlitCorrection(arc, 'RECT', rect, arm[0], mode, base_dir, night, super_arc=arc).correct()
    tilt = os.path.splitext(arc)[0] + '_Curvature.npz'
    OrderRectification(arc, flat, trace, arm, mode, base_dir, super_arc=arc, tilt_file=tilt, overwrite=True).perform()
    for frame in flats + arcs + science:
        OrderRectification(frame, flat, trace, arm, mode, base_dir, super_arc=arc, tilt_file=tilt).perform()
"""

from __future__ import annotations

import glob
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

import hrsreduce.utils.background_subtraction as BkgAlg

logger = logging.getLogger(__name__)


def _lanczos(t: np.ndarray, a: int) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    inside = np.abs(t) < a
    tt = t[inside]
    with np.errstate(invalid="ignore", divide="ignore"):
        out[inside] = np.where(tt == 0, 1.0, np.sinc(tt) * np.sinc(tt / a))
    return out


def _kernel_weights(frac: np.ndarray, kernel: str):
    """Weights and integer offsets of a separable 1-D kernel at fractional position ``frac``
    (0 <= frac < 1) relative to the floor pixel.  Returns (offsets[list], weights[list of arrays])."""
    if kernel == "linear":
        return [0, 1], [1.0 - frac, frac]
    if kernel.startswith("lanczos"):
        a = int(kernel[-1])
        offsets = list(range(-a + 1, a + 1))
        weights = [_lanczos(frac - k, a) for k in offsets]
        norm = np.sum(weights, axis=0)
        weights = [w / np.where(norm == 0, 1.0, norm) for w in weights]
        return offsets, weights
    raise ValueError(f"unknown kernel {kernel!r}")


def resample_strip(data: np.ndarray, var: np.ndarray | None, mask: np.ndarray,
                   x_src: np.ndarray, y_src: np.ndarray, kernel: str = "linear"):
    """Sample ``data`` at the (non-integer) positions (x_src, y_src) with a separable kernel.
    Masked source pixels contribute zero (as in the original polygon mapping) and flag the
    output pixel; the variance is propagated with the squared weights.  All arrays have the
    shape of x_src/y_src on output."""
    ny, nx = data.shape
    fx = np.floor(x_src); fy = np.floor(y_src)
    ox, wx = _kernel_weights(x_src - fx, kernel)
    oy, wy = _kernel_weights(y_src - fy, kernel)
    ix0 = fx.astype(int); iy0 = fy.astype(int)
    out = np.zeros(x_src.shape); outv = np.zeros(x_src.shape) if var is not None else None
    outm = np.zeros(x_src.shape, dtype=bool); cover = np.zeros(x_src.shape)
    for ky, wyk in zip(oy, wy):
        iy = iy0 + ky; vy = (iy >= 0) & (iy < ny); iyc = np.clip(iy, 0, ny - 1)
        for kx, wxk in zip(ox, wx):
            ix = ix0 + kx; vx = (ix >= 0) & (ix < nx); ixc = np.clip(ix, 0, nx - 1)
            w = wxk * wyk * (vx & vy)
            m = mask[iyc, ixc]
            out += np.where(m, 0.0, w * data[iyc, ixc])
            if var is not None:
                outv += np.where(m, 0.0, w * w * var[iyc, ixc])
            outm |= m & (np.abs(w) > 1e-9)
            cover += w
    # pixels whose stencil fell partly outside the detector
    outm |= np.abs(cover - 1.0) > 1e-6
    return out, outv, outm


class OrderRectification:
    """Rectify (and, with a tilt solution, straighten) the orders in one resampling."""

    NORMAL = 0
    VERTICAL = 1
    NoRECT = 2
    METHOD_NAME = "vertical-onepass"

    def __init__(self, sci_frame: str, flat_frame: str, order_trace_file: str, sarm: str, mode: str,
                 base_dir: str, super_arc: str | None = None, processes: int | None = None,
                 mapping_cache_dir: str | Path | None = None,
                 tilt_file: str | None = None, kernel: str = "linear", overwrite: bool = False):
        self.input_spectrum = sci_frame
        self.input_flat = flat_frame
        self.order_trace_file = order_trace_file
        self.rectification_method = self.VERTICAL
        self.arm = sarm
        self.mode = mode
        self.base_dir = base_dir
        self.logger = logger
        self.super_arc = super_arc
        self.tilt_file = tilt_file
        self.kernel = kernel
        self.overwrite = overwrite
        self.mapping_cache_hit = False

        if self.arm == "Blu":
            self.sarm = "H"; mask_name = "BPM_H.fits"
        elif self.arm == "Red":
            self.sarm = "R"; mask_name = "BPM_R.fits"
        else:
            raise ValueError(f"Unsupported arm: {self.arm}")
        mask_file = os.path.join(".", "hrsreduce", "utils", mask_name)
        if not os.path.isfile(mask_file):                       # run from anywhere, not only the package root
            mask_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils", mask_name)
        with fits.open(mask_file, memmap=True) as mask_hdul:
            self._bpm = np.asarray(mask_hdul[0].data, dtype=bool)

        with fits.open(self.input_spectrum, memmap=True) as hdul:
            data = np.ascontiguousarray(hdul[0].data, dtype=np.float64)
            self._spectrum_header = hdul[0].header.copy()
            self._data_type = hdul[0].header["OBSTYPE"]
            self._var = np.ascontiguousarray(hdul["VAR"].data, dtype=np.float64) if "VAR" in hdul else None
        if self._data_type == "Science":
            # background subtraction
            spectrum_data, _ = BkgAlg.BkgAlg(np.ma.masked_array(data, mask=self._bpm, copy=False),
                                             self.order_trace_file, self.arm, self.mode, self.logger)
            data = np.ascontiguousarray(np.ma.getdata(spectrum_data), dtype=np.float64)
        self._data = data
        self._image_shape = data.shape
        if self._bpm.shape != self._image_shape:
            raise ValueError(f"bad-pixel mask shape {self._bpm.shape} does not match the frame {self._image_shape}")

        if not order_trace_file:
            raise ValueError("order_trace_file is required")
        self.order_trace_data = pd.read_csv(order_trace_file, header=0, index_col=0)
        coeff_cols = [c for c in self.order_trace_data.columns if c.startswith("Coeff")]
        self._coeffs = self.order_trace_data[coeff_cols].to_numpy(dtype=float)        # Coeff0 .. CoeffN
        self.o_set = np.arange(self.order_trace_data.shape[0])

        self._tilt = self._shear = None
        if self.tilt_file:
            with np.load(self.tilt_file, allow_pickle=True) as tf:
                self._tilt = np.asarray(tf["tilt"], dtype=float)
                self._shear = np.asarray(tf["shear"], dtype=float) if "shear" in tf.files else np.zeros_like(self._tilt)
            if self._tilt.shape != (len(self.o_set), self._image_shape[1]):
                raise ValueError(f"tilt solution {self.tilt_file} has shape {self._tilt.shape}, expected "
                                 f"({len(self.o_set)}, {self._image_shape[1]}): different order table?")


    def _order_geometry(self, i: int, previous=None):
        """Centre row, edges and x range of orderlet ``i`` in the output, as the original module defines
        them: the centre row is floor of the mid-range of the trace (evaluated up to X2 + 1), the edges
        are clipped to the image, and the second orderlet of a pair is pushed up to leave a 2-row gap
        above the first (``previous`` = (y_center, upper) of the first orderlet of the pair)."""
        ny, nx = self._image_shape
        x1, x2 = int(self.order_trace_data["X1"].iloc[i]), int(self.order_trace_data["X2"].iloc[i])
        x = np.arange(max(0, x1), min(nx - 1, x2) + 1)
        y_mid = np.polyval(self._coeffs[i][::-1], x)                       # y of the trace along x
        # the original evaluates the trace up to X2 + 1 (its border column) for the centre row
        y_ext = np.polyval(self._coeffs[i][::-1], np.arange(max(0, x1), min(nx - 1, x2) + 2))
        y_center = int(math.floor(0.5 * (np.amax(y_ext) + np.amin(y_ext))))
        bottom = int(self.order_trace_data["BottomEdge"].iloc[i]); top = int(self.order_trace_data["TopEdge"].iloc[i])
        lower = min(bottom, y_center); upper = min(top, ny - 1 - y_center)
        if previous is not None:
            pre_center, pre_upper = previous
            if y_center < pre_center + pre_upper + lower + 2:
                y_center = pre_center + pre_upper + lower + 2
        return x, y_mid, y_center, lower, upper

    def _rectification_is_done(self) -> bool:
        """Only the presence of the RECT extension in the file itself counts.  The earlier shortcut that
        also accepted an existing <mode>_Master_Wave_<arm><frame>.fits for an arc skipped the rectification
        of a bgo file that had just been rebuilt from raw (no RECT inside), and SlitCorrection then failed
        with "Extension 'RECT' not found" on every night whose wavelength solution already existed."""
        if self.overwrite:
            return False
        with fits.open(self.input_spectrum, memmap=True) as hdul:
            return "RECT" in hdul


    def perform(self) -> str:
        if self.super_arc is None:
            raise ValueError("super_arc is required to name the rectified order table")
        order_output = os.path.splitext(self.super_arc)[0] + "_Orders_Rect.csv"
        if self._rectification_is_done():
            self.logger.info("OrderRectification: Already done...")
            return order_output

        ny, nx = self._image_shape
        rect = np.zeros((ny, nx)); rect_var = np.zeros((ny, nx)) if self._var is not None else None
        rect_mask = np.zeros((ny, nx), dtype=np.uint8)
        rows = []
        self.logger.info("OrderRectification: one-pass %s rectification of %d orderlets (%s kernel%s)",
                         "straightening" if self._tilt is not None else "vertical", len(self.o_set), self.kernel,
                         f", tilt from {os.path.basename(self.tilt_file)}" if self.tilt_file else "")
        previous = None
        for i in self.o_set:
            x, y_mid, y_center, lower, upper = self._order_geometry(int(i), previous if int(i) % 2 == 1 else None)
            previous = (y_center, upper)
            dy = np.arange(-lower, upper, dtype=float)
            y_src = y_mid[None, :] + dy[:, None]
            if self._tilt is not None:
                t = self._tilt[i, x]; s = self._shear[i, x]
                x_src = x[None, :] + dy[:, None] * t[None, :] + dy[:, None] ** 2 * s[None, :]
            else:
                x_src = np.broadcast_to(x[None, :].astype(float), y_src.shape)
            strip, strip_var, strip_mask = resample_strip(self._data, self._var, self._bpm, x_src, y_src, self.kernel)
            coeff0 = y_center - 20
            to_pos = coeff0 - lower
            if to_pos < 0 or to_pos + strip.shape[0] > ny:
                self.logger.warning("OrderRectification: orderlet %d falls outside the image, skipped", i)
                continue
            sl = slice(to_pos, to_pos + strip.shape[0])
            rect[sl, x] += strip                                             
            if rect_var is not None:
                rect_var[sl, x] += strip_var
            rect_mask[sl, x] |= strip_mask.astype(np.uint8)
            rows.append({"order": int(i), "Coeff0": coeff0, "BottomEdge": lower, "TopEdge": upper, "X1": 0, "X2": nx - 1})

        with fits.open(self.input_spectrum) as hdul:
            for name in ("RECT", "STRAIGHT", "RECT_VAR", "RECT_MASK"):
                while name in hdul:
                    del hdul[name]
            hdr = fits.Header()
            hdr["RECTMETH"] = (self.METHOD_NAME, "rectification: vertical, one resampling")
            hdr["RECTKERN"] = (self.kernel, "interpolation kernel")
            hdr["RECTTILT"] = (os.path.basename(self.tilt_file) if self.tilt_file else "none", "slit tilt folded into the mapping")
            hdul.append(fits.ImageHDU(data=rect, header=hdr, name="RECT"))
            if self._tilt is not None:
                hdul.append(fits.ImageHDU(data=rect, header=hdr, name="STRAIGHT"))   # already straight: same image
            if rect_var is not None:
                hdul.append(fits.ImageHDU(data=rect_var, name="RECT_VAR"))
            hdul.append(fits.ImageHDU(data=rect_mask, name="RECT_MASK"))
            hdul.writeto(self.input_spectrum, overwrite=True)

        pd.DataFrame(rows).to_csv(order_output)
        return order_output


__all__ = ["OrderRectification", "resample_strip"]
