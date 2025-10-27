import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from scipy.special import binom
from mpl_toolkits.mplot3d import axes3d, Axes3D
from astropy.io import fits
from scipy.signal import fftconvolve

from scipy import signal, datasets, ndimage


import numpy as np
from numpy.fft import fft2, ifft2, fftshift

def compute_offset_fft_subpixel(ref, target):
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


# --- Example usage ---
if __name__ == "__main__":
    from scipy.ndimage import shift

    size = 128
    ref = np.zeros((size, size))
    ref[40:60, 50:70] = 1.0

    # Apply known subpixel shift
    true_shift = (3.6, -7.2)  # (y, x)
    target = shift(ref, shift=true_shift, order=3)

    dx, dy = compute_offset_fft_subpixel(ref, target)
    print(f"Estimated offset: x = {dx:.3f}, y = {dy:.3f}")
    print(f"True offset:      x = {-true_shift[1]:.3f}, y = {-true_shift[0]:.3f}")

    
    ref_file1 = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/Super_Arcs/HR_Super_Arc_H20220701.fits'
    with fits.open(ref_file1) as hdu:
        ref_fig = hdu['FIBRE_P'].data
        ref_fig[np.isnan(ref_fig)] = 0

    ref_file2 = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2025/1011/reduced/bgoH202510110034.fits'
    with fits.open(ref_file2) as hdu:
        obs_fig = hdu['FIBRE_P'].data
        obs_fig[np.isnan(obs_fig)] = 0
        
        # Apply known subpixel shift
    true_shift = (0, -7.5)  # (y, x)
    target = shift(ref_fig, shift=true_shift, order=3)
    print(ref_fig.shape)

    dx, dy = compute_offset_fft_subpixel(ref_fig[0:20,:], obs_fig[0:20,:])
    print(f"Estimated offset: x = {dx:.3f}, y = {dy:.3f}")
