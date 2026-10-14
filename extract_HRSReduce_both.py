"""
Extract a one-dimensional spectrum from an HRSReduce data product (v1.1 or v1.3) and write it as a tab-separated file.

This is the version that reads both product layouts: the v1.3 merged tables (MERGED_LIN / MERGED_LOG with Flux and Var
columns for both fibres and the sky-subtracted spectrum) and the v1.1 image extensions (FIBRE_L_SPEC_MERGED etc. with a
linear WCS).  extract_HRSReduce_v13.py is the same tool for v1.3 products only.  Writes WAVE, FLUX, ERR, with the
wavelengths optionally corrected to the barycentre.

    python extract_HRSReduce_both.py R202608260033_product.fits                 # target, linear grid, BC applied, plot
    python extract_HRSReduce_both.py -t sub -g log --no-plot *_product.fits     # sky-subtracted, log grid, no plot

Output file: <object>_<BJD>_HRS_<mode>_<arm>[_sky|_sub][_log][_BC].dat   (the target on the linear grid keeps the old name)
"""
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.io import ascii

c_ms = 299792458.0

ARM_NAMES = {'HRDET': 'R', 'HBDET': 'B'}
MODE_NAMES = {'LOW RESOLUTION': 'LR', 'MEDIUM RESOLUTION': 'MR', 'HIGH RESOLUTION': 'HR', 'HIGH STABILITY': 'HS'}


def read_v13(hdu, spec_type, grid):
    """
    Read one merged spectrum from a v1.3 product.

    Parameters
    ----------
    hdu : astropy.io.fits.HDUList
        The open product.
    spec_type : str
        'target', 'sky' or 'sub'.
    grid : str
        'lin' for the MERGED_LIN table (constant Angstrom step), 'log' for MERGED_LOG (constant km/s step).

    Returns
    -------
    wave, flux, err : ndarray
        Wavelength (Angstrom, air, observed frame), flux (counts / blaze) and its error (sqrt of the Var column).
    err_source : str
        Where the error column came from.
    """
    header = hdu[0].header
    # the fibre holding the target is written by the pipeline (upper fibre in LR, lower otherwise)
    star = str(header.get('STARFIB', 'U' if header['OBSMODE'] == 'LOW RESOLUTION' else 'L')).strip()
    sky = 'U' if star == 'L' else 'L'
    column = {'target': star, 'sky': sky, 'sub': 'Sub'}[spec_type]

    table = hdu['MERGED_LIN' if grid == 'lin' else 'MERGED_LOG'].data
    wave = np.asarray(table['Wave'], dtype=float)
    flux = np.asarray(table['Flux_' + column], dtype=float)
    var = np.asarray(table['Var_' + column], dtype=float)

    if np.isfinite(var).any():
        err = np.sqrt(np.where(var > 0, var, np.nan))
        err_source = 'sqrt(Var column), variance propagated by the pipeline'
    else:
        # product written without variances (extraction before the variance was carried)
        err = np.sqrt(np.abs(flux))
        err_source = 'sqrt(|FLUX|), no variance in the product'
    return wave, flux, err, err_source


def read_v11(hdu, spec_type):
    """
    Read one merged spectrum from a v1.1 product (image extensions with a linear wavelength WCS).

    Returns
    -------
    wave, flux, err : ndarray
        Wavelength (Angstrom), flux and sqrt(|flux|) as the error, as the original extractor wrote them.
    err_source : str
    """
    header = hdu[0].header
    # the LR fibres swap the upper and lower traces for sky / target
    if header['OBSMODE'] == 'LOW RESOLUTION':
        names = {'target': 'FIBRE_U_SPEC_MERGED', 'sky': 'FIBRE_L_SPEC_MERGED', 'sub': 'SKY_SUBTRACTED_SPEC_MERGED'}
    else:
        names = {'target': 'FIBRE_L_SPEC_MERGED', 'sky': 'FIBRE_U_SPEC_MERGED', 'sub': 'SKY_SUBTRACTED_SPEC_MERGED'}
    ext = hdu[names[spec_type]]
    flux = np.asarray(ext.data, dtype=float).flatten()
    wave_step = float(ext.header['CDELT1'])
    wave_base = float(ext.header['CRVAL1'])
    reference_pixel = float(ext.header['CRPIX1'])
    wave = (np.arange(flux.size) - reference_pixel + 1) * wave_step + wave_base
    err = np.sqrt(np.abs(flux))
    return wave, flux, err, 'sqrt(|FLUX|), v1.1 product'


def main():
    parser = argparse.ArgumentParser(prog='extract_HRSReduce_both.py',
                                     description='----------------- Extract HRSReduce data products -----------------')
    parser.add_argument('file', nargs='+', type=str,
                        help='File name to read. Can be many strings, or use wildcards.')
    parser.add_argument('-t', '--type', default='target', choices=['target', 'sky', 'sub'],
                        help="Type of data to extract: 'target', 'sky' or 'sub' (target minus sky). Default is target.")
    parser.add_argument('-g', '--grid', default='lin', choices=['lin', 'log'],
                        help="Wavelength grid of the merged spectrum: 'lin' (constant Angstrom step, default) or "
                             "'log' (constant km/s step). v1.1 products only have the linear grid.")
    parser.add_argument('-b', '--bc', default=True, action=argparse.BooleanOptionalAction,
                        help='Apply the barycentric correction to the wavelengths (--no-bc to switch off). Default is on.')
    parser.add_argument('-p', '--plot', default=True, action=argparse.BooleanOptionalAction,
                        help='Plot the chosen data product (--no-plot to switch off). Default is on.')
    parser.add_argument('--keep-gaps', default=False, action='store_true',
                        help='Keep the bins without data (written as nan) instead of dropping them. Default drops them.')
    parser.add_argument('--header', default=False, action='store_true',
                        help='Prepend a block of # comment lines describing the spectrum to the output file.')
    args = parser.parse_args()

    for fname in args.file:
        with fits.open(fname) as hdu:
            header = hdu[0].header
            obj = str(header['OBJECT']).replace(' ', '_')
            BCV = float(header['BARYRV']) * 1000.0           # km/s -> m/s
            BJD = float(header['BJD'])
            out_arm = ARM_NAMES[header['DETNAM']]
            obs_mode = header['OBSMODE']
            mode = MODE_NAMES[obs_mode]
            version = str(header.get('PIPEVERS', 'v1.1 (no PIPEVERS keyword)'))

            if 'MERGED_LIN' in hdu:
                wave, flux, err, err_source = read_v13(hdu, args.type, args.grid)
                grid = args.grid
            elif 'FIBRE_L_SPEC_MERGED' in hdu:
                if args.grid == 'log':
                    print(f'{fname}: v1.1 product, only the linear grid exists; writing that.')
                wave, flux, err, err_source = read_v11(hdu, args.type)
                grid = 'lin'
            else:
                print(f'{fname}: no merged spectrum found (neither MERGED_LIN nor FIBRE_L_SPEC_MERGED). Skipping.')
                continue

        if not args.keep_gaps:
            keep = np.isfinite(flux)
            wave, flux, err = wave[keep], flux[keep], err[keep]

        if args.bc:
            wave_out = wave * (1.0 + BCV / c_ms)
        else:
            wave_out = wave
        # the target file keeps the name the v1.1 extractor gave it; sky and sub outputs are tagged so that they do not
        # overwrite it, and the log grid is tagged so that both grids can sit side by side
        out_fname = (f'{obj}_{BJD}_HRS_{mode}_{out_arm}' + ('' if args.type == 'target' else '_' + args.type)
                     + ('_log' if grid == 'log' else '') + ('_BC' if args.bc else '') + '.dat')

        ascii.write([wave_out, flux, err], out_fname, overwrite=True, names=['WAVE', 'FLUX', 'ERR'], format='tab')

        if args.header:
            header_lines = [
                '# Object: ' + obj,
                '# Instrument: HRS',
                '# Mode: ' + str(obs_mode),
                '# Arm: ' + str(header['DETNAM']),
                '# Pipeline: HRSReduce ' + version,
                '# Spectrum: ' + args.type + ', ' + ('linear' if grid == 'lin' else 'logarithmic') + ' wavelength grid',
                '# Units: WAVE=Angstrom (air), FLUX=counts/blaze, ERR=counts/blaze',
                '# ERR: ' + err_source,
                '# Barycentric correction (km/s): ' + str(header['BARYRV']),
                '# Barycentric correction applied: ' + str(args.bc),
                '# BJD: ' + str(BJD),
                '# -------------------------------------',
            ]
            with open(out_fname, 'r') as f:
                content = f.read()
            with open(out_fname, 'w') as f:
                f.write('\n'.join(header_lines) + '\n')
                f.write(content)

        print(f'{fname} -> {out_fname}  ({flux.size} points, {args.type}, {grid} grid, error from {err_source})')

        if args.plot:
            plt.figure(figsize=(10, 5))
            plt.plot(wave_out, flux, linewidth=1)
            plt.xlabel(r'Wavelength (Å)')
            plt.ylabel('Flux (arbitrary units)')
            plt.title('Spectrum of ' + obj)
            plt.show()


if __name__ == '__main__':
    main()
