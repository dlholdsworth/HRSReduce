"""Checks on raw flat-field frames before they are stacked.

A flat is not used when it is
  * saturated: more than MAX_SATURATED pixels at or above SATURATION_ADU in the raw frame. The raw frames are 16 bit
    (BZERO 32768), so the ADC stops at 65535 ADU. Isolated pixels at the limit (cosmic rays, hot pixels) are common in
    good flats: 0-6 per frame in the archive. Saturated flats have thousands to hundreds of thousands.
  * without light: the standard deviation of its data section, in electrons, is MIN_FLAT_SPREAD or less. This is
    the test MasterFlat.create_masterflat applies to the bias-subtracted frame (a constant bias does not change a
    standard deviation). Good flats spread by about 900 e- (HS 30 s) to 3500 e- (LR 10 s); the HS flats of Blue
    2018-04-24 spread by 78 e-.
Checking at selection lets reduce.py and FindNearestFiles go to the nearest night with usable flats.
"""
import numpy as np
from astropy.io import fits

SATURATION_ADU = 65000      # raw ADU; the 16-bit ADC limit is 65535
MAX_SATURATED = 100         # pixels at the limit allowed in a usable flat
MIN_FLAT_SPREAD = 200.      # e-; a flat must spread by more than this (also used by MasterFlat)


def _section(sec):
    """FITS section '[x1:x2,y1:y2]' (1-based, inclusive) as numpy slices, or None."""
    try:
        x, y = str(sec).strip().strip('[]').split(',')[:2]
        x1, x2 = (int(v) for v in x.split(':'))
        y1, y2 = (int(v) for v in y.split(':'))
        return slice(y1 - 1, y2), slice(x1 - 1, x2)
    except Exception:
        return None


def flat_quality(file):
    """(saturated pixels, spread in e-) of a raw flat."""
    with fits.open(file) as hdul:
        hdr, data = hdul[0].header, hdul[0].data
        n_sat = int(np.count_nonzero(data >= SATURATION_ADU))
        sec = _section(hdr.get('DATASEC'))
        light = data[sec] if sec is not None else data
        try:
            gain = float(str(hdr.get('GAIN', '1')).split()[0])
        except ValueError:
            gain = 1.
        spread = float(np.nanstd(light.astype(np.float32))) * gain
    return n_sat, spread


def usable_flats(files, logger):
    """The flats of `files` that are neither saturated nor without light. Each one left out is named in a warning."""
    good = []
    for file in files:
        n_sat, spread = flat_quality(file)
        if n_sat > MAX_SATURATED:
            logger.warning("Flat %s is saturated (%d pixels at or above %d ADU); not used", file, n_sat, SATURATION_ADU)
        elif spread <= MIN_FLAT_SPREAD:
            logger.warning("Flat %s has too little light (spread %.0f e-, needs more than %.0f); not used",
                           file, spread, MIN_FLAT_SPREAD)
        else:
            good.append(file)
    if files and not good:
        logger.warning("None of the %d flats in %s is usable", len(files), files[0].rsplit('/', 1)[0])
    return good
