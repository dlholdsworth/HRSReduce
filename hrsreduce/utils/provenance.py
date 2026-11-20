"""Recipe keys of the cached products.

The pipeline reuses a product it finds on disk (master bias, master flat, order trace, rectified images, variance,
Master Wave) instead of making it again. To stop products made by older code from being reused, each product carries
a recipe key: the version of the last code change that alters it. A product whose key is missing (made before 1.3.7)
or differs from the value below is rebuilt.

When a change alters a product, set its value here to the new version; products made before are then rebuilt on the
next run. A version bump that does not alter a product leaves its key alone, so nothing is redone needlessly.

Products in electrons (master bias, master flat, Master Wave) are also rebuilt when the gain they were made with
differs from the gain L0Corrections applies now (file_is_current with `gain`), so a change of
hrsreduce/L0_Corrections/gain_table.csv redoes only the products of the nights it changes.
"""
from astropy.io import fits

from hrsreduce import __version__

#            kind      header key   recipe   what made it change
RECIPES = {'bias':  ('RCPBIAS',  '1.3.3'),  # read noise from bias pairs (1.3.2); noise-model stacking (1.3.3)
           'flat':  ('RCPFLAT',  '1.3.6'),  # noise-model stacking (1.3.3), flat check (1.3.5), no red column-minimum background (1.3.6)
           'trace': ('RCPTRACE', '1.3.7'),  # reference tracer (1.3.6); top-pair span (1.3.7)
           'rect':  ('RCPRECT',  '1.3.7'),  # trace taken at the source column, real X1/X2 (1.3.7)
           'var':   ('RCPVAR',   '1.3.7'),  # no flat term; the master flat's own VAR = UNC^2 (1.3.7)
           'wave':  ('RCPWAVE',  '1.3.7'),  # 2027 HR/MR line lists (1.3.1); rectification fix and top-pair span (1.3.7)
           }


def stamp(header, kind):
    """Write the recipe key of `kind`, and the pipeline version, into a FITS header."""
    key, recipe = RECIPES[kind]
    header[key] = (recipe, f'hrsreduce recipe of the {kind} step')
    header['HRSVERS'] = (str(__version__), 'hrsreduce version that wrote this')


def is_current(header, kind):
    """True when the header carries the current recipe key of `kind`."""
    key, recipe = RECIPES[kind]
    return str(header.get(key, '')).strip() == recipe


def gain_of(header):
    """The gain (e-/ADU) a frame or product was made with: AVG_GAIN, else the first GAIN value (None if neither)."""
    for key in ('AVG_GAIN', 'GAIN'):
        try:
            return float(str(header[key]).split()[0])
        except (KeyError, ValueError, IndexError):
            continue
    return None


def frame_gain(path, ext=0):
    """gain_of() of a FITS file (None if it cannot be read)."""
    try:
        return gain_of(fits.getheader(path, ext))
    except Exception:
        return None


def file_is_current(path, kind, ext=0, gain=None):
    """True when extension `ext` of the FITS file carries the current recipe key of `kind` (False if unreadable) and,
    when `gain` is given, was made with that gain (to 1e-4)."""
    try:
        header = fits.getheader(path, ext)
    except Exception:
        return False
    if not is_current(header, kind):
        return False
    if gain is None:
        return True
    made = gain_of(header)
    return made is not None and abs(made - gain) <= 1e-4 * abs(gain)
