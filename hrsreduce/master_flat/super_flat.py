import numpy as np
import logging
import glob
from astropy.io import fits
import os
from datetime import date, timedelta

from hrsreduce.L0_Corrections.level0corrections import L0Corrections
from hrsreduce.master_bias.master_bias import MasterBias, SubtractBias
from hrsreduce.master_flat.master_flat import MasterFlat


logger = logging.getLogger(__name__)

MODEFULL = {'HS': 'HIGH STABILITY', 'HR': 'HIGH RESOLUTION', 'MR': 'MEDIUM RESOLUTION', 'LR': 'LOW RESOLUTION'}


class SuperFlat():
    """
    Build a super-flat from HRS flat-field exposures spanning multiple nights.

    Copy of original with with these changes:

    - A range may cross 1 January: each night's biases and reduced files are taken
      from that night's own year (the original used the year passed in for every night).
    - A range may cross data folders: `roots` lists the base directories to search. Each 
      night's products are written under the root that holds its raw data. The super-flat 
      is written under `base_dir`.
    - Flats of any width are accepted (older Blue frames are 2098 or 2100 columns
      wide); only the number of rows must match. L0 trims each frame to its DATASEC.
      Frames whose header has no NAMPS (Blue before December 2014) are skipped,
      because L0 cannot read them.
    - Only frames with OBSTYPE "Flat field" are used.
    - `exclude_nights` leaves out nights whose flats are known to be unlike a normal
      flat (other lamp or fibre set-up, dark frames, one-off order positions).
    - At most `max_flats` frames are combined, spread evenly through the range, so long
      ranges fit in memory. The orders do not move within a range, so this loses no
      positional information.
    - Exactly the selected flats are bias-subtracted and combined; leftovers from
      earlier runs in the reduced folders are not picked up.
    - A night with no usable biases, or whose processing fails, is skipped with a
      warning instead of stopping the run. With no usable flats at all, nothing is
      written and None is returned.
    - The super-flat header gets SF_START, SF_END and SF_NIGHT (range, nights used)
      and create_superflat() returns the path of the file written.

    Parameters
    ----------
    year : str
        Year of `start`.
    base_dir : str
        Base directory (with trailing slash) the super-flat is written under.
    arm : str
        "Blu" or "Red".
    mode : str
        "HS", "HR", "MR" or "LR".
    start, end : str
        First and last night of the range, YYYYMMDD. The super-flat is named after `start`.
    plot : bool, optional
    roots : list of str, optional
        Base directories to search for raw data, in order. Defaults to [base_dir].
    exclude_nights : iterable of str, optional
        Nights (YYYYMMDD) to leave out.
    max_flats : int, optional
        Largest number of flat frames to combine (default 30).
    """

    def __init__(self, year, base_dir, arm, mode, start, end, plot=False, roots=None,
                 exclude_nights=(), max_flats=30):
        self.flat_propid = "CAL_FLAT"
        self.year = year
        self.base_dir = base_dir
        self.roots = list(roots) if roots else [base_dir]
        self.plot = plot
        self.arm = arm
        if self.arm == "Blu":
            self.sarm = "H"
            self.ax2 = 4102
        else:
            self.sarm = "R"
            self.ax2 = 4112
        self.s_date = start
        self.e_date = end
        self.mode = mode
        self.modefull = MODEFULL[mode]
        self.tn = '00000000'
        self.exclude = set(exclude_nights)
        self.max_flats = max_flats
        logger.info('Started {}'.format(self.__class__.__name__))

    @staticmethod
    def date_range(start_date, end_date):
        return [start_date + timedelta(days=n) for n in range(int((end_date - start_date).days) + 1)]

    def _raw_dir(self, night):
        """(root, raw directory) holding this night's raw data, or (None, None)."""
        for root in self.roots:
            d = f"{root}{self.arm}/{night[:4]}/{night[4:]}/raw/"
            if glob.glob(d + "*.fits"):
                return root, d
        return None, None

    def create_superflat(self):
        logger.info('Creating Super Flat file for {} arm, {} mode between {} and {}'.format(
            self.arm, self.mode, self.s_date, self.e_date))

        start = date(int(self.s_date[0:4]), int(self.s_date[4:6]), int(self.s_date[6:8]))
        end = date(int(self.e_date[0:4]), int(self.e_date[4:6]), int(self.e_date[6:8]))

        # 1. Find the flats, night by night
        candidates = []        # (night, root, raw_dir, file, naxis1)
        for d in self.date_range(start, end):
            night = d.strftime("%Y%m%d")
            if night in self.exclude:
                continue
            root, raw_dir = self._raw_dir(night)
            if raw_dir is None:
                continue
            for f in sorted(glob.glob(raw_dir + "*.fits")):
                try:
                    h = fits.getheader(f)
                except Exception:
                    continue
                if (h.get("PROPID") == self.flat_propid and h.get("OBSMODE") == self.modefull
                        and h.get("OBSTYPE") == "Flat field" and h.get("NAXIS2") == self.ax2):
                    if "NAMPS" not in h:
                        logger.warning("Skipping {}: no NAMPS in header, L0 cannot read it".format(f))
                        continue
                    candidates.append((night, root, raw_dir, f, h["NAXIS1"]))

        if not candidates:
            logger.error("No {} {} flats found between {} and {}; no Super Flat made".format(
                self.arm, self.mode, self.s_date, self.e_date))
            return None

        # 2. Keep at most max_flats frames, spread evenly through the range
        if len(candidates) > self.max_flats:
            keep = np.unique(np.round(np.linspace(0, len(candidates) - 1, self.max_flats)).astype(int))
            logger.info("Using {} of {} flats, spread through the range".format(len(keep), len(candidates)))
            candidates = [candidates[i] for i in keep]

        # 3. Per night: L0 on the flats, master bias from that night's biases, bias subtraction
        flat_files, nights_used = [], []
        for night in sorted(set(c[0] for c in candidates)):
            sel = [c for c in candidates if c[0] == night]
            root, raw_dir = sel[0][1], sel[0][2]
            widths = set(c[4] for c in sel)
            red_dir = raw_dir[:-4] + "reduced/"
            os.makedirs(red_dir, exist_ok=True)
            try:
                flats = L0Corrections({'flat': [c[3] for c in sel]}, {'flat': night}, self.tn,
                                      raw_dir, red_dir, root, self.sarm).run()['flat']

                b_files = []
                for f in sorted(glob.glob(raw_dir + "*.fits")):
                    try:
                        h = fits.getheader(f)
                    except Exception:
                        continue
                    if ((h.get("OBSTYPE") == "Bias" or h.get("CCDTYPE") == "Bias") and h.get("EXPTIME") == 0.
                            and h.get("NAXIS2") == self.ax2 and h.get("NAXIS1") in widths and "NAMPS" in h):
                        b_files.append(f)
                if not b_files:
                    logger.warning("No biases on {}: its {} flats are not used".format(night, len(sel)))
                    continue
                biases = L0Corrections({'bias': b_files}, {'bias': night}, self.tn,
                                       raw_dir, red_dir, root, self.sarm).run()
                master_bias = MasterBias(biases["bias"], raw_dir, red_dir, self.arm, night, False).create_masterbias()
                flat_files += SubtractBias(master_bias, {'flat': flats}, root, self.arm, night, "flat").subtract()
                nights_used.append(night)
            except Exception as e:
                logger.warning("Skipping {} after an error: {}".format(night, e))
                continue

        if not flat_files:
            logger.error("No usable {} {} flats between {} and {}; no Super Flat made".format(
                self.arm, self.mode, self.s_date, self.e_date))
            return None

        # 4. Combine
        nights = {'flat': self.s_date}
        master_file = MasterFlat(flat_files, nights, ' ', ' ', self.base_dir, self.arm, self.tn, self.mode,
                                 self.plot, super=True).create_masterflat()
        with fits.open(master_file, mode='update') as hdul:
            hdr = hdul[0].header
            hdr['SF_START'] = (self.s_date, 'First night of the Super Flat range')
            hdr['SF_END'] = (self.e_date, 'Last night of the Super Flat range')
            hdr['SF_NIGHT'] = (len(nights_used), 'Number of nights with flats used')
        logger.info("Super Flat written: {} ({} flats from {} nights)".format(
            master_file, len(flat_files), len(nights_used)))
        return master_file
