import numpy as np
import glob
from astropy.io import fits
import logging

def find_files(input_dir):
    """
    Find all FITS files in the specified directory.

    This helper routine searches the input directory for files with the
    `.fits` extension, sorts them alphabetically, and returns the result as a
    NumPy array.

    Parameters
    ----------
    input_dir : str
        Directory to search for FITS files.

    Returns
    -------
    numpy.ndarray
        Sorted array of full file paths to FITS files found in the directory.
    """
    files = sorted(glob.glob(input_dir + "/*.fits"))
    files = np.array(files)
    return files


def SortFiles(input_dir, logger, arm, mode, CAL_RVST=False):
    """
    Sort HRS FITS files into calibration and science categories.

    This routine scans all FITS files in the input directory, checks their
    detector dimensions and FITS header metadata, and classifies them into
    lists of bias, flat, arc, LFC, and science exposures appropriate for the
    requested arm and observing mode.

    Files are first filtered by detector size so that only frames matching the
    selected spectrograph arm are considered. They are then separated by
    `OBSTYPE`, `PROPID`, `OBSMODE`, and, where relevant, `I2STAGE`.

    The returned file categories are:

        - bias
        - flat
        - arc
        - lfc
        - sci

    In High-Stability (`HS`) mode, the arc output is replaced by the subset of
    arc-like reference-fibre files appropriate to that mode.

    Parameters
    ----------
    input_dir : str
        Directory containing FITS files to classify.
    logger : logging.Logger
        Logger used for debug messages when files do not match expected
        categories.
    arm : str or list
        Spectrograph arm identifier. Only the first element/value is used.
    mode : str
        Observing mode, e.g. "HS", "HR", "MR", or "LR".
    CAL_RVST : bool, optional
        If True, only science files with `PROPID == "CAL_RVST"` are retained.

    Returns
    -------
    tuple
        Five lists of file paths:
            - bias files
            - flat files
            - arc files (or HS reference-fibre arcs in HS mode)
            - LFC files
            - science files
    """
    
    files = find_files(input_dir)
    bias_files = []
    flat_files = []
    arc_files = []
    lfc_files = []
    sci_files = []
    hs_arc_files = []

    if arm[0] == "H":
        ax1 = 2074
        ax2 = 4102
    if arm[0] == "R":
        ax1 = 4122
        ax2 = 4112
    
    if mode == "HS":
        modefull = 'HIGH STABILITY'
    if mode == "HR":
        modefull = 'HIGH RESOLUTION'
    if mode == "MR":
        modefull = 'MEDIUM RESOLUTION'
    if mode == "LR":
        modefull = 'LOW RESOLUTION'
        
    for file in files:
        with fits.open(file) as hdul:
            hdr = hdul[0].header
            if hdr["NAXIS1"] == ax1 and hdr["NAXIS2"] == ax2:
                try:
                    if np.logical_and((hdr["OBSTYPE"] == "Bias" or hdr["CCDTYPE"] == "Bias") ,(hdr["PROPID"] == "CAL_BIAS" or hdr["PROPID"] == "ENG_HRS")):
                        bias_files.append(file)
                        continue
                except KeyError:
                    continue
                    
                if hdr["OBSMODE"] == modefull:
                    if np.logical_and(hdr["OBSTYPE"] == "Flat field",hdr["PROPID"] == "CAL_FLAT"):
                        flat_files.append(file)
                    elif np.logical_and(hdr["OBSTYPE"] == "Arc",(hdr["PROPID"] == "CAL_ARC" or hdr["PROPID"] == "CAL_STABLE")):
                        arc_files.append(file)
                    elif hdr["OBSTYPE"] == "Science":
                        if CAL_RVST:
                            if hdr["PROPID"] == "CAL_RVST":
                                sci_files.append(file)
                        else:
                            sci_files.append(file)
                    elif hdr["OBSTYPE"] == "Arc" and hdr["I2STAGE"] == "ThAr->Fibre O" :
                        lfc_files.append(file)
                    else:
                        logger.debug("File %s does not match an expected value in %s, %s, %s or %s", file,"Flat field","Arc","Science","Comb")
                if mode == 'HS':
                    sci_files = []
                    if np.logical_and(hdr["OBSTYPE"] == "Arc", (hdr["PROPID"] == "ENG_HRS" or hdr["PROPID"] == "CAL_STABLE")):
                        if hdr["I2STAGE"] == "Reference Fibre":
                            hs_arc_files.append(file)

    if mode == "HS":
        return bias_files,flat_files,hs_arc_files,lfc_files,sci_files
    else:
        return bias_files,flat_files,arc_files,lfc_files,sci_files
