import numpy as np
import glob
from astropy.io import fits
import logging

def find_files(input_dir):
    """Find fits files in the given folder

    Parameters
    ----------
    input_dir : string
        directory to look for fits and fits.gz files in, may include bash style wildcards

    Returns
    -------
    files: array(string)
        absolute path filenames
    """
    files = sorted(glob.glob(input_dir + "/*.fits"))
    files = np.array(files)
    return files


def SortFiles(input_dir, logger, arm, mode):
    """
    Sort a set of fits files into different categories
    types are: bias, flat, wavecal, orderdef, spec

    Parameters
    ----------
    input_dir : str
        input directory containing the files to sort
    night : str
        observation night, possibly with wildcards
    mode : str
        instrument mode
    arm : str
        instrument arm
        
    Returns
    -------
    files_per_night : list[dict{str:dict{str:list[str]}}]
        a list of file sets, one entry per night, where each night consists of a dictionary with one entry per setting,
        each fileset has five lists of filenames: "bias", "flat", "arc", "lfc", "sci", organised in another dict
    nights_out : list[datetime]
        a list of observation times, same order as files_per_night
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
        
    for file in files:
        with fits.open(file) as hdul:
            hdr = hdul[0].header
            if hdr["NAXIS1"] == ax1 and hdr["NAXIS2"] == ax2:
                try:
                    if np.logical_and((hdr["OBSTYPE"] == "Bias" or hdr["CCDTYPE"] == "Bias") ,hdr["PROPID"] == "CAL_BIAS"):
                        bias_files.append(file)
                        continue
                except KeyError:
                    continue
                    
                if hdr["FIFPORT"] == mode:
                    if np.logical_and(hdr["OBSTYPE"] == "Flat field",hdr["PROPID"] == "CAL_FLAT"):
                        flat_files.append(file)
                    elif np.logical_and(hdr["OBSTYPE"] == "Arc",(hdr["PROPID"] == "CAL_ARC" or hdr["PROPID"] == "CAL_STABLE")):
                        arc_files.append(file)
                    elif hdr["OBSTYPE"] == "Science":
                        sci_files.append(file)
                    elif hdr["OBSTYPE"] == "Comb":
                        lfc_files.append(file)
                    else:
                        logger.debug("File %s does not match an expected value in %s, %s, %s or %s", file,"Flat field","Arc","Science","Comb")
                if mode == 'HS':
                    if np.logical_and(hdr["OBSTYPE"] == "Arc", hdr["PROPID"] == "CAL_STABLE"):
                        if hdr["I2STAGE"] == "Reference Fibre":
                            hs_arc_files.append(file)

    if mode == "HS":
        return bias_files,flat_files,hs_arc_files,lfc_files,sci_files
    else:
        return bias_files,flat_files,arc_files,lfc_files,sci_files
