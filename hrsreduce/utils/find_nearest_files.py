import arrow
import os.path
import glob
import logging

from .sort_files import SortFiles

def FindNearestFiles(type,night,m,base_dir,arm_colour,logger):
    """
    Locate the nearest available calibration files when none exist for a target night.

    This helper routine searches the nightly directory structure for calibration
    frames (bias, flat, arc, or LFC) when none are present for the requested
    observing night. The search proceeds outward in time, alternating between
    earlier and later nights, until suitable files are found.

    The routine relies on the `SortFiles` utility to classify files within each
    nightly raw-data directory and returns the first set of matching calibration
    files encountered.

    Parameters
    ----------
    type : str
        Calibration type to search for. Supported values are "Bias", "Flat",
        "Arc", or "LFC".
    night : str
        Target observing night in YYYYMMDD format.
    m : str
        Observing mode used when filtering files.
    base_dir : str
        Base reduction directory containing the nightly raw-data structure.
    arm_colour : str
        Spectrograph arm directory name ("Blu" or "Red").
    logger : logging.Logger
        Logger used for status messages during the search.

    Returns
    -------
    tuple
        A tuple containing:
            - list of matching calibration files
            - the observing night (YYYYMMDD) from which the files were selected
    """
    
    if type == "Bias":
        idx = 0
    if type == "Flat":
        idx = 1
    if type== "Arc":
        idx = 2
    if type == "LFC":
        idx = 3
        
    if arm_colour == "Blu":
        arm = ["H"]
    if arm_colour == "Red":
        arm = ["R"]

    files = []
    prev_night = night
    next_night = night
    while not files:
        prev_night = arrow.get(prev_night).shift(days=-1).format('YYYYMMDD')
        prev_year=prev_night[0:4]
        prev_mmdd=prev_night[4:8]
        prev_data_location = os.path.join(base_dir, arm_colour+'/'+prev_year+'/'+prev_mmdd+'/raw/')
        result = SortFiles(prev_data_location,logger,arm,mode=m)
        files = result[idx]
        if idx == 1 and len(files)<3:
            files = []
        else:
            if len(files)>0:
                logger.info(
                f"Using %s files found in folder: %s\n", type,
                prev_data_location,
                )
                files_night = str(prev_year+prev_mmdd)
                break

        next_night = arrow.get(next_night).shift(days=+1).format('YYYYMMDD')
        next_year=next_night[0:4]
        next_mmdd=next_night[4:8]
        next_data_location = os.path.join(base_dir, arm_colour+'/'+next_year+'/'+next_mmdd+'/raw/')
        result = SortFiles(next_data_location,logger,arm,mode=m)
        files = result[idx]
        if idx == 1 and len(files)<3:
            files = []
        else:
            if len(files)>0:
                logger.info(
                f"Using %s files found in folder: %s\n", type,
                next_data_location,
                )
                files_night = str(next_year+next_mmdd)
                break
            else:
                continue
    return files,files_night
