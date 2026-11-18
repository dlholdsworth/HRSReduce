import arrow
import os.path
import glob
import logging

from .sort_files import SortFiles
from .flat_check import usable_flats

MAX_SEARCH_DAYS = 365     # nights searched either side of the target night

def FindNearestFiles(type,night,m,base_dir,arm_colour,logger,new_flat=False):
    """
    Locate the nearest available calibration files when none exist for a target night.

    This helper routine searches the nightly directory structure for calibration
    frames (bias, flat, arc, or LFC) when none are present for the requested
    observing night. The search proceeds outward in time, alternating between
    earlier and later nights, for at most MAX_SEARCH_DAYS on either side.
    Saturated flats and flats without light are not counted; a flat set needs at least three usable frames.

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
        or ([], None), with an error logged, when nothing is found within MAX_SEARCH_DAYS.
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

    for step in range(1, MAX_SEARCH_DAYS + 1):
        for shift in (-step, step):          # the night before first, then the night after
            cand = arrow.get(night).shift(days=shift).format('YYYYMMDD')
            data_location = os.path.join(base_dir, arm_colour+'/'+cand[0:4]+'/'+cand[4:8]+'/raw/')
            if not os.path.isdir(data_location):
                continue
            result = SortFiles(data_location,logger,arm,mode=m,new_flat=new_flat)
            files = result[idx]
            if idx == 1:
                files = usable_flats(files, logger)
                if len(files) < 3:
                    continue
            if len(files) > 0:
                logger.info(
                f"Using %s files found in folder: %s\n", type,
                data_location,
                )
                return files, cand

    logger.error(f"No {type} files for the {arm_colour} arm ({m}) within {MAX_SEARCH_DAYS} days of {night}")
    return [], None
