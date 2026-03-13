import numpy as np
from astropy.io import fits
import logging
from astroscrappy import detect_cosmics
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

def CosmicRayMasking(all_files, arm, verbose=True):
    """
    Apply cosmic-ray rejection to science frames using the Astro-SCRAPPY algorithm.

    This routine scans a set of science exposures and applies the Laplacian
    cosmic-ray detection algorithm implemented in `astroscrappy.detect_cosmics`.
    Detected cosmic-ray pixels are replaced with cleaned values and the modified
    image is written back to the original FITS file.

    To avoid corrupting bright stellar spectra or saturated regions, the routine
    first estimates the mean signal level in the central detector column. If the
    mean count level exceeds a fixed threshold, the cosmic-ray cleaning step is
    skipped because strong stellar flux can cause false detections along order
    edges.

    Parameters
    ----------
    all_files : dict
        Dictionary containing lists of files grouped by frame type. The routine
        currently processes entries under the key `'sci'`.
    arm : str
        Spectrograph arm identifier ("H" or "R"). Used to select a default
        read-noise value if the header keyword is missing.
    verbose : bool, optional
        If True, report the number of detected cosmic-ray pixels for each frame.

    Returns
    -------
    list
        List of processed science files.
    """
    
    sigclip = 10.
    sigfrac = 0.3
    f_types = ['sci']

    for type in f_types:
        files = all_files[type]
        for file in files:
            with fits.open(file,mode='update') as hdu:
                gain = 1
                #Arbitary cut off as to whether to preceed or not. Bright stars cause issues for detection on order edges
                img_mid = int(hdu[0].data.shape[1]/2)
                mn_count = np.sum(hdu[0].data[:,img_mid:img_mid+1])/hdu[0].data.shape[0]
                if mn_count < 1000:
                    try:
                        readnoise = float(hdu[0].header["RONOISE"])
                    except:
                        if arm == "H":
                            readnoise = 4.2
                        if arm == "R":
                            readnoise = 3.6

                    # see astroscrappy docs: this pre-determined background image can
                    # improve performance of the algorithm.
                    inbkg = None

                    # detect cosmic rays in the resulting image
                    recov_mask, clean_img = detect_cosmics(
                        hdu[0].data,
                        inbkg=inbkg,
                        gain=gain,
                        readnoise=readnoise,
                        sigclip=sigclip,
                        sigfrac=sigfrac
                    )
                    hdu[0].data = clean_img
                    hdu[0].header['CRCLEAN'] = ("True", "Cosmic Ray cleaning applied")

                    if verbose:
                        N = recov_mask.sum()
                        logger.info('Number of CR masked pixels {} in {}'.format(N, file))
                    
                    hdu.flush()
                else:
                    hdu[0].header['CRCLEAN'] = ("False", "Cosmic Ray cleaning applied")
                    logger.info('CR process skipped, likely to interfere with science for file {}'.format(file))
                    hdu.flush()
    return files
