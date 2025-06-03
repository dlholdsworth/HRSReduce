import numpy as np
from astropy.io import fits
import logging
from astroscrappy import detect_cosmics
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

def CosmicRayMasking(all_files, arm, verbose=True):
    """Masks cosmic rays from input file.
    """
    sigclip = 6.
    sigfrac = 0.3
    f_types = ['sci', 'lfc','flat']

    for type in f_types:
        files = all_files[type]
        for file in files:
            with fits.open(file,mode='update') as hdu:
                if type == "flat":
                    sigclip = 10
                gain = 1
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

                if verbose:
                    N = recov_mask.sum()
                    logger.info('Number of CR masked pixels {}:'.format(N))
                
                hdu.flush()
    return files
