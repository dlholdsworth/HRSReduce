import numpy as np
import logging

from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground

import matplotlib.pyplot as plt

def BkgAlg(raw, order_mask,logger):
    """ Background subtraction
    Args:
        raw: raw image to have background calcualted for
        order_mask: NPZ file holding the x and y pixel locations of the orders

    Returns:
        raw: input image with background subtracted.
        bkg: Background image
    """

    orders = np.load(order_mask,allow_pickle=True)
    x_pix,y_pix=orders['x_pix'],orders['y_pix']
    
    #Create a mask using the order file, and cover the frame edges too.
    mask_val = np.full((raw.shape[0],raw.shape[1]),False)
    mask_val[y_pix,x_pix] = True
    mask_val[:,max(x_pix)-20:raw.shape[1]+1] = True
    mask_val[0:20,:] = True
    mask_val[raw.shape[0]-20:,:] = True
    
    clip = SigmaClip(sigma=3.)
    est = MedianBackground()
    bkg = np.zeros_like(raw)
    t_box = "(30, 30)"
    t_fs = "(9, 9)"
    box = eval(t_box)  # box size for estimating background
    fs = eval(t_fs)    # window size for 2D low resolution median filtering
    
#    mask_val = order_masks[ffi].astype(bool)
    if logger:
        logger.info(f"Background Subtraction box_size: "+ t_box + ' filter_size: '+t_fs)

    bkg[:, :] = Background2D(raw, box, mask=mask_val, filter_size=fs, sigma_clip=clip, bkg_estimator=est).background
    raw = raw - bkg
    
    return raw, bkg
