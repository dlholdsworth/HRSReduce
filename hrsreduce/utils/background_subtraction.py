import numpy as np
import pandas as pd
import logging

from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground

import matplotlib.pyplot as plt

def BkgAlg(raw, order_mask,arm,mode,logger):
    """ Background subtraction
    Args:
        raw: raw image to have background calcualted for
        order_mask: NPZ file holding the x and y pixel locations of the orders

    Returns:
        raw: input image with background subtracted.
        bkg: Background image
    """
    
    logger.info("Subtracting Background...")
    
    if arm == 'Red':
        t_box = "(9,9)"
        t_fs = "(79,79)"
    if arm == 'Blu':
        t_box = "(10,16)"
        t_fs = "(29,29)"
    
    #Create a mask using the order file, and cover the frame edges too.
    mask_val = np.full((raw.shape[0],raw.shape[1]),False)
    
    if order_mask:
        order_trace_data = pd.read_csv(order_mask, header=0, index_col=0)
        coeffs = order_trace_data.values
    
    if mode != 'LR':
        for ord in range(0,coeffs.shape[0],2):
            x=np.arange(coeffs[ord][8],coeffs[ord][9])
            ord_cen = coeffs[ord][0]+coeffs[ord][1]*x + coeffs[ord][2]*x**2 + coeffs[ord][3]*x**3 + coeffs[ord][4]*x**4 + coeffs[ord][5]*x**5
            for x_pix in x:
                x_pix = int(x_pix)
                mask_val[int(ord_cen[x_pix])-int(coeffs[ord][6]):int(ord_cen[x_pix])+int(coeffs[ord][7]),x_pix] = True
    else:
        for ord in range(1,coeffs.shape[0],2):
            x=np.arange(coeffs[ord][8],coeffs[ord][9])
            ord_cen = coeffs[ord][0]+coeffs[ord][1]*x + coeffs[ord][2]*x**2 + coeffs[ord][3]*x**3 + coeffs[ord][4]*x**4 + coeffs[ord][5]*x**5
            for x_pix in x:
                x_pix = int(x_pix)
                mask_val[int(ord_cen[x_pix])-int(coeffs[ord][6]):int(ord_cen[x_pix])+int(coeffs[ord][7]),x_pix] = True

    mask_val[0:20,:] = True
    mask_val[:,raw.shape[0]-20:raw.shape[0]] = True
    
    clip = SigmaClip(sigma=3.)
    est = MedianBackground()
    bkg = np.zeros_like(raw)
    box = eval(t_box)  # box size for estimating background
    fs = eval(t_fs)    # window size for 2D low resolution median filtering
    
    bkg[:, :] = Background2D(raw, box, mask=mask_val, filter_size=fs, sigma_clip=clip, bkg_estimator=est).background
    raw = raw - bkg
    
    return raw, bkg
