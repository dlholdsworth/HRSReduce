import hrsreduce.utils.background_subtraction as BKG
import numpy as np
from astropy.io import fits
import logging

logger = 'tmp'

data=fits.getdata("/Users/daniel/Desktop/SALT_HRS_DATA/Red/2024/0109/reduced/goR202401090026.fits")


order_mask="/Users/daniel/Desktop/SALT_HRS_DATA/Red/2024/0109/reduced/HS_Orders_Red20240109.npz"

res = BKG.background_subtraction(data,order_mask,logger)
