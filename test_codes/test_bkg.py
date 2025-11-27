import hrsreduce.utils.background_subtraction as BKG
import numpy as np
from astropy.io import fits
import logging

logger = 'tmp'

data=fits.getdata("/Users/daniel/Desktop/SALT_HRS_DATA/Red/2025/1117/reduced/bgoR202511170019.fits")


order_mask="/Users/daniel/Desktop/SALT_HRS_DATA/Red/2025/1114/reduced/HR_Orders_R20251114.npz"

res = BKG.background_subtraction(data,order_mask,logger)
