import hrsreduce.utils.background_subtraction as BKG
import numpy as np
from astropy.io import fits
import logging

logger = 'tmp'

data=fits.getdata("/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/0703/reduced/bgoH202307030029.fits")
order_mask="/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/0701/reduced/HR_Orders_H20230701.csv"
arm = 'H'

res = BKG.BkgAlg(data,order_mask,arm,logger)
