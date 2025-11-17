from hrsreduce.utils.frame_stacker import FrameStacker
from astropy.io import fits
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import logging
logger = logging.getLogger(__name__)

files = sorted(glob.glob("/Users/daniel/Desktop/SALT_HRS_DATA/Red/2025/1107/raw/R20251107002[4-6].fit"))

data = []

for file in files:
    with fits.open(file) as hdu:
        data.append(hdu[0].data)

fs = FrameStacker(data,2.1,logger)
avg,var,cnt,unc = fs.compute()
plt.imshow(avg,vmin=900,vmax=950)
plt.show()

plt.imshow(cnt)
plt.show()
