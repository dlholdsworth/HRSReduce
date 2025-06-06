import numpy as np

import hrsreduce

from hrsreduce.extraction.slit_correction import SlitCorrection
import time


if __name__ == '__main__':
    start = time.time()

    sci_file ="/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1018/reduced/bgoH202310180013.fits"
    order_file = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/0701/reduced/HS_Orders_H20230701_Rect.csv"
    master_flat = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/0701/reduced/HS_Master_Flat_H20230701.fits"
    m='HS'
    arm = 'H'
    base_dir = "/Users/daniel/Desktop/SALT_HRS_DATA/"
    header_ext = 'RECT'
    yyyymmdd = '20230704'
    plot=True
    extracted = SlitCorrection(sci_file,header_ext, order_file, arm,m, base_dir,yyyymmdd,plot=plot).correct()

    end = time.time()
    print("TIME ELAPSED",end - start)
