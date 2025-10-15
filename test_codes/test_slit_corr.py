import numpy as np

import hrsreduce

from hrsreduce.extraction.slit_correction import SlitCorrection
import time


if __name__ == '__main__':
    start = time.time()

    sci_file ="/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1018/reduced/bgoH202310180013.fits"
    order_file_rect = "/Users/daniel/Desktop/SALT_HRS_DATA/Red/2022/Super_Arcs/HR_Super_Arc_R20220701_Orders_Rect.csv"
    master_flat = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/0701/reduced/HS_Master_Flat_H20230701.fits"
    super_arc = "/Users/daniel/Desktop/SALT_HRS_DATA/Red/2022/Super_Arcs/HR_Super_Arc_R20220701.fits"
    
    m='HR'
    arm = ['R']
    base_dir = "/Users/daniel/Desktop/SALT_HRS_DATA/"
    header_ext = 'RECT'
    yyyymmdd = '20220729'
    plot=True
    SlitCorrection(super_arc,header_ext,order_file_rect,arm[0],m,base_dir,yyyymmdd,plot=plot,super_arc=super_arc).correct()

    end = time.time()
    print("TIME ELAPSED",end - start)
