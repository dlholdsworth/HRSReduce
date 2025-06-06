import logging
import hrsreduce
import time
from hrsreduce.extraction.order_rectification import OrderRectification


if __name__ == '__main__':
    start = time.time()
    master_flat = "/Users/daniel/Desktop/SALT_HRS_DATA/Red/2023/0312/reduced/HR_Master_Flat_R20230312.fits"
    master_flat = "/Users/daniel/Desktop/SALT_HRS_DATA/Red/2024/0109/reduced/HS_Master_Flat_R20240109.fits"

    master_flat = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/0312/reduced/LR_Master_Flat_H20230312.fits"
    master_flat = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/0701/reduced/HS_Master_Flat_H20230701.fits"
    sci_file = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1018/reduced/bgoH202310180013.fits"
    order_file = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/0701/reduced/HS_Orders_H20230701.csv"
    nights = {}
    nights['flat'] = '20230312'
    nights['flat'] = '20231201'
    base_dir = ("/Users/daniel/Desktop/SALT_HRS_DATA/")
    arm_colour = 'Blu'
    m='HS'
    plot=False
    extraction_method = 0
    extracted = OrderRectification(sci_file, master_flat,order_file,'H',m,base_dir).perform()

    end = time.time()
    print("TIME ELAPSED",end - start)
