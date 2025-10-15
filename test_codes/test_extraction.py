import logging
import hrsreduce

from hrsreduce.extraction.extraction import SpectralExtraction


if __name__ == '__main__':

    master_flat = "/Users/daniel/Desktop/SALT_HRS_DATA/Red/2023/0312/reduced/HR_Master_Flat_R20230312.fits"
    master_flat = "/Users/daniel/Desktop/SALT_HRS_DATA/Red/2024/0109/reduced/HS_Master_Flat_R20240109.fits"

    master_flat = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/0312/reduced/LR_Master_Flat_H20230312.fits"
    
    master_flat = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1201/reduced/HS_Master_Flat_H20231201.fits"
    order_file ="/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1201/reduced/HS_Orders_H20231201_Rect.csv"
    sci_file ="/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1204/reduced/bgoH202312040023.fits"
    
    master_flat = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1201/reduced/HS_Master_Flat_H20231201.fits"
    order_file ="/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1201/reduced/HS_Orders_H20231201_Rect.csv"
    sci_file ="/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1204/reduced/bgoH202312040023.fits"
    
    nights = {}
    nights['flat'] = '20230312'
    nights['flat'] = '20240109'
    base_dir = ("/Users/daniel/Desktop/SALT_HRS_DATA/")
    arm_colour = 'Blu'
    m='HR'
    plot=False
    extraction_method = 0
    extracted = SpectralExtraction(sci_file, master_flat,order_file, 'H',m, base_dir).extraction()
