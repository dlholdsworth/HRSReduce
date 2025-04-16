import logging
import hrsreduce

from hrsreduce.order_trace.order_trace import OrderTrace



master_flat = "/Users/daniel/Desktop/SALT_HRS_DATA/Red/2023/0312/reduced/HR_Master_Flat_R20230312.fits"
master_flat = "/Users/daniel/Desktop/SALT_HRS_DATA/Red/2024/0109/reduced/HS_Master_Flat_R20240109.fits"

master_flat = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/0312/reduced/LR_Master_Flat_H20230312.fits"
#master_flat = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2024/0109/reduced/HS_Master_Flat_H20240109.fits"
nights = {}
nights['flat'] = '20230312'
#nights['flat'] = '20240109'
base_dir = ("/Users/daniel/Desktop/SALT_HRS_DATA/")
arm_colour = 'Blu'
m='LR'
plot=False

#Create the Order file
order_file = OrderTrace(master_flat,nights,base_dir,arm_colour,m,plot).order_trace()
