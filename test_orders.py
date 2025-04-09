import logging
import hrsreduce

from hrsreduce.order_trace.order_trace import OrderTrace



master_flat = "/Users/daniel/Desktop/SALT_HRS_DATA/Red/2024/0109/reduced/HS_Master_Flat_R20240109.fits"

nights = {}
nights['flat'] = '20240109'
base_dir = ("/Users/daniel/Desktop/SALT_HRS_DATA/")
arm_colour = 'Red'
m='HS'
plot=False

#Create the Order file
order_file = OrderTrace(master_flat,nights,base_dir,arm_colour,m,plot).order_trace()
