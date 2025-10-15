import hrsreduce
from hrsreduce.ccf.ccf_EJB2 import CCF

sci_frame = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0730/reduced/bgoH202207300018.fits"
#sci_frame = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0730/reduced/H202207300018_product.fits"
sci_frame = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/1025/reduced/bgoH202210250017.fits"
#sci_frame = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0828/reduced/bgoH202208280016.fits"
ccf_mask = "/Users/daniel/Documents/Work/SALT_Pipeline/HRSReduce/hrsreduce/ccf/F9_espresso.txt"
ccf_mask = "/Users/daniel/Documents/Work/SALT_Pipeline/HRSReduce/hrsreduce/ccf/t8400.0logg4.44vsini50.txt"

sci_frame = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0828/reduced/bgoH202208280019.fits"

#sci_frame = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0729/reduced/bgoH202207290042.fits"
#ccf_mask = "/Users/daniel/Documents/Work/SALT_Pipeline/HRSReduce/hrsreduce/wave_cal/thar_list_int.txt"

CCF(sci_frame,ccf_mask).execute()


