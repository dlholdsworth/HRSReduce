import logging
import hrsreduce

from hrsreduce.master_flat.super_flat import SuperFlat

SuperFlat('2022','/Users/daniel/Desktop/SALT_HRS_DATA/','Red', 'HR','20220101','20220630').create_superflat()

SuperFlat('2022','/Users/daniel/Desktop/SALT_HRS_DATA/','Red', 'HR','20220701','20221231').create_superflat()

#SuperFlat('2023','/Users/daniel/Desktop/SALT_HRS_DATA/','Blu', 'LR','20230101','20230630').create_superflat()

#SuperFlat('2023','/Users/daniel/Desktop/SALT_HRS_DATA/','Blu', 'LR','20230701','20231231').create_superflat()

#2024 has different dates as the Tank was opened in August for FIF fault finding
#SuperFlat('2024','/Users/daniel/Desktop/SALT_HRS_DATA/','Blu', 'LR','20240823','20241231').create_superflat()

#SuperFlat('2024','/Users/daniel/Desktop/SALT_HRS_DATA/','Blu', 'LR','20240101','20240816').create_superflat()

#SuperFlat('2025','/Users/daniel/Desktop/SALT_HRS_DATA/','Blu', 'LR','20250101','20250630').create_superflat()
