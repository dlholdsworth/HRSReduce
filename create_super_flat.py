import logging
import hrsreduce

from hrsreduce.master_flat.super_flat import SuperFlat

arms = ['Red']
modes = ['HR','MR','LR']
modes = ['HS']

for mode in modes:
    for arm in arms:
    
#        SuperFlat('2015','/Volumes/SALT_DATA/',arm, mode,'20150101','20150630').create_superflat()
#        SuperFlat('2015','/Volumes/SALT_DATA/',arm, mode,'20150701','20151231').create_superflat()
#
#        SuperFlat('2022','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20220210','20220630').create_superflat()
#        SuperFlat('2022','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20220701','20221231').create_superflat()
#
#        SuperFlat('2023','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20230101','20230616').create_superflat()
#        SuperFlat('2023','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20230624','20231231').create_superflat()

        #This is where the flats were taken with 9 exposures for HS. End up running out of memory, so calculate a super in smaller chunks
        if mode == 'HS':
#                SuperFlat('2024','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20240101','20240229').create_superflat()
#                SuperFlat('2024','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20240301','20240430').create_superflat()
#                SuperFlat('2024','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20240501','20240630').create_superflat()
#
#                SuperFlat('2024','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20240701','20240816').create_superflat()
#                SuperFlat('2024','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20240823','20241031').create_superflat()
#                SuperFlat('2024','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20241101','20241231').create_superflat()
#                
                SuperFlat('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20250101','20250228').create_superflat()
                SuperFlat('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20250301','20250326').create_superflat()
#                SuperFlat('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20250519','20250519').create_superflat()
                SuperFlat('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20250520','20250831').create_superflat()
                SuperFlat('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20250901','20251031').create_superflat()
                SuperFlat('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20251101','20251211').create_superflat()
                SuperFlat('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20251220','20260115').create_superflat()
                SuperFlat('2026','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20260122','20260214').create_superflat()
        
        else:
            SuperFlat('2024','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20240101','20240816').create_superflat()
            SuperFlat('2024','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20240823','20241231').create_superflat()
            
            SuperFlat('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20250101','20250326').create_superflat()
            SuperFlat('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20250519','20250519').create_superflat()
            
            SuperFlat('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20250520','20251211').create_superflat()
            SuperFlat('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20251220','20260115').create_superflat()
            
            SuperFlat('2026','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20260122','20260214').create_superflat()
