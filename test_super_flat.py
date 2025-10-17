import logging
import hrsreduce

from hrsreduce.master_flat.super_flat import SuperFlat

arms = ['Red']
modes = ['HS']

for mode in modes:
    for arm in arms:

        #SuperFlat('2022','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20220101','20220630').create_superflat()

        #SuperFlat('2022','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20220701','20221231').create_superflat()

        #SuperFlat('2023','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20230101','20230630').create_superflat()

        #SuperFlat('2023','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20230701','20231231').create_superflat()

        #This is where the flats were taken with 9 exposures for HS. End up running out of memory, so calculate a super in smaller chunks
        if mode == 'HS':
                SuperFlat('2024','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20240101','20240229').create_superflat()
                SuperFlat('2024','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20240301','20240430').create_superflat()
                SuperFlat('2024','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20240501','20240630').create_superflat()
                #2024 has different dates as the Tank was opened in August for FIF fault finding
                SuperFlat('2024','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20240701','20240816').create_superflat()
                SuperFlat('2024','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20240823','20241031').create_superflat()
                SuperFlat('2024','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20241101','20241231').create_superflat()
                
                SuperFlat('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20250101','20250229').create_superflat()
                SuperFlat('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20250301','20250430').create_superflat()
                SuperFlat('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20250501','20250630').create_superflat()
                SuperFlat('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20250701','20250831').create_superflat()
                #SuperFlat('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20250901','20251031').create_superflat()
                #SuperFlat('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20251101','20251231').create_superflat()
            
        
        else:
            SuperFlat('2024','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20240101','20240816').create_superflat()
        
            #2024 has different dates as the Tank was opened in August for FIF fault finding
            SuperFlat('2024','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20240823','20241231').create_superflat()

            SuperFlat('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20250101','20250630').create_superflat()
