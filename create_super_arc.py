import logging
import hrsreduce

from hrsreduce.super_arc.super_arc import SuperArc

modes = ['MR','LR','HR']#,'MR','LR']
arms = ['Blu','Red']#'Red'

for mode in modes:
    for arm in arms:
    
#        SuperArc('2020','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20200101','20200630').create_superarc()
#        SuperArc('2020','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20200701','20201231').create_superarc()

#        SuperArc('2021','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20210101','20210630').create_superarc()
#        SuperArc('2021','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20210701','20211001').create_superarc()
#
#        SuperArc('2021','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20211105','20220112').create_superarc()
#        
#        SuperArc('2022','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20220210','20220630').create_superarc()
#        
#
#        SuperArc('2022','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20220701','20221231').create_superarc()
#
#        SuperArc('2023','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20230101','20230630').create_superarc()
#
#        SuperArc('2023','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20230701','20231231').create_superarc()
#        
#        SuperArc('2024','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20240101','20240816').create_superarc()
#
#        #2024 has different dates as the Tank was opened in August for FIF fault finding
#        SuperArc('2024','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20240823','20241231').create_superarc()
#
#        SuperArc('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20250101','20250630').create_superarc()
#        
#        SuperArc('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20250701','20251231').create_superarc()
        SuperArc('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20251010','20260301').create_superarc()
