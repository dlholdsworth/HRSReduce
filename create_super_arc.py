import logging
import hrsreduce

from hrsreduce.super_arc.super_arc import SuperArc

modes = ['HS']
arms = ['Blu','Red']

for mode in modes:
    for arm in arms:
 
#        SuperArc('2022','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20220210','20220630').create_superarc()
#        SuperArc('2022','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20220701','20221231').create_superarc()
#
#        SuperArc('2023','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20230101','20230616').create_superarc()
#        SuperArc('2023','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20230624','20231231').create_superarc()
#        
#        SuperArc('2024','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20240101','20240816').create_superarc()
#        SuperArc('2024','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20240823','20241231').create_superarc()
#        
#        SuperArc('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20250101','20250326').create_superarc()
#        SuperArc('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20250519','20250519').create_superarc()
        
        SuperArc('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20250520','20251211').create_superarc()
        SuperArc('2025','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20251220','20260115').create_superarc()
        
        SuperArc('2026','/Users/daniel/Desktop/SALT_HRS_DATA/',arm, mode,'20260122','20260214').create_superarc()
