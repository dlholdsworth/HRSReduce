import arrow
import os.path
import glob
import logging

from .sort_files import SortFiles

def FindNearestFiles(type,night,m,base_dir,arm_colour,logger):
    
    if type == "Bias":
        idx = 0
    if type == "Flat":
        idx = 1
    if type== "Arc":
        idx = 2
    if type == "LFC":
        idx = 3

    files = []
    prev_night = night
    next_night = night
    while not files:
        prev_night = arrow.get(prev_night).shift(days=-1).format('YYYYMMDD')
        prev_year=prev_night[0:4]
        prev_mmdd=prev_night[4:8]
        prev_data_location = os.path.join(base_dir, arm_colour+'/'+prev_year+'/'+prev_mmdd+'/raw/')
        result = SortFiles(prev_data_location,logger,mode=m)
        print("PREVIOUS",prev_night)

        files = result[idx]
        if m =="HS" and idx == 1 and len(files)<9:
            files = []
        else:
            logger.info(
            f"Using %s files found in folder: %s\n", type,
            prev_data_location,
            )
            files_night = str(prev_year+prev_mmdd)
            break
            
        next_night = arrow.get(next_night).shift(days=+1).format('YYYYMMDD')
        print("NEXT", next_night)
        next_year=next_night[0:4]
        next_mmdd=next_night[4:8]
        next_data_location = os.path.join(base_dir, arm_colour+'/'+next_year+'/'+next_mmdd+'/raw/')
        result = SortFiles(next_data_location,logger,mode=m)
        files = result[idx]
        if m =="HS" and idx == 1 and len(files)<9:
            files = []
        else:
            logger.info(
            f"Using %s files found in folder: %s\n", type,
            next_data_location,
            )
            files_night = str(next_year+next_mmdd)
            break
    return files,files_night
