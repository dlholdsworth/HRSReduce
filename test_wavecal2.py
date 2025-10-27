from hrsreduce.wave_cal.wave_cal import WavelengthCalibration
from hrsreduce.wave_cal.build_wavemodel import BuildWaveModel

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
thar_file = "/Users/daniel/Documents/Work/SALT_Pipeline/PyReduce-HRS/DLH_Codes_combined/2025_Mar/thar_best.fits"
arc_file = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2025/1011/reduced/bgoH202510110034.fits"
super_arc= "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2025/Super_Arcs/HR_Super_Arc_H20250101.fits"

#arc_file = "/Users/daniel/Desktop/SALT_HRS_DATA/Red/2025/1011/reduced/bgoR202510110034.fits"
#super_arc = "/Users/daniel/Desktop/SALT_HRS_DATA/Red/2025/Super_Arcs/HR_Super_Arc_R20250101.fits"


#arc_file2= "/Users/daniel/Desktop/SALT_HRS_DATA/Red/2022/1115/reduced/bgoR202211150026.fits"
arm = "H"
m="HR"
cal_type= "ThAr"
cal_line_list= './hrsreduce/wave_cal/thar_best.txt'
base_dir = "/Users/daniel/Desktop/SALT_HRS_DATA/"
plot= True

line_list = np.load("./hrsreduce/wave_cal/HR_H_linelist_P.npy",allow_pickle=True).item()



WavelengthCalibration(arc_file, super_arc, arm, m, base_dir,cal_type,cal_line_list,plot).execute()

