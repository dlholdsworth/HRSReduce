from hrsreduce.wave_cal.wave_cal import WavelengthCalibration
from hrsreduce.wave_cal.build_wavemodel import BuildWaveModel

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
thar_file = "/Users/daniel/Documents/Work/SALT_Pipeline/PyReduce-HRS/DLH_Codes_combined/2025_Mar/thar_best.fits"
#arc_file = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0717/reduced/bgoH202207170027.fits"

#arc_file = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0711/reduced/bgoH202207110027.fits"
super_arc= "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/Super_Arcs/HR_Super_Arc_H20220701.fits"

arc_file = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/1011/reduced/bgoH202310110028.fits"
#super_arc= "/Users/daniel/Desktop/SALT_HRS_DATA/Red/2022/Super_Arcs/HR_Super_Arc_R20220701.fits"

#arc_file = "/Users/daniel/Desktop/SALT_HRS_DATA/Red/2025/1011/reduced/bgoR202510110034.fits"
#arc_file = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0711/reduced/bgoH202207110027.fits"
#super_arc = "/Users/daniel/Desktop/SALT_HRS_DATA/Red/2025/Super_Arcs/HR_Super_Arc_R20250101.fits"


#arc_file2= "/Users/daniel/Desktop/SALT_HRS_DATA/Red/2022/1115/reduced/bgoR202211150026.fits"
arm = "H"
m="HR"
cal_type= "ThAr"
cal_line_list= './hrsreduce/wave_cal/thar_best.txt'
base_dir = "/Users/daniel/Desktop/SALT_HRS_DATA/"
plot= True

line_list = np.load("./hrsreduce/wave_cal/HR_R_linelist_O.npy",allow_pickle=True).item()



WavelengthCalibration(arc_file, super_arc, arm, m, base_dir,cal_type,plot).execute()
