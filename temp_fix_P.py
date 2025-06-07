import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as cst
file ="New_HS_H_linelist_O.npy"

peak_P = np.load(file,allow_pickle=True).item()
x=np.arange(2048)

output= {}

for ord in range(42):
    output[ord] = {}
    lines = peak_P[ord]['line_positions']
    waves = peak_P[ord]['known_wavelengths_vac']
    kk=np.where(lines > 0)[0]
    
    P_coeffs1 = np.polyfit(lines,waves,4)
    
    res = np.abs((np.polyval(P_coeffs1,lines)-waves)/waves *cst.c.value)
    ii=np.where((res) > 500)[0]
    tmp_l2 = lines
    tmp_w2= waves
    for i in range(10):
        tmp_lines = tmp_l2
        tmp_wave = tmp_w2
        tmp_res = res
        if len(ii) > 0:
            bad = np.max(tmp_res)
            jj=np.where(tmp_res != bad)[0]
            tmp_w2=tmp_wave[jj]
            tmp_l2=tmp_lines[jj]
            
            kk = np.where(lines > 0)[0]
            #new_wave = waves[kk]
            #new_line = lines[kk]
            P_coeffs = np.polyfit(tmp_l2,tmp_w2,4)
            fit= np.polyval(P_coeffs,x)
            res = np.abs((np.polyval(P_coeffs,tmp_l2)-tmp_w2)/tmp_w2 *cst.c.value)
            ii=np.where((res) > 500)[0]
        else:
            pass
    
    plt.title(str(ord))
    plt.plot(tmp_w2,res,'bo')
    plt.show()
    output[ord]['line_positions'] = tmp_l2
    output[ord]['known_wavelengths_vac'] = tmp_w2
    
np.save("./New_HS_H_linelist_O_clean",output)
    
