import numpy as np

import glob

output = {}

arm = 'H'

if arm == 'R':

    for ord in range(33):

        individual_file =sorted(glob.glob('HR_R_linelist_P_cl_'+str(ord)+'.npy'))[0]
        print(individual_file)
        lines = np.load(individual_file,allow_pickle=True).item()
        line_pix = lines[ord]['line_positions']
        line_wave =lines[ord]['known_wavelengths_air']
        
        output[ord] = {}
        output[ord]['line_positions'] = line_pix
        output[ord]['known_wavelengths_air'] = line_wave

    np.save("./HR_R_linelist_P.npy",output)
    
if arm == 'H':

    for ord in range(42):

        individual_file =sorted(glob.glob('./Intermediate_files/HR_H_linelist_O_500s_'+str(ord)+'.npy'))[0]
        print(individual_file)
        lines = np.load(individual_file,allow_pickle=True).item()
        line_pix = lines[ord]['line_positions']
        line_wave =lines[ord]['known_wavelengths_air']
        
        output[ord] = {}
        output[ord]['line_positions'] = line_pix
        output[ord]['known_wavelengths_air'] = line_wave

    np.save("./HR_H_linelist_O_500s.npy",output)

