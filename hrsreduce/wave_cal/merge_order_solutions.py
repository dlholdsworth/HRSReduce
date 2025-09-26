import numpy as np

import glob

output = {}

for ord in range(42):

    individual_file =sorted(glob.glob('HR_H_linelist_P_cl_'+str(ord)+'.npy'))[0]
    print(individual_file)
    lines = np.load(individual_file,allow_pickle=True).item()
    line_pix = lines[ord]['line_positions']
    line_wave =lines[ord]['known_wavelengths_air']
    
    output[ord] = {}
    output[ord]['line_positions'] = line_pix
    output[ord]['known_wavelengths_air'] = line_wave

np.save("./HR_H_linelist_P.npy",output)

