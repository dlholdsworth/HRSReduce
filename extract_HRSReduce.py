import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.io import ascii
import numpy as np

import argparse

c_ms = 299792458.0

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="extract_HRSReduce.py",
                                     description="----------------- Extract HRSReduce data products -----------------\n"
                                     )
                                            
    parser.add_argument('file', nargs='+',type=str,
                        help="File name to read. Can be many strings, or use wildcards.")
                        
    parser.add_argument('-t', '--type', default='target',
                        help="Type of data to extract. Can be 'target', 'sky', or 'sub'. Default is target.")
    
    parser.add_argument('-b', '--bc', default=True,
                        help="Apply the barycentric correction [True/False]? Default is True.")
                        
    parser.add_argument('-p', '--plot', default=True,
                        help="Plot the chosen data product [True/False]? Default is True.")
                        
    args = parser.parse_args()
    
    files = args.file
    type = args.type
    if type not in ['target','sky','sub']:
        print("\nWrong type given. Options are 'target', 'sky', or 'sub'.\n\nExiting.\n")
        exit()
    bc = args.bc
    plot = args.plot
    
    for fname in files:

        #Open the header
        hdu = fits.open(fname)
        header = hdu[0].header
        object = header["OBJECT"]
        #Replace any spaces with an underscore in the object name
        object = object.replace(" ", "_")
    
        #Extract info for BC and BJD
        BCV = float(header["BARYRV"])*1000.
        BJD = float(header["BJD"])
    
        #Determine the arm and mode
        arm = header["DETNAM"]
        obs_mode = header["OBSMODE"]
        
        if arm == 'HRDET':
            out_arm = "R"
        if arm == 'HBDET':
            out_arm = "B"
        if obs_mode == 'LOW RESOLUTION':
            mode = "LR"
        if obs_mode == 'MEDIUM RESOLUTION':
            mode = "MR"
        if obs_mode == 'HIGH RESOLUTION':
            mode = "HR"
        if obs_mode == 'HIGH STABILITY':
            mode = "HS"
            
        if mode =="LR":
        #Since the LR Fibres swap the upper and lower traces for Sky/target, treat LR mode differently.
            if type == 'sky':
                #Extract the sky spectrum (in Å)
                spec = hdu['FIBRE_L_SPEC_MERGED'].data #or can use HDU positional argument #7 (in pythonic counting)
                data_head = hdu['FIBRE_L_SPEC_MERGED'].header
            if type == 'target':
                #Extract the target spectrum (in Å)
                spec = hdu['FIBRE_U_SPEC_MERGED'].data #or can use HDU positional argument #8 (in pythonic counting)
                data_head = hdu['FIBRE_U_SPEC_MERGED'].header
            if type == 'sub':
                 #Extract the combined spectrum (in Å)
                spec = hdu['SKY_SUBTRACTED_SPEC_MERGED'].data #or can use HDU positional argument #9 (in pythonic counting)
                data_head = hdu['SKY_SUBTRACTED_SPEC_MERGED'].header
        else:
            if type == 'target':
                #Extract the sky spectrum (in Å)
                spec = hdu['FIBRE_L_SPEC_MERGED'].data #or can use HDU positional argument #7 (in pythonic counting)
                data_head = hdu['FIBRE_L_SPEC_MERGED'].header
            if type == 'sky':
                #Extract the target spectrum (in Å)
                spec = hdu['FIBRE_U_SPEC_MERGED'].data #or can use HDU positional argument #8 (in pythonic counting)
                data_head = hdu['FIBRE_U_SPEC_MERGED'].header
            if type == 'sub':
                 #Extract the combined spectrum (in Å)
                spec = hdu['SKY_SUBTRACTED_SPEC_MERGED'].data #or can use HDU positional argument #9 (in pythonic counting)
                data_head = hdu['SKY_SUBTRACTED_SPEC_MERGED'].header

        specaxis = str(1)
        flux = spec.flatten()
        wave_step = float(data_head['CDELT%s' % (specaxis)])
        wave_base = float(data_head['CRVAL%s' % (specaxis)])
        reference_pixel = float(data_head['CRPIX%s' % (specaxis)])
        xconv = lambda v: ((v-reference_pixel+1)*wave_step+wave_base)
        waveobs = xconv(np.arange(len(flux)))

        if bc == True:
            #Apply the BC
            wave_out = ((waveobs)*(1.0+(BCV/c_ms)))
            out_fname = (str(object)+"_"+(str(BJD)).strip("[]") + "_HRS_" + mode + "_" + str(out_arm)+"_BC.dat")
        else:
            wave_out = waveobs
            out_fname = (str(object)+"_"+(str(BJD)).strip("[]") + "_HRS_" + mode + "_" + str(out_arm)+".dat")
        
        ascii.write([wave_out, flux, np.sqrt(np.abs(flux))], out_fname, overwrite=True, names=['WAVE', 'FLUX', 'ERR'], format='tab')
        
        # Define your header block (lines must be strings)
        header_lines = [
        "# Object: "+str(object),
        "# Instrument: HRS",
        "# Mode: "+str(obs_mode),
        "# Units: WAVE=Angstrom, FLUX=arb, ERR=arb",
        "# Barycentric correction (km/s): "+str(header["BARYRV"]),
        "# Barycentric correction applied: "+str(bc),
        "# BJD: "+str(BJD),
        "# -------------------------------------"
            ]

#        # Prepend header
#        with open(out_fname, "r") as f:
#            content = f.read()
#
#        with open(out_fname, "w") as f:
#            f.write("\n".join(header_lines) + "\n")
#            f.write(content)
#        
        if plot == True:
            plt.figure(figsize=(10, 5))
            plt.plot(wave_out, flux, linewidth=1)

            plt.xlabel(r"Wavelength (Å)")
            plt.ylabel("Flux (arbitrary units)")
            plt.title("Spectrum of "+str(object))
            plt.show()
