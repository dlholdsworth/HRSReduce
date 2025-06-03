import numpy as np
import pandas as pd
from astropy import constants as cst, units as u
from astropy.io import fits
import datetime
import os

import matplotlib.pyplot as plt

import logging

from hrsreduce.wave_cal.alg import WaveCalAlg
from hrsreduce.wave_cal.build_wavemodel import BuildWaveModel

logger = logging.getLogger(__name__)

class WavelengthCalibration():

    def __init__(self, arc_file,sarm,mode,base_dir,cal_type,plot=False):
    
        self.file = arc_file
        self.arm = sarm
        self.mode = mode
        self.base_dir = base_dir
        self.logger = logger
        save_diagnostics = os.path.dirname(self.file)
        self.output_dir = save_diagnostics
        # start a logger
        self.logger.info('Started WavelengthCalibration')
        self.cal_type = cal_type
        
        with fits.open(self.file) as hdul:
            self.nord = hdul['Fibre_P'].header['NORDS']
        
        
        self.alg = WaveCalAlg(self.cal_type,self.logger,save_diagnostics = save_diagnostics)
        self.linelist_path_P = "./New_HS_H_linelist_P_clean.npy"
        self.linelist_path_O = "./New_HS_H_linelist_O_clean.npy"
        self.rough_wls = None
        self.plot = plot
        self.save_diagnostics = os.path.dirname(arc_file)
        self.save_wl_pixel_toggle = True
            
    def execute(self):
    
        with fits.open(self.file) as hdul:

            if self.cal_type == 'LFC' or 'ThAr':

                # Create dictionary for storing information about the WLS, fits of each order, fits of each line, etc.
                self.wls_dict = {
                    'wls_processing_date' : str(datetime.datetime.now()), # data and time that this dictionary was created
                    'cal_type' : self.cal_type, # LFC, ThAr
                    'order' : {} #one order_dict for each order
                }
                P_fluxs = hdul['Fibre_P'].data
                O_fluxs = hdul['Fibre_O'].data

                P_fluxs = np.nan_to_num(P_fluxs)
                O_fluxs = np.nan_to_num(O_fluxs)
                ord_lens = np.zeros(P_fluxs.shape[0])
                for o in range(len(ord_lens)):
                    ord_lens[o] = len(P_fluxs[o])
                #self.rough_wls, self.echelle_ord = BuildWaveModel(self.arm,self.mode,ord_lens).execute()
                
                #### lfc ####
                if self.cal_type == 'LFC':
                    line_list, wl_soln, orderlet_dict = self.calibrate_lfc(calflux, output_ext=output_ext)
                    # self.drift_correction(prefix, line_list, wl_soln)
                    self.wls_dict['orderlets'][orderlet_name]['norders'] = self.max_order-self.min_order+1
                    self.wls_dict['orderlets'][orderlet_name]['orders'] = orderlet_dict
 
                #### thar ####
                elif self.cal_type == 'ThAr':
                    if self.linelist_path_O is not None:
#                        peak_wavelengths_ang = pd.read_csv(self.linelist_path,
#                                                           header=None,
#                                                           names=['wave', 'weight'],
#                                                           sep='\s+')
                        #peak_wavelengths_ang = peak_wavelengths_ang.query('weight == 1')
                        peak_wavelengths_ang_P = np.load(self.linelist_path_P,allow_pickle=True).item()
                        peak_wavelengths_ang_O = np.load(self.linelist_path_O,allow_pickle=True).item()
                        
                        wls_P = []
                        wls_O = []
                        for o in range(len(ord_lens)):
                            fit_P = np.polyfit(peak_wavelengths_ang_P[o]['line_positions'],peak_wavelengths_ang_P[o]['known_wavelengths_vac'],3)
                            fit_O = np.polyfit(peak_wavelengths_ang_O[o]['line_positions'],peak_wavelengths_ang_O[o]['known_wavelengths_vac'],3)
                            
                            x_len = len(O_fluxs[o])
                            
                            wls_P.append(np.polyval(fit_P,np.arange(x_len)))
                            wls_O.append(np.polyval(fit_O,np.arange(x_len)))
                            
                        self.rough_wls_P = np.array(wls_P)
                        self.rough_wls_O = np.array(wls_O)

                        
                    else:
                        raise ValueError('ThAr run requires linelist_path')


                    wl_soln_P, wls_and_pixels_P, orderlet_dict_P,absolute_precision_P, order_precisions_P = self.alg.run_wavelength_cal(
                        P_fluxs,peak_wavelengths_ang=peak_wavelengths_ang_P,
                        rough_wls=self.rough_wls_P,fibre='P')
                    #Save the wavelength solution to a new fits extension
                    Ext_wave_P = fits.ImageHDU(data=wl_soln_P, name="WAVE_P")
                    Ext_wave_P.header["RV_PREC"] = ((absolute_precision_P),"Overall absolute precision (all orders) cm/s")
                    hdul.append(Ext_wave_P)
                    Ext_wave_P2 = fits.ImageHDU(name='WAVE_P_PRE', data=order_precisions_P)
                    hdul.append(Ext_wave_P2)
                    
                    wl_soln_O, wls_and_pixels_O, orderlet_dict_O, absolute_precision_O, order_precisions_O = self.alg.run_wavelength_cal(
                        O_fluxs,peak_wavelengths_ang=peak_wavelengths_ang_O,
                        rough_wls=self.rough_wls_O,fibre='O')
                    #Save the wavelength solution to a new fits extension
                    Ext_wave_O = fits.ImageHDU(data=wl_soln_O, name="WAVE_O")
                    Ext_wave_O.header["RV_PREC"] = ((absolute_precision_O),"Overall absolute precision (all orders) cm/s")
                    hdul.append(Ext_wave_O)
                    Ext_wave_O2 = fits.ImageHDU(name='WAVE_O_PRE', data=order_precisions_O)
                    hdul.append(Ext_wave_O2)
                    
#                    if self.save_wl_pixel_toggle == True:
#                        wlpixelwavedir = self.output_dir + '/wlpixelfiles/'
#                        if not os.path.exists(wlpixelwavedir):
#                            os.mkdir(wlpixelwavedir)
#                        file_name = wlpixelwavedir + self.cal_type + 'lines_' + os.path.splitext(os.path.basename(self.file))[0] + "_" + '.npy'
                        
                    hdul.writeto(self.file,overwrite='True')
#                    self.l1_obj[output_ext] = wl_soln
#                    self.wls_dict['orderlets'][orderlet_name]['norders'] = self.max_order-self.min_order+1
#                    self.wls_dict['orderlets'][orderlet_name]['orders'] = orderlet_dict

#
#                # Save WLS dictionary as a JSON file
#                if self.json_filename != None:
#                    print('*******************************************')
#                    print('Saving JSON file with WLS fit information: ' +  self.json_filename)
#                    json_dir = os.path.dirname(self.json_filename)
#                    if not os.path.isdir(json_dir):
#                        os.makedirs(json_dir)
#                    write_wls_json(self.wls_dict, self.json_filename)

            else:
                raise ValueError('cal_type {} not recognized. Available options are LFC, ThAr, & Etalon'.format(self.cal_type))
                                
        return
            ## where to save final polynomial solution
            
