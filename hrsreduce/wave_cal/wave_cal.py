import numpy as np
import pandas as pd
from astropy import constants as cst, units as u
import datetime
import os

import logging

from hrsreduce.wave_cal.alg import WaveCalAlg

logger = logging.getLogger(__name__)

class WavelengthCalibration():

    def __init__(self, arc_file,sarm,mode,base_dir):
    
        self.file = arc_file
        self.arm = sarm
        self.mode = mode
        self.base_dir = base_dir
        self.logger = logger
    
        # start a logger
        self.logger.info('Started WavelengthCalibration')
        self.cal_type = 'ThAr'
        
        self.alg = WaveCalibration(self.cal_type,self.logger)


    def execute(self):

        if self.cal_type == 'LFC' or 'ThAr':
            self.file_name_split = self.l1_obj.filename.split('_')[0]
            self.file_name = self.l1_obj.filename.split('.')[0]

            # Create dictionary for storing information about the WLS, fits of each order, fits of each line, etc.
            self.wls_dict = {
                #"name" = 'KP.20230829.12345.67_WLS', # unique string to identify WLS
                'wls_processing_date' : str(datetime.datetime.now()), # data and time that this dictionary was created
                'cal_type' : self.cal_type, # LFC, ThAr
                'orderlets' : {} #one orderlet_dict for each combination of chip and orderlet, e.g. 'RED_SCI1'
            }

            for i, prefix in enumerate(self.cal_orderlet_names):
                print('\nCalibrating orderlet {}.'.format(prefix))
                
                # Create a dictionary for each orderlet that will be filled in later
                full_name = prefix.replace('_FLUX', '') # like GREEN_SCI1
                orderlet_name = full_name.split('_')[1]
                chip_name = prefix.split('_')[0]
                self.wls_dict['chip'] = chip_name
                self.wls_dict['orderlets'][orderlet_name] = {
                    'full_name' : full_name, # e.g., RED_SCI1
                    'orderlet' : orderlet_name, # SCI1, SCI2, SCI3, SKY, CAL
                    'chip' : chip_name, # GREEN or RED
                }

                if self.save_diagnostics is not None:
                    self.alg.save_diagnostics_dir = '{}/{}/'.format(self.save_diagnostics, prefix)

                output_ext = self.output_ext[i]
                calflux = self.l1_obj[prefix]
                calflux = np.nan_to_num(calflux)
                        
                #### lfc ####
                if self.cal_type == 'LFC':
                    line_list, wl_soln, orderlet_dict = self.calibrate_lfc(calflux, output_ext=output_ext)
                    # self.drift_correction(prefix, line_list, wl_soln)
                    self.wls_dict['orderlets'][orderlet_name]['norders'] = self.max_order-self.min_order+1
                    self.wls_dict['orderlets'][orderlet_name]['orders'] = orderlet_dict
 
                #### thar ####
                elif self.cal_type == 'ThAr':
                    if not self.l1_obj.header['PRIMARY']['CAL-OBJ'].startswith('ThAr'):
                        pass # TODO: fix
                        # raise ValueError('Not a ThAr file!')
                    
                    if self.linelist_path is not None:
                        peak_wavelengths_ang = pd.read_csv(self.linelist_path,
                                                           header=None,
                                                           names=['wave', 'weight'],
                                                           sep='\s+')
                        peak_wavelengths_ang = peak_wavelengths_ang.query('weight == 1')
                    else:
                        raise ValueError('ThAr run requires linelist_path')

                    wl_soln, wls_and_pixels, orderlet_dict = self.alg.run_wavelength_cal(
                        calflux,peak_wavelengths_ang=peak_wavelengths_ang,
                        rough_wls=self.rough_wls
                    )
                    
                    if self.save_wl_pixel_toggle == True:
                        wlpixelwavedir = self.output_dir + '/wlpixelfiles/'
                        if not os.path.exists(wlpixelwavedir):
                            os.mkdir(wlpixelwavedir)
                        file_name = wlpixelwavedir + self.cal_type + 'lines_' + self.file_name + "_" + '{}.npy'.format(prefix)

                    self.l1_obj[output_ext] = wl_soln
                    self.wls_dict['orderlets'][orderlet_name]['norders'] = self.max_order-self.min_order+1
                    self.wls_dict['orderlets'][orderlet_name]['orders'] = orderlet_dict

  
            # Save WLS dictionary as a JSON file
            if self.json_filename != None:
                print('*******************************************')
                print('Saving JSON file with WLS fit information: ' +  self.json_filename)
                json_dir = os.path.dirname(self.json_filename)
                if not os.path.isdir(json_dir):
                    os.makedirs(json_dir)
                write_wls_json(self.wls_dict, self.json_filename)

        else:
            raise ValueError('cal_type {} not recognized. Available options are LFC, ThAr, & Etalon'.format(self.cal_type))
                            
        return Arguments(self.l1_obj)
            ## where to save final polynomial solution
            
