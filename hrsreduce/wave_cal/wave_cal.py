
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

    def __init__(self, arc_file,super_arc,sarm,mode,base_dir,cal_type,plot=False):
    
        self.file = arc_file
        self.super = super_arc
        self.arm = sarm[0]
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
        if self.mode == 'HS' and self.arm == 'H':
            self.linelist_path_P = "./hrsreduce/wave_cal/HS_H_linelist_P.npy"
            self.linelist_path_O = "./hrsreduce/wave_cal/HS_H_linelist_O.npy"
        if self.mode == 'HS' and self.arm == 'R':
            self.linelist_path_P = "./hrsreduce/wave_cal/HS_R_linelist_P.npy"
            self.linelist_path_O = "./hrsreduce/wave_cal/HS_R_linelist_O.npy"
        if self.mode == 'HR' and self.arm == 'H':
            self.linelist_path_P = "./hrsreduce/wave_cal/HR_H_linelist_P.npy"
            self.linelist_path_O = "./hrsreduce/wave_cal/HR_H_linelist_O.npy"
        if self.mode == 'HR' and self.arm == 'R':
            self.linelist_path_P = "./hrsreduce/wave_cal/HR_R_linelist_P.npy"
            self.linelist_path_O = "./hrsreduce/wave_cal/HR_R_linelist_O.npy"
        if self.mode == 'MR' and self.arm == 'H':
            self.linelist_path_P = "./hrsreduce/wave_cal/New_MR_H_linelist_P_clean.npy"
            self.linelist_path_O = "./hrsreduce/wave_cal/New_MR_H_linelist_O_clean.npy"
        if self.mode == 'MR' and self.arm == 'R':
            self.linelist_path_P = "./hrsreduce/wave_cal/New_MR_R_linelist_P_clean.npy"
            self.linelist_path_O = "./hrsreduce/wave_cal/New_MR_R_linelist_O_clean.npy"
        if self.mode == 'LR' and self.arm == 'H':
            self.linelist_path_P = "./hrsreduce/wave_cal/New_LR_H_linelist_P_clean.npy"
            self.linelist_path_O = "./hrsreduce/wave_cal/New_LR_H_linelist_O_clean.npy"
        if self.mode == 'LR' and self.arm == 'R':
            self.linelist_path_P = "./hrsreduce/wave_cal/New_LR_R_linelist_P_clean.npy"
            self.linelist_path_O = "./hrsreduce/wave_cal/New_LR_R_linelist_O_clean.npy"
            
            
        self.rough_wls = None
        self.plot = plot
        self.save_diagnostics = os.path.dirname(arc_file)
        self.save_wl_pixel_toggle = True
            
    def execute(self):
    
        with fits.open(self.file) as hdul:
            try:
                test = hdul['WAVE_P']
                wave_cal_done = True
#                hdul.pop('WAVE_P')
#                hdul.pop('WAVE_O')
            except:
                wave_cal_done = False
                
            if wave_cal_done:
                self.logger.info('Wavelength Calibration already done')
                
            else:
            
                if self.arm =='R':
                    ref_arc = "./hrsreduce/wave_cal/HR_Super_Arc_R_Reference.fits"
                if self.arm =='H':
                    ref_arc = "./hrsreduce/wave_cal/HR_Super_Arc_H_Reference.fits"
            
                with fits.open(ref_arc) as hdu_ref:
                    ref_P_fluxes = np.nan_to_num(hdu_ref['Fibre_P'].data)
                    ref_O_fluxes = np.nan_to_num(hdu_ref['Fibre_O'].data)

                if self.cal_type == 'LFC' or 'ThAr':
                    if self.mode == 'HS':

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

                                peak_wavelengths_ang_P = np.load(self.linelist_path_P,allow_pickle=True).item()
                                peak_wavelengths_ang_O = np.load(self.linelist_path_O,allow_pickle=True).item()
                                
                                wls_P = []
                                wls_O = []
                                for o in range(len(ord_lens)):
                                    fit_P = np.polyfit(peak_wavelengths_ang_P[o]['line_positions'],peak_wavelengths_ang_P[o]['known_wavelengths_air'],6)
                                    fit_O = np.polyfit(peak_wavelengths_ang_O[o]['line_positions'],peak_wavelengths_ang_O[o]['known_wavelengths_air'],6)
                                    
                                    x_len = len(O_fluxs[o])
                                    
                                    wls_P.append(np.polyval(fit_P,np.arange(x_len)))
                                    wls_O.append(np.polyval(fit_O,np.arange(x_len)))
                                    
                                self.rough_wls_P = np.array(wls_P)
                                self.rough_wls_O = np.array(wls_O)

                                
                            else:
                                raise ValueError('ThAr run requires linelist_path')


                            wl_soln_P, wls_and_pixels_P, orderlet_dict_P,absolute_precision_P, order_precisions_P = self.alg.run_wavelength_cal( P_fluxs,peak_wavelengths_ang=peak_wavelengths_ang_P, rough_wls=self.rough_wls_P,fibre='P',plot=self.plot)
                                
                                                
                            wl_soln_O, wls_and_pixels_O, orderlet_dict_O, absolute_precision_O, order_precisions_O = self.alg.run_wavelength_cal( O_fluxs,peak_wavelengths_ang=peak_wavelengths_ang_O, rough_wls=self.rough_wls_O,fibre='O',plot=self.plot)
                                
                    else:
                    
                        #Run a simplier wavelength solution for the non-specialised modes that takes the super arc, calculates the shift between local arc and super arc, applies the offset and uses that for the wavelength solution.
                        
                        P_fluxs = hdul['Fibre_P'].data
                        O_fluxs = hdul['Fibre_O'].data

                        P_fluxs = np.nan_to_num(P_fluxs)
                        O_fluxs = np.nan_to_num(O_fluxs)
                        ord_lens = np.zeros(P_fluxs.shape[0])
                        
                        for o in range(len(ord_lens)):
                            ord_lens[o] = len(P_fluxs[o])
                        
                        with fits.open(self.super) as Shdu:
                            Su_P_fluxes = np.nan_to_num(Shdu['Fibre_P'].data)
                            Su_O_fluxes = np.nan_to_num(Shdu['Fibre_O'].data)
                            
                        
                        wl_soln_P, order_precisions_P,absolute_precision_P  = self.alg.run_wavelength_cal_nonHS(P_fluxs,Su_P_fluxes,ref_P_fluxes,self.linelist_path_P,self.nord, self.arm)
                        self.logger.info("Overall absolute precision (all orders P): {} m/s".format(absolute_precision_P))
                        
                        wl_soln_O, order_precisions_O, absolute_precision_O = self.alg.run_wavelength_cal_nonHS(O_fluxs,Su_O_fluxes,ref_O_fluxes,self.linelist_path_O,self.nord, self.arm)
                        self.logger.info("Overall absolute precision (all orders O): {} m/s".format(absolute_precision_O))
                        
                    #Save the wavelength solution to a new fits extension
                    Ext_wave_P = fits.ImageHDU(data=wl_soln_P, name="WAVE_P")
                    Ext_wave_P.header["RV_PREC"] = ((absolute_precision_P),"Overall absolute precision (all orders) m/s")
                    hdul.append(Ext_wave_P)
                    Ext_wave_P2 = fits.ImageHDU(name='WAVE_P_PRE', data=order_precisions_P)
                    hdul.append(Ext_wave_P2)

                    #Save the wavelength solution to a new fits extension
                    Ext_wave_O = fits.ImageHDU(data=wl_soln_O, name="WAVE_O")
                    Ext_wave_O.header["RV_PREC"] = ((absolute_precision_O),"Overall absolute precision (all orders) m/s")
                    hdul.append(Ext_wave_O)
                    Ext_wave_O2 = fits.ImageHDU(name='WAVE_O_PRE', data=order_precisions_O)
                    hdul.append(Ext_wave_O2)
                    
                        
                    hdul.writeto(self.file,overwrite='True')

                else:
                    raise ValueError('cal_type {} not recognized. Available options are LFC or ThAr'.format(self.cal_type))
                                
        return
            
