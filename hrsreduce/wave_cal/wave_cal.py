
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
            self.nord = hdul['Fibre_U'].header['NORDS']
        
        self.alg = WaveCalAlg(self.cal_type,self.logger,save_diagnostics = save_diagnostics)
        if self.mode == 'HS' and self.arm == 'H':
            self.linelist_path_U = "./hrsreduce/wave_cal/HS_H_linelist_U.npy"
            self.linelist_path_L = "./hrsreduce/wave_cal/HS_H_linelist_L.npy"
        if self.mode == 'HS' and self.arm == 'R':
            self.linelist_path_U = "./hrsreduce/wave_cal/HS_R_linelist_U.npy"
            self.linelist_path_L = "./hrsreduce/wave_cal/HS_R_linelist_L.npy"
        if self.mode == 'HR' and self.arm == 'H':
            self.linelist_path_U = "./hrsreduce/wave_cal/HR_H_linelist_U.npy"
            self.linelist_path_L = "./hrsreduce/wave_cal/HR_H_linelist_L.npy"
        if self.mode == 'HR' and self.arm == 'R':
            self.linelist_path_U = "./hrsreduce/wave_cal/HR_R_linelist_U.npy"
            self.linelist_path_L = "./hrsreduce/wave_cal/HR_R_linelist_L.npy"
        if self.mode == 'MR' and self.arm == 'H':
            self.linelist_path_U = "./hrsreduce/wave_cal/New_MR_H_linelist_U_clean.npy"
            self.linelist_path_L = "./hrsreduce/wave_cal/New_MR_H_linelist_L_clean.npy"
        if self.mode == 'MR' and self.arm == 'R':
            self.linelist_path_U = "./hrsreduce/wave_cal/New_MR_R_linelist_U_clean.npy"
            self.linelist_path_L = "./hrsreduce/wave_cal/New_MR_R_linelist_L_clean.npy"
        if self.mode == 'LR' and self.arm == 'H':
            self.linelist_path_U = "./hrsreduce/wave_cal/New_LR_H_linelist_U_clean.npy"
            self.linelist_path_L = "./hrsreduce/wave_cal/New_LR_H_linelist_L_clean.npy"
        if self.mode == 'LR' and self.arm == 'R':
            self.linelist_path_U = "./hrsreduce/wave_cal/New_LR_R_linelist_U_clean.npy"
            self.linelist_path_L = "./hrsreduce/wave_cal/New_LR_R_linelist_L_clean.npy"
            
            
        self.rough_wls = None
        self.plot = plot
        self.save_diagnostics = os.path.dirname(arc_file)
        self.save_wl_pixel_toggle = True
            
    def execute(self):
    
        with fits.open(self.file) as hdul:
            try:
                test = hdul['WAVE_U']
                wave_cal_done = True
#                hdul.pop('WAVE_U')
#                hdul.pop('WAVE_L')
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
                    ref_U_fluxes = np.nan_to_num(hdu_ref['Fibre_U'].data)
                    ref_L_fluxes = np.nan_to_num(hdu_ref['Fibre_L'].data)

                if self.cal_type == 'LFC' or 'ThAr':
                    if self.mode == 'HS':

                        # Create dictionary for storing information about the WLS, fits of each order, fits of each line, etc.
                        self.wls_dict = {
                            'wls_processing_date' : str(datetime.datetime.now()), # data and time that this dictionary was created
                            'cal_type' : self.cal_type, # LFC, ThAr
                            'order' : {} #one order_dict for each order
                        }
                        U_fluxs = hdul['Fibre_U'].data
                        L_fluxs = hdul['Fibre_L'].data

                        U_fluxs = np.nan_to_num(U_fluxs)
                        L_fluxs = np.nan_to_num(L_fluxs)
                        ord_lens = np.zeros(U_fluxs.shape[0])
                        for o in range(len(ord_lens)):
                            ord_lens[o] = len(U_fluxs[o])
                        #self.rough_wls, self.echelle_ord = BuildWaveModel(self.arm,self.mode,ord_lens).execute()
                        
                        #### lfc ####
                        if self.cal_type == 'LFC':
                            line_list, wl_soln, orderlet_dict = self.calibrate_lfc(calflux, output_ext=output_ext)
                            # self.drift_correction(prefix, line_list, wl_soln)
                            self.wls_dict['orderlets'][orderlet_name]['norders'] = self.max_order-self.min_order+1
                            self.wls_dict['orderlets'][orderlet_name]['orders'] = orderlet_dict
         
                        #### thar ####
                        elif self.cal_type == 'ThAr':
                            if self.linelist_path_L is not None:

                                peak_wavelengths_ang_U = np.load(self.linelist_path_U,allow_pickle=True).item()
                                peak_wavelengths_ang_L = np.load(self.linelist_path_L,allow_pickle=True).item()
                                
                                wls_U = []
                                wls_L = []
                                for o in range(len(ord_lens)):
                                    fit_U = np.polyfit(peak_wavelengths_ang_U[o]['line_positions'],peak_wavelengths_ang_U[o]['known_wavelengths_air'],6)
                                    fit_L = np.polyfit(peak_wavelengths_ang_L[o]['line_positions'],peak_wavelengths_ang_L[o]['known_wavelengths_air'],6)
                                    
                                    x_len = len(L_fluxs[o])
                                    
                                    wls_U.append(np.polyval(fit_U,np.arange(x_len)))
                                    wls_L.append(np.polyval(fit_L,np.arange(x_len)))
                                    
                                self.rough_wls_U = np.array(wls_U)
                                self.rough_wls_L = np.array(wls_L)

                                
                            else:
                                raise ValueError('ThAr run requires linelist_path')


                            wl_soln_U, wls_and_pixels_U, orderlet_dict_U,absolute_precision_U, order_precisions_U = self.alg.run_wavelength_cal( U_fluxs,peak_wavelengths_ang=peak_wavelengths_ang_U, rough_wls=self.rough_wls_U,fibre='U',plot=self.plot)
                                
                                                
                            wl_soln_L, wls_and_pixels_L, orderlet_dict_L, absolute_precision_L, order_precisions_L = self.alg.run_wavelength_cal( L_fluxs,peak_wavelengths_ang=peak_wavelengths_ang_L, rough_wls=self.rough_wls_L,fibre='L',plot=self.plot)
                                
                    else:
                    
                        #Run a simplier wavelength solution for the non-specialised modes that takes the super arc, calculates the shift between local arc and super arc, applies the offset and uses that for the wavelength solution.
                        
                        U_fluxs = hdul['Fibre_U'].data
                        L_fluxs = hdul['Fibre_L'].data

                        U_fluxs = np.nan_to_num(U_fluxs)
                        L_fluxs = np.nan_to_num(L_fluxs)
                        ord_lens = np.zeros(U_fluxs.shape[0])
                        
                        for o in range(len(ord_lens)):
                            ord_lens[o] = len(U_fluxs[o])
                        
                        with fits.open(self.super) as Shdu:
                            Su_U_fluxes = np.nan_to_num(Shdu['Fibre_U'].data)
                            Su_L_fluxes = np.nan_to_num(Shdu['Fibre_L'].data)
                            
                        
                        wl_soln_U, order_precisions_U,absolute_precision_U  = self.alg.run_wavelength_cal_nonHS(U_fluxs,Su_U_fluxes,ref_U_fluxes,self.linelist_path_U,self.nord, self.arm)
                        self.logger.info("Overall absolute precision (all orders U): {} m/s".format(absolute_precision_U))
                        
                        wl_soln_L, order_precisions_L, absolute_precision_L = self.alg.run_wavelength_cal_nonHS(L_fluxs,Su_L_fluxes,ref_L_fluxes,self.linelist_path_L,self.nord, self.arm)
                        self.logger.info("Overall absolute precision (all orders L): {} m/s".format(absolute_precision_L))
                        
                    #Save the wavelength solution to a new fits extension
                    Ext_wave_U = fits.ImageHDU(data=wl_soln_U, name="WAVE_U")
                    Ext_wave_U.header["RV_PREC"] = ((absolute_precision_U),"Overall absolute precision (all orders) m/s")
                    hdul.append(Ext_wave_U)
                    Ext_wave_U2 = fits.ImageHDU(name='WAVE_U_PRE', data=order_precisions_U)
                    hdul.append(Ext_wave_U2)

                    #Save the wavelength solution to a new fits extension
                    Ext_wave_L = fits.ImageHDU(data=wl_soln_L, name="WAVE_L")
                    Ext_wave_L.header["RV_PREC"] = ((absolute_precision_L),"Overall absolute precision (all orders) m/s")
                    hdul.append(Ext_wave_L)
                    Ext_wave_L2 = fits.ImageHDU(name='WAVE_L_PRE', data=order_precisions_L)
                    hdul.append(Ext_wave_L2)
                    
                        
                    hdul.writeto(self.file,overwrite='True')

                else:
                    raise ValueError('cal_type {} not recognized. Available options are LFC or ThAr'.format(self.cal_type))
                                
        return
            
