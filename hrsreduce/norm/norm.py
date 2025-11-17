import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt

import logging
from hrsreduce.norm.alg import ContNormAlg

logger = logging.getLogger(__name__)

class ContNorm():
    """This module defines class `ContNorm` which provides methods
    to perform the event `Continuum Normalization`.

    Args:
        KPF1_Primitive: Parent class.
        action (keckdrpframework.models.action.Action): Contains positional arguments and keyword arguments passed by the `ContinuumNormalization` event issued in recipe.
        context (keckdrpframework.models.processing_context.ProcessingContext): Contains path of config file defined for `continuum_normalization` module in master config file associated with recipe.

    Attributes:
        l1_obj (kpfpipe.models.level1.KPF1): Instance of `KPF1`, assigned by `actions.args[0]`
        config (configparser.ConfigParser): Config context.
        logger (logging.Logger): Instance of logging.Logger
        alg (modules.wavelength_cal.src.alg.LFCWaveCalibration): Instance of `LFCWaveCalibration,` which has operation codes for LFC Wavelength Calibration.

    """
    def __init__(self,sci_file,wave_file,mst_flat) -> None:
        """
        ContNorm constructor.

        Args:
            action (Action): Contains positional arguments and keyword arguments passed by the `ContinuumNormalization` event issued in recipe:
              
                `action.args[0] (kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing level 1 spectrum
                `action.args[1] (kpfpipe.models.level1.KPF1)`: Instance of `KPF1` containing data type.

            context (ProcessingContext): Contains path of config file defined for `cont_norm` module in master config file associated with recipe.

        """

        #Start logger
        self.logger = logger
        self.logger.info('Started ContinuumNormalisation')
        
        self.sci = sci_file
        self.arc = wave_file
        self.flat = mst_flat

        #Continuum normalization algorithm setup
        self.alg=ContNormAlg(self.logger)

    #Perform
    def execute(self):
        """
        Performs continuum normalization by calling on ContNormAlg in alg.

        Returns:
            norm: Normalized spectrum.

        """

        #extract extensions
        if self.logger:
            self.logger.info("Continuum Normalization: Extracting SCIWAVE & SCIFLUX extensions")
            
        with fits.open(self.sci) as hdul:
            sciflux_U = hdul['FIBRE_U'].data
            sciflux_L = hdul['FIBRE_L'].data
            sciwave_U = hdul['WAVE_U'].data
            sciwave_L = hdul['WAVE_L'].data
            flatflux_U = hdul['BLAZE_U'].data
            flatflux_L = hdul['BLAZE_L'].data
        


        #run continuum normalization
        if self.logger:
            self.logger.info("Continuum Normalization: Extracting wavelength and flux data")
        norm_U, full_wave_U, full_spec_U,full_cont_U = self.alg.run_cont_norm(sciwave_U,sciflux_U,flatflux_U)
        
        plt.plot(full_wave_U,full_spec_U/full_cont_U,'r')
        for i in range(42):
            plt.plot(sciwave_U[i],norm_U[i])
        plt.show()
        
        norm_L, full_wave_L, full_spec_L,full_cont_L = self.alg.run_cont_norm(sciwave_L,sciflux_L,flatflux_L)
        
        
        

        #Write results to FITS
        if self.logger:
            self.logger.info("Continuum Normalization: Adding data to SCIENCE FITS file")
        with fits.open(self.sci) as hdul:
            Ext_norm_U = fits.ImageHDU(data=norm_U, name="NORM_U")
            hdul.append(Ext_norm_U)
            merged_U = np.array([full_wave_U,full_spec_U,full_cont_U])
            Ext_merg_U = fits.ImageHDU(data=merged_U, name="MERGED_U")
            Ext_merg_U.header["WAVE"] =  (0,"Column of Wave data")
            Ext_merg_U.header["SPEC"] =  (1,"Column of Spectrum data")
            Ext_merg_U.header["CONT"] =  (2,"Column of Continuum data")
            hdul.append(Ext_merg_U)
            Ext_norm_L = fits.ImageHDU(data=norm_L, name="NORM_L")
            hdul.append(Ext_norm_L)
            merged_L = np.array([full_wave_L,full_spec_L,full_cont_L])
            Ext_merg_L = fits.ImageHDU(data=merged_L, name="MERGED_L")
            Ext_merg_L.header["WAVE"] =  (0,"Column of Wave data")
            Ext_merg_L.header["SPEC"] =  (1,"Column of Spectrum data")
            Ext_merg_L.header["CONT"] =  (2,"Column of Continuum data")
            hdul.append(Ext_merg_L)
            hdul.writeto(self.sci,overwrite='True')
