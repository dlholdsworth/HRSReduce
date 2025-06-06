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
            sciflux_P = hdul['FIBRE_P'].data
            sciflux_O = hdul['FIBRE_O'].data
        with fits.open(self.arc) as hdul:
            sciwave_P = hdul['WAVE_P'].data
            sciwave_O = hdul['WAVE_O'].data
        with fits.open(self.flat) as hdul:
            flatflux_P = hdul['FIBRE_P'].data
            flatflux_O = hdul['FIBRE_O'].data
        


        #run continuum normalization
        if self.logger:
            self.logger.info("Continuum Normalization: Extracting wavelength and flux data")
        norm_P, full_wave_P, full_spec_P,full_cont_P = self.alg.run_cont_norm(sciwave_P,sciflux_P,flatflux_P)
        
        plt.plot(full_wave_P,full_spec_P/full_cont_P,'r')
        for i in range(42):
            plt.plot(sciwave_P[i],norm_P[i])
        plt.show()
        
        norm_O, full_wave_O, full_spec_O,full_cont_O = self.alg.run_cont_norm(sciwave_O,sciflux_O,flatflux_O)
        
        
        

        #Write results to FITS
        if self.logger:
            self.logger.info("Continuum Normalization: Adding data to SCIENCE FITS file")
        with fits.open(self.sci) as hdul:
            Ext_norm_P = fits.ImageHDU(data=norm_P, name="NORM_P")
            hdul.append(Ext_norm_P)
            merged_P = np.array([full_wave_P,full_spec_P,full_cont_P])
            Ext_merg_P = fits.ImageHDU(data=merged_P, name="MERGED_P")
            Ext_merg_P.header["WAVE"] =  (0,"Column of Wave data")
            Ext_merg_P.header["SPEC"] =  (1,"Column of Spectrum data")
            Ext_merg_P.header["CONT"] =  (2,"Column of Continuum data")
            hdul.append(Ext_merg_P)
            Ext_norm_O = fits.ImageHDU(data=norm_O, name="NORM_O")
            hdul.append(Ext_norm_O)
            merged_O = np.array([full_wave_O,full_spec_O,full_cont_O])
            Ext_merg_O = fits.ImageHDU(data=merged_O, name="MERGED_O")
            Ext_merg_O.header["WAVE"] =  (0,"Column of Wave data")
            Ext_merg_O.header["SPEC"] =  (1,"Column of Spectrum data")
            Ext_merg_O.header["CONT"] =  (2,"Column of Continuum data")
            hdul.append(Ext_merg_O)
            hdul.writeto(self.sci,overwrite='True')
