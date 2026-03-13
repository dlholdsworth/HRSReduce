import numpy as np
import logging
import matplotlib.pyplot as plt
from astropy.io import fits

logger = logging.getLogger(__name__)

#TODO: Include readout noise calculations for different amplifiers

class VarExts():
    """
    Construct and append a variance extension for a reduced science frame.

    This class assembles the total variance image associated with a science
    exposure by combining the relevant detector and calibration-noise terms.
    The variance model includes contributions from:

        - detector read-noise variance
        - master-bias variance
        - master-flat variance propagated into the science counts
        - photon-counting variance from the science image itself

    The resulting variance image is written to the science FITS file as a new
    `VAR` extension if one does not already exist.

    Parameters
    ----------
    sci_frame : str
        Path to the science FITS file for which the variance extension will be
        created.
    bias_frame : str
        Path to the master-bias FITS file.
    flat_frame : str
        Path to the master-flat FITS file.

    Attributes
    ----------
    sci_frame : str
        Path to the target science FITS file.
    bias_frame : str
        Path to the master-bias FITS file.
    flat_frame : str
        Path to the master-flat FITS file.
    logger : logging.Logger
        Logger used for status and diagnostic messages.

    Notes
    -----
    The read-noise contribution is currently assumed to be spatially uniform
    across the detector. A future update may include amplifier-dependent
    read-noise modelling.
    """

    def __init__(self, sci_frame, bias_frame,flat_frame):

        self.sci_frame = sci_frame
        self.bias_frame = bias_frame
        self.flat_frame = flat_frame
        self.logger = logger

        self.logger.info('Started {}'.format(self.__class__.__name__))

    def assemble_var_image(self, filename):
    
        var_img = None
    
        with fits.open(filename) as hdul:
            try:
                unc_img = np.array(hdul['UNC'].data)
                var_img = unc_img * unc_img
            except:
                self.logger.info('No UNC extension in file {}. Using None'.format(filename))
        
        return var_img
        
    def assemble_read_noise_var_image(self,filename):
    
        with fits.open(filename) as hdul:
            sci_image = hdul[0].data
            rn = float(hdul[0].header['RONOISE'])
            var = rn * rn
            ny = hdul[0].header['NAXIS2']
            nx = hdul[0].header['NAXIS1']
            var_img = np.full((ny,nx),var,dtype=float)
            
        return var_img, sci_image
        
    def write_to_file(self, img):
    
        var_hdu = fits.ImageHDU(data=img.astype(np.float32),name="VAR")
        var_hdu.header.insert(8,('COMMENT',"Variance of Frame"))
        
        with fits.open(self.sci_frame) as hdul:
            hdul.append(var_hdu)
            hdul.writeto(self.sci_frame,overwrite=True)

    def run(self):
    
        #Test if VAR ext exists
        with fits.open(self.sci_frame) as hdul:
            try:
                test = hdul['VAR'].data
                exists = True
            except:
                exists = False
    
        if not exists:
    
            # Assemble master-file variance images.
            bias_varimg = self.assemble_var_image(self.bias_frame)
            flat_varimg = self.assemble_var_image(self.flat_frame)
            
            # Assemble Readnoise variance image.
            rn_var, ccdimg = self.assemble_read_noise_var_image(self.sci_frame)
            
            ccdimg = np.where(ccdimg >= 0.0, ccdimg , 0.0)
            
            
            var_image = rn_var +\
                        bias_varimg +\
                        flat_varimg * ccdimg +\
                        ccdimg
                        
            self.write_to_file(var_image)
        
        return
