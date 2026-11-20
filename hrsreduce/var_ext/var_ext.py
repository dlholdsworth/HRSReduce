import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from astropy.io import fits
from hrsreduce.utils.provenance import stamp, is_current

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
        - photon-counting variance from the science image itself

    The frame is never divided by the flat, so the flat's noise is not part of the
    pixel variance (it enters through the extraction profile and the blaze).

    For the master flat itself the variance is its stacking uncertainty (UNC
    squared), which is already in the flat's units (e-/s).

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
        stamp(var_hdu.header, 'var')

        with fits.open(self.sci_frame) as hdul:
            while 'VAR' in hdul:              # a VAR of an older recipe is replaced
                del hdul['VAR']
            hdul.append(var_hdu)
            hdul.writeto(self.sci_frame,overwrite=True)

    def run(self):

        #Test if a VAR of the current recipe exists (an older one is made again)
        with fits.open(self.sci_frame) as hdul:
            exists = 'VAR' in hdul and is_current(hdul['VAR'].header, 'var')
            if 'VAR' in hdul and not exists:
                self.logger.info('VarExts: VAR of {} was made by an older recipe; made again'.format(os.path.basename(self.sci_frame)))

        if not exists:

            # The master flat itself: its variance is the stacking uncertainty, in the flat's own units (e-/s).
            # Read noise and photon noise in e- would not match those units.
            if os.path.abspath(self.sci_frame) == os.path.abspath(self.flat_frame):
                flat_varimg = self.assemble_var_image(self.flat_frame)
                if flat_varimg is not None:
                    self.write_to_file(flat_varimg)
                    return
                self.logger.warning('VarExts: no UNC in the master flat {}; its VAR is built like that of a science frame'.format(self.flat_frame))

            # Assemble the master-bias variance image.
            bias_varimg = self.assemble_var_image(self.bias_frame)
            if bias_varimg is None:
                self.logger.warning('VarExts: no UNC in the master bias {}; its variance is left out'.format(self.bias_frame))
                bias_varimg = 0.0

            # Assemble Readnoise variance image.
            rn_var, ccdimg = self.assemble_read_noise_var_image(self.sci_frame)

            # A frame whose scattered light was removed and kept in BKG: the photon noise is that of all the light
            # that fell on the pixel, so add it back here.
            with fits.open(self.sci_frame) as hdul:
                if 'BKG' in hdul and np.shape(hdul['BKG'].data) == np.shape(ccdimg):
                    ccdimg = ccdimg + np.array(hdul['BKG'].data)

            ccdimg = np.where(ccdimg >= 0.0, ccdimg , 0.0)

            # Read noise, master-bias uncertainty and photon noise, all in e-^2. The frame is never divided by the
            # flat, so the flat's uncertainty does not belong here (it was added as UNC_flat^2 x counts until 1.3.6,
            # mixing (e-/s)^2 with e-).
            var_image = rn_var +\
                        bias_varimg +\
                        ccdimg

            self.write_to_file(var_image)

        return
