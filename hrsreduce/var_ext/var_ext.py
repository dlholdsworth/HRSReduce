import numpy as np
import logging
import matplotlib.pyplot as plt
from astropy.io import fits

logger = logging.getLogger(__name__)

#TODO: Include readout noise calculations for different amplifiers

class VarExts():

    """
    Description:
        Input L0 filename and database primary key rId for the L0Files database table.
        Select the record from the ReadNoise database table, and square for the read-noise variances.
        Gather all the other variances, sum them all, and write the resulting total variance images
        to FITS extensions ['GREEN_VAR','RED_VAR'] in the associated 2D FITS file.

    Arguments:
        data_type (str): Type of data (e.g., KPF).
        l0_filename (str): Full path and filename of L0 FITS file within container.
        masterbias_path (str): Input master bias.
        masterdark_path (str): Input master dark.
        masterflat_path (str): Input master flat.
        rId (int): Primary database key of L0 FITS file in L0Files database record.
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
