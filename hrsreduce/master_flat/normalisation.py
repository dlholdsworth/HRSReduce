import logging
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import numpy.ma as ma
from scipy.stats import mode

logger = logging.getLogger (__name__)

class FlatNormalisation():

    """
    Normalise a straightened master flat on an order-by-order basis.

    This class uses an order-trace table to identify the spatial extent of
    each straightened echelle order in a flat-field image and then normalises
    each order independently. The normalised result is written back into the
    FITS file as a new extension for use in later flat-field correction steps.

    The normalisation is performed only within the traced order regions. For
    each order, the method extracts the corresponding subsection of the
    straightened flat, estimates a representative mean value from illuminated
    pixels, divides the order by that mean, and suppresses very low residual
    values.

    Parameters
    ----------
    flat : str
        Path to the FITS file containing the straightened flat-field image.
    orders : str
        Path to the CSV file containing traced-order geometry information.

    Attributes
    ----------
    flat : str
        Path to the flat-field FITS file to be normalised.
    order_file : str
        Path to the order-trace CSV file.
    logger : logging.Logger
        Logger used for status and debug messages.
    order_trace_data : pandas.DataFrame
        Reduced order-geometry table containing the trace centre, upper and
        lower order widths, and x-range for each straightened order.
    """
    
    def __init__(self,flat,orders):
    
        self.flat = flat
        self.order_file = orders
        self.logger = logger
        
        if self.order_file:
            self.order_trace_data = pd.read_csv(self.order_file, header=0, index_col=0)

            df_result_out = []
            for ord in range(self.order_trace_data.shape[0]):
                df_result = {}
                df_result['Coeff0'] = self.order_trace_data['Coeff0'][ord]
                df_result['BottomEdge'] = self.order_trace_data['BottomEdge'][ord]
                df_result['TopEdge'] = self.order_trace_data['TopEdge'][ord]
                df_result['X1'] = self.order_trace_data['X1'][ord]
                df_result['X2'] = self.order_trace_data['X2'][ord]
                df_result_out.append(df_result)
            
            self.order_trace_data = pd.DataFrame(df_result_out)
        
    def normalise(self):
        """
        Normalise each traced order in the straightened flat image.

        This method opens the flat-field FITS file and checks whether a
        `NORMALISED` extension already exists. If so, no further action is taken.
        Otherwise, it reads the `STRAIGHT` image extension and processes each
        traced order region independently.

        For each order, the method:
            1. extracts the rectangular subsection defined by the trace table
            2. identifies illuminated pixels above a fixed threshold
            3. computes the mean level of the unmasked order pixels
            4. divides the order by this mean to produce a locally normalised flat
            5. sets very small values to zero to suppress poorly illuminated
               regions

        The fully reconstructed normalised image is then appended to the FITS file
        as a new `NORMALISED` extension.

        Returns
        -------
        None
            The method updates the input FITS file in place.
        """
    
        with fits.open(self.flat) as hdu:
            self.logger.info("Running Flat normalisation...")
        
            #Check if flat is already normalised
            try:
                Normalised = hdu['NORMALISED']
                Norm_done = True
            except:
                Norm_done = False
            
            if Norm_done:
                if self.logger:
                    self.logger.info("Flat normaliseation already done...")

            
            else:
            
                flat_img = hdu['STRAIGHT'].data # Input data
                norm_flat = flat_img.copy() # Output data
                
                #Loop over the orders that define the straigetend orders
                for ord in range(self.order_trace_data.shape[0]):
                    #Extract the order from the image
                    
                    X1 = self.order_trace_data['X1'][ord]
                    X2 = self.order_trace_data['X2'][ord]
                    MID= self.order_trace_data['Coeff0'][ord]
                    Y1 = MID-self.order_trace_data['BottomEdge'][ord]
                    Y2 = MID+self.order_trace_data['TopEdge'][ord]
                    
                    flat = flat_img[Y1:Y2,X1:X2]
                    
                    np_om_ffi_bool = np.where(flat > 0.05, True, False)
                    np_om_ffi_shape = np.shape(np_om_ffi_bool)

                    # Compute mean of unmasked pixels in unnormalized flat.
                    unmx = ma.masked_array(flat, mask = ~ np_om_ffi_bool)    # Invert the mask for mask_array operation.
                    unnormalized_flat_mean = ma.getdata(unmx.mean()).item()
                    self.logger.debug('unnormalized_flat_mean = {}'.format(unnormalized_flat_mean))

                    # Normalize flat.
                    flat_n = flat / unnormalized_flat_mean                     # Normalize the master flat by the mean.
        #            flat_unc = unnormalized_flat_unc / unnormalized_flat_mean             # Normalize the uncertainties.

                    # Reset flat to 0 if flat < 0.01
                    flat_n = np.where(flat_n < 0.01, 0.0, flat_n)

                    #Insert the normalised orderback into the origial image
                    norm_flat[Y1:Y2,X1:X2] = flat_n
                
                #Save the new flat normalised image.
                Norm_hdu = fits.ImageHDU(data=norm_flat, name="NORMALISED")
                hdu.append(Norm_hdu)
                hdu.writeto(self.flat,overwrite='True')
                

