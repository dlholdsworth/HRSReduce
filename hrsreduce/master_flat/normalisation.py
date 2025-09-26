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
    
        with fits.open(self.flat) as hdu:
            self.logger.info("Running Flat normaliseation...")
        
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

                    # Reset flat to unity if flat < 0.1
                    flat_n = np.where(flat_n < 0.1, 1.0, flat_n)

                    #Insert the normalised orderback into the origial image
                    norm_flat[Y1:Y2,X1:X2] = flat_n
                
                #Save the new flat normalised image.
                Norm_hdu = fits.ImageHDU(data=norm_flat, name="NORMALISED")
                hdu.append(Norm_hdu)
                hdu.writeto(self.flat,overwrite='True')
                

