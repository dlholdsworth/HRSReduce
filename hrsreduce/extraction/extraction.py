import logging
import matplotlib.pyplot as plt

import multiprocessing as mp
import pandas as pd
import numpy as np
import os.path
from astropy.io import fits

# Local dependencies
from hrsreduce.extraction.alg import SpectralExtractionAlg
import hrsreduce.utils.background_subtraction as BkgAlg

#TODO: Run a test to see if the file has been processed (check FITS extension) to avoid running many times. (Not normally an issue, but testing/development will be slow otherwise)
#TODO: Add in VAR details to get a realistic error


logger = logging.getLogger(__name__)

class SpectralExtraction():

    def __init__(self, sci_frame, flat_frame,order_trace_file, sarm,mode,base_dir):

        self.input_spectrum = sci_frame
        self.input_flat = flat_frame
        self.order_trace_file = order_trace_file
        self.rectification_method = 0 # Normal to the order trace: 0, vertical:1, none: 2
        self.extraction_method = 0 # Optimal extraction: 0, sum: 1, no: 2
        self.arm = sarm
        self.mode = mode
        self.base_dir = base_dir
        self.logger = logger
        self.outlier_rejection = False
        spec_no_bk = sci_frame
        var_ext = 'VAR'
        
        # start a logger
        self.logger.info('Started SpectralExtraction')

        self.order_trace_data = None
        if self.order_trace_file:
            self.order_trace_data = pd.read_csv(self.order_trace_file, header=0, index_col=0)
            poly_degree = self.order_trace_data.shape[1]-5
            origin = [0,0]
            order_trace_header = {'STARTCOL': origin[0], 'STARTROW': origin[1], 'POLY_DEG': poly_degree}

        # Open the data files
        with fits.open(self.input_spectrum) as hdl:
            self.spec_header = hdl[0].header
            spec_flux = hdl[0].data
            var_data = hdl['VAR'].data
            
        # Calculate and subtract the background
        order_trace_npz = self.order_trace_file.replace(".csv",".npz")
        self.spec_flux,bkg = BkgAlg.BkgAlg(spec_flux,order_trace_npz,self.logger)

        with fits.open(self.input_flat) as hdl:
            self.flat_data = hdl[0].data
            self.flat_header = hdl[0].header

        try:
            self.alg = SpectralExtractionAlg(self.flat_data,
                                        self.flat_header,
                                        self.spec_flux,
                                        self.spec_header,
                                        self.order_trace_data,
                                        order_trace_header,
                                        config=None, logger=self.logger,
                                        rectification_method=self.rectification_method,
                                        extraction_method=self.extraction_method,
                                        ccd_index=None,
                                        orderlet_names=None,
                                        total_order_per_ccd=None,
                                        clip_file=None,
                                        do_outlier_rejection = self.outlier_rejection,
                                        outlier_flux=None,
                                        var_data=var_data)
        except Exception as e:
            self.alg = None



    def extraction(self):
        """
        Perform spectral extraction by calling method `extract_spectrum` from SpectralExtractionAlg and create adataframe to contain the analysis result.

        Returns:
            File name containing extracted results.

        """
        # rectification_method: SpectralExtractAlg.NoRECT(fastest) SpectralExtractAlg.VERTICAL, SpectralExtractAlg.NORMAL
        # extraction_method: 'optimal' (default), 'sum'

        if self.logger:
            self.logger.info("SpectralExtraction: rectifying and extracting order...")

        if self.alg is None:
            if self.logger:
                self.logger.info("SpectralExtraction: no extension data, order trace data or improper header.")
            return None

        all_o_sets = []
        first_trace_at = []
        s_order = 0

        self.o_set = np.arange(self.order_trace_data.shape[0])
 
        n_ord = int(self.order_trace_data.shape[0] / 2)
        
        for order_name in self.o_set:
            o_set, f_idx = self.get_order_set(s_order)
            all_o_sets.append(o_set)
            first_trace_at.append(f_idx)

        good_result = True

        #Run orders (pairs of fibres) in parallel.
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = [mp.Process(target=self.alg.extract_spectrum, args=(o_set[(2*i):(2*i)+2],i,return_dict)) for i in range(n_ord)]
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        #opt_ext_result = self.alg.extract_spectrum(order_set = o_set[0:1])
        #        assert('spectral_extraction_result' in opt_ext_result and
#                       isinstance(opt_ext_result['spectral_extraction_result'], pd.DataFrame))

#        data_df = opt_ext_result['spectral_extraction_result']


        data_df = pd.DataFrame.from_dict(return_dict[0]['spectral_extraction_result'])
        data_P = pd.DataFrame(columns=data_df.columns)
        data_O = pd.DataFrame(columns=data_df.columns)
        for i in range(0,n_ord):
            data_df = pd.concat([data_df,pd.DataFrame.from_dict(return_dict[i]['spectral_extraction_result'])],ignore_index=True)
            tmp = pd.DataFrame.from_dict(return_dict[i]['spectral_extraction_result'])
            data_P = pd.concat([data_P, tmp.iloc[[1]]],axis=0,ignore_index=True)
            data_O = pd.concat([data_O, tmp.iloc[[0]]],axis=0,ignore_index=True)


        good_result = good_result and data_df is not None

        if not good_result and self.logger:
            self.logger.info("SpectralExtraction: no spectrum extracted")
        elif good_result and self.logger:
            self.logger.info("SpectralExtraction: Receipt written")
            self.logger.info("SpectralExtraction: Done for {} orders!".format(int(n_ord*2)))
            
            out_file=os.path.splitext(str(os.path.dirname(self.input_spectrum))+"/HRS_E_"+str(os.path.basename(self.input_spectrum)))[0]+'.csv'
 
            data_df.to_csv(out_file)
            data_P_np=data_P.to_numpy()
            data_O_np=data_O.to_numpy()
            P_VAR = np.array(data_P_np)
            P_VAR = np.absolute(P_VAR)
            O_VAR = np.array(data_O_np)
            O_VAR = np.absolute(O_VAR)

               
            with fits.open(self.input_spectrum) as hdul:
                Ext_ords_P = fits.ImageHDU(data=data_P, name="FIBRE_P")
                Ext_ords_P.header["NORDS"] =  ((data_P.shape[0]),"Number of extracted orders")
                Ext_ords_P.header["E_MTHD"] = ((self.extraction_method),"Extraction Method. 0: optimal, 1: sum")
                Ext_ords_P.header["R_MTHD"] = ((self.rectification_method), "Rectification Method. 0: Norm, 1: Vert, 2: None")
                Ext_ords_P.header["FLATFILE"] = (str(os.path.basename(self.input_flat)),"Input flat for extraction")
                Ext_ords_P.header["ORDFILE"] = (str(os.path.basename(self.order_trace_file)),"Order trace file")
                hdul.append(Ext_ords_P)
                Ext_ords_P_VAR = fits.ImageHDU(data=P_VAR, name="FIBRE_P_VAR")
                hdul.append(Ext_ords_P_VAR)
                Ext_ords_O = fits.ImageHDU(data=data_O, name="FIBRE_O")
                Ext_ords_O.header["NORDS"] =  ((data_O.shape[0]),"Number of extracted orders")
                Ext_ords_O.header["E_MTHD"] = ((self.extraction_method),"Extraction Method. 0: optimal, 1: sum")
                Ext_ords_O.header["R_MTHD"] = ((self.rectification_method), "Rectification Method. 0: Norm, 1: Vert, 2: None")
                Ext_ords_O.header["FLATFILE"] = (str(os.path.basename(self.input_flat)),"Input flat for extraction")
                Ext_ords_O.header["ORDFILE"] = (str(os.path.basename(self.order_trace_file)),"Order trace file")
                hdul.append(Ext_ords_O)
                Ext_ords_O_VAR = fits.ImageHDU(data=O_VAR, name="FIBRE_O_VAR")
                hdul.append(Ext_ords_O_VAR)
                
                hdul.writeto(self.input_spectrum,overwrite='True')

        #return Arguments(self.output_level1) if good_result else Arguments(None)
#        return data_df
        
    def get_order_set(self, s_order):
        if self.o_set.size > 0:
            e_order = self.o_set.size

            o_set_ary = self.o_set[0:e_order] + s_order
            valid_idx = np.where(o_set_ary >= 0)[0]
            first_idx = valid_idx[0] if valid_idx.size > 0 else -1

            return o_set_ary, first_idx
        else:
            return o_set
