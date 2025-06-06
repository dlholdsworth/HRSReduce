# Standard dependencies
"""
    This module defines class OrderRectification which inherits from `KPF0_Primitive` and provides methods to perform
    the event on order rectification in the recipe.

    Description:
        * Method `__init__`:

            OrderRectification constructor, the following arguments are passed to `__init__`,

                - `action (keckdrpframework.models.action.Action)`: `action.args` contains positional arguments and
                  keyword arguments passed by the `OrderRectification` event issued in the recipe:

                    - `action.args[0] (kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing spectrum data for
                      spectral extraction.
                    - `action.args[1] (kpfpipe.models.level0.KPF0)`: Instance of `KPF0` containing flat data and order
                      trace result.
                    - `action.args['order_name'] (str|list, optional)`: Name or list of names of the order to be
                      processed. Defaults to 'SCI'.
                    - `action.args['start_order'] (int, optional)`: Index of the first order to be processed.
                      Defaults to 0.
                    - `action.args['max_result_order'] (int, optional)`: Total orders to be processed, Defaults to -1.
                    - `action.args['rectification_method'] (str, optional)`: Rectification method, '`norect`',
                      '`vertial`', or '`normal`', to rectify the curved order trace. Defaults to '`norect`',
                      meaning no rectification.
                    - `action.args['data_extension']: (str, optional)`: the name of the extension in spectrum containing data.
                    - `action.args['flat_extension']: (str, optional)`: the name of the extension in flat containing data.
                    - `action.args['clip_file'] (str, optional)`:  Prefix of clip file path. Defaults to None.
                      Clip file is used to store the polygon clip data for the rectification method
                      which is not NoRECT.

                - `context (keckdrpframework.models.processing_context.ProcessingContext)`: `context.config_path`
                  contains the path of the config file defined for the module of spectral extraction in the master
                  config file associated with the recipe.

            and the following attributes are defined to initialize the object,

                - `input_spectrum (kpfpipe.models.level0.KPF0)`: Instance of `KPF0`, assigned by `actions.args[0]`.
                - `input_flat (kpfpipe.models.level0.KPF0)`:  Instance of `KPF0`, assigned by `actions.args[1]`.
                - `order_name (str)`: Name of the order to be processed.
                - `start_order (int)`: Index of the first order to be processed.
                - `max_result_order (int)`: Total orders to be processed.
                - `rectification_method (int)`: Rectification method code as defined in `SpectralExtractionAlg`.
                - `extraction_method (str)`: Extraction method code as defined in `SpectralExtractionAlg`.
                - `config_path (str)`: Path of config file for spectral extraction.
                - `config (configparser.ConfigParser)`: Config context per the file defined by `config_path`.
                - `logger (logging.Logger)`: Instance of logging.Logger.
                - `clip_file (str)`: Prefix of clip file path.
                - `alg (modules.order_trace.src.alg.SpectralExtractionAlg)`: Instance of `SpectralExtractionAlg` which
                  has operation codes for the computation of spectral extraction.

        * Method `__perform`:

            OrderRectification returns the result in `Arguments` object which contains a level 0 data object (`KPF0`)
            with the rectification results replacing the raw image.

    Usage:
        For the recipe, the spectral extraction event is issued like::

            :
            lev0_data = kpf0_from_fits(input_lev0_file, data_type=data_type)
            op_data = OrderRectification(lev0_data, lev0_flat_data,
                                        order_name=order_name,
                                        rectification_method=rect_method,
                                        clip_file="/clip/file/folder/fileprefix")
            :

        where `op_data` is KPF0 object wrapped in `Arguments` class object.
"""


import configparser
import pandas as pd
import numpy as np
import logging
from astropy.io import fits
import multiprocessing as mp
import os.path
import matplotlib.pyplot as plt

import hrsreduce.utils.background_subtraction as BkgAlg

# Pipeline dependencies
# from kpfpipe.logger import start_logger
#from kpfpipe.primitives.level0 import KPF0_Primitive
#from kpfpipe.models.level0 import KPF0
#from kpfpipe.models.level1 import KPF1

# External dependencies
#from keckdrpframework.models.action import Action
#from keckdrpframework.models.arguments import Arguments
#from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
from hrsreduce.extraction.alg import SpectralExtractionAlg

logger = logging.getLogger(__name__)

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/spectral_extraction/configs/default.cfg'

class OrderRectification():
    default_args_val = {
                    'order_name': 'SCI',
                    'max_result_order': -1,
                    'start_order': 0,
                    'rectification_method': 'normal',  # 'norect', 'normal', 'vertical',
                    'clip_file': None,
                    'data_extension': 'DATA',
                    'flat_extension': 'DATA',
                    'trace_extension': None,
                    'trace_file': None,
                    'poly_degree': 3,
                    'origin': [0, 0]
            }

    NORMAL = 0
    VERTICAL = 1
    NoRECT = 2

    def __init__(self,sci_frame, flat_frame,order_trace_file,sarm,mode,base_dir,super_arc=None):

        self.input_spectrum = sci_frame
        self.input_flat = flat_frame
        self.order_trace_file = order_trace_file
        self.rectification_method = 0 # Normal to the order trace: 0, vertical:1, none: 2
        self.extraction_method = 2 # Optimal extraction: 0, sum: 1, no: 2
        self.arm = sarm
        self.mode = mode
        self.base_dir = base_dir
        self.logger = logger
        self.outlier_rejection = False
        spec_no_bk = sci_frame
        var_ext = 'VAR'
        self.data_ext = '0'
        self.flat_ext = '0'
        self.super_arc=super_arc

        self.logger = logger


        # Order trace algorithm setup
        with fits.open(self.input_spectrum) as hdul:
            spec_data = hdul[0].data
            spec_header = hdul[0].header
            
        # Calculate and subtract the background
        order_trace_npz = self.order_trace_file.replace(".csv",".npz")
        spec_data,bkg = BkgAlg.BkgAlg(spec_data,order_trace_npz,self.logger)

        with fits.open(self.input_flat) as hdul:
            flat_data = hdul[0].data
            flat_header = hdul[0].header
        
        # Calculate and subtract the background
        flat_data,bkg = BkgAlg.BkgAlg(flat_data,order_trace_npz,self.logger)

        self.order_trace_data = None
        if order_trace_file:
            self.order_trace_data = pd.read_csv(order_trace_file, header=0, index_col=0)
            poly_degree = 5
            origin = [0,0]
            order_trace_header = {'STARTCOL': origin[0], 'STARTROW': origin[1], 'POLY_DEG': poly_degree}
        elif order_trace_ext and hasattr(self.input_flat, order_trace_ext):
            self.order_trace_data = self.input_flat[order_trace_ext]
            order_trace_header = self.input_flat.header[order_trace_ext]

        self.alg = SpectralExtractionAlg(flat_data,
                                        flat_header,
                                        spec_data,
                                        spec_header,
                                        self.order_trace_data,
                                        order_trace_header,
                                        self.input_spectrum,
                                        config=None, logger=self.logger,
                                        rectification_method=self.rectification_method,
                                        extraction_method=self.extraction_method,
                                        clip_file=None)
        
    def get_order_set(self, s_order):
        if self.o_set.size > 0:
            e_order = self.o_set.size

            o_set_ary = self.o_set[0:e_order] + s_order
            valid_idx = np.where(o_set_ary >= 0)[0]
            first_idx = valid_idx[0] if valid_idx.size > 0 else -1

            return o_set_ary, first_idx
        else:
            return o_set



    def perform(self):
        """
        Perform rectification of the orders by calling method `extract_spectrum` from SpectralExtractionAlg.

        Returns:
            The order definition file name of the rectified orders, with a new extension of the rectified 2D orders

        """

        try:
            hdul=fits.open(self.input_spectrum)
            rectified = hdul['RECT']
            hdul.close
            rect_done = True
        except:
            rect_done = False
                
        if rect_done:
            if self.logger:
                self.logger.info("OrderRectification: Already done...")
            file_base = os.path.splitext(self.super_arc)[0]
            
        else:

            if self.logger:
                self.logger.info("OrderRectification: rectifying order...")
            '''
            all_order_names = self.orderlet_names if type(self.orderlet_names) is list else [self.orderlet_names]
            all_orders = []
            all_o_sets = []

            s_order = self.start_order if self.start_order is not None else 0

            for order_name in all_order_names:
                o_set = self.alg.get_order_set(order_name)
                if o_set.size > 0 :
                    o_set = self.get_order_set(o_set, s_order)
                all_o_sets.append(o_set)
            order_to_process = min([len(a_set) for a_set in all_o_sets])
            for a_set in all_o_sets:
                all_orders.extend(a_set[0:order_to_process])

            all_orders = np.sort(all_orders)
            '''
            total_orders = np.shape(self.order_trace_data)[0]
            all_orders = np.arange(0, total_orders, dtype=int)
            if self.logger:
                self.logger.info("OrderRectification: do " +
                                 SpectralExtractionAlg.rectifying_method[self.rectification_method] +
                                 " rectification on " + str(all_orders.size) + " orders")

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
            
            with fits.open(self.input_spectrum) as hdul:
                data= np.zeros([hdul[0].data.shape[0],hdul[0].data.shape[1]])
                rect_img = fits.ImageHDU(data=data, name="RECT")
                hdul.append(rect_img)
                hdul.writeto(self.input_spectrum,overwrite='True')

    #        #Run orders (pairs of fibres) in parallel.
            manager = mp.Manager()
            return_dict = manager.dict()
            processes = [mp.Process(target=self.alg.extract_spectrum, args=(o_set[(2*i):(2*i)+2],i,return_dict,self.input_spectrum)) for i in range(n_ord)]
            for process in processes:
                process.start()
            for process in processes:
                process.join()

            #return_dict = {}
            #opt_ext_result = self.alg.extract_spectrum(all_orders,1,return_dict,spectrum_file=self.input_spectrum)
            data_df = []
            for i in range(n_ord):
                data_df.append(return_dict[i])
            out_data = (pd.concat(data_df,axis=0,ignore_index=True))
            file_base = os.path.splitext(self.super_arc)[0]
            out_data.to_csv(file_base+"_Orders_Rect.csv")
        
        return (file_base+"_Orders_Rect.csv")


