"""
Order rectification wrapper for the extraction pipeline.

This module defines the ``OrderRectification`` class, which prepares science and
flat-frame data, loads order-trace information, and calls
``SpectralExtractionAlg`` to rectify curved echelle orders onto a straightened
2D representation.

Overview
--------
The class is responsible for:

    1. Loading the science and flat FITS frames.
    2. Applying a bad-pixel mask appropriate to the selected spectrograph arm.
    3. Optionally subtracting the background from science data.
    4. Loading the order-trace definition from a CSV trace file.
    5. Constructing a ``SpectralExtractionAlg`` instance configured for
       rectification-only mode.
    6. Running order rectification, optionally in parallel, and writing the
       rectified order definition table to disk.

Notes
-----
- In this implementation, the extraction algorithm is configured for
  rectification only:

      * ``rectification_method = 0``  -> normal-to-trace rectification

- The rectified image is written into a ``RECT`` extension in the input
  spectrum FITS file.
- A CSV file describing the rectified order locations is written alongside the
  associated output file base name.

Class
-----
``OrderRectification``

Constructor
-----------
``__init__(sci_frame, flat_frame, order_trace_file, sarm, mode, base_dir,
super_arc=None)``

Parameters
----------
sci_frame : str
    Path to the input science FITS frame.
flat_frame : str
    Path to the input flat-field FITS frame.
order_trace_file : str
    Path to the CSV file containing traced order polynomial coefficients and
    associated order geometry.
sarm : str
    Spectrograph arm identifier, typically ``'Blu'`` or ``'Red'``.
mode : str
    Instrument mode used when naming associated calibration products.
base_dir : str
    Base working directory for the reduction.
super_arc : str, optional
    Associated arc frame or reference file used to define the output file base
    name.

Primary attributes set during initialisation
--------------------------------------------
input_spectrum : str
    Input science FITS filename.
input_flat : str
    Input flat FITS filename.
order_trace_file : str
    Input order-trace CSV filename.
rectification_method : int
    Rectification mode code passed to ``SpectralExtractionAlg``.
extraction_method : int
    Extraction mode code passed to ``SpectralExtractionAlg``.
arm : str
    Original arm label, e.g. ``'Blu'`` or ``'Red'``.
sarm : str
    Short arm label used in filenames, e.g. ``'H'`` or ``'R'``.
mode : str
    Instrument mode.
base_dir : str
    Working directory.
logger : logging.Logger
    Module logger.
order_trace_data : pandas.DataFrame
    Table of order-trace coefficients and metadata loaded from CSV.
alg : SpectralExtractionAlg
    Configured extraction/rectification algorithm instance.

Main method
-----------
``perform()``

This method:

    - checks whether rectification has already been completed;
    - creates a ``RECT`` FITS extension if needed;
    - rectifies all traced orders, using multiprocessing over order pairs;
    - collects the rectification metadata returned by each worker; and
    - writes the final order definition table to
      ``<file_base>_Orders_Rect.csv``.

Returns
-------
str
    Filename of the CSV file containing the rectified order definitions.

Example
-------
Typical usage is:

    rectifier = OrderRectification(
        sci_frame,
        flat_frame,
        order_trace_file,
        sarm,
        mode,
        base_dir,
        super_arc=super_arc,
    )
    rect_file = rectifier.perform()

Output
------
The main output is a CSV table containing, for each rectified order:

    - order index
    - rectified central row coefficient
    - bottom edge
    - top edge
    - x-range of the rectified order
"""

import pandas as pd
import numpy as np
import logging
from astropy.io import fits
import multiprocessing as mp
import os.path
import glob

# Local dependencies
from hrsreduce.extraction.alg import SpectralExtractionAlg
import hrsreduce.utils.background_subtraction as BkgAlg

logger = logging.getLogger(__name__)


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
        if self.arm == 'Blu':
            self.sarm = 'H'
            mask_file = './hrsreduce/utils/BPM_H.fits'
            with fits.open(mask_file) as M_hdu:
                mask = M_hdu[0].data
                mask = mask.astype(bool)
        elif self.arm == 'Red':
            self.sarm = 'R'
            mask_file = './hrsreduce/utils/BPM_R.fits'
            with fits.open(mask_file) as M_hdu:
                mask = M_hdu[0].data
                mask = mask.astype(bool)
        else:
            self.logger.warning("No BPM file available. Using all pixels instead.")
            mask = False


        # Order trace algorithm setup
        with fits.open(self.input_spectrum) as hdul:
            spec_data = hdul[0].data
            spec_data = np.ma.masked_array(spec_data, mask=mask)
            spec_header = hdul[0].header
            data_type = hdul[0].header['OBSTYPE']
            
        # Calculate and subtract the background
        if data_type == "Science":
            spec_data,_ = BkgAlg.BkgAlg(spec_data,self.order_trace_file,self.arm,self.mode,self.logger)
        with fits.open(self.input_flat) as hdul:
            flat_data = hdul[0].data
            flat_data = np.ma.masked_array(flat_data, mask=mask)
            flat_header = hdul[0].header
#            if data_type == 'Arc':
#                flat_data /=flat_data

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
            data_type = hdul[0].header['OBSTYPE']
            rectified = hdul['RECT']
            hdul.close
            rect_done = True
        except:
            rect_done = False
            
        hdul=fits.open(self.input_spectrum)
        hdul.close
        if data_type == 'Arc':
            #Test if there is a Master_Wave file associated with the Arc
            path = os.path.dirname(self.input_spectrum)
            obs_date=(os.path.basename(self.input_spectrum)[-17:-5])
            dst = (path +"/"+str(self.mode)+"_Master_Wave_"+str(self.sarm)+str(obs_date)+".fits")
            master_wave = glob.glob(dst)
            if len(master_wave) != 0:
                rect_done = True
                
        if rect_done:
            if self.logger:
                self.logger.info("OrderRectification: Already done...")
            file_base = os.path.splitext(self.super_arc)[0]
            
        else:

            if self.logger:
                self.logger.info("OrderRectification: rectifying order...")

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

            #Run orders (pairs of fibres) in parallel.
 
            # Start worker processes (as many as nr of CPUs)
            pool = mp.Pool(mp.cpu_count())
            # Initialise extracts list
            extracts = []
            # Loop for orders...
            for i in range(n_ord):
                # Extract spectrum for order i
                extract = pool.apply_async(self.alg.extract_spectrum,args=(o_set[(2*i):(2*i)+2],i,{},self.input_spectrum,))
                # Add extract to extracts list
                extracts.append(extract)

            # Prevent any more tasks from being submitted to the pool
            pool.close()
            # Wait for worker processes to exit
            pool.join()

            # Initialise return dictionary
            return_dict={}
            # Loop for extracts...
            for extract in extracts:
                # Unpack extraction and add to return dictionary
                return_dict[extract.get()[0]] = extract.get()[1]

            data_df = []
            for i in range(n_ord):
                data_df.append(return_dict[i])
            out_data = (pd.concat(data_df,axis=0,ignore_index=True))
            file_base = os.path.splitext(self.super_arc)[0]
            out_data.to_csv(file_base+"_Orders_Rect.csv")
        
        return (file_base+"_Orders_Rect.csv")


