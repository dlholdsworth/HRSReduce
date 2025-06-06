import numpy as np
import math
import pandas as pd
from astropy.io import fits
from astropy.stats import mad_std
import re
import matplotlib.pyplot as plt
#from modules.Utils.config_parser import ConfigHandler
#from modules.Utils.alg_base import ModuleAlgBase
import os
import json


class SpectralExtractionAlg():
    """
    This module defines class 'SpectralExtractionAlg' and methods to perform the optimal or summation
    extraction which reduces 2D spectrum to 1D spectrum for each order, or perform the rectification on either 2D
    flat or spectrum.
    The process includes 2 steps. In the first step, the curved order from the 2D spectral data and/or flat data
    is straighted and output to a new 2D data set by using the specified rectification method.
    The second step performs either the optimal or summation extraction to reduce the 2D data made by
    the first step into 1D data for each order based on the specified extraction method.

    In the first step, the pixels along the normal or vertical direction of the order trace are collected and
    processed column by column by using one of three methods,

        - no rectification method: collecting the pixels within the edge size along the north-up direction
          of the order and taking the pixel flux in full to result the output pixel.
        - vertical rectification method: collecting the pixels within the edge size along the north-up direction
          of the order and performing fractional summation over the collected pixels to result the output pixel.
          The output pixel coverage is based on the vector along the vertical direction and
          the weighting for the fractional summation is based on the overlapping between the collected pixels
          and the output pixel.
        - normal rectification method: collecting the pixels within the edge size along the normal direction
          of the order and performing fractional summation over the collected pixels to result the output pixel.
          The output pixel coverage is based on the vector along the normal direction and the weighting for the
          fractional summation is based on the overlapping between the collected pixels and the output pixel.

    In the second step, either spectral extraction or no operation is made.
    If spectral extraction is made, either optimal or summation extraction is performed
    to reduce the 2D data along each order into 1D data. By using optimal extraction, the output pixel of the
    first step from the spectrum data are weighted and summed up column by column and the weighting is based on
    the associated output pixels of the first step from the flat data. By using summation extraction,
    the pixels are summed up directly without the weighting. If no operation is made, the 2D results of
    all orders from the first step are combined like the raw image with straight orders.

    Args:
        flat_data (numpy.ndarray): 2D flat data, raw data or rectified data.
        flat_header (fits.header.Header): fit header of flat data.
        spectrum_data (numpy.ndarray): 2D spectrum data, raw data or rectified data. None is allowed in case the
            instance is created to perform rectification on flat data only.
        spectrum_header (fits.header.Header): fits header of spectrum data. None is allowed in case no spectrum data.
        order_trace_data (Union[numpy.ndarray, pandas.DataFrame]): order trace data including polynomial coefficients,
            top/bottom edges and area coverage of the order trace.
        order_trace_header (dict): fits header of order trace extension.
        config (configparser.ConfigParser, optional): config context. Defaults to None.
        logger (logging.Logger, optional): Instance of logging.Logger. Defaults to None.
        rectification_method (int, optional): There are three methods used to collect pixels from orders of spectrum
                data and flat data for spectral extraction. Defaults to NoRECT.

                - SpectralExtractionAlg.NoRECT: Pixels at the north-up direction along the order are collected.
                  No rectification. (the fastest computation).
                - SpectralExtractionAlg.VERTICAL: Pixels at the north-up direction along the order are collected
                  to be rectified.
                - SpectralExtractionAlg.NORMAL: Pixels at the normal direction of the order are collected to
                  be rectified.
        extraction_method (int, optional): There are 2 extraction methods performing extraction on collected
                flux along the order or no extraction is made. Defaults to OPTIMAL.

                - SpectralExtractionAlg.OPTIMAL (i.e. 'optimal'): for optimal extraction.
                - SpectralExtractionAlg.SUM (i.e. 'sum'): for summation extraction.
                - SpectralExtractionAlg.NOEXTRACT (i.e. 'rectonly'): no reduction on rectified
                  (including VERTICAL, NORMAL or NoRECT rectification method) order trace.
        clip_file (str, optional): Prefix of clip file path. Defaults to None. Clip file is used to store the
            polygon clip data for the rectification method which is not NoRECT.
        logger_name (str, optional): Name of the logger defined for the SpectralExtractionAlg instance.

    Note:
        Any rectification method combined with extraction method `NOEXTRACT` means only rectification step and no
        extraction is involved and the output is 2D image with the straightened orders at the location of the
        original curved ones.

    Attributes:
        flat_flux (numpy.ndarray): Numpy array storing 2d flat data.
        flat_header (fits.header.Header): fit header of flat data.
        spectrum_flux (numpy.ndarray): Numpy array storing 2d spectrum data for spectral extraction. None is allowed.
        spectrum_header (fits.header.Header): Header of the fits for spectrum data. None is allowed.
        extraction_method (int): Extraction method.
        rectification_method (int): Rectification method.
        poly_order (int): Polynomial order for the approximation made on the order trace.
        origin (list): The origin of the image from the original raw image.
        instrument (str): Instrument of the observation.
        config_ins (ConfigHandler): Instance of ConfigHandler related to section for the instrument or 'PARAM' section.
        order_trace (numpy.ndarrary): Order trace results from order trace module including polynomial coefficients,
            top/bottom edges and  area coverage of the order trace.
        total_order (int): Total order in order trace object.
        order_coeffs (numpy.ndarray): Polynomial coefficients for order trace from higher to lower order.
        order_edges (numpy.ndarray): Bottom and top edges along order trace.
        order_xrange (numpy.ndarray): Left and right boundary of order trace.

        orderlet_names (list): A list containing orderlet names.
        total_image_orderlets (int): Total orderlet contained in the image.
        extracted_flux_pixels (numpy.ndarray): Container to hold the rectified data.
        is_raw_flat (bool): If the flat data is raw image or rectified image.
        is_raw_spectrum (bool): If the spectrum data is raw image or rectified image.
        clip_file_prefix (str): Prefix of the clip files.
        output_clip_area (bool): Flag to indicate if outputting the polygon clipping information to clip file or not.
        output_area_info (list): Container to store the dimension of the order at rectification step. The data of
            each order has to be added into the list in the ascending order of trace's vertical position.

    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions that are relevant to the interface.
        TypeError: If there is type error for `flat_data`, `spectrum_data`, or `order_trace_data`.
        TypeError: If there is type error for `spectrum_header` or `order_trace_header`.
        TypeError: If there is no flux data for spectral extraction.
        Exception: If the order size between spectrum data and order trace data is not the same.

    """

    X = 0
    Y = 1
    HORIZONTAL = 0
    NORMAL = 0
    VERTICAL = 1
    NoRECT = 2
    OPTIMAL = 0
    SUM = 1
    NOEXTRACT = 2
    FOX = 3
    V_UP = 0
    H_RIGHT = 1
    V_DOWN = 2
    H_LEFT = 3
    V1 = 'v_1'
    V2 = 'v_2'
    SDATA = 0
    FDATA = 1
    RECTIFYKEY = 'RECTIFYM'
    RAWSIZEKEY = 'RAWSIZE'
    name = 'SpectralExtraction'
    rectifying_method = ['normal', 'vertical', 'norect']
    extracting_method = ['optimal', 'summ', 'rectonly', 'fox']  # fox: flat-relative optimal extraction

    def __init__(self, flat_data, flat_header, spectrum_data, spectrum_header,  order_trace_data, order_trace_header,
                spectrum_file,config=None, logger=None,
                 rectification_method=NoRECT, extraction_method=OPTIMAL, ccd_index=None,
                 total_order_per_ccd=None, orderlet_names=None, clip_file=None, logger_name=None,
                 do_outlier_rejection=False, outlier_flux=None, var_data=None):

        if not isinstance(flat_data, np.ndarray):
            raise TypeError('flat data type error, cannot construct object from SpectralExtractionAlg')
        if spectrum_data is not None and not isinstance(spectrum_data, np.ndarray):
            raise TypeError('flux data type error, cannot construct object from SpectralExtractionAlg')
        if (spectrum_data is None or not spectrum_data.any()) and extraction_method in [self.SUM, self.OPTIMAL, self.FOX]:
            raise TypeError("no flux data for spectral extraction, cannot construct object from SpectralExtractAlg")
        if not isinstance(order_trace_data, np.ndarray) and not isinstance(order_trace_data, pd.DataFrame):
            raise TypeError('flux data type error, cannot construct object from SpectralExtractionAlg')
        if not isinstance(spectrum_header, fits.header.Header) and extraction_method in [self.SUM, self.OPTIMAL, self.FOX] :
            raise TypeError('flux header type error, cannot construct object from SpectralExtractionAlg')
        if not isinstance(flat_header, fits.header.Header):
            raise TypeError('flat header type error, cannot construct object from SpectralExtractionAlg')
        if not isinstance(order_trace_header, dict) and not isinstance(order_trace_header, fits.header.Header):
            raise TypeError('type: ' + str(type(order_trace_header)) +
                            ' flux header type error, cannot construct object from SpectralExtractionAlg')
        if rectification_method < SpectralExtractionAlg.NORMAL or rectification_method > SpectralExtractionAlg.NoRECT:
            raise TypeError('illegal rectification method code, cannot construct object from SpectralExtractionAlg')
        if extraction_method < SpectralExtractionAlg.OPTIMAL or rectification_method > SpectralExtractionAlg.NOEXTRACT:
            raise TypeError('illegal extraction method code, cannot construct object from SpectralExtractionAlg')

        #ModuleAlgBase.__init__(self, logger_name or self.name, config, logger)

        self.flat_flux = flat_data
        self.flat_header = flat_header
        self.spectrum_flux = spectrum_data   # None is allowed
        self.spectrum_header = spectrum_header
        self.logger = logger
        self.extraction_method = extraction_method
        self.rectification_method = rectification_method

        self.poly_order = order_trace_header['POLY_DEG'] if 'POLY_DEG' in order_trace_header else 3
 
        # origin of the image
        self.origin = [order_trace_header['STARTCOL'] if 'STARTCOL' in order_trace_header else 0,
                       order_trace_header['STARTROW'] if 'STARTROW' in order_trace_header else 0]

#        ins = self.config_param.get_config_value('instrument', '') if self.config_param is not None else ''
#        self.instrument = ins.upper()
#        # section of instrument or 'PARAM'
#        self.config_ins = ConfigHandler(config, ins, self.config_param)
#
        self.total_order = np.shape(order_trace_data)[0]
        if isinstance(order_trace_data, pd.DataFrame):
            self.order_trace = order_trace_data.values
        else:
            self.order_trace = np.array(order_trace_data)
        self.order_coeffs = np.flip(self.order_trace[:, 0:self.poly_order+1], axis=1)
        self.order_edges = None
        self.order_xrange = None

        self.orderlet_names = orderlet_names
        self.total_image_orderlets = len(orderlet_names) \
            if ((isinstance(orderlet_names, list)) and (len(orderlet_names) > 0)) else None
        self.total_order_per_ccd = total_order_per_ccd
        self.extracted_flux_pixels = None
        self.is_raw_flat = self.RECTIFYKEY not in self.flat_header
        try:
            self.spectrum_header['RECT']
            self.is_raw_spectrum = False
        except:
            self.is_raw_spectrum = True
        #self.is_raw_spectrum = False #self.RECTIFYKEY not in spectrum_header if spectrum_header is not None else True
        self.clip_file_prefix = clip_file
        self.output_clip_area = False
        self.output_area_info = list()
        self.poly_clip_dict = dict()
        self.poly_clip_update = False
        self.ccd_index = ccd_index
        self.do_outlier_rejection = do_outlier_rejection
        self.outlier_flux = outlier_flux
        self.order_name = None
        self.var_data = var_data

    def get_config_value(self, prop, default=''):
        """ Get defined value from the config file.

        Search the value of the specified property from instrument section. The default value is returned if not found.

        Args:
            prop (str): Name of the parameter to be searched.
            default (Union[int, float, str], optional): Default value for the searched parameter.

        Returns:
            Union[int, float, str]: Value for the searched parameter.

        """
        return self.config_ins.get_config_value(prop, default)

    def get_order_edges(self, idx=0):
        """ Get the top and bottom edges of the specified order.

        Args:
            idx (Union([int, numpy.ndarray]), optional): Index of the order in the order trace array. Defaults to zero.

        Returns:
            numpy.ndarray: Bottom and top edges of the order, `idx`. The first in the array is the bottom edge,
            and the second in the array is the top edge.

        """
        if self.order_edges is None:
            trace_col = np.shape(self.order_trace)[1]
            if trace_col >= self.poly_order+3:
                self.order_edges = self.order_trace[:, self.poly_order+1: self.poly_order+3]
            else:
                self.order_edges = np.repeat(np.ones((1, 2))*self.get_config_value('width_default', 6),
                                             self.total_order, axis=0)

        return self.order_edges[idx, :] if isinstance(idx, np.ndarray) or self.total_order > idx >= 0 \
            else self.order_edges[0, :]

    def get_order_xrange(self, idx=0):
        """ Get the left and right x boundaries of the specified order.

        Args:
            idx (int, optional): Index of the order in the order trace array. Defaults to zero.

        Returns:
            numpy.ndarray: Left and right boundaries of order, `idx`. The first in the array is the left end,
            and the second in the array is the right end.

        """
        dim_width, dim_height = self.get_spectrum_size()
        if self.order_xrange is None:
            trace_col = np.shape(self.order_trace)[1]
            if trace_col >= self.poly_order + 5:
                self.order_xrange = self.order_trace[:, self.poly_order + 3: self.poly_order + 5].astype(int)
            else:
                self.order_xrange = np.repeat(np.array([0, dim_width-1], dtype=int).reshape((1, 2)),
                                              self.total_order, axis=0)
        if idx >= self.total_order or idx < 0:
            idx = 0
        return self.order_xrange[idx, :]

    def get_spectrum_size(self):
        """ Get the dimension of the spectrum data.

        Returns:
            tuple: Dimension of the data,

                * (*int*): Width of the data.
                * (*int*): Height of the data.

        """
        if self.is_raw_flat:
            dim_h, dim_w = np.shape(self.flat_flux)
        elif self.is_raw_spectrum and self.spectrum_flux is not None:
            dim_h, dim_w = np.shape(self.spectrum_flux)
        else:
            h, w = self.flat_header.get(self.RAWSIZEKEY).split(',')
            dim_h = int(h)
            dim_w = int(w)
        return dim_w, dim_h

    def get_spectrum_order(self):
        """ Get the total order of the order trace or the spectrum data.

        Returns:
            int: The total order.

        """
        return self.total_order

    def get_instrument(self):
        """ Get imaging instrument.

        Returns:
            str: Instrument name.
        """
        return self.instrument

    def var_of(self, x, y):
        return 1.0 if self.var_data is None else self.var_data[y, x]

    def update_spectrum_flux(self, bleeding_cure_file=None):
        """ Update the spectrum flux per specified bleeding cure file or 'nan_pixels' set in config file.

        Args:
            bleeding_cure_file (str, optional): Filename of bleeding cure file if there is. Defaults to None.

        Returns:
            None.

        """
        if not self.is_raw_spectrum:
            return

        dim_width, dim_height = self.get_spectrum_size()
        if bleeding_cure_file is not None:
            correct_data, correct_header = fits.getdata(bleeding_cure_file, header=True)
            if np.shape(correct_data) == np.shape(self.spectrum_flux):
                correct_method = self.get_config_value('correct_method')
                if correct_method == 'sub':
                    self.spectrum_flux = self.spectrum_flux - correct_data

        nan_pixels = False #DLH self.get_config_value('nan_pixels').replace(' ', '')
        if nan_pixels:
            pixel_groups = re.findall("^\\((.*)\\)$", nan_pixels)
            
            if len(pixel_groups) > 0:            # group of nan pixels is set
                res = [i.start()+1 for i in re.finditer('\\],\\[', pixel_groups[0])]
                res.insert(0, -1)
                res.append(len(pixel_groups[0]))
                idx_groups = [pixel_groups[0][res[i]+1:res[i+1]] for i in range(len(res)-1)]
                for group in idx_groups:
                    idx_set = re.findall("^\\[([0-9]*):([0-9]*),([0-9]*):([0-9]*)\\]$", group)
                    if len(idx_set) > 0 and len(idx_set[0]) == 4:
                        y_idx, x_idx = idx_set[0][0:2], idx_set[0][2:4]
                        r_idx = [int(y_idx[i]) if y_idx[i] else (0 if i == 0 else dim_height) for i in range(2)]
                        c_idx = [int(x_idx[i]) if x_idx[i] else (0 if i == 0 else dim_width) for i in range(2)]
                        self.spectrum_flux[r_idx[0]:r_idx[1], c_idx[0]:c_idx[1]] = np.nan


    def allocate_memory_flux(self, order_set, total_data_group, s_rate_y=1):
        """Allocate memory to hold the extracted flux from either rectified or not-rectified order trace

        Args:
            order_set (numpy.ndarray): List of the trace index eligible for optimal extraction process.
            total_data_group (int): Total data set to process.
            s_rate_y (int, optional): Sampling rate along y axis from input domain to output domain for 2D data.
                Defaults to 1.

        Returns:
            numpy.ndarray: allocated 2D area to hold the extracted flux from the order trace.
        """
        if np.size(order_set) == 0:
            return self.extracted_flux_pixels

        if self.extracted_flux_pixels is None or np.size(order_set) > 0:
            order_edges = self.get_order_edges(order_set)
            max_lower_edge = np.amax(order_edges[:, 0])
            max_upper_edge = np.amax(order_edges[:, 1])
            widths = self.get_output_pos(np.array([max_lower_edge, max_upper_edge]), s_rate_y).astype(int)
            y_size = np.sum(widths)
            self.extracted_flux_pixels = np.zeros((total_data_group, y_size, 1))

        return self.extracted_flux_pixels

    def extraction_handler(self, out_data, height, data_group, t_mask=None, f_var=None):
        """Perform the spectral extraction on one column of rectification data based on the extraction method.

        Args:
            out_data (np.ndarray) : The rectification data of 1 column.
            height (int): Height of the rectification data.
            data_group (list): Container contains the data set for the process.

        Returns:
            dict: Spectral extraction or rectification result of one column.
        """

        if self.extraction_method == SpectralExtractionAlg.OPTIMAL:
            return self.optimal_extraction(out_data[self.SDATA][0:height], out_data[self.FDATA][0:height], height, 1, t_mask, f_var)
        elif self.extraction_method == SpectralExtractionAlg.SUM:
            return self.summation_extraction(out_data[self.SDATA][0:height])
        elif self.extraction_method == SpectralExtractionAlg.FOX:
            return self.flat_relative_optimal_extraction(out_data[self.SDATA][0:height], out_data[self.FDATA][0:height],
                                                         height, 1, t_mask, f_var)

        # data_group should contain only the data set to be rectified.
        return {'extraction': out_data[data_group[0]['idx']][0:height]}

    def compute_order_area(self, c_order, rectified_group,  output_x_dim, output_y_dim, s_rate, w_border=False):
        """
            Compute the order center y location, upper edge and lower edge, x coverage at input and output domain,
            y position of fitting polynomial of the order, and the polygon clipping data from the clip file
            if there is.

        Args:
            c_order (int): Order index.
            rectified_group (list): Data set which is rectified.
            output_x_dim (int): Horizontal dimension in output domain.
            output_y_dim (int): Vertical dimension in output domain.
            s_rate (list): Sampling rate.
            w_border (bool): Flag to indicate if the pixel specified as the right end of xrange is included for the
                            computation.

        Returns:
            tuple: dimension for rectification computation,

                * **y_output_mid** (*int*): y position to locate the center of the rectified order.
                * **lower_width** (*int*): lower edge of the rectified curve.
                * **upper_width** (*int*): upper edge of the rectified curve.
                * **y_mid** (*numpy.ndarray*): y position of the polynomial fitting for the order.
                * **x_step** (*numpy.ndarray*): x coordinate of pixels in input domain along the order.
                * **x_output_step** (*numpy.ndarray*): x coordinate of pixels in output domain along the order.
                * **clip_areas** (*numpy.ndarray*): polygon clipping information read from the clip file for the order.

        """
        border = 1 if w_border else 0
        xrange = self.get_order_xrange(c_order)
        x_o = self.origin[self.X]

        # construct coordinate map between input and output
        x_output_step = np.arange(0, output_x_dim + border, dtype=int)     # x step in output domain including border
        x_step = self.get_input_pos(x_output_step, s_rate[self.X])  # x step in input domain

        # x step coverage compliant to xrange
        x_step = x_step[np.where(np.logical_and(x_step >= (xrange[0] + x_o), x_step <= (xrange[1] + x_o + 1)))[0]]
        x_output_step = self.get_output_pos(x_step, s_rate[self.X]).astype(int)

        # prepare for order area calculation
        y_mid = None               # order trace
        clip_areas = None          # clip data for rectification method
        poly_file = self.get_clip_file(c_order)

        gap = 2
        # get order information from poly clip file
        if self.rectification_method != self.NoRECT and poly_file and os.path.exists(poly_file):
            y_output_mid, lower_width, upper_width, clip_areas = self.read_clip_file(poly_file)
        else:
            # get order information from rectified lev0 fits
            order_key = 'ORD_' + str(c_order)
            rectified_header = self.spectrum_header \
                if len(rectified_group) > 0 and rectified_group[0]['idx'] == self.SDATA else self.flat_header

            coeffs = self.order_coeffs[c_order]
            # order location and size along y axis
            y_mid = np.polyval(coeffs, x_step - x_o) + self.origin[self.Y]  # spectral trace value at mid point

            # check if already have data in rectified group
            # if self.rectification_method != self.NoRECT and len(rectified_group) > 0 and order_key in rectified_header:
            if self.extraction_method != self.NOEXTRACT and len(rectified_group) > 0 and order_key in rectified_header:
                order_info = rectified_header.get(order_key)
                [y_output_mid, lower_width, upper_width] = [int(s) for s in order_info.split(',')]
            else:    # NoRect need the data of y_mid
                widths = self.get_order_edges(c_order)
                # the central position of the order, a hint for locating the order in output domain
                y_output_mid = math.floor(np.mean(np.array([np.amax(y_mid), np.amin(y_mid)])) * s_rate[self.Y])
                # y_output_mid = math.floor(np.polyval(coeffs,  center_x - x_o) * s_rate[self.Y])

                # output grid along y
                output_widths = self.get_output_pos(widths, s_rate[self.Y]).astype(int)  # width of output
                upper_width = min(output_widths[1], output_y_dim - 1 - y_output_mid)
                lower_width = min(output_widths[0], y_output_mid)

                if self.output_area_info:
                    pre_y_center = self.output_area_info[-1].get('y_center')
                    pre_upper_width = self.output_area_info[-1].get('upper_width')
                    if y_output_mid < pre_y_center + pre_upper_width + lower_width + gap:
                        y_output_mid = pre_y_center + pre_upper_width + lower_width + gap

        self.output_area_info.append({'y_center': y_output_mid, 'upper_width': upper_width,
                                      'lower_width': lower_width})

        return y_output_mid, lower_width, upper_width, y_mid, x_step, x_output_step, clip_areas

    def data_extraction_for_optimal_extraction(self, data_group,
                            y_mid, y_output_mid,
                            input_widths, output_widths,
                            input_x, x_output_step, mask_height, is_debug=False):
        """

        Args:
            data_group (list): a list containing spectrum data and flat_data
            y_mid (numpy.ndarray): y location of the trace in input domain
            y_output_mid (int): y location of the trace in output domain
            input_widths (numpy.ndarray): from lower to upper width in input domain
            output_widths (numpy.ndarray): from lower to upper width in output domain
            input_x (numpy.ndarry): x positions covering trace in input domain
            x_output_step (numpy.ndarray): x positions covering trace in output domain
            mask_height (int): height of the returned mask denoting the outlier pixels of the trace

        Returns:
            numpy.ndarray: a 2D mask denoting the rejected outlier with the same size of height and width as
                           mask_height and the size of input_x.

        """
        # collect all data
        input_x_dim, input_y_dim = self.get_spectrum_size()
        p_height = mask_height
        p_width = input_x.size
        s_dt = None
        f_dt = None
        is_sdata_raw = True

        for dt in data_group:
            if dt['idx'] == self.SDATA:
                s_dt = dt
                is_sdata_raw = dt['is_raw_data']
            else:
                f_dt = dt

        s_data = np.zeros((p_height, p_width), dtype=float) if s_dt is not None else None
        f_data = np.zeros((p_height, p_width), dtype=float) if f_dt is not None else None
        v_data = np.zeros((p_height, p_width), dtype=float) if self.var_data is not None else None

        for i_x, p_x in enumerate(input_x):
            y_input = np.floor(input_widths + y_mid[i_x]).astype(int)
            y_input_idx = np.where((y_input <= (input_y_dim - 1)) & (y_input >= 0))[0]
            y_input = y_input[y_input_idx]

            if s_data is not None:
                if s_dt['is_raw_data']:
                    s_data[y_input_idx, i_x] = s_dt['data'][y_input, p_x]
                else:
                    s_data[y_input_idx, i_x] = s_dt['data'][output_widths[y_input_idx] + y_output_mid, x_output_step[i_x]]

            if f_data is not None:
                if f_dt['is_raw_data']:
                    f_data[y_input_idx, i_x] = f_dt['data'][y_input, p_x]
                else:
                    f_data[y_input_idx, i_x] = f_dt['data'][output_widths[y_input_idx] + y_output_mid, x_output_step[i_x]]

            if v_data is not None:
                v_data[y_input_idx, i_x] = self.var_data[y_input, p_x]

        return s_data, f_data, is_sdata_raw, v_data

    def outlier_rejection_for_optimal_extraction(self, s_data, f_data, input_x, x_output_step, is_sdata_raw,
                                                 input_widths, output_widths,  y_mid, y_output_mid,
                                                 do_column=True, is_debug=False):
        """

        Args:
            s_data (numpy.ndarray): extracted spectrum data of one order
            f_data (numpy.ndarray): extracted flat data of one order
            input_x (numpy.ndarray):  x positions covering trace in input domain
            x_output_step (numpy.ndarray): x positions covering trace in output domain
            is_sdata_raw (bool): if the spectrum is the original or rectified.
            input_widths (numpy.ndarray): from lower to upper width in input domain
            output_widths  (numpy.ndarray): from lower to upper width in output domain
            y_mid (numpy.ndarray): y location of the trace in input domain
            y_output_mid (int): y location of the trace in output domain
            do_columm (bool, optional): do outlier rejection column by column.
            is_debug (bool, optional): output debug message

        Returns:
            t_mask (numpy.ndarray):  a 2D mask denoting the rejected outlier with the same size as that of s_data.
            s_data_outlier (numpy.ndarray): a 2D spectrum data in which the outlier is replaced by the interpolation of
                                    the neighbors which are not outliers or the scaled profile.

        """
        input_x_dim, input_y_dim = self.get_spectrum_size()
        outlier_sigma = self.get_config_value("outlier_sigma", 5.0)
        p_height = input_widths.size
        p_width = input_x.size

        t_mask, scaled_profile = self.outlier_rejection_local_profile(s_data, f_data, input_x,
                                                         outlier_sigma, y_mid, input_widths, input_y_dim, is_debug)
        s_data_outlier = np.copy(s_data)

        if t_mask is None:
            return t_mask, s_data_outlier

        if t_mask is not None and is_debug:
            total_outlier = np.where(t_mask == 0)[0].size
            print('total outlier :', total_outlier, ' out of ', p_height*p_width, ' pixels: ',
                  total_outlier/(p_height*p_width))

        for x in range(p_width):
            y_inputs = np.floor(input_widths + y_mid[x]).astype(int)
            y_sel_idx = np.where((y_inputs >= 0) & (y_inputs < input_y_dim))[0]
            y_out_idx = np.where(t_mask[y_sel_idx, x] == 0)[0]   # check any outlier t_mask = 0

            if y_out_idx.size == 0:          # no mask found
                continue
            else:
                y_out_idx = y_sel_idx[y_out_idx]

            s_data_outlier[y_out_idx, x] = 0.0

            if self.outlier_flux is not None:
                if is_sdata_raw:
                    original_x = input_x[x]                 # a number
                    original_y = y_inputs[y_out_idx]        # an array
                else:
                    original_x = x_output_step[x]
                    original_y = (output_widths[y_out_idx] + y_output_mid)

                self.outlier_flux[original_y, original_x] = 1

        return t_mask, s_data_outlier


    @staticmethod
    def outlier_rejection_fix_profile(s_data, f_data, input_x, outlier_sigma, y_mid, input_widths, input_y_dim, is_debug=False):
        """

        Args:
            s_data (numpy.ndarray): spectrum pixels of one order trace (starightened trace)
            f_data (numpy.ndarray): flat pixels of one order trace (straightened trace)
            input_x (numpy.ndarray): x positions that cover the trace in input domain
            outlier_sigma (float): thresold for the deviation to find the outlier.
            y_mid (numpy.ndarray): y location of the trace in input domain
            input_widths (numpy.ndarray): from lower to upper width in input domain
            input_y_dim (int): height of original spectrum
            is_debug (bool): prints out the analysis result if it is True.

        Returns:
            np.ndarray: t_mask in which each pixel indicates if the trace pixel is an outlier(0) or not (1).

        """
        t_mask = np.ones_like(s_data, dtype=int)
        p_h, p_w = np.shape(s_data)

        profile_order = np.nanmedian(f_data, axis=1)
        profile_median = np.nanmedian(profile_order)

        if profile_median == 0:
            profile_median = 1
        scaled_profile = np.zeros_like(f_data)

        for i_x, p_x in enumerate(input_x):

            y_input = np.floor(input_widths + y_mid[i_x]).astype(int)
            y_valid_idx = np.where((y_input <= (input_y_dim - 1)) & (y_input >= 0))[0]

            p_median = np.nanmedian(profile_order[y_valid_idx]) if y_valid_idx.size < y_input.size else profile_median

            total_rejection = p_h//2
            for i in range(total_rejection):
                p_non_outlier = np.where(t_mask[y_valid_idx, i_x] != 0)[0]

                if p_non_outlier.size == 0:
                    break

                eff_y = y_valid_idx[p_non_outlier]
                c_flux = s_data[eff_y, i_x]
                c_median = np.nanmedian(c_flux)
                scaled_profile[:, i_x] = profile_order * (c_median/p_median)
                c_residuals = c_flux - scaled_profile[eff_y, i_x]
                c_std = mad_std(c_residuals)
                if c_std == 0.0:
                    c_std = mad_std(np.unique(c_residuals))
                if c_std == 0.0:
                    break

                c_deviation = np.absolute(c_residuals/c_std)
                outside_p = (np.min(scaled_profile) > s_data[eff_y, i_x]) | (s_data[eff_y, i_x] > np.max(scaled_profile))
                outlier_idx = np.where((c_deviation >= outlier_sigma) & outside_p)[0]
                if outlier_idx.size > 0:
                    max_idx = outlier_idx[np.argmax(c_deviation[outlier_idx])]
                    t_mask[eff_y[max_idx], i_x] = 0
                    # t_mask[eff_y[outlier_idx[0]], i_x] = 0
                else:
                    break          # converge

        if np.where(t_mask == 0)[0].size == 0:
            t_mask = None
        else:
            if is_debug:
                print("outlier at ", np.where(t_mask == 0)[1]+input_x[0])

        return t_mask, scaled_profile

    @staticmethod
    def outlier_rejection_local_profile(s_data, f_data, input_x, outlier_sigma, y_mid, input_widths, input_y_dim, is_debug=False):
        """

        Args:
            s_data (numpy.ndarray): spectrum pixels of one order trace (starightened trace)
            f_data (numpy.ndarray): flat pixels of one order trace (straightened trace)
            input_x (numpy.ndarray): x positions that cover the trace in input domain
            outlier_sigma (float): thresold for the deviation to find the outlier.
            y_mid (numpy.ndarray): y location of the trace in input domain
            input_widths (numpy.ndarray): from lower to upper width in input domain
            input_y_dim (int): height of original spectrum
            is_debug (bool): prints out the analysis result if it is True.

        Returns:
            np.ndarray: t_mask in which each pixel indicates if the trace pixel is an outlier(0) or not (1).

        """

        t_mask = np.ones_like(s_data, dtype=int)
        p_h, p_w = np.shape(s_data)

        scaled_profile = np.zeros_like(f_data)

        for i_x, p_x in enumerate(input_x):
            y_input = np.floor(input_widths + y_mid[i_x]).astype(int)
            y_valid_idx = np.where((y_input <= (input_y_dim - 1)) & (y_input >= 0))[0]

            profile_order = f_data[:, i_x]
            profile_median = np.nanmedian(profile_order[y_valid_idx])
            total_rejection = y_valid_idx.size//2
            i = 0
            while i < total_rejection:
                p_non_outlier = np.where(t_mask[y_valid_idx, i_x] != 0)[0]

                if p_non_outlier.size == 0:
                    break

                eff_y = y_valid_idx[p_non_outlier]
                c_flux = s_data[eff_y, i_x]
                c_median = np.nanmedian(c_flux)

                scaled_profile[:, i_x] = profile_order * (c_median/profile_median)
                c_residuals = c_flux - scaled_profile[eff_y, i_x]
                c_std = mad_std(c_residuals)
                if c_std == 0.0:
                    c_std = mad_std(np.unique(c_residuals))
                if c_std == 0.0:
                    break

                c_deviation = np.absolute(c_residuals/c_std)
                outside_p = (np.min(scaled_profile) > s_data[eff_y, i_x]) | (s_data[eff_y, i_x] > np.max(scaled_profile))
                outlier_idx = np.where((c_deviation >= outlier_sigma) & outside_p)[0]
                if outlier_idx.size > 0:
                    max_idx = outlier_idx[np.argmax(c_deviation[outlier_idx])]         # max_idx: within area of eff_y
                    t_mask[eff_y[max_idx], i_x] = 0
                    i += 1
                else:
                    break          # converge

        if np.where(t_mask == 0)[0].size == 0:
            t_mask = None
        else:
            if is_debug:
                print("outlier at ", np.where(t_mask == 0)[1] + input_x[0])
        return t_mask, scaled_profile


    def collect_and_extract_spectrum_curve(self, data_group, order_idx, s_rate=1):
        """ Collect and extract the spectral data along the order per polynomial fit data and no rectification.

        Args:
            data_group (list): Set of input data type from various sources such as spectral data and flat data.
            order_idx (int): Index of the order.
            s_rate (Union[list, float], optional): Sampling rate from input domain to output domain for 2D data.
                Defaults to 1.

        Returns:
            dict: Information of non rectified data from the order including the dimension, like::

                {
                    'y_center': int
                        # the vertical position where to locate the order in output domain
                    'widths': list
                        # adjusted bottom and top edges, i.e. [<bottom edge>, <top edge>]
                    'extracted_data': numpy.ndarray
                        # spectral extraction (1D data) or rectification result (2D data)
                }

        """
        
        input_x_dim, input_y_dim = self.get_spectrum_size()
        sampling_rate = s_rate if isinstance(s_rate, list) else [s_rate, s_rate]

        output_x_dim = input_x_dim * sampling_rate[self.X]
        output_y_dim = input_y_dim * sampling_rate[self.Y]

        raw_group = list()
        rectified_group = list()

        for dt in data_group:
            if dt['is_raw_data']:
                raw_group.append(dt)
            else:
                rectified_group.append(dt)

        y_output_mid, lower_width, upper_width, y_mid, x_step, x_output_step,  _ = \
            self.compute_order_area(order_idx, rectified_group, output_x_dim, output_y_dim, sampling_rate)

        y_size = upper_width + lower_width
        total_data_group = 2  # No. of data set for out_data

        # memory allocated to hold data for one columnn
        out_data = self.allocate_memory_flux(np.array([]), total_data_group, sampling_rate[self.Y])

        # x_output_step aligned with input_x,
        # input_widths, y_input, y_output_widths aligned with [-lower_width, ..., upper_width]
        input_widths = np.array([self.get_input_pos(y_o, sampling_rate[self.Y])
                                 for y_o in range(-lower_width, upper_width)])
        input_x = np.floor(x_step).astype(int)

        # container to hold extracted or rectified result for one order
        extracted_data = np.zeros((y_size if self.extraction_method == self.NOEXTRACT else 1, output_x_dim))
        y_output_widths = np.arange(-lower_width, upper_width)      # in parallel to input_widths

        is_debug = False
        mask_height = np.shape(out_data)[1]

        #if self.order_name == 'GREEN_SCI_FLUX1':
        #    if order_idx == 10:
        #        is_debug = True
        s_data, f_data, is_sdata_raw, f_var = self.data_extraction_for_optimal_extraction(data_group, y_mid, y_output_mid,
                                                                     input_widths, y_output_widths,
                                                                     input_x, x_output_step, mask_height, is_debug)
        if self.do_outlier_rejection and \
                (self.extraction_method == SpectralExtractionAlg.OPTIMAL or
                 self.extraction_method == SpectralExtractionAlg.FOX) and \
                self.rectification_method == SpectralExtractionAlg.NoRECT and \
                s_data is not None and f_data is not None:
            t_mask, s_data_outlier = self.outlier_rejection_for_optimal_extraction(s_data, f_data,
                                                            input_x, x_output_step, is_sdata_raw,
                                                            input_widths, y_output_widths,
                                                            y_mid, y_output_mid,
                                                            do_column=True, is_debug=is_debug)
        else:
            s_data_outlier = None
            t_mask = None

        for s_x, o_x in enumerate(x_output_step):               # ox: 0...x_dim-1, out_data: 0...x_dim-1, corners: 0...

            if f_data is not None:
                out_data[self.FDATA][:] = f_data[:, s_x][:, np.newaxis]
            if s_data is not None:
                out_data[self.SDATA][:] = s_data[:, s_x][:, np.newaxis] \
                    if s_data_outlier is None else s_data_outlier[:, s_x][:, np.newaxis]

            extracted_result = self.extraction_handler(out_data, y_size, data_group,
                                                       t_mask[:, s_x][:, np.newaxis] if t_mask is not None else None,
                                                       f_var[:, s_x][:, np.newaxis] if f_var is not None else None)
            extracted_data[:, o_x:o_x+1] = extracted_result['extraction']

        # out data starting from origin [0, 0] contains the reduced flux associated with the data range
        result_data = {'y_center': y_output_mid,
                       'widths': [lower_width, upper_width],
                       'extracted_data': extracted_data}

        return result_data

    def rectify_and_extract_spectrum_curve(self, data_group, order_idx, s_rate=1):
        """ Rectify and extraction the order trace based on the pixel collection method.

        Parameters:
            data_group (list): Set of input data from various sources such as spectral data and flat data.
            order_idx (int): Index of the order.
            s_rate (Union[list, float], optional): Sampling rate from input domain to output domain for 2D data.
                Defaults to 1.

        Returns:
            dict:  Information of rectified data from the order including the dimension, like::

                {
                    'y_center': int
                        # the vertical position where to locate the order in output domain.
                    'widths': list
                         # adjusted bottom and top edges, i.e. [<bottom edge>, <top edge>].
                    'extracted_data': numpy.ndarray
                        # spectral extraction (1D data) or rectification result (2D data)
                }

                """
        raw_group = list()
        rectified_group = list()
        for dt in data_group:
            if dt['is_raw_data']:
                raw_group.append(dt)
            else:
                rectified_group.append(dt)

        # output dimension
        input_x_dim, input_y_dim = self.get_spectrum_size()
        sampling_rate = s_rate if isinstance(s_rate, list) else [s_rate, s_rate]

        output_x_dim = input_x_dim * sampling_rate[self.X]
        output_y_dim = input_y_dim * sampling_rate[self.Y]

        y_output_mid, lower_width, upper_width, y_mid, x_step, x_output_step, clip_areas = \
            self.compute_order_area(order_idx, rectified_group, output_x_dim, output_y_dim, sampling_rate, True)

        read_poly_file = False
        all_input_corners = None
        poly_file = self.get_clip_file(order_idx)

        if len(raw_group) > 0 and clip_areas is not None:
            read_poly_file = True
        else:
            if len(raw_group) > 0 and poly_file:
                p_dir = os.path.dirname(poly_file)
                if not p_dir:
                    p_dir = '.'
                self.output_clip_area = os.access(p_dir, os.W_OK)

            # prepare corners for clipping computation

        if len(raw_group) > 0 and (not read_poly_file):
            x_o = self.origin[self.X]
            coeffs = self.order_coeffs[order_idx]

            # y step in vertical or normal direction along the order
            if self.rectification_method == self.NORMAL:
                # curve norm along x in input domain
                y_norm_step = self.poly_normal(x_step - x_o, coeffs, sampling_rate[self.Y])
            else:  # vertical direction
                # vertical norm along x in input domain
                y_norm_step = self.vertical_normal(x_step - x_o, sampling_rate[self.Y])

            corners_at_mid = np.vstack((x_step, y_mid)).T  # for x and y in data range, relative to 2D origin [0, 0]

            # corners along the order at output domain
            all_input_corners = np.zeros((upper_width + lower_width + 1, x_step.size, 2))  # for x & y
            all_input_corners[lower_width] = corners_at_mid.copy()

            for o_y in range(1, upper_width + 1):
                next_upper_corners = self.go_vertical(all_input_corners[lower_width + o_y - 1], y_norm_step, 1)
                all_input_corners[lower_width + o_y] = next_upper_corners

            for o_y in range(1, lower_width + 1):
                next_lower_corners = self.go_vertical(all_input_corners[lower_width - o_y + 1], y_norm_step, -1)
                all_input_corners[lower_width - o_y] = next_lower_corners

            if self.output_clip_area:
                clip_areas = dict()

        y_size = upper_width + lower_width
        v2_borders = None
        v1_borders = None
        h_borders = None

        upper_pixels = list(range(lower_width, upper_width + lower_width))  # [0, 1,...,lower_width-1, ..., y_size-1]
        lower_pixels = list(range(lower_width - 1, -1, -1))
        y_output_widths = np.arange(-lower_width, upper_width)
        out_data = self.allocate_memory_flux(np.array([]), 2, sampling_rate[self.Y])
        extracted_data = np.zeros((y_size if self.extraction_method == self.NOEXTRACT else 1, output_x_dim))

        flux_v = np.zeros(len(raw_group)) if len(raw_group) > 0 else None
        raw_data_group = [dt['data'] for dt in raw_group] if len(raw_group) > 0 else None
        for i, o_x in enumerate(x_output_step[0:-1]):  # for x output associated with the data range
            # if i % 100 == 0:
            #    print(i, end=" ")
            if len(raw_group) > 0:
                # if not read from clip file
                if not read_poly_file:
                    if i == 0:
                        v1_borders = self.collect_v_borders(all_input_corners, i)
                    else:
                        v1_borders = v2_borders

                    v2_borders = self.collect_v_borders(all_input_corners, i + 1)
                    h_borders = self.collect_h_borders(v1_borders, v2_borders)

                # each border: vertex_1, vertex_2, direction, intersect_with_borders={direction, pos, loc}
                for pixel_list in [upper_pixels, lower_pixels]:
                    for o_y in pixel_list:
                        # if read from clip file
                        if read_poly_file:
                            input_pixels = clip_areas[str(o_y)][str(o_x)]
                            flux_v.fill(0.0)
                            total_area = 0
                            for i_p in input_pixels:
                                total_area += i_p[2]
                                for n in range(len(raw_group)):
                                    flux_v[n] += raw_group[n]['data'][i_p[1], i_p[0]] * i_p[2]

                            flux = flux_v/total_area if total_area != 0.0 else flux_v
                        else:
                            borders = [
                                v1_borders[o_y].copy(), h_borders[o_y + 1].copy(),
                                v2_borders[o_y].copy(), h_borders[o_y].copy()
                            ]
                            # adjust v1 and v2 in clockwise direction
                            for n in [self.V_DOWN, self.H_LEFT]:
                                borders[n][self.V1], borders[n][self.V2] = borders[n][self.V2], borders[n][self.V1]

                            flux, area = self.compute_flux_for_output_pixel(borders, raw_data_group,
                                                                            len(raw_data_group))
                            if self.output_clip_area:
                                if o_y not in clip_areas:
                                    clip_areas[int(o_y)] = dict()
                                clip_areas[int(o_y)][int(o_x)] = area

                        for n in range(len(raw_group)):
                            out_data[raw_group[n]['idx']][o_y, 0] = flux[n]

            for dt in rectified_group:
                out_data[dt['idx']][0:y_size, 0] = dt['data'][y_output_widths + y_output_mid, o_x]
            extracted_result = self.extraction_handler(out_data, y_size, data_group)
            extracted_data[:, o_x:o_x + 1] = extracted_result['extraction']


        if self.output_clip_area:
            self.poly_clip_update = True
            self.write_clip_file(poly_file, int(y_output_mid), [int(lower_width), int(upper_width)], clip_areas,
                                 order_idx)

        result_data = {'y_center': y_output_mid,
                       'widths': [lower_width, upper_width],
                       'extracted_data': extracted_data}
        return result_data

    def intersect_cell_borders(self, vertex_1, vertex_2):
        """Find out the intersection of a line segment with the horizontal and vertical grid lines.

        Args:
            vertex_1 (list): End point 1 of the line segment.
            vertex_2 (list): Eng point 2 of the line segment.

        Returns:
            dict: A dict instance containing the intersect information, like::

                {
                    'min_cell_x': int
                        # min x of the cell coverage of the line segment.
                    'max_cell_x': int
                        # max x of the cell coverage of the line segment.
                    'min_cell_y': int
                        # min y of the cell coverage of the line segment.
                    'max_cell_y': int
                        # max y of the cell coverage of the line segment.
                    'v_borders': dict
                        # collection of intersection with vertical grid lines, like:
                        {
                            <grid_line_x_position>:
                            {
                                'direction': VERTICAL,
                                'pos': float,
                                      # intersected position along y axis.
                                'axis_idx': self.X
                                      # coordinate index in 'loc' of the grid line.
                                'loc': list,
                                      # cell location of the intersected position.
                            }
                            :
                        }

                    'h_borders': dict
                        # collection of intersection with horizontal grid lines, like:
                         {
                            <grid_line_y_position>:
                            {
                                'direction': HORIZONTAL,
                                'pos': float,
                                        # intersected position along x axis.
                                'axis_idx': self.Y
                                        # coordinate index in 'loc' of the grid line.
                                'loc': list,
                                        # cell location of the intersected position.
                            }
                            :
                        }
                }

        """
        x_dim, y_dim = self.get_spectrum_size()

        x_ends = [vertex_1[self.X], vertex_2[self.X]]
        y_ends = [vertex_1[self.Y], vertex_2[self.Y]]

        x_min = min(x_ends[0], x_ends[1])
        x_max = max(x_ends[0], x_ends[1])
        y_min = min(y_ends[0], y_ends[1])
        y_max = max(y_ends[0], y_ends[1])

        min_cell_x = max(0, math.floor(x_min))
        max_cell_x = min(x_dim, math.ceil(x_max))
        min_cell_y = max(0, math.floor(y_min))
        max_cell_y = min(y_dim, math.ceil(y_max))

        h_borders = dict()
        v_borders = dict()

        # def make_border(direct, pos, pos_cells):
        # pos could be fractional, loc falls on the position increment by one
        def make_border(direction, loc, pos):
            if direction == self.VERTICAL:
                loc_x = loc
                loc_y = math.floor(pos)
                axis_idx = self.X
            else:
                loc_y = loc
                loc_x = math.floor(pos)
                axis_idx = self.Y

            new_border = {'direction': direction, 'pos': pos, 'axis_idx': axis_idx,  'loc': [loc_x, loc_y]}
            return new_border

        def intersect_xline(x1, x3, y3, x4, y4):
            if y3 == y4:        # vertial x line interact with horozontal line
                return y3
            if x1 == x3:        # one end meets the vertical line
                return y3
            if x1 == x4:
                return y4

            c_xy = x3 * y4 - x4 * y3
            den = (x3 - x4)
            num_y = x1 * (y3 - y4) + c_xy      # x1*(x3-x4) + x3*y4-x4*y3
            return num_y/den

        def intersect_yline(y1, x3, y3, x4, y4):
            if x3 == x4:       # horizontal y line interacts with vertical line
                return x3
            if y1 == y3:
                return x3
            if y1 == y4:
                return x4

            c_xy = x3 * y4 - x4 * y3
            den = y3 - y4
            num_x = y1 * (x3 - x4) - c_xy       # y1*(x4-x3)+x3*y4-x4*y3
            return num_x/den

        # end_xy = x_ends[0]*y_ends[1]-x_ends[1]*y_ends[0]   # x3*y4-x4*y3
        # find intersection with vertical cell border (vertical cell line)
        for c_x in range(min_cell_x, max_cell_x+1):
            if x_min <= c_x <= x_max:
                if x_ends[0] != x_ends[1]:          # not a vertical line overlapping cell border
                    y_p = intersect_xline(c_x, x_ends[0], y_ends[0], x_ends[1], y_ends[1])

                    i_border = make_border(self.VERTICAL, c_x, y_p)
                    v_borders[c_x] = i_border

        # find intersection with horizontal cell border (horizontal cell line)
        for c_y in range(min_cell_y, max_cell_y+1):
            if y_min <= c_y <= y_max:
                if y_ends[0] != y_ends[1]:          # not a horizontal line overlapping cell border
                    x_p = intersect_yline(c_y, x_ends[0], y_ends[0], x_ends[1], y_ends[1])

                    i_border = make_border(self.HORIZONTAL, c_y, x_p)
                    h_borders[c_y] = i_border

        intersect_borders = {
                    'orig_v1': vertex_1,
                    'orig_v2': vertex_2,
                    'min_cell_x': min_cell_x,
                    'max_cell_x': max_cell_x,
                    'min_cell_y': min_cell_y,
                    'max_cell_y': max_cell_y,
                    'v_borders': v_borders,
                    'h_borders': h_borders
            }
        return intersect_borders

    def collect_v_borders(self, corners: np.ndarray, start_idx: int):
        """ Collect vertical borders of output pixels.

        Find vertical borders of the rectified cells along either normal or vertical direction
        from the order trace.

        Args:
            corners (list): A set of corners of rectified cells along the normal or vertical
                direction of the order.
            start_idx (int): Starting index of corners of the polygon representing the rectified
                output cell.

        Returns:
            list: Collection of all vertical borders along either normal or vertical direction
            from an order trace, like::

                [
                    {
                        'v1': list   # end point 1 of a border
                        'v2': list   # end point 2 of a border
                        'direction': VERTICAL
                        'intersect_with_borders': dict
                                     # grid line intersection of the border,
                                     # see 'intersect_cell_borders' for the detail
                    }
                    :
                ]

        """

        all_borders = list()
        total_borders = np.shape(corners)[0] - 1

        for idx in range(total_borders):
            vertex_1 = [corners[idx, start_idx, self.X], corners[idx, start_idx, self.Y]]
            vertex_2 = [corners[idx+1, start_idx, self.X], corners[idx+1, start_idx, self.Y]]

            v1 = self.V1
            v2 = self.V2

            border = {
                    v1: vertex_1,
                    v2: vertex_2,
                    'direction': self.VERTICAL,
                    'intersect_with_borders': self.intersect_cell_borders(vertex_1, vertex_2)
            }

            all_borders.append(border)

        return all_borders

    def collect_h_borders(self, v1_borders, v2_borders):
        """Collect horizontal borders of the output pixels.

        Find horizontal borders of the rectified cells along either normal or vertical direction from the order trace.

        Args:
            v1_borders: Left side of vertical borders of the rectified cells collected by
                        :func:`~alg.SpectralExtractionAlg.collect_v_borders()`
            v2_borders: Right side of vertical borders of the rectified cells collected by
                        :func:`~alg.SpectralExtractionAlg.collect_v_borders()`

        Returns:
            list: Collection of all horizontal borders along either normal or vertical direction
            from an order trace, like::

                [
                    {
                        'v1': list,   # end point 1 of a border
                        'v2': list,   # end point 2 of a border
                        'direction': HORIZONTAL,
                        'intersect_with_borders': dict,
                                      # grid line intersection of the border,
                                      # see 'intersect_cell_borders' for the detail
                    }
                    :
                ]

        """
        all_borders = list()
        total_borders = len(v1_borders) + 1
        v1 = self.V1
        v2 = self.V2

        for idx in range(total_borders):
            if idx < (total_borders - 1):
                vertex_1 = v1_borders[idx][v1]
                vertex_2 = v2_borders[idx][v1]
            else:
                vertex_1 = v1_borders[idx-1][v2]
                vertex_2 = v2_borders[idx-1][v2]

            border = {
                    v1: vertex_1,
                    v2: vertex_2,
                    'direction': self.HORIZONTAL,
                    'intersect_with_borders': self.intersect_cell_borders(vertex_1, vertex_2)
            }
            all_borders.append(border)
        return all_borders

    def get_flux_from_order(self, data_group, order_idx, s_rate=1):
        """Collect the data along the order by either rectifying the pixels or not and doing spectral
        extraction on the the rectified results or not based on the extraction method.

        The data collection is based on the following 2 types of methods,

            - rectification method: the pixels along the order are selected depending on the edge size
              (i.e. `widths`) and the direction (i.e. `norm_direction`). With that, all pixels appearing at
              either vertical or normal direction of the order are collected, weighted and summed up.
              The weighting for each pixel is based on the area of that pixel contributing to the pixel
              after rectification.
            - no rectification method: the pixels along the vertical direction of the order are collected
              in full depending on the edge size.

        Parameters:
            data_group (list): List containing 2D spectral data and/or 2D flat data
                and each data set is either raw data or rectified data.
            order_idx (int, optional): Order index.
            s_rate (Union[list, float], optional): sampling rate from input domain to
                output domain for 2D data. Defaults to 1.

        Returns:
            dict: Information related to the order data, like::

                {
                    'extracted_data': numpy.ndarray
                        # extracted spectrum data or rectified spectrum data or
                        # flat data from the order
                        # using specified rectification method and extraction method.
                    'edges': list
                        # lower and upper edges.
                    'out_y_center': int
                        # y center position where 'extracted_data' (only for rectified data)
                        # should be located in the output image.
                }

        Raises:
            AttributeError: The ``Raises`` section is a list of all exceptions that are relevant
                to the interface.
            Exception: If there is no data for spectral extraction or rectification.
            Exception: If there is wrong order index number.
        """

        if len(data_group) == 0:
            raise Exception("no data for spectral extraction or rectification")
        if order_idx < 0 or order_idx >= self.get_spectrum_order():
            raise Exception("wrong order index number")

        if self.rectification_method == self.NoRECT:
            flux_results = self.collect_and_extract_spectrum_curve(data_group, order_idx, s_rate)
        else:  # vertical or normal direction
            flux_results = self.rectify_and_extract_spectrum_curve(data_group, order_idx, s_rate)

        if flux_results is None:
            raise Exception("get flux error on order "+str(order_idx))

        return {'extracted_data': flux_results['extracted_data'],
                'edges': flux_results.get('widths'),
                'out_y_center': flux_results.get('y_center')}

    @staticmethod
    def optimal_extraction(s_data, f_data, data_height, data_width, t_mask=None, f_var=None):
        """ Do optimal extraction on collected pixels along the order.

        This optimal extraction method does the calculation based on the variance of the spectrum data and the
        weighting based on the flat data.

        Args:
            s_data (numpy.ndarray): 2D spectral data collected for one order.
            f_data (numpy.ndarray): 2D flat data collected for one order.
            data_height (int): Height of the 2D data for optimal extraction.
            data_width (int): Width of the 2D data for optimal extraction.
            f_var (numpy.ndarray, optional): flux variance collected for one order.

        Returns:
            dict: Information of optimal extraction result, like::

                {
                    'extraction': numpy.ndarray   # optimal extraction result.
                }

        Raises:
            AttributeError: The ``Raises`` section is a list of all exceptions that are relevant to the interface.
            Exception: If there is unmatched size between collected order data and associated flat data.
            Exception: If the data size doesn't match to the given dimension.

        """

        if np.shape(s_data) != np.shape(f_data):
            raise Exception("unmatched size between collected order data and associated flat data")

        if np.shape(s_data)[0] != data_height or np.shape(s_data)[1] != data_width:
            raise Exception("unmatched data size with the given dimension")

        w_data = np.zeros((1, data_width))

        # taking weighted summation on spectral data of each column,
        # the weight is based on flat data.
        # formula: sum((f/sum(f)) * s/variance)/sum((f/sum(f))^2)/variance) ref. Horne 1986
        if t_mask is not None:
            pixel_mask = t_mask
        else:
            pixel_mask = np.ones_like(f_data)

        # test needed using variance from varaiance extension if the data from variance extension is available
        # d_var = np.where(np.isnan(s_data), 1.0, s_data) if f_var is None else f_var[0:data_height, 1]
        d_var = np.full((data_height, 1), 1.0)  # set the variance to be 1.0
        w_sum = np.sum(f_data[0:data_height, :], axis=0)
        nz_idx = np.where(w_sum != 0.0)[0]
        if nz_idx.size > 0:
            p_data = f_data[0:data_height, nz_idx]/w_sum[nz_idx]
            num = pixel_mask[0:data_height, nz_idx] * p_data * s_data[0:data_height, nz_idx]/d_var
            dem = pixel_mask[0:data_height, nz_idx] * np.power(p_data, 2)/d_var
            w_data[0, nz_idx] = (np.sum(num, axis=0)/np.sum(dem, axis=0))

        # for x in range(0, data_width):
        #    w_sum = sum(f_data[:, x])
        #    if w_sum != 0.0:                        # pixel is not out of range
        #        p_data = f_data[:, x] / w_sum
        #        num = p_data * s_data[:, x] / d_var
        #        dem = np.power(p_data, 2) / d_var
        #        w_data[0, x] = np.sum(num) / np.sum(dem)

        return {'extraction': w_data}

    @staticmethod
    def summation_extraction(s_data):
        """ Spectrum extraction by summation on collected pixels (rectified or non-rectified)

        This summation extraction method does the calculation to extract the spectrum by summing the flux data.

        Args:
            s_data (numpy.ndarray): Collected data for spectrum extraction.

        Returns:
            dict: Information related to the order data, like::

                {
                    'extraction': numpy.ndarray   # summation extraction result.
                }

        """
        out_data = np.sum(s_data, axis=0).reshape(1, -1)

        return {'extraction': out_data}

    @staticmethod
    def flat_relative_optimal_extraction(s_data, f_data, data_height, data_width, t_mask=None, f_var=None):
        """ Do flat relative optimal extraction on collected pixels along the order.

        This flat relative optimal extraction method does the calculation to extract the spectrum relative to the
        flat spectrum.

        Args:
            s_data (numpy.ndarray): 2D spectral data collected for one order.
            f_data (numpy.ndarray): 2D flat data collected for one order.
            data_height (int): Height of the 2D data for flat relative optimal extraction.
            data_width (int): Width of the 2D data for flat relative optimal extraction.
            t_mask (numpy.ndarray, option): 2D mask. Default to None

        Returns:
            dict: Information of flat relative optimal extraction result, like::

                {
                    'extraction': numpy.ndarray   # flat relative optimal extraction result.
                }
        Raises:
            AttributeError: The ``Raises`` section is a list of all exceptions that are relevant to the interface.
            Exception: If there is unmatched size between collected order data and associated flat data.
            Exception: If the data size doesn't match to the given dimension.

        """

        if np.shape(s_data) != np.shape(f_data):
            raise Exception("unmatched size between collected order data and associated flat data")

        if np.shape(s_data)[0] != data_height or np.shape(s_data)[1] != data_width:
            raise Exception("unmatched data size with the given dimension")

        w_data = np.zeros((1, data_width))

        # taking flat relative optimal extraction
        # formula s_x/f_x = sum(Wx,y * F_x,y * S_x,y)/sum(Wx,y * F_x,y * F_x,y) where Wx,y = Mx,y/(sigma_x,y)^2

        if t_mask is not None:
            pixel_mask = t_mask[0:data_height, :]
        else:
            pixel_mask = np.ones_like(s_data)

        # get variance (sigma^2) from variance extension or from flux data
        d_var = np.where(np.isnan(s_data)|(s_data==0.0), 1.0, s_data) if f_var is None else f_var[0:data_height]
        if np.where(d_var == 0.0)[0].size > 0:
            d_var = np.where(d_var==0.0, 1.0, d_var)
        # d_var = np.full((data_height, data_width), 1.0)  # set the variance to be 1.0
        w_xy = pixel_mask/d_var
        s_alter_data = np.where(np.isnan(s_data), 0.0, s_data)

        num_sum = np.sum(w_xy * f_data * s_alter_data, axis=0)
        dem_sum = np.sum(w_xy * f_data * f_data, axis=0)
        nz_idx = np.where(dem_sum != 0.0)[0]

        if nz_idx.size > 0:
            w_data[0, nz_idx] = num_sum[nz_idx]/dem_sum[nz_idx]

        return {'extraction': w_data}

    @staticmethod
    def fill_2d_with_data(from_data, to_data, to_pos, from_pos=0, height=1):
        """ Fill a band of 2D data into another 2D container starting from and to specified vertical positions.

        Args:
            from_data (numpy.ndarray): Band of data to be copied from.
            to_data (numpy.ndarray): 2D area to copy the data to.
            to_pos (int): The vertical position of `to_data` where `from_data` is copied to.
            from_pos (int): the vertical position of `from_data` where the data is copied from. The default is 0.
            height (int): the height of the data to be copied from. The default is 1.

        Returns:
            numpy.ndarray: 2D data with `from_data` filled in.

        """

        to_y_dim = np.shape(to_data)[0]
        from_y_dim = np.shape(from_data)[0]

        if to_pos < 0 or to_pos >= to_y_dim or from_pos < 0 or from_pos >= from_y_dim or \
                from_pos+height > from_y_dim or to_pos+height > to_y_dim:  # out of range, do nothing
            return to_data

        to_data[to_pos:to_pos+height, :] = from_data[from_pos:from_pos + height, :]
        return to_data

    @staticmethod
    def vertical_normal(pos_x, sampling_rate):
        """ Calculate the vertical vector at the specified x position per vertical sampling rate.

        Args:
            pos_x (numpy.ndarray): x position.
            sampling_rate (float): Vertical sampling rate.

        Returns:
            numpy.ndarray: Vertical vector at position `pos_x`.

        """

        v_norms = np.zeros((pos_x.size, 2))
        d = 1.0/sampling_rate
        v_norms[:, 1] = d

        return v_norms

    @staticmethod
    def poly_normal(pos_x, coeffs, sampling_rate=1):
        """ Calculate the normal vector at the specified x position per vertical sampling rate.

        Args:
            pos_x (numpy.ndarray): x position.
            coeffs (numpy.ndarray): Coefficients of the polynomial fit to the order from higher order to lower.
            sampling_rate (int, optional): Vertical sampling rate. Defaults to 1.

        Returns:
            numpy.ndarray: Normal vectors at positions in `pos_x`.

        """

        d_coeff = np.polyder(coeffs)
        tan = np.polyval(d_coeff, pos_x)
        v_norms = np.zeros((pos_x.size, 2))
        norm_d = np.sqrt(tan*tan+1)*sampling_rate
        v_norms[:, 0] = -tan/norm_d
        v_norms[:, 1] = 1.0/norm_d

        return v_norms

    @staticmethod
    def get_input_pos(output_pos, s_rate):
        """ Get associated position of input domain per output position and sampling rate.

        Args:
            output_pos (Union([numpy.ndarray, float]): Position of output domain.
            s_rate (flat): Sampling ratio between input domain and output domain, i.e. *input*s_rate = output*.

        Returns:
            Union([numpy.ndarray, float]): Position of input domain.

        """

        return output_pos/s_rate

    @staticmethod
    def get_output_pos(input_pos: np.ndarray, s_rate: float):
        """ Get associated output position per input position and sampling rate.

        Args:
            input_pos (Union[numpy.ndarray, float]): Position of input domain.
            s_rate (float): Sampling rate.

        Returns:
            Union([numpy.ndarray, float]): Position of output domain.

        """
        return np.floor(input_pos*s_rate)     # position at output cell domain is integer based

    @staticmethod
    def go_vertical(crt_pos, norm, direction=1):
        """ Get new positions by starting from a set of positions and traversing along the specified direction.

        Args:
            crt_pos (numpy.ndarray): Current positions.
            norm (numpy.ndarray): Vector to traverse.
            direction (int, optional): Traverse direction. Defaults to 1.

        Returns:
            numpy.ndarray: New position at given traversing direction.

        """

        new_pos = crt_pos + direction * norm

        return new_pos

    def compute_flux_for_output_pixel(self, borders, input_data, total_data_group):
        """ Compute weighted flux covered by one output pixel.

            Compute the weighted flux per spectrum data and flat data covered for the area of one output pixel.
            The area is represented by polygon borders and the polygon is collected in either normal direction or
            vertical direction to the order. The weight depends on the area overlap between the output pixel and the
            input pixel.

            Args:
                borders(list): Borders of the coverage of one output pxiel.
                input_data(list): Input data group. each element is numpy.ndarray.
                total_data_group(int): total data group

            Returns:
                list : Flux value on observation data and flat data for the area covered by one output pixel.

        """
        x_1 = min([border['intersect_with_borders']['min_cell_x'] for border in borders])
        x_2 = max([border['intersect_with_borders']['max_cell_x'] for border in borders])
        y_1 = min([border['intersect_with_borders']['min_cell_y'] for border in borders])
        y_2 = max([border['intersect_with_borders']['max_cell_y'] for border in borders])

        flux_polygon, total_area_polygon, clipped_areas = self.compute_flux_from_polygon_clipping2(borders,
                                                        [x_1, x_2, y_1, y_2],input_data, total_data_group)
        return flux_polygon, clipped_areas

    def compute_flux_from_polygon_clipping2(self, borders, clipper_borders, input_data, total_data_group):
        """ Compute flux on pixels covered by one polygon formed in the normal or vertical direction of the order.

        Collect input pixels covered by the output pixel and compute weighted summation on the input pixels.

        Args:
            borders(list): Borders of the coverage of one output pxiel.
            clipper_borders (list): Rectangular area covered by the clipper boundaries.
            input_data (list): Imaging data - list of numpy.ndarray containing spectrum data and flat data.
            total_data_group (int): total data group.

        Returns:
            tuple: Weighted summation of flux for spectrum data and flat data over the area covered by
            the output pixel.

            * **flux** (*list*): Weighted summation of the flux for spectrum and flat data.
            * **total_area** (*float*): Total overlapping area between the collected pixels and the output pixel.
            * **clip_areas** (*list*): List of input pixels overlapping with the output pixels and the overlapping areas
        """
        x_1, x_2, y_1, y_2 = clipper_borders
        total_area = 0.0
        flux = np.zeros(total_data_group)

        clipped_areas = list()
        for x in range(x_1, x_2):
            for y in range(y_1, y_2):
                # stime = time.time()
                # if len([d_group[y, x] for d_group in input_data if d_group[y, x] != 0.0]) > 0:
                new_corners = self.polygon_clipping2(borders, [[x, y], [x, y+1], [x+1, y+1], [x+1, y]], 4)
                area = self.polygon_area(new_corners)
                total_area += area
                if area > 0.0 and self.output_clip_area:
                    clipped_areas.append((int(x), int(y), float(area)))
                for n in range(total_data_group):
                    if input_data[n][y, x] != 0.0:
                        flux[n] += area * input_data[n][y, x]

        new_flux = flux/total_area if total_area != 0.0 else flux
        return new_flux, total_area, clipped_areas

    def polygon_clipping2(self, borders, clipper_points, clipper_size):
        """ Clip a polygon by an area enclosed by a set of straight lines of 2D domain.

        Args:
            borders(list): Borders of the coverage of one output pixel in counterclockwise order.
            clipper_points (list): Corners of clipping area in counterclockwise order.
            clipper_size (int): Total sides of the clipping area.

        Returns:
            numpy.ndarray: Corners of the polygon after clipping in counterclockwise order.

        """
        new_poly_borders = [b.copy() for b in borders]

        # print('\nclipper_points: ', clipper_points)
        # do clipping on each side of the clipper
        for i in range(clipper_size):
            k = (i+1) % clipper_size
            # print('\ni = ', i )
            new_poly_borders = self.clip2(new_poly_borders, clipper_points[i][0], clipper_points[i][1],
                                          clipper_points[k][0], clipper_points[k][1], i)
            # for b in new_poly_borders:
            #    print(b)

        new_corners = self.remove_duplicate_point2(new_poly_borders)

        return np.array(new_corners)

    @staticmethod
    def remove_duplicate_point2(borders):
        """ Remove the duplicate points from a list of corner points of a polygon.

        Args:
            borders (list): Borders of a polygon.

        Returns:
            list: Corner points of the polygon.

        """
        new_corners = []
        for b in borders:
            v1 = b[SpectralExtractionAlg.V1]
            v2 = b[SpectralExtractionAlg.V2]
            if v1 not in new_corners:
                new_corners.append(v1)
            if v2 not in new_corners:
                new_corners.append(v2)

        return new_corners

    @staticmethod
    def polygon_area(corners):
        """ Calculate the polygon area per polygon corners.

        Args:
            corners (numpy.ndarray): Corners of a polygon.

        Returns:
            float: Area of the polygon.

        """
        polygon_size = np.shape(corners)[0]
        area = 0.0
        for i in range(polygon_size):
            k = (i+1) % polygon_size
            area += corners[i, 0]*corners[k, 1] - corners[k, 0]*corners[i, 1]

        return abs(area)/2

    def clip2(self, poly_borders, x1, y1, x2, y2, clipper_direction):
        """ Clipping a polygon by a vector on 2D domain.

        The end points in the borders of the polygons, `poly_borders`, may be replaced by the intersected points
        between the clipping vector and the polygon borders.

        Args:
            poly_borders (list): List of borders of the polygon.
                Each border is a dict instance as described in :func:`~alg.SpectralExtractionAlg.collect_v_borders()`
                or :func:`~alg.SpectralExtractionAlg.collect_h_borders()`.
            x1 (int): x of end point 1 of the vector.
            y1 (int): y of end point 1 of the vector.
            x2 (int): x of end point 2 of the vector.
            y2 (int): y of end point 2 of the vector.
            clipper_direction (int): Clipping vector direction.

        Returns:
            list: Updated borders of the polygon after being clipped by the vector.

        """
        poly_size = len(poly_borders)
        new_poly_borders = list()

        def set_pos(border_pos, point_pos, is_bigger_inside=True):
            p_pos = 0
            if point_pos > border_pos:
                p_pos = -1 if is_bigger_inside else 1
            elif point_pos < border_pos:
                p_pos = 1 if is_bigger_inside else -1
            return p_pos

        # pick the reserved one if two intersection points are found
        def get_intersect_point(border_set, clipper_at, clipper_axis):
            p_axis = self.X if clipper_axis == self.Y else self.Y
            if clipper_at in border_set:
                res_point = [0.0, 0.0]
                res_point[p_axis] = border_set[clipper_at]['pos']
                res_point[clipper_axis] = clipper_at
                return res_point
            else:
                return None

        v1 = self.V1
        v2 = self.V2

        for i in range(poly_size):
            p_border = poly_borders[i]

            i_idx = v1
            k_idx = v2
            ix, iy = p_border[i_idx][0], p_border[i_idx][1]
            kx, ky = p_border[k_idx][0], p_border[k_idx][1]

            intersect_borders = p_border['intersect_with_borders']
            v_borders = intersect_borders['v_borders']
            h_borders = intersect_borders['h_borders']

            if clipper_direction == self.V_UP or clipper_direction == self.V_DOWN:
                i_pos = set_pos(x1, ix, clipper_direction == self.V_UP)
                k_pos = set_pos(x1, kx, clipper_direction == self.V_UP)
            else:
                i_pos = set_pos(y1, iy, clipper_direction == self.H_LEFT)
                k_pos = set_pos(y1, ky, clipper_direction == self.H_LEFT)

            if i_pos <= 0 and k_pos <= 0:                 # from inside|border to inside|border, keep the border
                new_poly_borders.append(p_border)
            # from outside to inside or inside to outside, replace outside with clipped pt.
            elif (i_pos > 0 and k_pos < 0) or (i_pos < 0 and k_pos > 0):
                if x1 == x2:
                    i_point = get_intersect_point(v_borders, x1, self.X)
                else:           # y1 == y2
                    i_point = get_intersect_point(h_borders, y1, self.Y)

                if i_point is not None:
                    if i_pos > 0:
                        p_border[i_idx] = i_point      # clipping point
                    else:
                        p_border[k_idx] = i_point
                    new_poly_borders.append(p_border)
                else:
                    print("no intersect found ", x1, x2, y1, y2)

        new_poly_size = len(new_poly_borders)
        ret_poly_borders = list()

        for i in range(new_poly_size):
            k = (i+1) % new_poly_size
            p1 = new_poly_borders[i][v2]
            p2 = new_poly_borders[k][v1]

            ret_poly_borders.append(new_poly_borders[i])

            if p1 != p2:
                intersect_borders = self.intersect_cell_borders(p1, p2)
                a_border = {
                        v1: p1,
                        v2: p2,
                        'direction': -1,
                        'intersect_with_borders': intersect_borders
                }
                ret_poly_borders.append(a_border)

        return ret_poly_borders

    @staticmethod
    def line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
        """ Find the intersection of two lines on 2D by assuming there is the intersetion between these 2 lines.

        Args:
            x1 (int): x of end point 1 of line 1.
            y1 (int): y of end point 1 of line 1.
            x2 (int): x of end point 2 of line 1.
            y2 (int): y of end point 2 of line 1.
            x3 (int): x of end point 1 of line 2.
            y3 (int): y of end point 1 of line 2.
            x4 (int): x of end point 2 of line 2.
            y4 (int): y of end point 2 of line 2.

        Returns:
            list: Intersect point containing values of x and y coordinates.

        """
        den = (x1-x2)*(y3-y4) - (x3-x4)*(y1-y2)
        num_x = (x1*y2 - x2*y1) * (x3 - x4) - (x1 - x2) * (x3*y4 - x4*y3)
        num_y = (x1*y2 - x2*y1) * (y3 - y4) - (y1 - y2) * (x3*y4 - x4*y3)

        return [num_x/den, num_y/den]

    def write_data_to_dataframe(self, result_data, first_row=None):
        """ Write spectral extraction result to an instance of Pandas DataFrame.

        Args:
            result_data (numpy.ndarray): Spectral extraction result.  Each row of the array corresponds to the reduced
                1D data of one order.

        Returns:
            Pandas.DataFrame: Instance of DataFrame containing the extraction result plus the following attributes:

                - *MJD-OBS*: modified Julian date of the observation.
                - *EXPTIME*: exposure time of the observation.
                - *TOTALORD*: total order in the result data.
                - *DIMWIDTH*: Width of the order in the result data.

        """
        header_keys = list(self.spectrum_header.keys())
        flux_header = self.spectrum_header

        mjd = 0.0
        m_d = 2400000.5
        if 'SSBJD100' in header_keys:
            mjd = flux_header['SSBJD100'] - m_d
        elif 'OBSJD' in header_keys:
            mjd = flux_header['OBSJD'] - m_d
        elif 'OBS MJD' in header_keys:
            mjd = flux_header['OBS MJD']

        expt = 'ELAPSED'
        if expt in header_keys:
            exptime = flux_header[expt]
        else:
            exptime = 600.0

        total_order, dim_width = np.shape(result_data)
        # result_table = {'order_'+str(i): result_data[i, :] for i in range(total_order) }
        # df_result = pd.DataFrame(result_table)
        df_result = pd.DataFrame(result_data)
        if mjd != 0.0:
            df_result.attrs['MJD-OBS'] = mjd
            df_result.attrs['OBSJD'] = mjd + 2400000.5
        df_result.attrs['ELAPSED'] = exptime
        df_result.attrs['TOTALORD'] = total_order
        df_result.attrs['FIRSTORD'] = self.start_row_index() if first_row is None else first_row
        df_result.attrs['FROMIMGX'] = self.origin[self.X]
        df_result.attrs['FROMIMGY'] = self.origin[self.Y]

        return df_result

    def write_rectified_data_to_dataframe(self, result_data, rectification_result):
        """ Write rectification result to an instance of Pandas DataFrame.

        Args:
            result_data (numpy.ndarray): Rectification result.
                The 2D result data is the rectified order trace by using one of the rectification method.
            rectification_result (list): Result metadata for each recfitifed order

        Returns:
            Pandas.DataFrame: Instance of DataFrame containing the extraction result plus the following attributes:

                - *TOTALORD*: total order in the result data.
                - *RECTIFYM*: rectification method
                - *RAWSIZE*:  original raw image size
                - *ORD_nnn*:  location information of order nnn, y_center,lower_width,upper_width

        """
        df_result = pd.DataFrame(result_data)
        df_result.attrs['TOTALORD'] = len(rectification_result)
        df_result.attrs[self.RECTIFYKEY] = self.rectifying_method[self.rectification_method]
        w, h = self.get_spectrum_size()
        df_result.attrs[self.RAWSIZEKEY] = str(h)+','+str(w)

        for i in range(len(rectification_result)):
            order_rect = rectification_result[i].get('rectification')
            order_idx = rectification_result[i].get("order")
            y_center = order_rect.get("out_y_center")
            edge_low, edge_top = order_rect.get('edges')
            df_result.attrs['ORD_' + str(order_idx)] = \
                (str(y_center) + ',' + str(edge_low) + ',' + str(edge_top),
                 'y,lower_edge,upper_edge')

        return df_result

    def get_total_orderlets_from_image(self):
        """ Get total orderlets from level 0 image, defined in config

        Returns:
            int: total orderdelettes.
        """
        if self.total_image_orderlets is None:
            self.total_image_orderlets = self.get_config_value("total_image_orderlets", 1)

        return self.total_image_orderlets

    def get_orderlet_names(self):
        """ Get Orderlette names defined in config.

        Returns:
            list: list of orderlet names
        """
        if self.orderlet_names is None:
            o_list = self.get_config_value('orderlet_names', "['SCI']")
            if isinstance(o_list, list):
                self.orderlet_names = o_list
            elif isinstance(o_list, str):
                self.orderlet_names = o_list.split(',')

        return self.orderlet_names

    def get_total_order_for(self, ccd_index=None):
        """

        Args:
            ccd_index (int): ccd index for possible ccd array. eg. ccd_index = 0 is for green else for red for KPF.

        Returns:
            int: total orders for the specified ccd.
        """
        if ccd_index is None:
            return None

        if self.total_order_per_ccd is None:
            total_order_per_ccd = self.get_config_value('total_order_per_ccd', '[]')
            if isinstance(total_order_per_ccd, list):
                self.total_order_per_ccd = total_order_per_ccd
            elif isinstance(total_order_per_ccd, str):
                self.total_order_per_ccd = [int(t) for t in total_order_per_ccd.split(',')]

        return self.total_order_per_ccd[ccd_index] if ccd_index < len(self.total_order_per_ccd) else None

    def get_orderlet_index(self, order_name):
        """ Find the index of the order name in the orderlet name list.

        Args:
            order_name (str): Fiber name

        Returns:
            int: index of order name in the orderlet name list. If not existing, return is 0.

        """
        all_names = self.get_orderlet_names()
        traces_per_order = self.get_total_orderlets_from_image()
        order_name_idx = all_names.index(order_name) if order_name in all_names else 0
        order_name_idx = order_name_idx % traces_per_order

        return order_name_idx

    def start_row_index(self):
        """ The row index for the flux of the first order in the output.

        Returns:
            int: the row index.
        """
        return self.get_config_value('start_order', 0)

    def get_order_set(self, order_name=''):
        """ Get the list of the trace index eligible for spectral extraction process.

        Args:
            order_name (str): Fiber name.

        Returns:
            list: list of the trace index.

        """
        orderlet_index = self.get_orderlet_index(order_name)
        traces_per_order = self.get_total_orderlets_from_image()

        total_order_per_ccd = self.get_total_order_for(self.ccd_index) if self.ccd_index is not None else None
        total_traces = total_order_per_ccd * traces_per_order if total_order_per_ccd is not None else self.total_order

        if orderlet_index < traces_per_order:
            o_set = np.arange(orderlet_index, total_traces, traces_per_order, dtype=int)
        else:
            o_set = np.array([])

        return o_set

    @staticmethod
    def write_clip_file(poly_file, y_center=None, edges=None, clip_areas=None, order=None):
        """Write polygon clipping information to file.

        Args:
            poly_file (str): File name of the clip file
            y_center (int): Y center location.
            edges (list): Lower edge and uppper edge of the order
            clip_areas (dict): Object containing polygon clipping information in format like::

                {
                    {
                        y_loc_1: {x_loc_1: [(x_1, y_1, area_1), ..., (x_n, y_n, area_n)],
                                  ...,  x_loc_n: [(...),..., (...)]},
                            :
                        y_loc_n: {x_loc_1: [...],..., x_loc_n: [...]}
                    }

                    # where output pixel [x_loc_i, y_loc_i] overlaps with
                    # input pixel [x_i, y_i] and the overlapping area is area_n.
                }
            order (int): Order to be written.
        Returns:
            The polygon clipping information is written to .npy file.
        """
        order_clip = dict()
        if y_center is not None:
            order_clip['y_center'] = y_center
        if edges is not None:
            order_clip['edges'] = edges
        if clip_areas is not None:
            order_clip['clip_areas'] = clip_areas
        # order_clip_array = np.array(list(order_clip.items()))

        # if order is not None:
        #    self.poly_clip_dict[int(order)] = order_clip

        # f = open(poly_file, "wb")
        if poly_file is not None:
            with open(poly_file, "w") as outfile:
                json.dump(order_clip, outfile)

        #    order_set = self.poly_clip_dict.keys()
        #    for od in order_set:
        #        poly_file_order = poly_file + str(od) + '.json'
        #        with open(poly_file_order, "w") as outfile:
        #            json.dump(self.poly_clip_dict[od], outfile)

        # np.save(poly_file, order_clip_array)
        # f.close()

    @staticmethod
    def read_clip_file(poly_file):
        """Read order polygon data from clip file.

        Args:
            poly_file (str): File name of the clip file

        Returns:
            tuple: Polygon clip information read from the clip file.

                * **y_center** (*int*): y location of the rectified order.
                * **lower_width** (*int*): lower edge of the rectified order.
                * **upper_width** (*int*): upper edge of the rectified order.
                * **clip_areas** (*dict*): polygon clip information for the order.

        """
        with open(poly_file) as clip_input:
            order_poly = json.load(clip_input)

        # if self.poly_clip_dict is None or len(self.poly_clip_dict)==0:
        #    with open(poly_file) as clip_input:
        #        f = json.load(clip_input)
        #        self.poly_clip_dict = f

        # order_poly = self.poly_clip_dict[str(c_order)]

        y_center = order_poly['y_center'] if order_poly else None
        widths = order_poly['edges'] if order_poly else [None, None]
        clip_areas = order_poly['clip_areas'] if order_poly else None

        return y_center, widths[0], widths[1], clip_areas

    def reset_clip_file(self):
        """Reset the flag to output the clip information.
        """
        self.output_clip_area = False

    def get_clip_file(self, order_idx=None):
        """Compute the full path of the clip file for the specified order per clip file prefix.

        Args:
            order_idx: Index of the order.

        Returns:
            str: full path of the clip file.

        """
        # crt_order_clip_file = self.clip_file_prefix + '_order_' + str(order_idx) + '.npy' \
        #    if (self.clip_file_prefix and order_idx is not None) else None
        if order_idx is None:
            return self.clip_file_prefix + '_order_' if self.clip_file_prefix else None
        else:
            crt_order_clip_file = self.clip_file_prefix + '_order_' + str(order_idx) + '.json' \
                if self.clip_file_prefix else None

        return crt_order_clip_file

    def update_output_size(self, order_set, order_result, result_height, s_rate=1):
        """Compute the dimension for outputting the spectral extraction or rectification result.

        Args:
            order_set (list): Collection of order index.
            order_result (dict):  Object contains the rectified curve size information.
            result_height (dict): Height of the spectral extraction.
            s_rate (Union[list, int], optional): Sampling rate.

        Returns:
            tuple: Dimension of output data, height and width.

        """
        output_data_width, s_height = self.get_spectrum_size()
        if len(order_set) <= 0:
            output_data_height = result_height
        else:
            sampling_rate = s_rate if isinstance(s_rate, list) else [s_rate, s_rate]
            output_data_width = output_data_width * sampling_rate[self.X]
            s_height = s_height * sampling_rate[self.Y]
            if self.extraction_method != self.NOEXTRACT:
                output_data_height = result_height
            else:
                last_order = order_result[order_set[-1]]
                output_data_height = max(last_order.get('out_y_center') + last_order.get('edges')[1], s_height)

        return output_data_height, output_data_width

    @staticmethod
    def compute_variance(flux_data):
        var_ext = np.array(flux_data)
        var_ext = np.absolute(var_ext)
        return var_ext

    def extract_spectrum(self,
                         order_set,
                         proc_no,
                         return_dict,
                         spectrum_file,
                         order_name=None,
                         show_time=False,
                         print_debug=None,
                         bleeding_file=None,
                         first_index=0):
        """ Spectral extraction from 2D flux to 1D. Rectification step is optional.

        Args:
            order_set (numpy.ndarray, optional): Set of orders to extract. Defaults to None for all orders.
            order_name (str, optional): Name of the orderlet to be processed.
            show_time (bool, optional):  Show running time of the steps. Defaults to False.
            print_debug (str, optional): Print debug information to stdout if it is provided as empty string,
                a file with path `print_debug` if it is non empty string, or no print if it is None.
                Defaults to None.
            bleeding_file (str, optioanl): Bleeding cure file, such as that for PARAS data. Defaults to None.

        Returns:
            dict: Spectral extraction result from 2D spectrum data, like::

                    {
                        'spectral_extraction_result':  Padas.DataFrame
                            # table storing spectral extraction or rectification result.
                            # each row of the table containing the spectral extraction
                            # or rectification only.
                            # result for all orders set in order_set.
                        'rectification_on': string
                            # for the case of doing rectification on 'spectrum' or 'flat' only
                    }

        """
        self.order_name = order_name
        #self.add_file_logger(print_debug)
        #self.enable_time_profile(show_time)
        if self.spectrum_flux is not None:
            self.update_spectrum_flux(bleeding_file)

        if order_set is None:
            order_set = self.get_order_set(order_name)
        '''
        print('SpectralExtractionAlg: do ', self.rectifying_method[self.rectification_method],
                     'rectification and ',
                     self.extracting_method[self.extraction_method], 'extraction on ',
                     order_set.size, ' orders')
        '''

#        t_start = self.start_time()
        noop = False
        data_df = None
        rectification_on = ''

        # rectification on spectrum & flat if extraction method is set & data is raw flat
        if self.extraction_method == SpectralExtractionAlg.NOEXTRACT:
            rectification_on = 'spectrum' if self.spectrum_flux is not None else 'flat'

        if self.spectrum_flux is not None:
            # spectrum rectified yet
            if not self.is_raw_spectrum and self.extraction_method == SpectralExtractionAlg.NOEXTRACT:  # spec rectified
                data_df = self.write_data_to_dataframe(self.spectrum_flux)
                noop = True
        elif self.extraction_method == SpectralExtractionAlg.NOEXTRACT:    # do rectification on flat
            if not self.is_raw_flat:
                noop = True
                data_df = self.write_data_to_dataframe(self.flat_flux)
        else:                                               # no operation if do spectral extraction on null spectrum
            noop = True

        if noop:
            return {'spectral_extraction_result': data_df, 'rectification_on': rectification_on}

        # rectification_on == flat or spectrum  => do rectification on flat or spectrum, no extraction
        #                     data_group contains one data set
        # rectification_on = "" => do extraction on spectrum
        #                     data_group contains spectrum and flat data
        data_group = list()     # containing the data set needed for spectral extraction or rectification only

        if rectification_on != 'flat':
            data_group.append({'data': self.spectrum_flux, 'is_raw_data': self.is_raw_spectrum, 'idx': self.SDATA})
        if rectification_on != 'spectrum':
            data_group.append({'data': self.flat_flux, 'is_raw_data': self.is_raw_flat, 'idx': self.FDATA})

        # this is for not self.NOEXTRACT (not rectification)
        start_row_at = first_index if first_index is not None else self.start_row_index()

        total_order_per_ccd = self.get_total_order_for(self.ccd_index) if self.ccd_index is not None else None
        if total_order_per_ccd is not None:
            result_height = max(total_order_per_ccd, order_set.size + start_row_at)
        else:
            result_height = order_set.size + start_row_at if order_set.size > 0 else 0

        # re-allocate the space to hold the rectified spectrum for the order
        self.allocate_memory_flux(order_set, 2)
        order_result = dict()
        self.output_area_info = list()

        # for idx_out in range(2):
        for idx_out in range(order_set.size):
            c_order = order_set[idx_out]
            self.reset_clip_file()
            #print('SpectralExtractionAlg: ', c_order, ' edges: ', self.get_order_edges(c_order),
            #             ' xrange: ', self.get_order_xrange(c_order))

            order_flux = self.get_flux_from_order(data_group, c_order)
            

            # prepare out_data to contain rectification result
            order_result[c_order] = order_flux
            #t_start = self.time_check(t_start, '**** time [' + str(c_order) + ']: ')

        # if self.poly_clip_update:
        #    self.write_clip_file(self.get_clip_file())
        out_data_height, out_data_width = self.update_output_size(order_set, order_result, result_height)
        out_data = np.zeros((out_data_height, out_data_width))
        order_rectification_result = list()
        # produce output data

        for idx_out in range(order_set.size):
            c_order = order_set[idx_out]
            to_pos = idx_out+start_row_at if self.extraction_method != self.NOEXTRACT else \
                order_result[c_order].get('out_y_center') - order_result[c_order].get('edges')[0]
            y_size = np.sum(order_result[c_order].get('edges')) if self.extraction_method == self.NOEXTRACT else 1
            self.fill_2d_with_data(order_result[c_order].get('extracted_data'), out_data, to_pos, 0, height=y_size)
            if self.extraction_method == self.NOEXTRACT:
                order_rectification_result.append({"order": c_order, "rectification": order_result[c_order]})
            #t_start = self.time_check(t_start, '**** time ['+str(c_order)+']: ')
 

        #Write the rectified image to the input file
        if self.extraction_method == self.NOEXTRACT:
            with fits.open(spectrum_file,mode='update') as hdul:
                hdul["RECT"].data +=out_data
                hdul.flush()
                
            df_result_out = []
            for i in range(len(order_rectification_result)):
                order_rect = order_rectification_result[i].get("rectification")
                order_idx=(order_rectification_result[i].get("order"))
                edge_low,edge_top=(order_rect.get("edges"))
                y_center=(order_rect.get("out_y_center"))
                df_result={}
                df_result['order']= order_idx
                df_result['Coeff0'] = y_center
                df_result['BottomEdge'] = edge_low
                df_result['TopEdge'] = edge_top
                df_result['X1'] = 0
                df_result['X2'] =self.spectrum_flux.shape[1]-1
                df_result_out.append(df_result)

            df_result = pd.DataFrame(df_result_out)
            return_dict[proc_no] = df_result
                
        else:
            data_df = self.write_data_to_dataframe(out_data, first_row=start_row_at)
            
            return_dict[proc_no] = {'spectral_extraction_result': data_df}
            #return {'spectral_extraction_result': data_df, 'rectification_on': rectification_on,
            #        'outlier_rejection_result':self.outlier_flux}
                

