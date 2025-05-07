from astropy import units as u, constants as cst
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import Legendre
from modules.Utils.utils import DummyLogger
import os
import time
import pandas as pd
import scipy
from scipy import signal


class WaveCalibration:

    def __init__(
        self, cal_type, clip_peaks_toggle, quicklook, min_order, max_order, save_diagnostics=None,
        config=None, logger=None
    ):
        """Initializes WaveCalibration class.
        Args:
            clip_peaks_toggle (bool): Whether or not to clip any peaks. True to clip, false to not clip.
            quicklook (bool): Whether or not to run quicklook-specific algorithmic steps. False runs non-quicklook, full pipeline version.
            min_order (int): minimum order to fit
            max_order (int): maximum order to fit
            save_diagnostics (str) : Directory in which to save diagnostic plots and information. Defaults to None, which results
                in no saved diagnostics info.
            config (configparser.ConfigParser, optional): Config context.
                Defaults to None.
            logger (logging.Logger, optional): Instance of logging.Logger.
                Defaults to None.
        """
        self.cal_type = cal_type
        self.clip_peaks_toggle = clip_peaks_toggle
        self.quicklook = quicklook
        self.min_order = min_order
        self.max_order = max_order
        self.save_diagnostics_dir = save_diagnostics
        configpull = ConfigHandler(config,'PARAM')
        self.figsave_name = configpull.get_config_value('drift_figsave_name','instrument_drift')
        self.red_skip_orders = configpull.get_config_value('red_skip_orders')
        self.green_skip_orders = configpull.get_config_value('green_skip_orders')
        self.chi_2_threshold = configpull.get_config_value('chi_2_threshold')
        self.skip_orders = configpull.get_config_value('skip_orders',None)
        self.quicklook_steps = configpull.get_config_value('quicklook_steps',10)
        self.min_wave = configpull.get_config_value('min_wave',3800)
        self.max_wave = configpull.get_config_value('max_wave',9300)
        self.fit_order = configpull.get_config_value('fit_order',9)
        self.fit_type = configpull.get_config_value('fit_type', 'Legendre')
        self.n_sections = configpull.get_config_value('n_sections',1)
        self.clip_peaks_toggle = configpull.get_config_value('clip_peaks',False)
        self.clip_below_median  = configpull.get_config_value('clip_below_median',True)
        self.peak_height_threshold = configpull.get_config_value('peak_height_threshold',1.5)
        self.sigma_clip = configpull.get_config_value('sigma_clip',2.1)
        self.fit_iterations = configpull.get_config_value('fit_iterations',5)
        self.logger = logger
        self.etalon_mask_in = configpull.get_config_value('master_etalon_file',None)


    def run_wavelength_cal(
        self, calflux, rough_wls=None, our_wavelength_solution_for_order=None,
        peak_wavelengths_ang=None, lfc_allowed_wls=None,input_filename=None):
        """ Runs all wavelength calibration algorithm steps in order.
        Args:
            calflux (np.array): (N_orders x N_pixels) array of L1 flux data of a
                calibration source
            rough_wls (np.array): (N_orders x N_pixels) array of wavelength
                values describing a "rough" wavelength solution. Always None for
                lamps. For LFC, this is generally a lamp-derived solution.
                For Etalon, this is generally an LFC-derived solution. Default None.
            peak_wavelengths_ang (dict of dicts): dictionary of order number-dict
                pairs. Each order number corresponds to a dict containing
                an array of expected line positions (pixel values) under the key
                "line_positions" and an array of corresponding wavelengths in
                Angstroms under the key "known_wavelengths_vac". This value must be
                set for lamps. Can be set or not set for LFC and Etalon. If set to None,
                then peak finding is not run. Defaults to None. Ex:
                    {51: {
                            "line_positions" : array([500.2, ... 8000.3]),
                            "known_wavelengths_vac" : array([3633.1, ... 3570.1])
                        }
                    }
        
            lfc_allowed_wls (np.array): array of all allowed wavelengths for the
                LFC, computed using the order_flux equation. Should be None unless we
                are calibrating an LFC frame. Defaults to None.
        Examples:
            1: Calibrating an LFC frame using a rough ThAr solution,
               with no previous LFC frames to inform this one:
                rough_wls -> ThAr-derived wavelength solution
                lfc_allowed_wls -> wavelengths computed from comb eq
            2: Calibrating an LFC frame using a rough ThAr solution,
               given information about expected mode position:
                rough_wls -> ThAr-derived wavelength solution
                lfc_allowed_wls -> wavelengths computed from comb eq
                peak_wavelengths_ang -> LFC mode wavelengths and their
                    expected pixel locations
            3: Calibrating a lamp frame:
                peak_wavelengths_ang -> lamp line wavelengths in vacuum and their
                    expected rough pixel locations
            4: Calibrating an Etalon frame using an LFC-derived solution, with
               no previous Etalon frames to inform this one:
                rough_wls -> LFC-derived wavelength solution
            5: Calibrating an Etalon frame using an LFC-derived solution and
               at least one other Etalon frame to inform this one:
                rough_wls -> LFC-derived wavelength solution
                peak_wavelengths_ang -> Etalon peak wavelengths and their
                    expected pixel locations
        Returns:
            tuple of:
                np.array: Calculated polynomial solution
                np.array: (N_orders x N_pixels) array of the computed wavelength
                    for each pixel.
                dictionary: information about the fits for each line and order (orderlet_dict)
        """
        self.filename=input_filename
        # create directories for diagnostic plots
        if type(self.save_diagnostics_dir) == str:
            if not os.path.isdir(self.save_diagnostics_dir):
                os.makedirs(self.save_diagnostics_dir)
            if not os.path.isdir(self.save_diagnostics_dir + '/order_diagnostics'):
                os.makedirs(self.save_diagnostics_dir + '/order_diagnostics')

        if self.quicklook == False:
            order_list = self.remove_orders(step=1)
            n_orders = len(order_list)

            # masked_calflux = self.mask_array_neid(calflux, n_orders)
            masked_calflux = calflux # TODO: fix

            # perform wavelength calibration
            poly_soln, wls_and_pixels, orderlet_dict = self.fit_many_orders(
                masked_calflux, order_list, rough_wls=rough_wls,
                comb_lines_angstrom=lfc_allowed_wls,
                expected_peak_locs=peak_wavelengths_ang, peak_wavelengths_ang=peak_wavelengths_ang,
                our_wavelength_solution_for_order=our_wavelength_solution_for_order, print_update=True, plt_path=self.save_diagnostics_dir
            )

            # make a plot of all of the precise new wls minus the rough input  wls
            if self.save_diagnostics_dir is not None and rough_wls is not None:
                # don't do this for etalon exposures, where we're either not
                # deriving a new wls or using drift to do so
                if self.cal_type != 'Etalon':
                    fig, ax = plt.subplots(2,1, figsize=(12,5))
                    for i in order_list:
                        wls_i = poly_soln[i, :]
                        rough_wls_i = rough_wls[i,:]
                        ax[0].plot(wls_i - rough_wls_i, color='grey', alpha=0.5)

                        pixel_sizes = rough_wls_i[1:] - rough_wls_i[:-1]
                        ax[1].plot(
                            (wls_i[:-1] - rough_wls_i[:-1]) / pixel_sizes,
                            color='grey', alpha=0.5
                        )

                    ax[0].set_title('Derived WLS - Approx WLS')
                    ax[0].set_xlabel('Pixel')
                    ax[0].set_ylabel('[$\\rm \AA$]')
                    ax[1].set_xlabel('Pixel')
                    ax[1].set_ylabel('[Pixel]')
                    plt.tight_layout()
                    plt.savefig(
                        '{}/all_wls.png'.format(self.save_diagnostics_dir),
                        dpi=250
                    )

        if self.quicklook == True:
            #TODO
            order_list = self.remove_orders(step = self.quicklook_steps)
            n_orders = len(order_list)
            
            #masked_calflux = self.mask_array_neid(calflux,n_orders)
            masked_calflux = calflux
            
            poly_soln, wls_and_pixels, orderlet_dict = self.fit_many_orders(
                masked_calflux, order_list, rough_wls=rough_wls,
                comb_lines_angstrom=lfc_allowed_wls,
                expected_peak_locs=peak_wavelengths_ang, peak_wavelengths_ang=peak_wavelengths_ang,
                print_update=True, plt_path=self.save_diagnostics_dir ###CHECK THIS TODO
            )

        return poly_soln, wls_and_pixels, orderlet_dict
