from astropy import units as u, constants as cst
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy.interpolate import interp1d
import os
import time
import pandas as pd
import scipy
from scipy import signal
import warnings
from scipy.optimize.minpack import curve_fit
from scipy.special import erf


class WaveCalAlg:

    def __init__(
        self, cal_type, logger, save_diagnostics=None, config=None,plot=False):
        """Initializes WaveCalibration class.
        Args:
            clip_peaks_toggle (bool): Whether or not to clip any peaks. True to clip, false to not clip.
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
#        self.clip_peaks_toggle = clip_peaks_toggle
#        self.min_order = min_order
#        self.max_order = max_order
        self.save_diagnostics_dir = save_diagnostics
#        configpull = ConfigHandler(config,'PARAM')
#        self.figsave_name = configpull.get_config_value('drift_figsave_name','instrument_drift')
        self.red_skip_orders = None #configpull.get_config_value('red_skip_orders')
        self.green_skip_orders = None #configpull.get_config_value('green_skip_orders')
        self.chi_2_threshold = 2 #configpull.get_config_value('chi_2_threshold')
        self.skip_orders = None #configpull.get_config_value('skip_orders',None)
        self.quicklook_steps = 1 #configpull.get_config_value('quicklook_steps',10)
        self.min_wave = 3600 #configpull.get_config_value('min_wave',3800)
        self.max_wave = 9500 #configpull.get_config_value('max_wave',9300)
        self.fit_order = 9 #configpull.get_config_value('fit_order',9)
        self.fit_type = 'Legendre' #configpull.get_config_value('fit_type', 'Legendre')
        self.n_sections = 1 #configpull.get_config_value('n_sections',1)
        self.clip_peaks_toggle = False #configpull.get_config_value('clip_peaks',False)
        self.clip_below_median  = True #configpull.get_config_value('clip_below_median',True)
        self.peak_height_threshold = 1.5 #configpull.get_config_value('peak_height_threshold',1.5)
        self.sigma_clip = 2.1 #configpull.get_config_value('sigma_clip',2.1)
        self.fit_iterations = 5 #configpull.get_config_value('fit_iterations',5)
        self.logger = logger
        self.etalon_mask_in = None #configpull.get_config_value('master_etalon_file',None)
        self.plot = plot

    def run_wavelength_cal(
        self, calflux, rough_wls=None, our_wavelength_solution_for_order=None,
        peak_wavelengths_ang=None, lfc_allowed_wls=None,input_filename=None,fibre =None,plot=False):
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
        self.fibre = fibre
        # create directories for diagnostic plots
#        if type(self.save_diagnostics_dir) == str:
#            if not os.path.isdir(self.save_diagnostics_dir):
#                os.makedirs(self.save_diagnostics_dir)
#            if not os.path.isdir(self.save_diagnostics_dir + '/order_diagnostics'):
#                os.makedirs(self.save_diagnostics_dir + '/order_diagnostics')


        n_orders = calflux.shape[0]
        order_list = np.arange(n_orders)

        # masked_calflux = self.mask_array_neid(calflux, n_orders)
        masked_calflux = calflux # TODO: fix

        # perform wavelength calibration
        poly_soln, wls_and_pixels, orderlet_dict, absolute_precision, order_precisions = self.fit_many_orders(
            masked_calflux, order_list, rough_wls=rough_wls,
            comb_lines_angstrom=lfc_allowed_wls,
            expected_peak_locs=peak_wavelengths_ang, peak_wavelengths_ang=peak_wavelengths_ang,
            our_wavelength_solution_for_order=our_wavelength_solution_for_order, print_update=True, plt_path=self.save_diagnostics_dir
        )

        # make a plot of all of the precise new wls minus the rough input  wls
        if self.save_diagnostics_dir is not None and rough_wls is not None:
            if plot:
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
                        '{}/all_wls_{}.png'.format(self.save_diagnostics_dir,self.fibre),
                        dpi=500
                    )
                    plt.close()


        return poly_soln, wls_and_pixels, orderlet_dict, absolute_precision, order_precisions

    def fit_many_orders(
        self, cal_flux, order_list, rough_wls=None, comb_lines_angstrom=None,
        expected_peak_locs=None,peak_wavelengths_ang=None, our_wavelength_solution_for_order=None, plt_path=None, print_update=False):
        """
        Iteratively performs wavelength calibration for all orders.
        Args:
            cal_flux (np.array): (n_orders x n_pixels) array of calibrator fluxes
                for which to derive a wavelength solution
            order_list (list of int): list order to compute wls for
            rough_wls (np.array): (N_orders x N_pixels) array of wavelength
                values describing a "rough" wavelength solution. Always None for
                lamps. For LFC, this is generally a lamp-derived solution.
                For Etalon, this is generally an LFC-derived solution. Default None.
            comb_lines_angstrom (np.array): array of all allowed wavelengths for the
                LFC, computed using the order_flux equation. Should be None unless we
                are calibrating an LFC frame. Default None.
            expected_peak_locs (dict): dictionary of order number-dict
                pairs. See description in run_wavelength_cal().
            plt_path (str): if set, all diagnostic plots will be saved in this
                directory. If None, no plots will be made.
            print_update (bool): whether subfunctions should print updates.
        Returns:
            tuple of:
                np.array of float: (N_orders x N_pixels) derived wavelength
                    solution for each pixel
                dict: the peaks and wavelengths used for wavelength cal. Keys
                    are ints representing order numbers, values are 2-tuples of:
                        - lists of wavelengths corresponding to peaks
                        - the corresponding (fractional) pixels on which the
                          peaks fall
                dict: the orderlet dictionary, that is folded into wls_dict at a higher level
        """
        
        # Construct dictionary for each order in wlsdict
        orderlet_dict = {}
        for order_num in order_list:
            orderlet_dict[order_num] = {"ordernum" : order_num}

        # Plot 2D extracted spectra
        if plt_path is not None and self.plot:
            plt.figure(figsize=(20,10), tight_layout=True)
            im = plt.imshow(cal_flux, aspect='auto',origin='lower')
            im.set_clim(0, 20000)
            plt.xlabel('Pixel')
            plt.ylabel('Order Number')
            plt.savefig('{}/extracted_spectra_{}.png'.format(plt_path,self.fibre), dpi=600)
            plt.close()

        # Define variables to be used later
        order_precisions = []
        num_detected_peaks = []
        wavelengths_and_pixels = {}
        poly_soln_final_array = np.zeros(np.shape(cal_flux))

        # Iterate over orders
        for order_num in order_list:
            if print_update:
                print('\nRunning order # {}'.format(order_num))

            if plt_path is not None:
                order_plt_path = '{}/order_diagnostics/order{}'.format(plt_path, order_num)
                if not os.path.isdir(order_plt_path):
                    os.makedirs(order_plt_path)

                plt.figure(figsize=(20,10), tight_layout=True)
                #plt.plot(cal_flux[order_num,:], color='k', alpha=0.5)
                plt.plot(cal_flux[order_num,:], color='k', linewidth = 0.5)
                plt.title('Order # {}'.format(order_num), fontsize=36)
                plt.xlabel('Pixel', fontsize=28)
                plt.ylabel('Flux', fontsize=28)
                plt.yscale('symlog')
                plt.tick_params(axis='both', direction='inout', length=6, width=3, colors='k', labelsize=24)
                plt.savefig('{}/order_spectrum_{}.png'.format(order_plt_path,self.fibre), dpi=500)
                plt.close()
            else:
                order_plt_path = None

            order_flux = cal_flux[order_num,:]
            rough_wls_order = rough_wls[order_num,:]
            n_pixels = len(order_flux)
            
            # Add information for this order to the orderlet dictionary
            orderlet_dict[order_num]['flux'] = order_flux
            orderlet_dict[order_num]['initial_wls'] = rough_wls_order
 #           orderlet_dict[order_num]['echelle_order'] = echelle_ord[order_num]
            orderlet_dict[order_num]['n_pixels'] = n_pixels
            orderlet_dict[order_num]['lines'] = {}

            # check if there's flux in the orderlet (e.g., SKY order 0 is off of the GREEN CCD)
            npixels_wflux = len([x for x in order_flux if x != 0])
            if npixels_wflux == 0:
                self.logger.warn('This order has no flux, defaulting to rough WLS')
                continue

            if self.cal_type == 'Etalon':  # For etalon
                etalon_mask = pd.read_csv(self.etalon_mask_in, names=['wave','weight'], sep='\s+')
                wls, fitted_peak_pixels = self.find_etalon_peaks(order_flux,rough_wls_order,etalon_mask) # returns original mask and new mask positions for one order.
                wls=wls.tolist()

            # find, clip, and compute precise wavelengths for peaks.
            # this code snippet will only execute for Etalon and LFC frames.
            elif expected_peak_locs is None:
                skip_orders_wls = None
                if self.red_skip_orders and max(order_list) == 31:  # KPF max order for red chip (update if changed in KPF.cfg)
                    skip_orders_wls = np.fromstring(self.red_skip_orders, dtype=int, sep=',')
                elif self.green_skip_orders and max(order_list) == 34:  # KPF max order for green chip (update if changed in KPF.cfg)
                    skip_orders_wls = np.fromstring(self.green_skip_orders, dtype=int, sep=',')

                if skip_orders_wls is not None:
                    try:
                        if order_num in skip_orders_wls:
                            raise Exception(f'Order {order_num} is skipped in the config, defaulting to rough WLS')
                    except Exception as e:
                        print(e)
                        poly_soln_final_array[order_num, :] = rough_wls_order
                        wavelengths_and_pixels[order_num] = {
                            'known_wavelengths_vac': rough_wls_order,
                            'line_positions': []
                        }
                        continue

                try:
                    fitted_peak_pixels, detected_peak_pixels, \
                        detected_peak_heights, gauss_coeffs, lines_dict = self.find_peaks_in_order(
                        order_flux, plot_path=order_plt_path
                    )
                    orderlet_dict[order_num]['lines'] = lines_dict
                    
                except TypeError as e:
                    self.logger.warn('Not enough peaks found in order, defaulting to rough WLS')
                    self.logger.warn('TypeError = ' + str(e))
                    poly_soln_final_array[order_num,:] = rough_wls_order
                    wavelengths_and_pixels[order_num] = {
                        'known_wavelengths_vac': rough_wls_order,
                        'line_positions':[]
                    }
                    order_dict = {}
                    continue

                if self.clip_peaks_toggle:
                    good_peak_idx = self.clip_peaks(
                        order_flux, fitted_peak_pixels, detected_peak_pixels,
                        gauss_coeffs, detected_peak_heights,
                        clip_below_median=self.clip_below_median,
                        plot_path=order_plt_path, print_update=print_update
                    )
                else:
                    good_peak_idx = np.arange(len(detected_peak_pixels))

                if self.cal_type == 'LFC':
                    try:
                        wls, _, good_peak_idx = self.mode_match(
                            order_flux, fitted_peak_pixels, good_peak_idx,
                            rough_wls_order, comb_lines_angstrom,
                            print_update=print_update, plot_path=order_plt_path
                        )
                    except:
                        poly_soln_final_array[order_num,:] = rough_wls_order
                        wavelengths_and_pixels[order_num] = {
                            'known_wavelengths_vac': rough_wls_order,
                            'line_positions':[]
                        }
                        order_dict = {}
                        continue
                elif self.cal_type == 'Etalon':

                    assert comb_lines_angstrom is None, '`comb_lines_angstrom` \
                        should not be set for Etalon frames.'

                    wls = np.interp(
                        fitted_peak_pixels[good_peak_idx], np.arange(n_pixels)[rough_wls_order>0],
                        rough_wls_order[rough_wls_order>0]
                    )

                fitted_peak_pixels = fitted_peak_pixels[good_peak_idx]

                # Mark lines with bad fits and lambda_fit for each line in dictionary:
                '''
                good_line_ind = 0
                for l in np.arange(len(lines_dict)):
                    if l not in good_peak_idx:
                        orderlet_dict[order_num]['lines'][l]['quality'] = 'bad' #TODO: add this functionality to ThAr dictionaries
                    else:
                        orderlet_dict[order_num]['lines'][l]['lambda_fit'] = wls[good_line_ind]
                        good_line_ind += 1
                '''
            # use expected peak locations to compute updated precise wavelengths for each pixel
            # (only ThAr)
            else:
                if order_plt_path is not None:
                    plot_toggle = True
                else:
                    plot_toggle = False

                min_order_wave = np.min(rough_wls_order)
                max_order_wave = np.max(rough_wls_order)
#                line_wavelengths = expected_peak_locs.query(f'{min_order_wave} < wave < {max_order_wave}')['wave'].values
                line_wavelengths = expected_peak_locs[order_num]['known_wavelengths_vac']
                ii = np.where(np.logical_and(line_wavelengths > min_order_wave, line_wavelengths < max_order_wave))[0]
                
                line_wavelengths = line_wavelengths[ii]
                
                pixels_order = np.arange(0, len(rough_wls_order))
                wave_to_pix = interp1d(rough_wls_order, pixels_order,
                                       assume_sorted=False)
                line_pixels_expected = wave_to_pix(line_wavelengths)

                sorted_indices = np.argsort(line_pixels_expected)
                line_wavelengths = line_wavelengths[sorted_indices]
                
                line_pixels_expected = line_pixels_expected[sorted_indices]

                line_wavelengths = np.array([
                    line_wavelengths[i] for i in
                    np.arange(1, len(line_pixels_expected)) if
                    line_pixels_expected[i] != line_pixels_expected[i-1]
                ])
                line_pixels_expected = np.array([
                    line_pixels_expected[i] for i in
                    np.arange(1, len(line_pixels_expected)) if
                    line_pixels_expected[i] != line_pixels_expected[i-1]
                ])
                wls, gauss_coeffs, lines_dict = self.line_match(
                    order_flux, line_wavelengths, line_pixels_expected,
                    plot_toggle, order_plt_path
                )
                
                orderlet_dict[order_num]['lines'] = lines_dict
                
                fitted_peak_pixels = gauss_coeffs[1,:]

            # if we don't have an etalon frame, we won't use drift to calculate the wls
            # To-do for Etalon: add line_dicts
            if self.cal_type != 'Etalon':
                if expected_peak_locs is None:
                    peak_heights = detected_peak_heights[good_peak_idx]
                else:
                    peak_heights = fitted_peak_pixels

                # calculate the wavelength solution for the order
                polynomial_wls, leg_out = self.fit_polynomial(
                    wls, rough_wls_order, peak_wavelengths_ang, order_list, n_pixels, fitted_peak_pixels, peak_heights=peak_heights,
                    plot_path=order_plt_path, fit_iterations=self.fit_iterations,
                    sigma_clip=self.sigma_clip)
                poly_soln_final_array[order_num,:] = polynomial_wls

                if plt_path is not None:
                    fig, ax = plt.subplots(2, 1, figsize=(12,5))
                    ax[0].set_title('Precise WLS - Rough WLS')
                    ax[0].plot(np.arange(n_pixels), leg_out(np.arange(n_pixels)) - rough_wls_order, color='k')
                    ax[0].set_ylabel('[$\\rm \AA$]')
                    pixel_sizes = rough_wls_order[1:] - rough_wls_order[:-1]
                    ax[1].plot(np.arange(n_pixels - 1),
                              (leg_out(np.arange(n_pixels - 1)) - rough_wls_order[:-1]) / pixel_sizes, color='k')
                    ax[1].set_ylabel('[Pixels]')
                    ax[1].set_xlabel('Pixel')
                    plt.tight_layout()
                    plt.savefig('{}/precise_vs_rough_{}.png'.format(order_plt_path,self.fibre), dpi=500)
                    plt.close()

                # compute various RV precision values for order
                rel_precision, abs_precision = self.calculate_rv_precision(
                    fitted_peak_pixels, wls, leg_out, rough_wls_order, our_wavelength_solution_for_order, rough_wls_order, plot_path=order_plt_path,
                    print_update=print_update
                )
                order_precisions.append(abs_precision)
                num_detected_peaks.append(len(fitted_peak_pixels))

                # Add to dictionary for this order
                orderlet_dict[order_num]['fitted_wls'] = polynomial_wls
                orderlet_dict[order_num]['rel_precision_cms'] = rel_precision
                orderlet_dict[order_num]['abs_precision_cms'] = abs_precision
                orderlet_dict[order_num]['num_detected_peaks'] = len(fitted_peak_pixels)
                orderlet_dict[order_num]['known_wavelengths_vac'] = wls
                orderlet_dict[order_num]['line_positions'] = fitted_peak_pixels

            # compute drift, and use this to update the wavelength solution
            else:
                pass
                
            wavelengths_and_pixels[order_num] = {
                'known_wavelengths_vac':wls,
                'line_positions':fitted_peak_pixels
            }

        # for lamps and LFC, we can compute absolute precision across all orders
        if self.cal_type != 'Etalon':
            squared_resids = (np.array(order_precisions) * num_detected_peaks)**2
            sum_of_squared_resids = np.sum(squared_resids)
            overall_std_error = (np.sqrt(sum_of_squared_resids) / np.sum(num_detected_peaks))
            #orderlet_dict['overall_std_error_cms'] = overall_std_error
            print('\n\n\nOverall absolute precision (all orders): {:2.2f} cm/s\n\n\n'.format(overall_std_error))

        return poly_soln_final_array, wavelengths_and_pixels, orderlet_dict, overall_std_error, order_precisions

    def line_match(self, flux, linelist, line_pixels_expected, plot_toggle, savefig, gaussian_fit_width=10):
        """
        Given a linelist of known wavelengths of peaks and expected pixel locations
        (from a previous wavelength solution), returns precise, updated pixel locations
        for each known peak wavelength.
        Args:
            flux (np.array): flux of order
            linelist (np.array of float): wavelengths of lines to be fit (Angstroms)
            line_pixels_expected (np.array of float): expected pixels for each wavelength
                (Angstroms); must be same length as `linelist`
            plot_toggle (bool): if True, make and save plots.
            savefig (str): path to directory where plots will be saved
            gaussian_fit_width (int): pixel +/- range to use for Gaussian fitting
        Retuns:
            tuple of:
                np.array: same input linelist, with unfit lines removed
                np.array: array of size (4, n_peaks) containing best-fit
                    Gaussian parameters [a, mu, sigma**2, const] for each detected peak
                dictionary: a dictionary of information about the lines fit within this order
        """
        if self.cal_type == 'ThAr':
            gaussian_fit_width = 5
        num_input_lines = len(linelist)
        num_pixels = len(flux)
        successful_fits = []
        lines_dict = {}

        missed_lines = 0
        coefs = np.zeros((4,num_input_lines))
        for i in np.arange(num_input_lines):
            line_location = line_pixels_expected[i]
            peak_pixel = np.floor(line_location).astype(int)
            # don't fit saturated lines
            if peak_pixel < len(flux) and flux[peak_pixel] <= 1e6:
                if peak_pixel < gaussian_fit_width:
                    first_fit_pixel = 0
                else:
                    first_fit_pixel = peak_pixel - gaussian_fit_width
                
                if peak_pixel + gaussian_fit_width > num_pixels:
                    last_fit_pixel = num_pixels
                else:
                    last_fit_pixel = peak_pixel + gaussian_fit_width

                # fit gaussian to matched peak location
                result, line_dict = self.fit_gaussian_integral(
                    np.arange(first_fit_pixel,last_fit_pixel),
                    flux[first_fit_pixel:last_fit_pixel]
                )

                #add_to_line_dict = False
                if result is not None:
                    coefs[:, i] = result
                    successful_fits.append(i)  # Append index of successful fit
                    line_dict['lambda_fit'] = linelist[i]
                    lines_dict[str(i)] = line_dict  # Add line dictionary to lines dictionary
                else:
                    missed_lines += 1

                amp = coefs[0,i]
                if amp < 0:
                    missed_lines += 1
                    coefs[:,i] = np.nan

            else:
                coefs[:,i] = np.nan
                missed_lines += 1

        linelist = linelist[successful_fits]
        coefs = coefs[:, successful_fits]
        linelist = linelist[np.isfinite(coefs[0,:])]
        coefs = coefs[:, np.isfinite(coefs[0,:])]
        
        print('{}/{} lines not fit.'.format(missed_lines, num_input_lines))
        if plot_toggle:

            n_zoom_sections = 10
            zoom_section_pixels = num_pixels // n_zoom_sections

            zoom_section_pixels = (num_pixels // n_zoom_sections)
            _, ax_list = plt.subplots(n_zoom_sections,1,figsize=(10, 20))
            ax_list[0].set_title('({} missed lines)'.format(missed_lines))
            for i, ax in enumerate(ax_list):

                # plot the flux
                ax.plot(
                    np.arange(num_pixels)[i*zoom_section_pixels:(i+1)*zoom_section_pixels],
                    flux[i*zoom_section_pixels:(i+1)*zoom_section_pixels],color='k'
                )

                # #  plot the fitted peak maxima as points
                # ax.scatter(
                #     coefs[1,:][
                #         (coefs[1,:] > i * zoom_section_pixels) &
                #         (coefs[1,:] < (i+1) * zoom_section_pixels)
                #     ],
                #     coefs[0,:][
                #         (coefs[1,:] > i * zoom_section_pixels) &
                #         (coefs[1,:] < (i+1) * zoom_section_pixels)
                #     ] +
                #     coefs[3,:][
                #         (coefs[1,:] > i * zoom_section_pixels) &
                #         (coefs[1,:] < (i+1) * zoom_section_pixels)
                #     ],
                #     color='red'
                # )

                # overplot the Gaussian fits
                for j in np.arange(num_input_lines-missed_lines):

                    # if peak in range:
                    if (
                        (coefs[1,j] > i * zoom_section_pixels) &
                        (coefs[1,j] < (i+1) * zoom_section_pixels)
                    ):

                        xs = np.floor(coefs[1,j]) - gaussian_fit_width + \
                            np.linspace(
                                0,
                                2 * gaussian_fit_width,
                                2 * gaussian_fit_width
                            )
                        gaussian_fit = self.integrate_gaussian(
                            xs, coefs[0,j], coefs[1,j], coefs[2,j], coefs[3,j]
                        )

                        ax.plot(xs, gaussian_fit, alpha=0.5, color='red')

            plt.tight_layout()
            plt.savefig('{}/spectrum_and_gaussian_fits_{}.png'.format(savefig,self.fibre), dpi=500)
            plt.close()

        return linelist, coefs, lines_dict

    def fit_gaussian_integral(self, x, y):
        """
        Fits a continuous Gaussian to a discrete set of x and y datapoints
        using scipy.curve_fit
        
        Args:
            x (np.array): x data to be fit
            y (np.array): y data to be fit
        Returns a tuple of:
            list: best-fit parameters [a, mu, sigma**2, const]
            line_dict: dictionary of best-fit parameters, wav, flux, model, etc.
        """
        
        line_dict = {} # initialize dictionary to store fit parameters, etc.

        x = np.ma.compressed(x)
        y = np.ma.compressed(y)
        i = np.argmax(y[len(y) // 4 : len(y) * 3 // 4]) + len(y) // 4
        
        p0 = [y[i], x[i], 1.5, np.min(y)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(self.integrate_gaussian, x, y, p0=p0, maxfev=1000000)
            pcov[np.isinf(pcov)] = 0 # convert inf to zero
            pcov[np.isnan(pcov)] = 0 # convert nan to zero
            line_dict['amp']   = popt[0] # optimized parameters
            line_dict['mu']    = popt[1] # "
            line_dict['sig']   = popt[2] # "
            line_dict['const'] = popt[3] # ""
            line_dict['covar'] = pcov    # covariance
            line_dict['data']  = y
            line_dict['model'] = self.integrate_gaussian(x, *popt)
            line_dict['quality'] = 'good' # fits are assumed good until marked bad elsewhere
            

        if self.cal_type == 'ThAr':
            # Quality Checks for Gaussian Fits
            
            if max(y) == 0:
                print('Amplitude is 0')
                return(None, line_dict)
            
            chi_squared_threshold = int(self.chi_2_threshold)

            # Calculate chi^2
            predicted_y = self.integrate_gaussian(x, *popt)
            chi_squared = np.sum(((y - predicted_y) ** 2) / np.var(y))
            line_dict['chi2'] = chi_squared

            # Calculate RMS of residuals for Gaussian fit
            rms_residual = np.sqrt(np.mean(np.square(y - predicted_y)))
            line_dict['rms'] = np.sqrt(np.mean(np.square(rms_residual)))

            #rms_threshold = 1000 # RMS Quality threshold
            #disagreement_threshold = 1000 # Disagreement with initial guess threshold
            #asymmetry_threshold = 1000 # Asymmetry in residuals threshold

            # Calculate disagreement between Gaussian fit and initial guess
            #disagreement = np.abs(popt[1] - p0[1])
            line_dict['mu_diff'] = popt[1] - p0[1] # disagreement between Gaussian fit and initial guess

            ## Check for asymmetry in residuals
            #residuals = y - predicted_y
            #left_residuals = residuals[:len(residuals)//2]
            #right_residuals = residuals[len(residuals)//2:]
            #asymmetry = np.abs(np.mean(left_residuals) - np.mean(right_residuals))
            
            # Run checks against defined quality thresholds
            if (chi_squared > chi_squared_threshold):
                print("Chi squared exceeded the threshold for this line. Line skipped")
                return None, line_dict

            # Check if the Gaussian amplitude is positive, the peak is higher than the wings, or the peak is too high
            if popt[0] <= 0 or popt[0] <= popt[3] or popt[0] >= 500*max(y):
                line_dict['quality'] = 'bad_amplitude'  # Mark the fit as bad due to bad amplitude or U shaped gaussian
                print('Bad amplitude detected')
                return None, line_dict

        return (popt, line_dict)
    
    def integrate_gaussian(self, x, a, mu, sig, const, int_width=0.5):
        """
        Returns the integral of a Gaussian over a specified symmetric range.
        Gaussian given by:
        g(x) = a * exp(-(x - mu)**2 / (2 * sig**2)) + const
        Args:
            x (float): the central value over which the integral will be calculated
            a (float): the amplitude of the Gaussian
            mu (float): the mean of the Gaussian
            sig (float): the standard deviation of the Gaussian
            const (float): the Gaussian's offset from zero (i.e. the value of
                the Gaussian at infinity).
            int_width (float): the width of the range over which the integral will
                be calculated (i.e. if I want to calculate from 0.5 to 1, I'd set
                x = 0.75 and int_width = 0.25).
        Returns:
            float: the integrated value
        """

        integrated_gaussian_val = a * 0.5 * (
            erf((x - mu + int_width) / (np.sqrt(2) * sig)) -
            erf((x - mu - int_width) / (np.sqrt(2) * sig))
            ) + (const * 2 * int_width)
        
        return integrated_gaussian_val

    def fit_polynomial(self, wls, rough_wls_order, peak_wavelengths_ang, order_list, n_pixels, fitted_peak_pixels, fit_iterations=5, sigma_clip=2.1, peak_heights=None, plot_path=None):
        """
        Given precise wavelengths of detected LFC order_flux lines, fits a
        polynomial wavelength solution.
        Args:
            wls (np.array): the known, precise wavelengths of the detected peaks,
                either from fundamental physics or a previous wavelength solution.
            n_pixels (int): number of pixels in the order
            fitted_peak_pixels (np.array): array of true detected peak locations as
                determined by Gaussian fitting.
            fit_iterations (int): number of sigma-clipping iterations in the polynomial fit
            sigma_clip (float): clip outliers in fit with residuals greater than sigma_clip away from fit
            peak_heights (np.array): heights of peaks (either detected heights or
                fitted heights). We use this to weight the peaks in the polynomial
                fit, assuming Poisson errors.
            plot_path (str): if defined, the path to the output directory for
                diagnostic plots. If None, plots are not made.
        Returns:
            tuple of:
                np.array: calculated wavelength solution for the order (i.e.
                    wavelength value for each pixel in the order)
                func: a Python function that, given an array of pixel locations,
                    returns the Legendre polynomial wavelength solutions
        """
        weights = 1 / np.sqrt(peak_heights)
        if self.fit_type.lower() not in ['legendre', 'spline']:
            raise NotImplementedError("Fit type must be either legendre or spline")
        
        if self.fit_type.lower() == 'legendre' or self.fit_type.lower() == 'spline':

            _, unique_idx, count = np.unique(fitted_peak_pixels, return_index=True, return_counts=True)
            unclipped_idx = np.where(
                (fitted_peak_pixels > 0)
            )[0]
            unclipped_idx = np.intersect1d(unclipped_idx, unique_idx[count < 2])
            
            sorted_idx = np.argsort(fitted_peak_pixels[unclipped_idx])
            x, y, w = fitted_peak_pixels[unclipped_idx][sorted_idx], wls[unclipped_idx][sorted_idx], weights[unclipped_idx][sorted_idx]

            for i in range(fit_iterations):
                if self.fit_type.lower() == 'legendre':
                    if self.cal_type == 'ThAr':
                        # fit ThAr based on 4/30 WLS
                        rough_wls_int = interp1d(np.arange(n_pixels), rough_wls_order, kind='linear', fill_value="extrapolate")
                        
                        def polynomial_func(x, c0, c1, c2, c3, c4, c5):
                            """
                            Polynomial function to fit.
                            Args:
                                x (np.array): Pixel values.
                                c0, c1, c2, c3 (float): Coefficients of the polynomial.
                            Returns:
                                np.array: Evaluated polynomial.
                            """
                            return rough_wls_int(x) + c0 + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4 + c5 * x**5
                        
                        # Using curve_fit to find the best-fit values of {c0, c1}
                        popt, _ = curve_fit(polynomial_func, x, y)

                        # Create the wavelength solution for the order
                        our_wavelength_solution_for_order = polynomial_func(np.arange(len(rough_wls_order)), *popt)
                        leg_out = Legendre.fit(np.arange(n_pixels), our_wavelength_solution_for_order, 9)
                    
                    if self.cal_type == 'LFC':
                        leg_out = Legendre.fit(x, y, self.fit_order, w=w)
                        our_wavelength_solution_for_order = leg_out(np.arange(n_pixels))
                if self.fit_type == 'spline':
                    leg_out = UnivariateSpline(x, y, w, k=5)
                    our_wavelength_solution_for_order = leg_out(np.arange(n_pixels))
                
                res = y - leg_out(x)
                good = np.where(np.abs(res) <= sigma_clip*np.std(res))
                x = x[good]
                y = y[good]
                w = w[good]
                res = res[good]
            
#            plt.plot(x, res, 'k.')
#            plt.axhline(0, color='b', lw=2)
#            plt.xlabel('Pixel')
#            plt.ylabel('Fit residuals [$\AA$]')
#            plt.tight_layout()
#            plt.savefig('{}/polyfit.png'.format(plot_path))
#            plt.close()
            
            if plot_path is not None and self.cal_type =='ThAr':
                approx_dispersion = (our_wavelength_solution_for_order[int(len(rough_wls_order)/2)] - our_wavelength_solution_for_order[int(len(rough_wls_order)/2)+100])/100
                #fig, ax1 = plt.subplots(tight_layout=True, figsize=(8, 4))
                
                # Range of interest b/c CCF chops off first/last 500 pixels
                pixel_range = np.arange(500, int(len(rough_wls_order))-500)
                rough_wls_int_range = rough_wls_int(pixel_range)
                wavelength_solution_range = our_wavelength_solution_for_order[500:int(len(rough_wls_order))-500]

                # Create the plot
                fig, ax1 = plt.subplots(tight_layout=True, figsize=(8, 4))
                ax1.plot(
                    pixel_range,
                    rough_wls_int_range - wavelength_solution_range,
                    color='k'
                )
                ax1.set_xlabel('Pixel')
                ax1.set_ylabel(r'Wavelength Difference ($\AA$)')
                ax2 = ax1.twinx()
                ax2.set_ylabel("Difference (pixels) \nusing dispersion " + r'$\approx$' + '{0:.2}'.format(approx_dispersion) + r' $\AA$/pixel')
                ax2.set_ylim(ax1.get_ylim())
                ax1_ticks = ax1.get_yticks()
                ax2.set_yticklabels([str(round(tick / approx_dispersion, 2)) for tick in ax1_ticks])
                plt.savefig('{}/interp_vs_our_wls_{}.png'.format(plot_path,self.fibre), dpi=500)
                plt.close()
        else:
            raise ValueError('Only set up to perform Legendre fits currently! Please set fit_type to "Legendre"')

        return our_wavelength_solution_for_order, leg_out

    def calculate_rv_precision(
        self, fitted_peak_pixels, wls, leg_out, rough_wls, our_wavelength_solution_for_order, rough_wls_order,
        print_update=True, plot_path=None
    ):
        """
        Calculates 1) RV precision from the difference between the known (from
        physics) wavelengths of pixels containing peak flux values and the
        fitted wavelengths of the same pixels, generated using a polynomial
        wavelength solution ("absolute RV precision") and 2) RV precision from
        the difference between the "master" wavelength solution and our
        fitted wavelength solution ("relative RV precision")
        Args:
            fitted_peak_pixels (np.array of float): array of true detected peak locations as
                determined by Gaussian fitting (already clipped)
            wls (np.array of float): precise wavelengths of `fitted_peak_pixels`,
                from fundamental physics or another wavelength solution.
            leg_out (func): a Python function that, given an array of pixel
                locations, returns the Legendre polynomial wavelength solutions
            rough_wls (np.array of float): rough wavelength values for each
                pixel in the order [Angstroms]
            print_update (bool): If true, prints standard error per order.
            plot_path (str): if defined, the path to the output directory for
                diagnostic plots. If None, plots are not made.
        Returns:
            tuple of:
                float: absolute RV precision in cm/s
                float: relative RV precision in cm/s
        """
        our_wls_peak_pos = leg_out(fitted_peak_pixels)
        # absolute/polynomial precision of order = difference between fundemental wavelengths
        # and our wavelength solution wavelengths for (fractional) peak pixels
        abs_residual = ((our_wls_peak_pos - wls) * scipy.constants.c) / wls
        abs_precision_cm_s = 100 * np.nanstd(abs_residual)/np.sqrt(len(fitted_peak_pixels))
        # the above line should use RMS not STD

        # relative RV precision of order = difference between rough wls wavelengths
        # and our wavelength solution wavelengths for all pixels
        n_pixels = len(rough_wls)
        our_wavelength_solution_for_order = leg_out(np.arange(n_pixels))
        rel_residual = (our_wavelength_solution_for_order[rough_wls>0] -  rough_wls[rough_wls>0]) * scipy.constants.c /rough_wls[rough_wls>0]
        rel_precision_cm_s = 100 * np.std(rel_residual)/np.sqrt(len(rough_wls[rough_wls>0]))
        if print_update:
            print('Absolute standard error (this order): {:.2f} cm/s'.format(abs_precision_cm_s))
            print('Relative standard error (this order): {:.2f} cm/s'.format(rel_precision_cm_s))
        
        if plot_path is not None:
            fig, ax = plt.subplots(2,1) #figsize=(20,16), tight_layout=True
            ax[0].plot(abs_residual)
            ax[0].set_xlabel('Pixel')
            ax[0].set_ylabel('Absolute Error [m/s]')
            ax[1].plot(rel_residual)
            ax[1].set_xlabel('Pixel')
            ax[1].set_ylabel('Relative Error [m/s]')
            plt.savefig('{}/rv_precision_{}.png'.format(plot_path,self.fibre), dpi=500)
            plt.close()

        return rel_precision_cm_s, abs_precision_cm_s
