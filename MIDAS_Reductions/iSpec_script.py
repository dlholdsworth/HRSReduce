#!/usr/bin/env python
#
#    This file is part of the Integrated Spectroscopic Framework (iSpec).
#    Copyright 2011-2012 Sergi Blanco Cuaresma - http://www.marblestation.com
#
#    iSpec is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    iSpec is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with iSpec. If not, see <http://www.gnu.org/licenses/>.
#
import os
import sys
import numpy as np
import logging
import multiprocessing
from multiprocessing import Pool
import glob
from astropy.io import fits
import matplotlib.pyplot as plt

################################################################################
#--- iSpec directory -------------------------------------------------------------
ispec_dir = '/Users/daniel/Packages/iSpec_v20201001/'
#ispec_dir = '/home/virtual/iSpec/'
sys.path.insert(0, os.path.abspath(ispec_dir))
import ispec


#--- Change LOG level ----------------------------------------------------------
LOG_LEVEL = "warning"
#LOG_LEVEL = "info"
logger = logging.getLogger() # root logger, common for all
logger.setLevel(logging.getLevelName(LOG_LEVEL.upper()))
################################################################################

 #star_spectrum = ispec.read_spectrum("56773*rest.dat")

def read_write_spectrum():
    #--- Reading spectra -----------------------------------------------------------
    logging.info("Reading spectra")
#    star_spectrum = ispec.read_spectrum("H20170907_BC_merged_nm.dat")
    ##--- Save spectrum ------------------------------------------------------------
    logging.info("Saving spectrum...")
    ispec.write_spectrum(star_spectrum, "H20170907_BC_merged_nm.fits")

def convert_air_to_vacuum():
    #    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Converting wavelengths from air to vacuum and viceversa -------------------
    star_spectrum_vacuum = ispec.air_to_vacuum(star_spectrum)
    star_spectrum_air = ispec.vacuum_to_air(star_spectrum_vacuum)

def plot():
    #    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    mu_cas_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_muCas.txt.gz")
    #--- Plotting (requires graphical interface) -----------------------------------
    logging.info("Plotting...")
    ispec.plot_spectra([star_spectrum, mu_cas_spectrum])
    ispec.show_histogram(star_spectrum['flux'])

def cut_spectrum_from_range():
    #    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Cut -----------------------------------------------------------------------
    logging.info("Cutting...")

    # - Keep points between two given wavelengths
    wfilter = ispec.create_wavelength_filter(star_spectrum, wave_base=470.0, wave_top=680.0)
    cutted_star_spectrum = star_spectrum[wfilter]

def cut_spectrum_from_segments():
    #    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Cut -----------------------------------------------------------------------
    logging.info("Cutting...")
    # Keep only points inside a list of segments
    segments = ispec.read_segment_regions(ispec_dir + "/input/regions/fe_lines_segments.txt")
    wfilter = ispec.create_wavelength_filter(star_spectrum, regions=segments)
    cutted_star_spectrum = star_spectrum[wfilter]


def determine_radial_velocity_with_mask():
    #    mu_cas_spectrum = ispec.read_spectrum("H20170907_BC_merged_nm.dat")
    #--- Radial Velocity determination with linelist mask --------------------------
    logging.info("Radial velocity determination with linelist mask...")
    # - Read atomic data
    #mask_file = ispec_dir + "input/linelists/CCF/Narval.Sun.370_1048nm/mask.lst"
    #mask_file = ispec_dir + "input/linelists/CCF/Atlas.Arcturus.372_926nm/mask.lst""
    #mask_file = ispec_dir + "input/linelists/CCF/Atlas.Sun.372_926nm/mask.lst"
    mask_file = ispec_dir + "input/linelists/CCF/HARPS_SOPHIE.A0.350_1095nm/mask.lst"
    #mask_file = ispec_dir + "input/linelists/CCF/HARPS_SOPHIE.F0.360_698nm/mask.lst"
    #mask_file = ispec_dir + "input/linelists/CCF/HARPS_SOPHIE.G2.375_679nm/mask.lst"
    #mask_file = ispec_dir + "input/linelists/CCF/HARPS_SOPHIE.K0.378_679nm/mask.lst"
    #mask_file = ispec_dir + "input/linelists/CCF/HARPS_SOPHIE.K5.378_680nm/mask.lst"
    #mask_file = ispec_dir + "input/linelists/CCF/HARPS_SOPHIE.M5.400_687nm/mask.lst"
    #mask_file = ispec_dir + "input/linelists/CCF/Synthetic.Sun.350_1100nm/mask.lst"
    #mask_file = ispec_dir + "input/linelists/CCF/VALD.Sun.300_1100nm/mask.lst"
    ccf_mask = ispec.read_cross_correlation_mask(mask_file)

    models, ccf = ispec.cross_correlate_with_mask(star_spectrum, ccf_mask, \
                            lower_velocity_limit=40, upper_velocity_limit=90, \
                            velocity_step=.5, mask_depth=0.01, \
                                                  fourier=False)

    # Number of models represent the number of components
    components = len(models)
    # First component:
    rv = np.round(models[0].mu(), 2) # km/s
    rv_err = np.round(models[0].emu(), 2) # km/s

    models, ccf = ispec.cross_correlate_with_mask(star_spectrum, ccf_mask, \
                            lower_velocity_limit=rv-10., upper_velocity_limit=rv+10., \
                            velocity_step=.05, mask_depth=0.01, \
                                                  fourier=False)

    rv = np.round(models[0].mu(), 3) # km/s
    rv_err = np.round(models[0].emu(), 3) # km/s

    return rv


def determine_radial_velocity_with_template():
    #    mu_cas_spectrum = ispec.read_spectrum("H20170907_BC_merged_nm.dat")
    #--- Radial Velocity determination with template -------------------------------
    logging.info("Radial velocity determination with template...")
    # - Read synthetic template
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Arcturus.372_926nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Sun.372_926nm/template.txt.gz")
    template = ispec.read_spectrum("template.txt")
    template = resample_spectrum(template,np.min(template['waveobs']),np.max(template['waveobs']))
    
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Synth.Sun.300_1100nm/template.txt.gz")

    try:
        models, ccf = ispec.cross_correlate_with_template(star_spectrum, template, \
                            lower_velocity_limit=-100, upper_velocity_limit=100, \
                            velocity_step=.10, fourier=False)

        # Number of models represent the number of components
        components = len(models)
        # First component:
        rv = np.round(models[0].mu(), 3) # km/s
        rv_err = np.round(models[0].emu(), 3) # km/s
        print(rv,rv_err)
    except:
        pass

def correct_radial_velocity():
    #    mu_cas_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_muCas.txt.gz")
    #--- Radial Velocity correction ------------------------------------------------
    logging.info("Radial velocity correction...")
    #    rv = -96.40 # km/s
    star_spectrum_cor = ispec.correct_velocity(star_spectrum, rv)
    return star_spectrum_cor


def determine_tellurics_shift_with_mask():
    #    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Telluric velocity shift determination from spectrum --------------------------
    logging.info("Telluric velocity shift determination...")
    # - Telluric
    telluric_linelist_file = ispec_dir + "/input/linelists/CCF/Synth.Tellurics.500_1100nm/mask.lst"
    telluric_linelist = ispec.read_telluric_linelist(telluric_linelist_file, minimum_depth=0.0)

    models, ccf = ispec.cross_correlate_with_mask(star_spectrum, telluric_linelist, \
                            lower_velocity_limit=-100, upper_velocity_limit=100, \
                            velocity_step=0.5, mask_depth=0.01, \
                            fourier = False,
                            only_one_peak = True)

    bv = np.round(models[0].mu(), 2) # km/s
    bv_err = np.round(models[0].emu(), 2) # km/s

def determine_tellurics_shift_with_template():
    #    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Telluric velocity shift determination from spectrum --------------------------
    logging.info("Telluric velocity shift determination...")
    # - Read synthetic template
    template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Synth.Tellurics.350_1100nm/template.txt.gz")

    models, ccf = ispec.cross_correlate_with_template(star_spectrum, template, \
                            lower_velocity_limit=-100, upper_velocity_limit=100, \
                            velocity_step=0.5, fourier=False, \
                            only_one_peak = True)

    bv = np.round(models[0].mu(), 2) # km/s
    bv_err = np.round(models[0].emu(), 2) # km/s

def degrade_resolution():
    #    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Resolution degradation ----------------------------------------------------
    logging.info("Resolution degradation...")
    from_resolution = 80000
    to_resolution = 47000
    convolved_star_spectrum = ispec.convolve_spectrum(star_spectrum, to_resolution, \
                                                    from_resolution=from_resolution)

def smooth_spectrum():
    #    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Smoothing spectrum (resolution will be affected) --------------------------
    logging.info("Smoothing spectrum...")
    resolution = 80000
    smoothed_star_spectrum = ispec.convolve_spectrum(star_spectrum, resolution)

def resample_spectrum(star_spectrum,w_min,w_max):
    #    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Resampling  --------------------------------------------------------------
    logging.info("Resampling...")
    wavelengths = np.arange(w_min, w_max, 0.01)
    #resampled_star_spectrum = ispec.resample_spectrum(star_spectrum, wavelengths, method="spline", zero_edges=True)
    resampled_star_spectrum = ispec.resample_spectrum(star_spectrum, wavelengths, method="linear", zero_edges=True)

    return resampled_star_spectrum

def coadd_spectra():
    # WARNING: This example co-adds spectra from different stars, in a real life situation
    #          the logical thing is to co-add spectra from the same star/instrument
    #    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    mu_cas_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_muCas.txt.gz")
    #--- Resampling and combining --------------------------------------------------
    logging.info("Resampling and comibining...")
    wavelengths = np.arange(470.0, 680.0, 0.001)
    resampled_star_spectrum = ispec.resample_spectrum(star_spectrum, wavelengths, zero_edges=True)
    resampled_mu_cas_spectrum = ispec.resample_spectrum(mu_cas_spectrum, wavelengths, zero_edges=True)
    # Coadd previously resampled spectra
    coadded_spectrum = ispec.create_spectrum_structure(resampled_star_spectrum['waveobs'])
    coadded_spectrum['flux'] = resampled_star_spectrum['flux'] + resampled_mu_cas_spectrum['flux']
    coadded_spectrum['err'] = np.sqrt(np.power(resampled_star_spectrum['err'],2) + \
                                    np.power(resampled_mu_cas_spectrum['err'],2))


def merge_spectra():
    # WARNING: This example merges spectra from different stars, in a real life situation
    #          the logical thing is to merge spectra from the same star/instrument
    #          and different wavelength ranges
    #--- Mergin spectra ------------------------------------------------------------
    logging.info("Mergin spectra...")
    left_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    right_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_muCas.txt.gz")
    merged_spectrum = np.hstack((left_spectrum, right_spectrum))


def normalize_spectrum_using_continuum_regions():
    """
    Consider only continuum regions for the fit, strategy 'median+max'
    """
    #    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")

    #--- Continuum fit -------------------------------------------------------------
    model = "Splines" # "Polynomy"
    degree = 2
    nknots = None # Automatic: 1 spline every 5 nm
    from_resolution = 80000

    # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
    order='median+max'
    median_wave_range=0.05
    max_wave_range=1.0

    continuum_regions = ispec.read_continuum_regions(ispec_dir + "/input/regions/fe_lines_continuum.txt")
    star_continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=from_resolution, \
                            continuum_regions=continuum_regions, nknots=nknots, degree=degree, \
                            median_wave_range=median_wave_range, \
                            max_wave_range=max_wave_range, \
                            model=model, order=order, \
                            automatic_strong_line_detection=True, \
                            strong_line_probability=0.5, \
                            use_errors_for_fitting=True)

    #--- Continuum normalization ---------------------------------------------------
    logging.info("Continuum normalization...")
    normalized_star_spectrum = ispec.normalize_spectrum(star_spectrum, star_continuum_model, consider_continuum_errors=False)
    # Use a fixed value because the spectrum is already normalized
    star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")


def normalize_spectrum_in_segments():
    """
    Fit continuum in each segment independently, strategy 'median+max'
    """
    #    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")

    #--- Continuum fit -------------------------------------------------------------
    model = "Splines" # "Polynomy"
    degree = 2
    nknots = 1
    from_resolution = 80000

    # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
    order='median+max'
    median_wave_range=0.05
    max_wave_range=1.0

    segments = ispec.read_segment_regions(ispec_dir + "/input/regions/fe_lines_segments.txt")
    star_continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=from_resolution, \
                            independent_regions=segments, nknots=nknots, degree=degree,\
                            median_wave_range=median_wave_range, \
                            max_wave_range=max_wave_range, \
                            model=model, order=order, \
                            automatic_strong_line_detection=True, \
                            strong_line_probability=0.5, \
                            use_errors_for_fitting=True)

    #--- Continuum normalization ---------------------------------------------------
    logging.info("Continuum normalization...")
    normalized_star_spectrum = ispec.normalize_spectrum(star_spectrum, star_continuum_model, consider_continuum_errors=False)
    # Use a fixed value because the spectrum is already normalized
    star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")

def normalize_whole_spectrum_with_template():
    """
    Use a template to normalize the whole spectrum
    """
    #    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    synth_spectrum = ispec.read_spectrum("J1917_B_UVES_syn.dat")

    #--- Continuum fit -------------------------------------------------------------
    model = "Template"
    nknots = None # Automatic: 1 spline every 5 nm (in this case, used to apply a gaussian filter)
    from_resolution = 80000
    median_wave_range=5.0

    #strong_lines = ispec.read_line_regions(ispec_dir + "/input/regions/strong_lines/absorption_lines.txt")
    #strong_lines = ispec.read_line_regions(ispec_dir + "/input/regions/relevant/relevant_line_masks.txt")
    strong_lines = None
    star_continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=from_resolution, \
                                ignore=strong_lines, \
                                nknots=nknots, \
                                median_wave_range=median_wave_range, \
                                model=model, \
                                template=synth_spectrum)

    #--- Continuum normalization ---------------------------------------------------
    logging.info("Continuum normalization...")
    normalized_star_spectrum = ispec.normalize_spectrum(star_spectrum, star_continuum_model, consider_continuum_errors=False)
    # Use a fixed value because the spectrum is already normalized
    star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")

    return normalized_star_spectrum

def normalize_whole_spectrum():
    """
    Use the whole spectrum, strategy 'median+max'
    """
    #    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")

    #--- Continuum fit -------------------------------------------------------------
    model = "Splines" # "Polynomy"
    degree = 2
    nknots = None # Automatic: 1 spline every 5 nm
    from_resolution = 80000

    # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
    order='median+max'
    median_wave_range=0.05
    max_wave_range=1.0

    star_continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=from_resolution, \
                                nknots=nknots, degree=degree, \
                                median_wave_range=median_wave_range, \
                                max_wave_range=max_wave_range, \
                                model=model, order=order, \
                                automatic_strong_line_detection=True, \
                                strong_line_probability=0.5, \
                                use_errors_for_fitting=True)

    #--- Continuum normalization ---------------------------------------------------
    logging.info("Continuum normalization...")
    normalized_star_spectrum = ispec.normalize_spectrum(star_spectrum, star_continuum_model, consider_continuum_errors=False)
    # Use a fixed value because the spectrum is already normalized
    star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")


def normalize_whole_spectrum_ignoring_prefixed_strong_lines():
    """
    Use the whole spectrum but ignoring some strong lines, strategy 'median+max'
    """
    #    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")

    #--- Continuum fit -------------------------------------------------------------
    model = "Splines" # "Polynomy"
    degree = 2
    nknots = None # Automatic: 1 spline every 5 nm
    from_resolution = 80000

    # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
    order='median+max'
    median_wave_range=0.05
    max_wave_range=1.0

    strong_lines = ispec.read_line_regions(ispec_dir + "/input/regions/strong_lines/absorption_lines.txt")
    star_continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=from_resolution, \
                                ignore=strong_lines, \
                                nknots=nknots, degree=degree, \
                                median_wave_range=median_wave_range, \
                                max_wave_range=max_wave_range, \
                                model=model, order=order, \
                                automatic_strong_line_detection=True, \
                                strong_line_probability=0.5, \
                                use_errors_for_fitting=True)

    #--- Continuum normalization ---------------------------------------------------
    logging.info("Continuum normalization...")
    normalized_star_spectrum = ispec.normalize_spectrum(star_spectrum, star_continuum_model, consider_continuum_errors=False)
    # Use a fixed value because the spectrum is already normalized
    star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")


def filter_cosmic_rays():
    #    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Continuum fit -------------------------------------------------------------
    model = "Splines" # "Polynomy"
    degree = 2
    nknots = None # Automatic: 1 spline every 5 nm
    from_resolution = 80000

    # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
    order='median+max'
    median_wave_range=0.05
    max_wave_range=1.0

    star_continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=from_resolution, \
                                nknots=nknots, degree=degree, \
                                median_wave_range=median_wave_range, \
                                max_wave_range=max_wave_range, \
                                model=model, order=order, \
                                automatic_strong_line_detection=True, \
                                strong_line_probability=0.5, \
                                use_errors_for_fitting=True)
    #--- Filtering cosmic rays -----------------------------------------------------
    # Spectrum should be already normalized
    cosmics = ispec.create_filter_cosmic_rays(star_spectrum, star_continuum_model, \
                                            resampling_wave_step=0.001, window_size=15, \
                                            variation_limit=0.01)
    clean_star_spectrum = star_spectrum[~cosmics]


def find_continuum_regions():
    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Continuum fit -------------------------------------------------------------
    model = "Splines" # "Polynomy"
    degree = 2
    nknots = None # Automatic: 1 spline every 5 nm
    from_resolution = 80000

    # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
    order='median+max'
    median_wave_range=0.05
    max_wave_range=1.0

    star_continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=from_resolution, \
                                nknots=nknots, degree=degree, \
                                median_wave_range=median_wave_range, \
                                max_wave_range=max_wave_range, \
                                model=model, order=order, \
                                automatic_strong_line_detection=True, \
                                strong_line_probability=0.5, \
                                use_errors_for_fitting=True)
    #--- Find continuum regions ----------------------------------------------------
    logging.info("Finding continuum regions...")
    resolution = 80000
    sigma = 0.001
    max_continuum_diff = 1.0
    fixed_wave_step = 0.05
    star_continuum_regions = ispec.find_continuum(star_spectrum, resolution, \
                                        max_std_continuum = sigma, \
                                        continuum_model = star_continuum_model, \
                                        max_continuum_diff=max_continuum_diff, \
                                        fixed_wave_step=fixed_wave_step)
    ispec.write_continuum_regions(star_continuum_regions, "example_star_fe_lines_continuum.txt")


def find_continuum_regions_in_segments():
    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Continuum fit -------------------------------------------------------------
    model = "Splines" # "Polynomy"
    degree = 2
    nknots = None # Automatic: 1 spline every 5 nm
    from_resolution = 80000

    # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
    order='median+max'
    median_wave_range=0.05
    max_wave_range=1.0

    star_continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=from_resolution, \
                                nknots=nknots, degree=degree, \
                                median_wave_range=median_wave_range, \
                                max_wave_range=max_wave_range, \
                                model=model, order=order, \
                                automatic_strong_line_detection=True, \
                                strong_line_probability=0.5, \
                                use_errors_for_fitting=True)
    #--- Find continuum regions in segments ----------------------------------------
    logging.info("Finding continuum regions...")
    resolution = 80000
    sigma = 0.001
    max_continuum_diff = 1.0
    fixed_wave_step = 0.05
    # Limit the search to given segments
    segments = ispec.read_segment_regions(ispec_dir + "/input/regions/fe_lines_segments.txt")
    limited_star_continuum_regions = ispec.find_continuum(star_spectrum, resolution, \
                                            segments=segments, max_std_continuum = sigma, \
                                            continuum_model = star_continuum_model, \
                                            max_continuum_diff=max_continuum_diff, \
                                            fixed_wave_step=fixed_wave_step)
    ispec.write_continuum_regions(limited_star_continuum_regions, \
            "example_limited_star_continuum_region.txt")


def find_linemasks(code = "spectrum"):
    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Radial Velocity determination with template -------------------------------
    logging.info("Radial velocity determination with template...")
    # - Read synthetic template
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Arcturus.372_926nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Sun.372_926nm/template.txt.gz")
    template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/NARVAL.Sun.370_1048nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Synth.Sun.300_1100nm/template.txt.gz")

    models, ccf = ispec.cross_correlate_with_template(star_spectrum, template, \
                            lower_velocity_limit=-200, upper_velocity_limit=200, \
                            velocity_step=1.0, fourier=False)

    # Number of models represent the number of components
    components = len(models)
    # First component:
    rv = np.round(models[0].mu(), 2) # km/s
    rv_err = np.round(models[0].emu(), 2) # km/s
    #--- Radial Velocity correction ------------------------------------------------
    logging.info("Radial velocity correction... %.2f +/- %.2f" % (rv, rv_err))
    star_spectrum = ispec.correct_velocity(star_spectrum, rv)
    #--- Telluric velocity shift determination from spectrum --------------------------
    logging.info("Telluric velocity shift determination...")
    # - Telluric
    telluric_linelist_file = ispec_dir + "/input/linelists/CCF/Synth.Tellurics.500_1100nm/mask.lst"
    telluric_linelist = ispec.read_telluric_linelist(telluric_linelist_file, minimum_depth=0.0)

    models, ccf = ispec.cross_correlate_with_mask(star_spectrum, telluric_linelist, \
                            lower_velocity_limit=-100, upper_velocity_limit=100, \
                            velocity_step=0.5, mask_depth=0.01, \
                            fourier = False,
                            only_one_peak = True)

    vel_telluric = np.round(models[0].mu(), 2) # km/s
    vel_telluric_err = np.round(models[0].emu(), 2) # km/s
    #--- Continuum fit -------------------------------------------------------------
    model = "Splines" # "Polynomy"
    degree = 2
    nknots = None # Automatic: 1 spline every 5 nm
    from_resolution = 80000

    # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
    order='median+max'
    median_wave_range=0.05
    max_wave_range=1.0

    star_continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=from_resolution, \
                                nknots=nknots, degree=degree, \
                                median_wave_range=median_wave_range, \
                                max_wave_range=max_wave_range, \
                                model=model, order=order, \
                                automatic_strong_line_detection=True, \
                                strong_line_probability=0.5, \
                                use_errors_for_fitting=True)
    #--- Find linemasks ------------------------------------------------------------
    logging.info("Finding line masks...")
    atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.1100_2400nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_hfs_iso.420_920nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_nohfs_noiso.420_920nm/atomic_lines.tsv"

    telluric_linelist_file = ispec_dir + "/input/linelists/CCF/Synth.Tellurics.500_1100nm/mask.lst"

    # Read
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(star_spectrum['waveobs']), wave_top=np.max(star_spectrum['waveobs']))
    atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01] # Select lines that have some minimal contribution in the sun
    #telluric_linelist = ispec.read_telluric_linelist(telluric_linelist_file, minimum_depth=0.01)
    #vel_telluric = 17.79  # km/s
    #telluric_linelist = None
    #vel_telluric = None

    resolution = 80000
    smoothed_star_spectrum = ispec.convolve_spectrum(star_spectrum, resolution)
    min_depth = 0.05
    max_depth = 1.00
    star_linemasks = ispec.find_linemasks(star_spectrum, star_continuum_model, \
                            atomic_linelist=atomic_linelist, \
                            max_atomic_wave_diff = 0.005, \
                            telluric_linelist=telluric_linelist, \
                            vel_telluric=vel_telluric, \
                            minimum_depth=min_depth, maximum_depth=max_depth, \
                            smoothed_spectrum=smoothed_star_spectrum, \
                            check_derivatives=False, \
                            discard_gaussian=False, discard_voigt=True, \
                            closest_match=False)
    # Exclude lines that have not been successfully cross matched with the atomic data
    # because we cannot calculate the chemical abundance (it will crash the corresponding routines)
    rejected_by_atomic_line_not_found = (star_linemasks['wave_nm'] == 0)
    star_linemasks = star_linemasks[~rejected_by_atomic_line_not_found]

    # Exclude lines with EW equal to zero
    rejected_by_zero_ew = (star_linemasks['ew'] == 0)
    star_linemasks = star_linemasks[~rejected_by_zero_ew]

    # Select only iron lines
    iron = star_linemasks['element'] == "Fe 1"
    iron = np.logical_or(iron, star_linemasks['element'] == "Fe 2")
    iron_star_linemasks = star_linemasks[iron]

    # Write regions with only masks limits and note:
    ispec.write_line_regions(star_linemasks, "example_star_linemasks.txt")
    # Write iron regions with only masks limits and note:
    ispec.write_line_regions(iron_star_linemasks, "example_star_fe_linemasks.txt")
    # Write regions with masks limits, cross-matched atomic data and fit data
    ispec.write_line_regions(star_linemasks, "example_star_fitted_linemasks.txt", extended=True)
    recover_star_linemasks = ispec.read_line_regions("example_star_fitted_linemasks.txt")
    # Write regions with masks limits and cross-matched atomic data (fit data fields are zeroed)
    zeroed_star_linemasks = ispec.reset_fitted_data_fields(star_linemasks)
    ispec.write_line_regions(zeroed_star_linemasks, "example_star_zeroed_fitted_linemasks.txt", extended=True)

    # Write only atomic data for the selected regions:
    ispec.write_atomic_linelist(star_linemasks, "example_star_atomic_linelist.txt")


def calculate_barycentric_velocity():
    mu_cas_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_muCas.txt.gz")
    #--- Barycentric velocity correction from observation date/coordinates ---------
    logging.info("Calculating barycentric velocity correction...")
    day = 15
    month = 2
    year = 2012
    hours = 0
    minutes = 0
    seconds = 0
    ra_hours = 19
    ra_minutes = 50
    ra_seconds = 46.99
    dec_degrees = 8
    dec_minutes = 52
    dec_seconds = 5.96

    # Project velocity toward star
    barycentric_vel = ispec.calculate_barycentric_velocity_correction((year, month, day, \
                                    hours, minutes, seconds), (ra_hours, ra_minutes, \
                                    ra_seconds, dec_degrees, dec_minutes, dec_seconds))
    #--- Correcting barycentric velocity -------------------------------------------
    corrected_spectrum = ispec.correct_velocity(mu_cas_spectrum, barycentric_vel)

def estimate_snr_from_flux():
    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    ## WARNING: To compare SNR estimation between different spectra, they should
    ##          be homogeneously sampled (consider a uniform re-sampling)
    #--- Estimate SNR from flux ----------------------------------------------------
    logging.info("Estimating SNR from fluxes...")
    num_points = 10
    estimated_snr = ispec.estimate_snr(star_spectrum['flux'], num_points=num_points)

def estimate_snr_from_err():
    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Estimate SNR from errors --------------------------------------------------
    logging.info("Estimating SNR from errors...")
    efilter = star_spectrum['err'] > 0
    filtered_star_spectrum = star_spectrum[efilter]
    if len(filtered_star_spectrum) > 1:
        estimated_snr = np.median(filtered_star_spectrum['flux'] / filtered_star_spectrum['err'])
    else:
        # All the errors are set to zero and we cannot calculate SNR using them
        estimated_snr = 0


def estimate_errors_from_snr():
    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Calculate errors based on SNR ---------------------------------------------
    snr = 100
    star_spectrum['err'] = star_spectrum['flux'] / snr


def clean_spectrum():
    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Clean fluxes and errors ---------------------------------------------------
    logging.info("Cleaning fluxes and errors...")
    flux_base = 0.0
    flux_top = 1.0
    err_base = 0.0
    err_top = 1.0
    ffilter = (star_spectrum['flux'] > flux_base) & (star_spectrum['flux'] <= flux_top)
    efilter = (star_spectrum['err'] > err_base) & (star_spectrum['err'] <= err_top)
    wfilter = np.logical_and(ffilter, efilter)
    clean_star_spectrum = star_spectrum[wfilter]


def clean_telluric_regions():
    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Telluric velocity shift determination from spectrum --------------------------
    logging.info("Telluric velocity shift determination...")
    # - Telluric
    telluric_linelist_file = ispec_dir + "/input/linelists/CCF/Synth.Tellurics.500_1100nm/mask.lst"
    telluric_linelist = ispec.read_telluric_linelist(telluric_linelist_file, minimum_depth=0.0)

    models, ccf = ispec.cross_correlate_with_mask(star_spectrum, telluric_linelist, \
                            lower_velocity_limit=-100, upper_velocity_limit=100, \
                            velocity_step=0.5, mask_depth=0.01, \
                            fourier = False,
                            only_one_peak = True)

    bv = np.round(models[0].mu(), 2) # km/s
    bv_err = np.round(models[0].emu(), 2) # km/s

    #--- Clean regions that may be affected by tellurics ---------------------------
    logging.info("Cleaning tellurics...")

    telluric_linelist_file = ispec_dir + "/input/linelists/CCF/Synth.Tellurics.500_1100nm/mask.lst"
    telluric_linelist = ispec.read_telluric_linelist(telluric_linelist_file, minimum_depth=0.0)

    # - Filter regions that may be affected by telluric lines
    #bv = 0.0
    min_vel = -30.0
    max_vel = +30.0
    # Only the 25% of the deepest ones:
    dfilter = telluric_linelist['depth'] > np.percentile(telluric_linelist['depth'], 75)
    tfilter = ispec.create_filter_for_regions_affected_by_tellurics(star_spectrum['waveobs'], \
                                telluric_linelist[dfilter], min_velocity=-bv+min_vel, \
                                max_velocity=-bv+max_vel)
    clean_star_spectrum = star_spectrum[~tfilter]


def adjust_line_masks():
    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Adjust line masks ---------------------------------------------------------
    resolution = 80000
    smoothed_star_spectrum = ispec.convolve_spectrum(star_spectrum, resolution)
    line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/fe_lines.txt")
    linemasks = ispec.adjust_linemasks(smoothed_star_spectrum, line_regions, max_margin=0.5)

def create_segments_around_linemasks():
    #---Create segments around linemasks -------------------------------------------
    line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/fe_lines.txt")
    segments = ispec.create_segments_around_lines(line_regions, margin=0.25)

def fit_lines_determine_ew_and_crossmatch_with_atomic_data(use_ares=False):
    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Radial Velocity determination with template -------------------------------
    logging.info("Radial velocity determination with template...")
    # - Read synthetic template
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Arcturus.372_926nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Sun.372_926nm/template.txt.gz")
    template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/NARVAL.Sun.370_1048nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Synth.Sun.300_1100nm/template.txt.gz")

    models, ccf = ispec.cross_correlate_with_template(star_spectrum, template, \
                            lower_velocity_limit=-200, upper_velocity_limit=200, \
                            velocity_step=1.0, fourier=False)

    # Number of models represent the number of components
    components = len(models)
    # First component:
    rv = np.round(models[0].mu(), 2) # km/s
    rv_err = np.round(models[0].emu(), 2) # km/s
    #--- Radial Velocity correction ------------------------------------------------
    logging.info("Radial velocity correction... %.2f +/- %.2f" % (rv, rv_err))
    star_spectrum = ispec.correct_velocity(star_spectrum, rv)
    #--- Telluric velocity shift determination from spectrum --------------------------
    logging.info("Telluric velocity shift determination...")
    # - Telluric
    telluric_linelist_file = ispec_dir + "/input/linelists/CCF/Synth.Tellurics.500_1100nm/mask.lst"
    telluric_linelist = ispec.read_telluric_linelist(telluric_linelist_file, minimum_depth=0.0)

    models, ccf = ispec.cross_correlate_with_mask(star_spectrum, telluric_linelist, \
                            lower_velocity_limit=-100, upper_velocity_limit=100, \
                            velocity_step=0.5, mask_depth=0.01, \
                            fourier = False,
                            only_one_peak = True)

    vel_telluric = np.round(models[0].mu(), 2) # km/s
    vel_telluric_err = np.round(models[0].emu(), 2) # km/s
    #--- Resolution degradation ----------------------------------------------------
    # NOTE: The line selection was built based on a solar spectrum with R ~ 47,000 and VALD atomic linelist.
    from_resolution = 80000
    to_resolution = 47000
    star_spectrum = ispec.convolve_spectrum(star_spectrum, to_resolution, from_resolution)
    #--- Continuum fit -------------------------------------------------------------
    model = "Splines" # "Polynomy"
    degree = 2
    nknots = None # Automatic: 1 spline every 5 nm
    from_resolution = 80000

    # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
    order='median+max'
    median_wave_range=0.05
    max_wave_range=1.0

    star_continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=from_resolution, \
                                nknots=nknots, degree=degree, \
                                median_wave_range=median_wave_range, \
                                max_wave_range=max_wave_range, \
                                model=model, order=order, \
                                automatic_strong_line_detection=True, \
                                strong_line_probability=0.5, \
                                use_errors_for_fitting=True)
    #--- Normalize -------------------------------------------------------------
    normalized_star_spectrum = ispec.normalize_spectrum(star_spectrum, star_continuum_model, consider_continuum_errors=False)
    # Use a fixed value because the spectrum is already normalized
    star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")
    #--- Fit lines -----------------------------------------------------------------
    logging.info("Fitting lines...")
    atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.1100_2400nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_hfs_iso.420_920nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_nohfs_noiso.420_920nm/atomic_lines.tsv"


    # Read
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(star_spectrum['waveobs']), wave_top=np.max(star_spectrum['waveobs']))
    atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01] # Select lines that have some minimal contribution in the sun

    #telluric_linelist_file = ispec_dir + "/input/linelists/CCF/Synth.Tellurics.500_1100nm/mask.lst"
    #telluric_linelist = ispec.read_telluric_linelist(telluric_linelist_file, minimum_depth=0.01)
    #vel_telluric = 17.79 # km/s
    #telluric_linelist = None
    #vel_telluric = None

    line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/spectrum_synth_turbospectrum_synth_sme_synth_moog_synth_synthe_synth_moog_ew_ispec_width_ew_ispec_good_for_params_all.txt")
    line_regions = ispec.adjust_linemasks(normalized_star_spectrum, line_regions, max_margin=0.5)

    linemasks = ispec.fit_lines(line_regions, normalized_star_spectrum, star_continuum_model, \
                                atomic_linelist = atomic_linelist, \
                                #max_atomic_wave_diff = 0.005, \
                                max_atomic_wave_diff = 0.00, \
                                telluric_linelist = telluric_linelist, \
                                smoothed_spectrum = None, \
                                check_derivatives = False, \
                                vel_telluric = vel_telluric, discard_gaussian=False, \
                                discard_voigt=True, \
                                free_mu=True, crossmatch_with_mu=False, closest_match=False)
    # Discard lines that are not cross matched with the same original element stored in the note
    linemasks = linemasks[linemasks['element'] == line_regions['note']]

    # Exclude lines that have not been successfully cross matched with the atomic data
    # because we cannot calculate the chemical abundance (it will crash the corresponding routines)
    rejected_by_atomic_line_not_found = (linemasks['wave_nm'] == 0)
    linemasks = linemasks[~rejected_by_atomic_line_not_found]

    # Exclude lines with EW equal to zero
    rejected_by_zero_ew = (linemasks['ew'] == 0)
    linemasks = linemasks[~rejected_by_zero_ew]

    # Exclude lines that may be affected by tellurics
    rejected_by_telluric_line = (linemasks['telluric_wave_peak'] != 0)
    linemasks = linemasks[~rejected_by_telluric_line]

    if use_ares:
        # Replace the measured equivalent widths by the ones computed by ARES
        old_linemasks = linemasks.copy()
        ### Different rejection parameters (check ARES papers):
        ##   - http://adsabs.harvard.edu/abs/2007A%26A...469..783S
        ##   - http://adsabs.harvard.edu/abs/2015A%26A...577A..67S
        #linemasks = ispec.update_ew_with_ares(normalized_star_spectrum, linemasks, rejt="0.995", tmp_dir=None, verbose=0)
        #linemasks = ispec.update_ew_with_ares(normalized_star_spectrum, linemasks, rejt="3;5764,5766,6047,6052,6068,6076", tmp_dir=None, verbose=0)
        snr = 50
        linemasks = ispec.update_ew_with_ares(normalized_star_spectrum, linemasks, rejt="%s" % (snr), tmp_dir=None, verbose=0)

    ew = linemasks['ew']
    ew_err = linemasks['ew_err']

    # Save linemasks (line masks + atomic cross-matched information + fit information)
    ispec.write_line_regions(linemasks, "example_fitted_atomic_linemask.txt", extended=True)


def fit_lines_already_crossmatched_with_atomic_data_and_determine_ew(use_ares=False):
    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Radial Velocity determination with template -------------------------------
    logging.info("Radial velocity determination with template...")
    # - Read synthetic template
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Arcturus.372_926nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Sun.372_926nm/template.txt.gz")
    template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/NARVAL.Sun.370_1048nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Synth.Sun.300_1100nm/template.txt.gz")

    models, ccf = ispec.cross_correlate_with_template(star_spectrum, template, \
                            lower_velocity_limit=-200, upper_velocity_limit=200, \
                            velocity_step=1.0, fourier=False)

    # Number of models represent the number of components
    components = len(models)
    # First component:
    rv = np.round(models[0].mu(), 2) # km/s
    rv_err = np.round(models[0].emu(), 2) # km/s
    #--- Radial Velocity correction ------------------------------------------------
    logging.info("Radial velocity correction... %.2f +/- %.2f" % (rv, rv_err))
    star_spectrum = ispec.correct_velocity(star_spectrum, rv)
    #--- Telluric velocity shift determination from spectrum --------------------------
    logging.info("Telluric velocity shift determination...")
    # - Telluric
    telluric_linelist_file = ispec_dir + "/input/linelists/CCF/Synth.Tellurics.500_1100nm/mask.lst"
    telluric_linelist = ispec.read_telluric_linelist(telluric_linelist_file, minimum_depth=0.0)

    models, ccf = ispec.cross_correlate_with_mask(star_spectrum, telluric_linelist, \
                            lower_velocity_limit=-100, upper_velocity_limit=100, \
                            velocity_step=0.5, mask_depth=0.01, \
                            fourier = False,
                            only_one_peak = True)

    vel_telluric = np.round(models[0].mu(), 2) # km/s
    vel_telluric_err = np.round(models[0].emu(), 2) # km/s
    #--- Resolution degradation ----------------------------------------------------
    # NOTE: The line selection was built based on a solar spectrum with R ~ 47,000 and VALD atomic linelist.
    from_resolution = 80000
    to_resolution = 47000
    star_spectrum = ispec.convolve_spectrum(star_spectrum, to_resolution, from_resolution)
    #--- Continuum fit -------------------------------------------------------------
    model = "Splines" # "Polynomy"
    degree = 2
    nknots = None # Automatic: 1 spline every 5 nm
    from_resolution = to_resolution

    # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
    order='median+max'
    median_wave_range=0.05
    max_wave_range=1.0

    star_continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=from_resolution, \
                                nknots=nknots, degree=degree, \
                                median_wave_range=median_wave_range, \
                                max_wave_range=max_wave_range, \
                                model=model, order=order, \
                                automatic_strong_line_detection=True, \
                                strong_line_probability=0.5, \
                                use_errors_for_fitting=True)
    #--- Normalize -------------------------------------------------------------
    normalized_star_spectrum = ispec.normalize_spectrum(star_spectrum, star_continuum_model, consider_continuum_errors=False)
    # Use a fixed value because the spectrum is already normalized
    star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")

    #--- Read lines with atomic data ------------------------------------------------
    line_regions_with_atomic_data = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/spectrum_synth_turbospectrum_synth_sme_synth_moog_synth_synthe_synth_moog_ew_ispec_width_ew_ispec_good_for_params_all_extended.txt")

    smoothed_star_spectrum = ispec.convolve_spectrum(star_spectrum, 2*to_resolution)
    line_regions_with_atomic_data = ispec.adjust_linemasks(smoothed_star_spectrum, line_regions_with_atomic_data, max_margin=0.5)

    #telluric_linelist_file = ispec_dir + "/input/linelists/CCF/Synth.Tellurics.500_1100nm/mask.lst"
    #telluric_linelist = ispec.read_telluric_linelist(telluric_linelist_file, minimum_depth=0.01)
    #vel_telluric = 17.79 # km/s
    #telluric_linelist = None
    #vel_telluric = None

    #--- Fit the lines but do NOT cross-match with any atomic linelist since they already have that information
    linemasks = ispec.fit_lines(line_regions_with_atomic_data, normalized_star_spectrum, star_continuum_model, \
                                atomic_linelist = None, \
                                max_atomic_wave_diff = 0.005, \
                                telluric_linelist = telluric_linelist, \
                                check_derivatives = False, \
                                vel_telluric = vel_telluric, discard_gaussian=False, \
                                smoothed_spectrum=None, \
                                discard_voigt=True, \
                                free_mu=True, crossmatch_with_mu=False, closest_match=False)

    # Discard bad masks
    flux_peak = normalized_star_spectrum['flux'][linemasks['peak']]
    flux_base = normalized_star_spectrum['flux'][linemasks['base']]
    flux_top = normalized_star_spectrum['flux'][linemasks['top']]
    bad_mask = np.logical_or(linemasks['wave_peak'] <= linemasks['wave_base'], linemasks['wave_peak'] >= linemasks['wave_top'])
    bad_mask = np.logical_or(bad_mask, flux_peak >= flux_base)
    bad_mask = np.logical_or(bad_mask, flux_peak >= flux_top)
    linemasks = linemasks[~bad_mask]

    # Exclude lines with EW equal to zero
    rejected_by_zero_ew = (linemasks['ew'] == 0)
    linemasks = linemasks[~rejected_by_zero_ew]

    # Exclude lines that may be affected by tellurics
    rejected_by_telluric_line = (linemasks['telluric_wave_peak'] != 0)
    linemasks = linemasks[~rejected_by_telluric_line]

    if use_ares:
        # Replace the measured equivalent widths by the ones computed by ARES
        old_linemasks = linemasks.copy()
        ### Different rejection parameters (check ARES papers):
        ##   - http://adsabs.harvard.edu/abs/2007A%26A...469..783S
        ##   - http://adsabs.harvard.edu/abs/2015A%26A...577A..67S
        #linemasks = ispec.update_ew_with_ares(normalized_star_spectrum, linemasks, rejt="0.995", tmp_dir=None, verbose=0)
        #linemasks = ispec.update_ew_with_ares(normalized_star_spectrum, linemasks, rejt="3;5764,5766,6047,6052,6068,6076", tmp_dir=None, verbose=0)
        snr = 50
        linemasks = ispec.update_ew_with_ares(normalized_star_spectrum, linemasks, rejt="%s" % (snr), tmp_dir=None, verbose=0)


def synthesize_spectrum(code="spectrum"):
    #--- Synthesizing spectrum -----------------------------------------------------
    # Parameters
    teff = 5771.0
    logg = 4.44
    MH = 0.00
    microturbulence_vel = ispec.estimate_vmic(teff, logg, MH) # 1.07
    macroturbulence = ispec.estimate_vmac(teff, logg, MH) # 4.21
    vsini = 1.60 # Sun
    limb_darkening_coeff = 0.6
    resolution = 300000
    wave_step = 0.001

    # Wavelengths to synthesis
    #regions = ispec.read_segment_regions(ispec_dir + "/input/regions/fe_lines_segments.txt")
    regions = None
    wave_base = 515.0 # Magnesium triplet region
    wave_top = 525.0


    # Selected model amtosphere, linelist and solar abundances
    #model = ispec_dir + "/input/atmospheres/MARCS/modeled_layers_pack.dump"
    model = ispec_dir + "/input/atmospheres/MARCS.GES/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/MARCS.APOGEE/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.APOGEE/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Castelli/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kurucz/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kirby/modeled_layers_pack.dump"

    atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.1100_2400nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_hfs_iso.420_920nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_nohfs_noiso.420_920nm/atomic_lines.tsv"

    isotope_file = ispec_dir + "/input/isotopes/SPECTRUM.lst"

    # Load chemical information and linelist
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=wave_base, wave_top=wave_top)
    atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01] # Select lines that have some minimal contribution in the sun

    isotopes = ispec.read_isotope_data(isotope_file)

    solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.2007/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2005/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.1998/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Anders.1989/stdatom.dat"

    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)
    # Load SPECTRUM abundances
    fixed_abundances = None # No fixed abundances
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)

    # Validate parameters
    if not ispec.valid_atmosphere_target(modeled_layers_pack, teff, logg, MH):
        msg = "The specified effective temperature, gravity (log g) and metallicity [M/H] \
                fall out of theatmospheric models."
        print (msg)

    # Enhance alpha elements + CNO abundances following MARCS standard composition
    alpha_enhancement, c_enhancement, n_enhancement, o_enhancement = ispec.determine_abundance_enchancements(MH)
    abundances = ispec.enhance_solar_abundances(solar_abundances, alpha_enhancement, c_enhancement, n_enhancement, o_enhancement)

    # Prepare atmosphere model
    atmosphere_layers = ispec.interpolate_atmosphere_layers(modeled_layers_pack, teff, logg, MH, code=code)

    # Synthesis
    synth_spectrum = ispec.create_spectrum_structure(np.arange(wave_base, wave_top, wave_step))
    synth_spectrum['flux'] = ispec.generate_spectrum(synth_spectrum['waveobs'], \
            atmosphere_layers, teff, logg, MH, atomic_linelist, isotopes, abundances, \
            fixed_abundances, microturbulence_vel = microturbulence_vel, \
            macroturbulence=macroturbulence, vsini=vsini, limb_darkening_coeff=limb_darkening_coeff, \
            R=resolution, regions=regions, verbose=1,
            code=code)
    ##--- Save spectrum ------------------------------------------------------------
    logging.info("Saving spectrum...")
    synth_filename = "example_synth_%s.fits" % (code)
    ispec.write_spectrum(synth_spectrum, synth_filename)


def add_noise_to_spectrum():
    """
    Add noise to an spectrum (ideally to a synthetic one) based on a given SNR.
    """
    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Adding poisson noise -----------------------------------------------------
    snr = 100
    distribution = "poisson" # "gaussian"
    noisy_star_spectrum = ispec.add_noise(star_spectrum, snr, distribution)

def generate_new_random_realizations_from_spectrum():
    """
    Considering fluxes as mean values and errors as standard deviation, generate
    N new random spectra.
    """
    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")

    number = 10
    spectra = ispec.random_realizations(star_spectrum, number, distribution="poisson")

def precompute_synthetic_grid(code="spectrum"):
    precomputed_grid_dir = "example_grid_%s/" % (code)

    # - Read grid ranges from file
    #ranges_filename = "input/grid/grid_ranges.txt"
    #ranges = ascii.read(ranges_filename, delimiter="\t")
    # - or define them directly here (example of only 2 reference points):
    ranges = np.recarray((2,),  dtype=[('Teff', int), ('logg', float), ('MH', float)])
    ranges['Teff'][0] = 5500
    ranges['logg'][0] = 4.5
    ranges['MH'][0] = 0.0
    ranges['Teff'][1] = 3500
    ranges['logg'][1] = 1.5
    ranges['MH'][1] = 0.0

    # Wavelengths
    initial_wave = 470.0
    final_wave = 680.0
    step_wave = 0.001
    wavelengths = np.arange(initial_wave, final_wave, step_wave)

    to_resolution = 80000 # Individual files will not be convolved but the grid will be (for fast comparison)
    number_of_processes = 1 # It can be parallelized for computers with multiple processors


    # Selected model amtosphere, linelist and solar abundances
    #model = ispec_dir + "/input/atmospheres/MARCS/modeled_layers_pack.dump"
    model = ispec_dir + "/input/atmospheres/MARCS.GES/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/MARCS.APOGEE/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.APOGEE/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Castelli/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kurucz/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kirby/modeled_layers_pack.dump"

    atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.1100_2400nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_hfs_iso.420_920nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_nohfs_noiso.420_920nm/atomic_lines.tsv"

    isotope_file = ispec_dir + "/input/isotopes/SPECTRUM.lst"

    # Load chemical information and linelist
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=initial_wave, wave_top=final_wave)
    atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01] # Select lines that have some minimal contribution in the sun

    isotopes = ispec.read_isotope_data(isotope_file)

    solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.2007/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2005/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.1998/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Anders.1989/stdatom.dat"

    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)
    # Load SPECTRUM abundances
    fixed_abundances = None # No fixed abundances
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)


    ispec.precompute_synthetic_grid(precomputed_grid_dir, ranges, wavelengths, to_resolution, \
                                    modeled_layers_pack, atomic_linelist, isotopes, solar_abundances, \
                                    segments=None, number_of_processes=number_of_processes, \
                                    code=code)


def determine_astrophysical_parameters_using_synth_spectra(code="spectrum"):
    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Radial Velocity determination with template -------------------------------
    logging.info("Radial velocity determination with template...")
    # - Read synthetic template
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Arcturus.372_926nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Sun.372_926nm/template.txt.gz")
    template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/NARVAL.Sun.370_1048nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Synth.Sun.300_1100nm/template.txt.gz")

    models, ccf = ispec.cross_correlate_with_template(star_spectrum, template, \
                            lower_velocity_limit=-200, upper_velocity_limit=200, \
                            velocity_step=1.0, fourier=False)

    # Number of models represent the number of components
    components = len(models)
    # First component:
    rv = np.round(models[0].mu(), 2) # km/s
    rv_err = np.round(models[0].emu(), 2) # km/s
    #--- Radial Velocity correction ------------------------------------------------
    logging.info("Radial velocity correction... %.2f +/- %.2f" % (rv, rv_err))
    star_spectrum = ispec.correct_velocity(star_spectrum, rv)
    #--- Resolution degradation ----------------------------------------------------
    # NOTE: The line selection was built based on a solar spectrum with R ~ 47,000 and VALD atomic linelist.
    from_resolution = 80000
    to_resolution = 47000
    star_spectrum = ispec.convolve_spectrum(star_spectrum, to_resolution, from_resolution)
    #--- Continuum fit -------------------------------------------------------------
    model = "Splines" # "Polynomy"
    degree = 2
    nknots = None # Automatic: 1 spline every 5 nm
    from_resolution = to_resolution

    # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
    order='median+max'
    median_wave_range=0.05
    max_wave_range=1.0

    star_continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=from_resolution, \
                                nknots=nknots, degree=degree, \
                                median_wave_range=median_wave_range, \
                                max_wave_range=max_wave_range, \
                                model=model, order=order, \
                                automatic_strong_line_detection=True, \
                                strong_line_probability=0.5, \
                                use_errors_for_fitting=True)
    #--- Normalize -------------------------------------------------------------
    normalized_star_spectrum = ispec.normalize_spectrum(star_spectrum, star_continuum_model, consider_continuum_errors=False)
    # Use a fixed value because the spectrum is already normalized
    star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")
    #--- Model spectra ----------------------------------------------------------
    # Parameters
    initial_teff = 5750.0
    initial_logg = 4.5
    initial_MH = 0.00
    initial_vmic = ispec.estimate_vmic(initial_teff, initial_logg, initial_MH)
    initial_vmac = ispec.estimate_vmac(initial_teff, initial_logg, initial_MH)
    initial_vsini = 2.0
    initial_limb_darkening_coeff = 0.6
    initial_R = to_resolution
    initial_vrad = 0
    max_iterations = 6

    # Selected model amtosphere, linelist and solar abundances
    #model = ispec_dir + "/input/atmospheres/MARCS/modeled_layers_pack.dump"
    model = ispec_dir + "/input/atmospheres/MARCS.GES/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/MARCS.APOGEE/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.APOGEE/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Castelli/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kurucz/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kirby/modeled_layers_pack.dump"

    atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.1100_2400nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_hfs_iso.420_920nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_nohfs_noiso.420_920nm/atomic_lines.tsv"

    solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.2007/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2005/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.1998/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Anders.1989/stdatom.dat"

    isotope_file = ispec_dir + "/input/isotopes/SPECTRUM.lst"

    # Load chemical information and linelist
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(star_spectrum['waveobs']), wave_top=np.max(star_spectrum['waveobs']))
    atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01] # Select lines that have some minimal contribution in the sun

    isotopes = ispec.read_isotope_data(isotope_file)


    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)

    # Load SPECTRUM abundances
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)

    # Free parameters
    #free_params = ["teff", "logg", "MH", "vmic", "vmac", "vsini", "R", "vrad", "limb_darkening_coeff"]
    free_params = ["teff", "logg", "MH", "vmic", "R"]

    # Free individual element abundance
    free_abundances = None
    linelist_free_loggf = None

    # Line regions
    line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/spectrum_synth_turbospectrum_synth_sme_synth_moog_synth_synthe_synth_good_for_params_all.txt")
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/spectrum_synth_turbospectrum_synth_sme_synth_moog_synth_synthe_synth_good_for_params_all_extended.txt")
    # Select only some lines to speed up the execution (in a real analysis it is better not to do this)
    line_regions = line_regions[np.logical_or(line_regions['note'] == 'Ti 1', line_regions['note'] == 'Ti 2')]
    line_regions = ispec.adjust_linemasks(normalized_star_spectrum, line_regions, max_margin=0.5)
    # Read segments if we have them or...
    #segments = ispec.read_segment_regions(ispec_dir + "/input/regions/fe_lines_segments.txt")
    # ... or we can create the segments on the fly:
    segments = ispec.create_segments_around_lines(line_regions, margin=0.25)

    ### Add also regions from the wings of strong lines:
    ## H beta
    #hbeta_lines = ispec.read_line_regions(ispec_dir + "input/regions/wings_Hbeta.txt")
    #hbeta_segments = ispec.read_segment_regions(ispec_dir + "input/regions/wings_Hbeta_segments.txt")
    #line_regions = np.hstack((line_regions, hbeta_lines))
    #segments = np.hstack((segments, hbeta_segments))
    ## H alpha
    #halpha_lines = ispec.read_line_regions(ispec_dir + "input/regions/wings_Halpha.txt")
    #halpha_segments = ispec.read_segment_regions(ispec_dir + "input/regions/wings_Halpha_segments.txt")
    #line_regions = np.hstack((line_regions, halpha_lines))
    #segments = np.hstack((segments, halpha_segments))
    ## Magnesium triplet
    #mgtriplet_lines = ispec.read_line_regions(ispec_dir + "input/regions/wings_MgTriplet.txt")
    #mgtriplet_segments = ispec.read_segment_regions(ispec_dir + "input/regions/wings_MgTriplet_segments.txt")
    #line_regions = np.hstack((line_regions, mgtriplet_lines))
    #segments = np.hstack((segments, mgtriplet_segments))

    obs_spec, modeled_synth_spectrum, params, errors, abundances_found, loggf_found, status, stats_linemasks = \
            ispec.model_spectrum(normalized_star_spectrum, star_continuum_model, \
            modeled_layers_pack, atomic_linelist, isotopes, solar_abundances, free_abundances, linelist_free_loggf, initial_teff, \
            initial_logg, initial_MH, initial_vmic, initial_vmac, initial_vsini, \
            initial_limb_darkening_coeff, initial_R, initial_vrad, free_params, segments=segments, \
            linemasks=line_regions, \
            enhance_abundances=True, \
            use_errors = True, \
            vmic_from_empirical_relation = False, \
            vmac_from_empirical_relation = True, \
            max_iterations=max_iterations, \
            tmp_dir = None, \
            code=code)
    ##--- Save results -------------------------------------------------------------
    logging.info("Saving results...")
    dump_file = "example_results_synth_%s.dump" % (code)
    logging.info("Saving results...")
    ispec.save_results(dump_file, (params, errors, abundances_found, loggf_found, status, stats_linemasks))
    # If we need to restore the results from another script:
    params, errors, abundances_found, loggf_found, status, stats_linemasks = ispec.restore_results(dump_file)

    logging.info("Saving synthetic spectrum...")
    synth_filename = "example_modeled_synth_%s.fits" % (code)
    ispec.write_spectrum(modeled_synth_spectrum, synth_filename)



def determine_astrophysical_parameters_using_synth_spectra_and_precomputed_grid(code="spectrum"):
    ############################################################################
    # WARNING !!!
    #  This routine depends on the previous precomputation of the synthetic grid
    ############################################################################
    precomputed_grid_dir = "example_grid_%s/" % (code)

    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Radial Velocity determination with template -------------------------------
    logging.info("Radial velocity determination with template...")
    # - Read synthetic template
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Arcturus.372_926nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Sun.372_926nm/template.txt.gz")
    template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/NARVAL.Sun.370_1048nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Synth.Sun.300_1100nm/template.txt.gz")

    models, ccf = ispec.cross_correlate_with_template(star_spectrum, template, \
                            lower_velocity_limit=-200, upper_velocity_limit=200, \
                            velocity_step=1.0, fourier=False)

    # Number of models represent the number of components
    components = len(models)
    # First component:
    rv = np.round(models[0].mu(), 2) # km/s
    rv_err = np.round(models[0].emu(), 2) # km/s
    #--- Radial Velocity correction ------------------------------------------------
    logging.info("Radial velocity correction... %.2f +/- %.2f" % (rv, rv_err))
    star_spectrum = ispec.correct_velocity(star_spectrum, rv)

    #--- Resolution degradation ----------------------------------------------------
    # NOTE: The line selection was built based on a solar spectrum with R ~ 47,000 and VALD atomic linelist.
    from_resolution = 80000
    to_resolution = 47000
    star_spectrum = ispec.convolve_spectrum(star_spectrum, to_resolution, from_resolution)

    #--- Continuum fit -------------------------------------------------------------
    model = "Splines" # "Polynomy"
    degree = 2
    nknots = None # Automatic: 1 spline every 5 nm
    from_resolution = to_resolution

    # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
    order='median+max'
    median_wave_range=0.05
    max_wave_range=1.0

    star_continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=from_resolution, \
                                nknots=nknots, degree=degree, \
                                median_wave_range=median_wave_range, \
                                max_wave_range=max_wave_range, \
                                model=model, order=order, \
                                automatic_strong_line_detection=True, \
                                strong_line_probability=0.5, \
                                use_errors_for_fitting=True)
    #--- Normalize -------------------------------------------------------------
    normalized_star_spectrum = ispec.normalize_spectrum(star_spectrum, star_continuum_model, consider_continuum_errors=False)
    # Use a fixed value because the spectrum is already normalized
    star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")

    #--- Model spectra ----------------------------------------------------------
    # Parameters
    initial_R = to_resolution
    max_iterations = 6

    # Selected model amtosphere, linelist and solar abundances
    #model = ispec_dir + "/input/atmospheres/MARCS/modeled_layers_pack.dump"
    model = ispec_dir + "/input/atmospheres/MARCS.GES/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/MARCS.APOGEE/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.APOGEE/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Castelli/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kurucz/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kirby/modeled_layers_pack.dump"

    atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.1100_2400nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_hfs_iso.420_920nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_nohfs_noiso.420_920nm/atomic_lines.tsv"

    solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.2007/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2005/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.1998/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Anders.1989/stdatom.dat"

    isotope_file = ispec_dir + "/input/isotopes/SPECTRUM.lst"

    # Load chemical information and linelist
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(star_spectrum['waveobs']), wave_top=np.max(star_spectrum['waveobs']))
    atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01] # Select lines that have some minimal contribution in the sun

    isotopes = ispec.read_isotope_data(isotope_file)


    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)

    # Load SPECTRUM abundances
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)

    # Free parameters
    #free_params = ["teff", "logg", "MH", "vmic", "vmac", "vsini", "R", "vrad", "limb_darkening_coeff"]
    free_params = ["teff", "logg", "MH", "vmic", "R"]

    # Free individual element abundance
    free_abundances = None
    linelist_free_loggf = None

    # Line regions
    line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/spectrum_synth_turbospectrum_synth_sme_synth_moog_synth_synthe_synth_good_for_params_all.txt")
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/spectrum_synth_turbospectrum_synth_sme_synth_moog_synth_synthe_synth_good_for_params_all_extended.txt")
    # Select only some lines to speed up the execution (in a real analysis it is better not to do this)
    line_regions = line_regions[np.logical_or(line_regions['note'] == 'Ti 1', line_regions['note'] == 'Ti 2')]
    line_regions = ispec.adjust_linemasks(normalized_star_spectrum, line_regions, max_margin=0.5)
    # Read segments if we have them or...
    segments = ispec.read_segment_regions(ispec_dir + "/input/regions/fe_lines_segments.txt")
    # ... or we can create the segments on the fly:
    #segments = ispec.create_segments_around_lines(line_regions, margin=0.25)

    ### Add also regions from the wings of strong lines:
    # H beta
    hbeta_lines = ispec.read_line_regions(ispec_dir + "input/regions/wings_Hbeta.txt")
    hbeta_segments = ispec.read_segment_regions(ispec_dir + "input/regions/wings_Hbeta_segments.txt")
    line_regions = np.hstack((line_regions, hbeta_lines))
    segments = np.hstack((segments, hbeta_segments))
    # H alpha
    halpha_lines = ispec.read_line_regions(ispec_dir + "input/regions/wings_Halpha.txt")
    halpha_segments = ispec.read_segment_regions(ispec_dir + "input/regions/wings_Halpha_segments.txt")
    line_regions = np.hstack((line_regions, halpha_lines))
    segments = np.hstack((segments, halpha_segments))
    # Magnesium triplet
    mgtriplet_lines = ispec.read_line_regions(ispec_dir + "input/regions/wings_MgTriplet.txt")
    mgtriplet_segments = ispec.read_segment_regions(ispec_dir + "input/regions/wings_MgTriplet_segments.txt")
    line_regions = np.hstack((line_regions, mgtriplet_lines))
    segments = np.hstack((segments, mgtriplet_segments))

    initial_vrad = 0
    initial_teff, initial_logg, initial_MH, initial_vmic, initial_vmac, initial_vsini, initial_limb_darkening_coeff = \
            ispec.estimate_initial_ap(normalized_star_spectrum, precomputed_grid_dir, initial_R, line_regions)
    print ("Initial estimation:", initial_teff, initial_logg, initial_MH, \
            initial_vmic, initial_vmac, initial_vsini, initial_limb_darkening_coeff)

    #--- Change LOG level ----------------------------------------------------------
    LOG_LEVEL = "warning"
    #LOG_LEVEL = "info"
    logger = logging.getLogger() # root logger, common for all
    logger.setLevel(logging.getLevelName(LOG_LEVEL.upper()))

    obs_spec, modeled_synth_spectrum, params, errors, abundances_found, loggf_found, status, stats_linemasks = \
            ispec.model_spectrum(normalized_star_spectrum, star_continuum_model, \
            modeled_layers_pack, atomic_linelist, isotopes, solar_abundances, free_abundances, linelist_free_loggf, initial_teff, \
            initial_logg, initial_MH, initial_vmic, initial_vmac, initial_vsini, \
            initial_limb_darkening_coeff, initial_R, initial_vrad, free_params, segments=segments, \
            linemasks=line_regions, \
            enhance_abundances=True, \
            precomputed_grid_dir = precomputed_grid_dir, \
            use_errors = True, \
            vmic_from_empirical_relation = False, \
            vmac_from_empirical_relation = True, \
            max_iterations=max_iterations, \
            tmp_dir = None, \
            code=code)

    #--- Change LOG level ----------------------------------------------------------
    #LOG_LEVEL = "warning"
    LOG_LEVEL = "info"
    logger = logging.getLogger() # root logger, common for all
    logger.setLevel(logging.getLevelName(LOG_LEVEL.upper()))

    ##--- Save results -------------------------------------------------------------
    logging.info("Saving results...")
    dump_file = "example_results_synth_precomputed_%s.dump" % (code)
    logging.info("Saving results...")
    ispec.save_results(dump_file, (params, errors, abundances_found, loggf_found, status, stats_linemasks))
    # If we need to restore the results from another script:
    params, errors, abundances_found, loggf_found, status, stats_linemasks = ispec.restore_results(dump_file)

    logging.info("Saving synthetic spectrum...")
    synth_filename = "example_modeled_synth_precomputed_%s.fits" % (code)
    ispec.write_spectrum(modeled_synth_spectrum, synth_filename)




def determine_abundances_using_synth_spectra(code="spectrum"):
    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Radial Velocity determination with template -------------------------------
    logging.info("Radial velocity determination with template...")
    # - Read synthetic template
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Arcturus.372_926nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Sun.372_926nm/template.txt.gz")
    template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/NARVAL.Sun.370_1048nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Synth.Sun.300_1100nm/template.txt.gz")

    models, ccf = ispec.cross_correlate_with_template(star_spectrum, template, \
                            lower_velocity_limit=-200, upper_velocity_limit=200, \
                            velocity_step=1.0, fourier=False)

    # Number of models represent the number of components
    components = len(models)
    # First component:
    rv = np.round(models[0].mu(), 2) # km/s
    rv_err = np.round(models[0].emu(), 2) # km/s
    #--- Radial Velocity correction ------------------------------------------------
    logging.info("Radial velocity correction... %.2f +/- %.2f" % (rv, rv_err))
    star_spectrum = ispec.correct_velocity(star_spectrum, rv)

    #--- Resolution degradation ----------------------------------------------------
    # NOTE: The line selection was built based on a solar spectrum with R ~ 47,000 and VALD atomic linelist.
    from_resolution = 80000
    to_resolution = 47000
    star_spectrum = ispec.convolve_spectrum(star_spectrum, to_resolution, from_resolution)

    #--- Continuum fit -------------------------------------------------------------
    model = "Splines" # "Polynomy"
    degree = 2
    nknots = None # Automatic: 1 spline every 5 nm
    from_resolution = to_resolution

    # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
    order='median+max'
    median_wave_range=0.05
    max_wave_range=1.0

    star_continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=from_resolution, \
                                nknots=nknots, degree=degree, \
                                median_wave_range=median_wave_range, \
                                max_wave_range=max_wave_range, \
                                model=model, order=order, \
                                automatic_strong_line_detection=True, \
                                strong_line_probability=0.5, \
                                use_errors_for_fitting=True)
    #--- Normalize -------------------------------------------------------------
    normalized_star_spectrum = ispec.normalize_spectrum(star_spectrum, star_continuum_model, consider_continuum_errors=False)
    # Use a fixed value because the spectrum is already normalized
    star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")
    #--- Model spectra ----------------------------------------------------------
    # Parameters
    initial_teff = 5771.0
    initial_logg = 4.44
    initial_MH = 0.00
    initial_vmic = ispec.estimate_vmic(initial_teff, initial_logg, initial_MH)
    initial_vmac = ispec.estimate_vmac(initial_teff, initial_logg, initial_MH)
    initial_vsini = 1.60 # Sun
    initial_limb_darkening_coeff = 0.6
    initial_R = to_resolution
    initial_vrad = 0
    max_iterations = 6

    # Selected model amtosphere, linelist and solar abundances
    #model = ispec_dir + "/input/atmospheres/MARCS/modeled_layers_pack.dump"
    model = ispec_dir + "/input/atmospheres/MARCS.GES/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/MARCS.APOGEE/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.APOGEE/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Castelli/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kurucz/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kirby/modeled_layers_pack.dump"

    atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.1100_2400nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_hfs_iso.420_920nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_nohfs_noiso.420_920nm/atomic_lines.tsv"

    solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.2007/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2005/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.1998/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Anders.1989/stdatom.dat"

    isotope_file = ispec_dir + "/input/isotopes/SPECTRUM.lst"

    # Load chemical information and linelist
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(star_spectrum['waveobs']), wave_top=np.max(star_spectrum['waveobs']))
    atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01] # Select lines that have some minimal contribution in the sun

    isotopes = ispec.read_isotope_data(isotope_file)



    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)

    # Load SPECTRUM abundances
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)


    # Free parameters
    #free_params = ["teff", "logg", "MH", "vmic", "vmac", "vsini", "R", "vrad", "limb_darkening_coeff"]
    free_params = ["vrad"]
    #free_params = []

    # Free individual element abundance (WARNING: it should be coherent with the selected line regions!)
    chemical_elements_file = ispec_dir + "/input/abundances/chemical_elements_symbols.dat"
    chemical_elements = ispec.read_chemical_elements(chemical_elements_file)

    element_name = "Ca"
    free_abundances = ispec.create_free_abundances_structure([element_name], chemical_elements, solar_abundances)
    free_abundances['Abund'] += initial_MH # Scale to metallicity

    linelist_free_loggf = None

    # Line regions
    line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/limited_but_with_missing_elements_%s_synth_good_for_abundances_all_extended.txt" % (code,))
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/limited_but_with_missing_elements_spectrum_synth_good_for_abundances_all_extended.txt")
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/limited_but_with_missing_elements_turobspectrum_synth_good_for_abundances_all_extended.txt")
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/limited_but_with_missing_elements_sme_synth_good_for_abundances_all_extended.txt")
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/limited_but_with_missing_elements_moog_synth_good_for_abundances_all_extended.txt")
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/limited_but_with_missing_elements_synthe_synth_good_for_abundances_all_extended.txt")
    # Select only the lines to get abundances from
    line_regions = line_regions[np.logical_or(line_regions['note'] == element_name+' 1', line_regions['note'] == element_name+' 2')]
    line_regions = ispec.adjust_linemasks(normalized_star_spectrum, line_regions, max_margin=0.5)

    # Read segments if we have them or...
    #segments = ispec.read_segment_regions(ispec_dir + "/input/regions/fe_lines_segments.txt")
    # ... or we can create the segments on the fly:
    segments = ispec.create_segments_around_lines(line_regions, margin=0.25)

    obs_spec, modeled_synth_spectrum, params, errors, abundances_found, loggf_found, status, stats_linemasks = \
            ispec.model_spectrum(normalized_star_spectrum, star_continuum_model, \
            modeled_layers_pack, atomic_linelist, isotopes, solar_abundances, free_abundances, linelist_free_loggf, initial_teff, \
            initial_logg, initial_MH, initial_vmic, initial_vmac, initial_vsini, \
            initial_limb_darkening_coeff, initial_R, initial_vrad, free_params, segments=segments, \
            linemasks=line_regions, \
            enhance_abundances=True, \
            use_errors = True, \
            vmic_from_empirical_relation = False, \
            vmac_from_empirical_relation = False, \
            max_iterations=max_iterations, \
            tmp_dir = None, \
            code=code)

    ##--- Save results -------------------------------------------------------------
    dump_file = "example_results_synth_abundances_%s.dump" % (code)
    logging.info("Saving results...")
    ispec.save_results(dump_file, (params, errors, abundances_found, loggf_found, status, stats_linemasks))
    # If we need to restore the results from another script:
    params, errors, abundances_found, loggf_found, status, stats_linemasks = ispec.restore_results(dump_file)

    logging.info("Saving synthetic spectrum...")
    synth_filename = "example_modeled_synth_abundances_%s.fits" % (code)
    ispec.write_spectrum(modeled_synth_spectrum, synth_filename)


def determine_abundances_line_by_line_using_synth_spectra(code="spectrum"):
    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Radial Velocity determination with template -------------------------------
    logging.info("Radial velocity determination with template...")
    # - Read synthetic template
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Arcturus.372_926nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Sun.372_926nm/template.txt.gz")
    template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/NARVAL.Sun.370_1048nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Synth.Sun.300_1100nm/template.txt.gz")

    models, ccf = ispec.cross_correlate_with_template(star_spectrum, template, \
                            lower_velocity_limit=-200, upper_velocity_limit=200, \
                            velocity_step=1.0, fourier=False)

    # Number of models represent the number of components
    components = len(models)
    # First component:
    rv = np.round(models[0].mu(), 2) # km/s
    rv_err = np.round(models[0].emu(), 2) # km/s
    #--- Radial Velocity correction ------------------------------------------------
    logging.info("Radial velocity correction... %.2f +/- %.2f" % (rv, rv_err))
    star_spectrum = ispec.correct_velocity(star_spectrum, rv)

    #--- Resolution degradation ----------------------------------------------------
    # NOTE: The line selection was built based on a solar spectrum with R ~ 47,000 and VALD atomic linelist.
    from_resolution = 80000
    to_resolution = 47000
    star_spectrum = ispec.convolve_spectrum(star_spectrum, to_resolution, from_resolution)

    #--- Continuum fit -------------------------------------------------------------
    model = "Splines" # "Polynomy"
    degree = 2
    nknots = None # Automatic: 1 spline every 5 nm
    from_resolution = to_resolution

    # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
    order='median+max'
    median_wave_range=0.05
    max_wave_range=1.0

    star_continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=from_resolution, \
                                nknots=nknots, degree=degree, \
                                median_wave_range=median_wave_range, \
                                max_wave_range=max_wave_range, \
                                model=model, order=order, \
                                automatic_strong_line_detection=True, \
                                strong_line_probability=0.5, \
                                use_errors_for_fitting=True)
    #--- Normalize -------------------------------------------------------------
    normalized_star_spectrum = ispec.normalize_spectrum(star_spectrum, star_continuum_model, consider_continuum_errors=False)
    # Use a fixed value because the spectrum is already normalized
    star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")
    #--- Model spectra ----------------------------------------------------------
    # Parameters
    initial_teff = 5771.0
    initial_logg = 4.44
    initial_MH = 0.00
    initial_vmic = ispec.estimate_vmic(initial_teff, initial_logg, initial_MH)
    initial_vmac = ispec.estimate_vmac(initial_teff, initial_logg, initial_MH)
    initial_vsini = 1.60 # Sun
    initial_limb_darkening_coeff = 0.6
    initial_R = to_resolution
    initial_vrad = 0
    max_iterations = 6

    # Selected model amtosphere, linelist and solar abundances
    #model = ispec_dir + "/input/atmospheres/MARCS/modeled_layers_pack.dump"
    model = ispec_dir + "/input/atmospheres/MARCS.GES/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/MARCS.APOGEE/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.APOGEE/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Castelli/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kurucz/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kirby/modeled_layers_pack.dump"

    atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.1100_2400nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_hfs_iso.420_920nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_nohfs_noiso.420_920nm/atomic_lines.tsv"

    solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.2007/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2005/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.1998/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Anders.1989/stdatom.dat"

    isotope_file = ispec_dir + "/input/isotopes/SPECTRUM.lst"

    # Load chemical information and linelist
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(star_spectrum['waveobs']), wave_top=np.max(star_spectrum['waveobs']))
    atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01] # Select lines that have some minimal contribution in the sun

    isotopes = ispec.read_isotope_data(isotope_file)



    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)

    # Load SPECTRUM abundances
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)


    # Free parameters
    #free_params = ["teff", "logg", "MH", "vmic", "vmac", "vsini", "R", "vrad", "limb_darkening_coeff"]
    free_params = ["vrad"]
    #free_params = []

    # Free individual element abundance (WARNING: it should be coherent with the selected line regions!)
    chemical_elements_file = ispec_dir + "/input/abundances/chemical_elements_symbols.dat"
    chemical_elements = ispec.read_chemical_elements(chemical_elements_file)

    # Line regions
    line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/limited_but_with_missing_elements_%s_synth_good_for_abundances_all_extended.txt" % (code,))
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/limited_but_with_missing_elements_spectrum_synth_good_for_abundances_all_extended.txt")
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/limited_but_with_missing_elements_turobspectrum_synth_good_for_abundances_all_extended.txt")
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/limited_but_with_missing_elements_sme_synth_good_for_abundances_all_extended.txt")
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/limited_but_with_missing_elements_moog_synth_good_for_abundances_all_extended.txt")
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/limited_but_with_missing_elements_synthe_synth_good_for_abundances_all_extended.txt")
    # Select only the lines to get abundances from
    line_regions = line_regions[:5]
    line_regions = ispec.adjust_linemasks(normalized_star_spectrum, line_regions, max_margin=0.5)

    output_dirname = "example_abundance_line_by_line_%s" % (code,)
    ispec.mkdir_p(output_dirname)
    for i, line in enumerate(line_regions):
        # Directory and file names
        #element_name = "_".join(line['element'].split())
        element_name = "_".join(line['note'].split())
        common_filename = "example_" + code + "_individual_" + element_name + "_%.4f" % line['wave_peak']

        # Free individual element abundance (WARNING: it should be coherent with the selected line regions!)
        #free_abundances = ispec.create_free_abundances_structure([line['element'].split()[0]], chemical_elements, solar_abundances)
        free_abundances = ispec.create_free_abundances_structure([line['note'].split()[0]], chemical_elements, solar_abundances)
        free_abundances['Abund'] += initial_MH # Scale to metallicity

        linelist_free_loggf = None

        # Line by line
        individual_line_regions = line_regions[i:i+1] # Keep recarray structure

        # Segment
        segments = ispec.create_segments_around_lines(individual_line_regions, margin=0.25)
        wfilter = ispec.create_wavelength_filter(normalized_star_spectrum, regions=segments) # Only use the segment

        obs_spec, modeled_synth_spectrum, params, errors, abundances_found, loggf_found, status, stats_linemasks = \
                ispec.model_spectrum(normalized_star_spectrum[wfilter], star_continuum_model, \
                modeled_layers_pack, atomic_linelist, isotopes, solar_abundances, free_abundances, linelist_free_loggf, initial_teff, \
                initial_logg, initial_MH, initial_vmic, initial_vmac, initial_vsini, \
                initial_limb_darkening_coeff, initial_R, initial_vrad, free_params, segments=segments, \
                linemasks=individual_line_regions, \
                enhance_abundances=True, \
                use_errors = True, \
                vmic_from_empirical_relation = False, \
                vmac_from_empirical_relation = False, \
                max_iterations=max_iterations, \
                tmp_dir = None, \
                code=code)


        ##--- Save results -------------------------------------------------------------
        dump_file = output_dirname + "/" + common_filename + ".dump"
        logging.info("Saving results...")
        ispec.save_results(dump_file, (params, errors, abundances_found, loggf_found, status, stats_linemasks))
        # If we need to restore the results from another script:
        #params, errors, abundances_found, loggf_found, status, stats_linemasks = ispec.restore_results(dump_file)

        logging.info("Saving synthetic spectrum...")
        synth_filename = output_dirname + "/" + common_filename + ".fits"
        ispec.write_spectrum(modeled_synth_spectrum, synth_filename)
        #ispec.write_line_regions(individual_line_regions, output_dirname + "/" + common_filename + "_linemask.txt")


def determine_loggf_line_by_line_using_synth_spectra(code="spectrum"):
    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Radial Velocity determination with template -------------------------------
    logging.info("Radial velocity determination with template...")
    # - Read synthetic template
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Arcturus.372_926nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Sun.372_926nm/template.txt.gz")
    template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/NARVAL.Sun.370_1048nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Synth.Sun.300_1100nm/template.txt.gz")

    models, ccf = ispec.cross_correlate_with_template(star_spectrum, template, \
                            lower_velocity_limit=-200, upper_velocity_limit=200, \
                            velocity_step=1.0, fourier=False)

    # Number of models represent the number of components
    components = len(models)
    # First component:
    rv = np.round(models[0].mu(), 2) # km/s
    rv_err = np.round(models[0].emu(), 2) # km/s
    #--- Radial Velocity correction ------------------------------------------------
    logging.info("Radial velocity correction... %.2f +/- %.2f" % (rv, rv_err))
    star_spectrum = ispec.correct_velocity(star_spectrum, rv)

    #--- Resolution degradation ----------------------------------------------------
    # NOTE: The line selection was built based on a solar spectrum with R ~ 47,000 and VALD atomic linelist.
    from_resolution = 80000
    to_resolution = 47000
    star_spectrum = ispec.convolve_spectrum(star_spectrum, to_resolution, from_resolution)

    #--- Continuum fit -------------------------------------------------------------
    model = "Splines" # "Polynomy"
    degree = 2
    nknots = None # Automatic: 1 spline every 5 nm
    from_resolution = to_resolution

    # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
    order='median+max'
    median_wave_range=0.05
    max_wave_range=1.0

    star_continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=from_resolution, \
                                nknots=nknots, degree=degree, \
                                median_wave_range=median_wave_range, \
                                max_wave_range=max_wave_range, \
                                model=model, order=order, \
                                automatic_strong_line_detection=True, \
                                strong_line_probability=0.5, \
                                use_errors_for_fitting=True)
    #--- Normalize -------------------------------------------------------------
    normalized_star_spectrum = ispec.normalize_spectrum(star_spectrum, star_continuum_model, consider_continuum_errors=False)
    # Use a fixed value because the spectrum is already normalized
    star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")
    #--- Model spectra ----------------------------------------------------------
    # Parameters
    initial_teff = 5771.0
    initial_logg = 4.44
    initial_MH = 0.00
    initial_vmic = ispec.estimate_vmic(initial_teff, initial_logg, initial_MH)
    initial_vmac = ispec.estimate_vmac(initial_teff, initial_logg, initial_MH)
    initial_vsini = 1.60 # Sun
    initial_limb_darkening_coeff = 0.6
    initial_R = to_resolution
    initial_vrad = 0
    max_iterations = 6

    # Selected model amtosphere, linelist and solar abundances
    #model = ispec_dir + "/input/atmospheres/MARCS/modeled_layers_pack.dump"
    model = ispec_dir + "/input/atmospheres/MARCS.GES/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/MARCS.APOGEE/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.APOGEE/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Castelli/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kurucz/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kirby/modeled_layers_pack.dump"

    atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.1100_2400nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_hfs_iso.420_920nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_nohfs_noiso.420_920nm/atomic_lines.tsv"

    solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.2007/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2005/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.1998/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Anders.1989/stdatom.dat"

    isotope_file = ispec_dir + "/input/isotopes/SPECTRUM.lst"

    # Load chemical information and linelist
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(star_spectrum['waveobs']), wave_top=np.max(star_spectrum['waveobs']))
    atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01] # Select lines that have some minimal contribution in the sun

    isotopes = ispec.read_isotope_data(isotope_file)



    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)

    # Load SPECTRUM abundances
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)


    # Free parameters
    #free_params = ["teff", "logg", "MH", "vmic", "vmac", "vsini", "R", "vrad", "limb_darkening_coeff"]
    #free_params = ["vrad"]
    free_params = []

    # Free individual element abundance (WARNING: it should be coherent with the selected line regions!)
    chemical_elements_file = ispec_dir + "/input/abundances/chemical_elements_symbols.dat"
    chemical_elements = ispec.read_chemical_elements(chemical_elements_file)

    # Line regions
    line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/spectrum_synth_turbospectrum_synth_sme_synth_moog_synth_synthe_synth_good_for_params_all_extended.txt")
    # Select only the lines to get abundances from
    line_regions = line_regions[:5]
    line_regions = ispec.adjust_linemasks(normalized_star_spectrum, line_regions, max_margin=0.5)

    output_dirname = "example_loggf_line_by_line_%s" % (code,)
    ispec.mkdir_p(output_dirname)
    for i, line in enumerate(line_regions):
        # Directory and file names
        #element_name = "_".join(line['element'].split())
        element_name = "_".join(line['note'].split())
        common_filename = "example_" + code + "_individual_" + element_name + "_%.4f" % line['wave_peak']

        # Free individual element abundance (WARNING: it should be coherent with the selected line regions!)
        free_abundances = None

        # Line by line
        individual_line_regions = line_regions[i:i+1] # Keep recarray structure
        linelist_free_loggf = line_regions[i:i+1] # Keep recarray structure

        # Filter the line that we want to determine the loggf from the global atomic linelist
        lfilter = atomic_linelist['element'] == linelist_free_loggf['element'][0]
        for key in ['wave_nm', 'lower_state_eV', 'loggf', 'stark', 'rad', 'waals']:
            lfilter = np.logical_and(lfilter, np.abs(atomic_linelist[key] - linelist_free_loggf[key][0]) < 1e-9)
        if len(atomic_linelist[lfilter]) > 1:
            logging.warn("Filtering more than one line!")
        if len(atomic_linelist[lfilter]) == 1:
            logging.warn("No line filtered!")

        # Segment
        segments = ispec.create_segments_around_lines(individual_line_regions, margin=0.25)
        wfilter = ispec.create_wavelength_filter(normalized_star_spectrum, regions=segments) # Only use the segment

        obs_spec, modeled_synth_spectrum, params, errors, abundances_found, loggf_found, status, stats_linemasks = \
                ispec.model_spectrum(normalized_star_spectrum[wfilter], star_continuum_model, \
                modeled_layers_pack, atomic_linelist[~lfilter], isotopes, solar_abundances, free_abundances, linelist_free_loggf, initial_teff, \
                initial_logg, initial_MH, initial_vmic, initial_vmac, initial_vsini, \
                initial_limb_darkening_coeff, initial_R, initial_vrad, free_params, segments=segments, \
                linemasks=individual_line_regions, \
                enhance_abundances=True, \
                use_errors = True, \
                vmic_from_empirical_relation = False, \
                vmac_from_empirical_relation = False, \
                max_iterations=max_iterations, \
                tmp_dir = None, \
                code=code)



        ##--- Save results -------------------------------------------------------------
        dump_file = output_dirname + "/" + common_filename + ".dump"
        logging.info("Saving results...")
        ispec.save_results(dump_file, (params, errors, abundances_found, loggf_found, status, stats_linemasks))
        # If we need to restore the results from another script:
        #params, errors, abundances_found, loggf_found, status, stats_linemasks = ispec.restore_results(dump_file)

        logging.info("Saving synthetic spectrum...")
        synth_filename = output_dirname + "/" + common_filename + ".fits"
        ispec.write_spectrum(modeled_synth_spectrum, synth_filename)
        #ispec.write_line_regions(individual_line_regions, output_dirname + "/" + common_filename + "_linemask.txt")

        linelist_file = output_dirname + "/" + common_filename + ".txt"
        ispec.write_atomic_linelist(loggf_found['linelist'], linelist_filename=linelist_file)




def determine_astrophysical_parameters_from_ew(code="width", use_lines_already_crossmatched_with_atomic_data=True, use_ares=False):
    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Radial Velocity determination with template -------------------------------
    logging.info("Radial velocity determination with template...")
    # - Read synthetic template
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Arcturus.372_926nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Sun.372_926nm/template.txt.gz")
    template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/NARVAL.Sun.370_1048nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Synth.Sun.300_1100nm/template.txt.gz")

    models, ccf = ispec.cross_correlate_with_template(star_spectrum, template, \
                            lower_velocity_limit=-200, upper_velocity_limit=200, \
                            velocity_step=1.0, fourier=False)

    # Number of models represent the number of components
    components = len(models)
    # First component:
    rv = np.round(models[0].mu(), 2) # km/s
    rv_err = np.round(models[0].emu(), 2) # km/s
    #--- Radial Velocity correction ------------------------------------------------
    logging.info("Radial velocity correction... %.2f +/- %.2f" % (rv, rv_err))
    star_spectrum = ispec.correct_velocity(star_spectrum, rv)
    #--- Telluric velocity shift determination from spectrum --------------------------
    logging.info("Telluric velocity shift determination...")
    # - Telluric
    telluric_linelist_file = ispec_dir + "/input/linelists/CCF/Synth.Tellurics.500_1100nm/mask.lst"
    telluric_linelist = ispec.read_telluric_linelist(telluric_linelist_file, minimum_depth=0.0)

    models, ccf = ispec.cross_correlate_with_mask(star_spectrum, telluric_linelist, \
                            lower_velocity_limit=-100, upper_velocity_limit=100, \
                            velocity_step=0.5, mask_depth=0.01, \
                            fourier = False,
                            only_one_peak = True)

    vel_telluric = np.round(models[0].mu(), 2) # km/s
    vel_telluric_err = np.round(models[0].emu(), 2) # km/s
    #--- Resolution degradation ----------------------------------------------------
    # NOTE: The line selection was built based on a solar spectrum with R ~ 47,000 and VALD atomic linelist.
    from_resolution = 80000
    to_resolution = 47000
    star_spectrum = ispec.convolve_spectrum(star_spectrum, to_resolution, from_resolution)
    #--- Continuum fit -------------------------------------------------------------
    model = "Splines" # "Polynomy"
    degree = 2
    nknots = None # Automatic: 1 spline every 5 nm
    #from_resolution = 80000
    from_resolution = to_resolution

    # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
    order='median+max'
    median_wave_range=0.05
    max_wave_range=1.0

    star_continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=from_resolution, \
                                nknots=nknots, degree=degree, \
                                median_wave_range=median_wave_range, \
                                max_wave_range=max_wave_range, \
                                model=model, order=order, \
                                automatic_strong_line_detection=True, \
                                strong_line_probability=0.5, \
                                use_errors_for_fitting=True)
    #--- Normalize -------------------------------------------------------------
    normalized_star_spectrum = ispec.normalize_spectrum(star_spectrum, star_continuum_model, consider_continuum_errors=False)
    # Use a fixed value because the spectrum is already normalized
    star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")

    #telluric_linelist_file = ispec_dir + "/input/linelists/CCF/Synth.Tellurics.500_1100nm/mask.lst"
    #telluric_linelist = ispec.read_telluric_linelist(telluric_linelist_file, minimum_depth=0.01)
    #vel_telluric = 17.79 # km/s
    #telluric_linelist = None
    #vel_telluric = None

    if use_lines_already_crossmatched_with_atomic_data:
        #--- Read lines and adjust them ------------------------------------------------
        if code in ['width', 'moog']:
            line_regions_with_atomic_data = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/moog_ew_ispec_width_ew_ispec_good_for_params_all_extended.txt")
        else:
            line_regions_with_atomic_data = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/spectrum_synth_turbospectrum_synth_sme_synth_moog_synth_synthe_synth_moog_ew_ispec_width_ew_ispec_good_for_params_all_extended.txt")

        # Select only iron lines
        line_regions_with_atomic_data = line_regions_with_atomic_data[np.logical_or(line_regions_with_atomic_data['note'] == "Fe 1", line_regions_with_atomic_data['note'] == "Fe 2")]

        smoothed_star_spectrum = ispec.convolve_spectrum(star_spectrum, 2*to_resolution)
        line_regions_with_atomic_data = ispec.adjust_linemasks(smoothed_star_spectrum, line_regions_with_atomic_data, max_margin=0.5)

        #--- Fit the lines but do NOT cross-match with any atomic linelist since they already have that information
        linemasks = ispec.fit_lines(line_regions_with_atomic_data, normalized_star_spectrum, star_continuum_model, \
                                    atomic_linelist = None, \
                                    max_atomic_wave_diff = 0.005, \
                                    telluric_linelist = telluric_linelist, \
                                    check_derivatives = False, \
                                    vel_telluric = vel_telluric, discard_gaussian=False, \
                                    smoothed_spectrum=None, \
                                    discard_voigt=True, \
                                    free_mu=True, crossmatch_with_mu=False, closest_match=False)
    else:
        #--- Fit lines -----------------------------------------------------------------
        logging.info("Fitting lines...")
        atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv"
        #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.1100_2400nm/atomic_lines.tsv"
        #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_hfs_iso.420_920nm/atomic_lines.tsv"
        #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_nohfs_noiso.420_920nm/atomic_lines.tsv"

        # Read
        atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(star_spectrum['waveobs']), wave_top=np.max(star_spectrum['waveobs']))

        if code in ['width', 'moog']:
            line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/moog_ew_ispec_width_ew_ispec_good_for_params_all.txt")
        else:
            line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/spectrum_synth_turbospectrum_synth_sme_synth_moog_synth_synthe_synth_moog_ew_ispec_width_ew_ispec_good_for_params_all.txt")

        line_regions = ispec.adjust_linemasks(normalized_star_spectrum, line_regions, max_margin=0.5)

        linemasks = ispec.fit_lines(line_regions, normalized_star_spectrum, star_continuum_model, \
                                    atomic_linelist = atomic_linelist, \
                                    #max_atomic_wave_diff = 0.005, \
                                    max_atomic_wave_diff = 0.00, \
                                    telluric_linelist = telluric_linelist, \
                                    smoothed_spectrum = None, \
                                    check_derivatives = False, \
                                    vel_telluric = vel_telluric, discard_gaussian=False, \
                                    discard_voigt=True, \
                                    free_mu=True, crossmatch_with_mu=False, closest_match=False)
        # Discard lines that are not cross matched with the same original element stored in the note
        linemasks = linemasks[linemasks['element'] == line_regions['note']]

        # Exclude lines that have not been successfully cross matched with the atomic data
        # because we cannot calculate the chemical abundance (it will crash the corresponding routines)
        rejected_by_atomic_line_not_found = (linemasks['wave_nm'] == 0)
        linemasks = linemasks[~rejected_by_atomic_line_not_found]


    # Discard bad masks
    flux_peak = normalized_star_spectrum['flux'][linemasks['peak']]
    flux_base = normalized_star_spectrum['flux'][linemasks['base']]
    flux_top = normalized_star_spectrum['flux'][linemasks['top']]
    bad_mask = np.logical_or(linemasks['wave_peak'] <= linemasks['wave_base'], linemasks['wave_peak'] >= linemasks['wave_top'])
    bad_mask = np.logical_or(bad_mask, flux_peak >= flux_base)
    bad_mask = np.logical_or(bad_mask, flux_peak >= flux_top)
    linemasks = linemasks[~bad_mask]

    # Exclude lines with EW equal to zero
    rejected_by_zero_ew = (linemasks['ew'] == 0)
    linemasks = linemasks[~rejected_by_zero_ew]

    # Exclude lines that may be affected by tellurics
    rejected_by_telluric_line = (linemasks['telluric_wave_peak'] != 0)
    linemasks = linemasks[~rejected_by_telluric_line]

    if use_ares:
        # Replace the measured equivalent widths by the ones computed by ARES
        old_linemasks = linemasks.copy()
        ### Different rejection parameters (check ARES papers):
        ##   - http://adsabs.harvard.edu/abs/2007A%26A...469..783S
        ##   - http://adsabs.harvard.edu/abs/2015A%26A...577A..67S
        #linemasks = ispec.update_ew_with_ares(normalized_star_spectrum, linemasks, rejt="0.995", tmp_dir=None, verbose=0)
        #linemasks = ispec.update_ew_with_ares(normalized_star_spectrum, linemasks, rejt="3;5764,5766,6047,6052,6068,6076", tmp_dir=None, verbose=0)
        snr = 50
        linemasks = ispec.update_ew_with_ares(normalized_star_spectrum, linemasks, rejt="%s" % (snr), tmp_dir=None, verbose=0)


    #--- Model spectra from EW --------------------------------------------------
    # Parameters
    initial_teff = 5777.0
    initial_logg = 4.44
    initial_MH = 0.00
    initial_vmic = ispec.estimate_vmic(initial_teff, initial_logg, initial_MH)
    max_iterations = 10

    # Selected model amtosphere, linelist and solar abundances
    #model = ispec_dir + "/input/atmospheres/MARCS/modeled_layers_pack.dump"
    model = ispec_dir + "/input/atmospheres/MARCS.GES/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/MARCS.APOGEE/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.APOGEE/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Castelli/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kurucz/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kirby/modeled_layers_pack.dump"

    atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.1100_2400nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_hfs_iso.420_920nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_nohfs_noiso.420_920nm/atomic_lines.tsv"

    solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.2007/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2005/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.1998/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Anders.1989/stdatom.dat"


    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)

    # Load SPECTRUM abundances
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)


    # Validate parameters
    if not ispec.valid_atmosphere_target(modeled_layers_pack, initial_teff, initial_logg, initial_MH):
        msg = "The specified effective temperature, gravity (log g) and metallicity [M/H] \
                fall out of theatmospheric models."
        print (msg)

    # Reduced equivalent width
    # Filter too weak/strong lines
    # * Criteria presented in paper of GALA
    #efilter = np.logical_and(linemasks['ewr'] >= -5.8, linemasks['ewr'] <= -4.65)
    efilter = np.logical_and(linemasks['ewr'] >= -6.0, linemasks['ewr'] <= -4.3)
    # Filter high excitation potential lines
    # * Criteria from Eric J. Bubar "Equivalent Width Abundance Analysis In Moog"
    efilter = np.logical_and(efilter, linemasks['lower_state_eV'] <= 5.0)
    efilter = np.logical_and(efilter, linemasks['lower_state_eV'] >= 0.5)
    ## Filter also bad fits
    efilter = np.logical_and(efilter, linemasks['rms'] < 1.00)
    # no flux
    noflux = normalized_star_spectrum['flux'][linemasks['peak']] < 1.0e-10
    efilter = np.logical_and(efilter, np.logical_not(noflux))
    unfitted = linemasks['fwhm'] == 0
    efilter = np.logical_and(efilter, np.logical_not(unfitted))

    results = ispec.model_spectrum_from_ew(linemasks[efilter], modeled_layers_pack, \
                        solar_abundances, initial_teff, initial_logg, initial_MH, initial_vmic, \
                        free_params=["teff", "logg", "vmic"], \
                        adjust_model_metalicity=True, \
                        max_iterations=max_iterations, \
                        enhance_abundances=True, \
                        #outliers_detection = "robust", \
                        #outliers_weight_limit = 0.90, \
                        outliers_detection = "sigma_clipping", \
                        #sigma_level = 3, \
                        tmp_dir = None, \
                        code=code)
    params, errors, status, x_over_h, selected_x_over_h, fitted_lines_params, used_linemasks = results

    ##--- Save results -------------------------------------------------------------
    logging.info("Saving results...")
    dump_file = "example_results_ew_%s.dump" % (code)

    ispec.save_results(dump_file, (params, errors, status, x_over_h, selected_x_over_h, fitted_lines_params, used_linemasks))
    # If we need to restore the results from another script:
    params, errors, status, x_over_h, selected_x_over_h, fitted_lines_param, used_linemasks = ispec.restore_results(dump_file)



def determine_abundances_from_ew(code="spectrum", use_lines_already_crossmatched_with_atomic_data=True, use_ares=False):
    star_spectrum = ispec.read_spectrum(ispec_dir + "/input/spectra/examples/NARVAL_Sun_Vesta-1.txt.gz")
    #--- Radial Velocity determination with template -------------------------------
    logging.info("Radial velocity determination with template...")
    # - Read synthetic template
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Arcturus.372_926nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Sun.372_926nm/template.txt.gz")
    template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/NARVAL.Sun.370_1048nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Synth.Sun.300_1100nm/template.txt.gz")

    models, ccf = ispec.cross_correlate_with_template(star_spectrum, template, \
                            lower_velocity_limit=-200, upper_velocity_limit=200, \
                            velocity_step=1.0, fourier=False)

    # Number of models represent the number of components
    components = len(models)
    # First component:
    rv = np.round(models[0].mu(), 2) # km/s
    rv_err = np.round(models[0].emu(), 2) # km/s
    #--- Radial Velocity correction ------------------------------------------------
    logging.info("Radial velocity correction... %.2f +/- %.2f" % (rv, rv_err))
    star_spectrum = ispec.correct_velocity(star_spectrum, rv)
    #--- Telluric velocity shift determination from spectrum --------------------------
    logging.info("Telluric velocity shift determination...")
    # - Telluric
    telluric_linelist_file = ispec_dir + "/input/linelists/CCF/Synth.Tellurics.500_1100nm/mask.lst"
    telluric_linelist = ispec.read_telluric_linelist(telluric_linelist_file, minimum_depth=0.0)

    models, ccf = ispec.cross_correlate_with_mask(star_spectrum, telluric_linelist, \
                            lower_velocity_limit=-100, upper_velocity_limit=100, \
                            velocity_step=0.5, mask_depth=0.01, \
                            fourier = False,
                            only_one_peak = True)

    vel_telluric = np.round(models[0].mu(), 2) # km/s
    vel_telluric_err = np.round(models[0].emu(), 2) # km/s
    #--- Resolution degradation ----------------------------------------------------
    # NOTE: The line selection was built based on a solar spectrum with R ~ 47,000 and VALD atomic linelist.
    from_resolution = 80000
    to_resolution = 47000
    star_spectrum = ispec.convolve_spectrum(star_spectrum, to_resolution, from_resolution)
    #--- Continuum fit -------------------------------------------------------------
    model = "Splines" # "Polynomy"
    degree = 2
    nknots = None # Automatic: 1 spline every 5 nm
    #from_resolution = 80000
    from_resolution = to_resolution

    # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
    order='median+max'
    median_wave_range=0.05
    max_wave_range=1.0

    star_continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=from_resolution, \
                                nknots=nknots, degree=degree, \
                                median_wave_range=median_wave_range, \
                                max_wave_range=max_wave_range, \
                                model=model, order=order, \
                                automatic_strong_line_detection=True, \
                                strong_line_probability=0.5, \
                                use_errors_for_fitting=True)
    #--- Normalize -------------------------------------------------------------
    normalized_star_spectrum = ispec.normalize_spectrum(star_spectrum, star_continuum_model, consider_continuum_errors=False)
    # Use a fixed value because the spectrum is already normalized
    star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")

    #telluric_linelist_file = ispec_dir + "/input/linelists/CCF/Synth.Tellurics.500_1100nm/mask.lst"
    #telluric_linelist = ispec.read_telluric_linelist(telluric_linelist_file, minimum_depth=0.01)
    #vel_telluric = 17.79 # km/s
    #telluric_linelist = None
    #vel_telluric = None

    if use_lines_already_crossmatched_with_atomic_data:
        #--- Read lines and adjust them ------------------------------------------------
        if code in ['width', 'moog']:
            line_regions_with_atomic_data = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/moog_ew_ispec_width_ew_ispec_good_for_params_all_extended.txt")
        else:
            line_regions_with_atomic_data = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/spectrum_synth_turbospectrum_synth_sme_synth_moog_synth_synthe_synth_moog_ew_ispec_width_ew_ispec_good_for_params_all_extended.txt")

        # Select only iron lines
        line_regions_with_atomic_data = line_regions_with_atomic_data[np.logical_or(line_regions_with_atomic_data['note'] == "Fe 1", line_regions_with_atomic_data['note'] == "Fe 2")]

        smoothed_star_spectrum = ispec.convolve_spectrum(star_spectrum, 2*to_resolution)
        line_regions_with_atomic_data = ispec.adjust_linemasks(smoothed_star_spectrum, line_regions_with_atomic_data, max_margin=0.5)

        #--- Fit the lines but do NOT cross-match with any atomic linelist since they already have that information
        linemasks = ispec.fit_lines(line_regions_with_atomic_data, normalized_star_spectrum, star_continuum_model, \
                                    atomic_linelist = None, \
                                    max_atomic_wave_diff = 0.005, \
                                    telluric_linelist = telluric_linelist, \
                                    check_derivatives = False, \
                                    vel_telluric = vel_telluric, discard_gaussian=False, \
                                    smoothed_spectrum=None, \
                                    discard_voigt=True, \
                                    free_mu=True, crossmatch_with_mu=False, closest_match=False)
    else:
        #--- Fit lines -----------------------------------------------------------------
        logging.info("Fitting lines...")
        atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv"
        #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.1100_2400nm/atomic_lines.tsv"
        #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_hfs_iso.420_920nm/atomic_lines.tsv"
        #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_nohfs_noiso.420_920nm/atomic_lines.tsv"

        # Read
        atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(star_spectrum['waveobs']), wave_top=np.max(star_spectrum['waveobs']))

        if code in ['width', 'moog']:
            line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/moog_ew_ispec_width_ew_ispec_good_for_params_all.txt")
        else:
            line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/spectrum_synth_turbospectrum_synth_sme_synth_moog_synth_synthe_synth_moog_ew_ispec_width_ew_ispec_good_for_params_all.txt")

        line_regions = ispec.adjust_linemasks(normalized_star_spectrum, line_regions, max_margin=0.5)

        linemasks = ispec.fit_lines(line_regions, normalized_star_spectrum, star_continuum_model, \
                                    atomic_linelist = atomic_linelist, \
                                    #max_atomic_wave_diff = 0.005, \
                                    max_atomic_wave_diff = 0.00, \
                                    telluric_linelist = telluric_linelist, \
                                    smoothed_spectrum = None, \
                                    check_derivatives = False, \
                                    vel_telluric = vel_telluric, discard_gaussian=False, \
                                    discard_voigt=True, \
                                    free_mu=True, crossmatch_with_mu=False, closest_match=False)
        # Discard lines that are not cross matched with the same original element stored in the note
        linemasks = linemasks[linemasks['element'] == line_regions['note']]

        # Exclude lines that have not been successfully cross matched with the atomic data
        # because we cannot calculate the chemical abundance (it will crash the corresponding routines)
        rejected_by_atomic_line_not_found = (linemasks['wave_nm'] == 0)
        linemasks = linemasks[~rejected_by_atomic_line_not_found]


    # Discard bad masks
    flux_peak = normalized_star_spectrum['flux'][linemasks['peak']]
    flux_base = normalized_star_spectrum['flux'][linemasks['base']]
    flux_top = normalized_star_spectrum['flux'][linemasks['top']]
    bad_mask = np.logical_or(linemasks['wave_peak'] <= linemasks['wave_base'], linemasks['wave_peak'] >= linemasks['wave_top'])
    bad_mask = np.logical_or(bad_mask, flux_peak >= flux_base)
    bad_mask = np.logical_or(bad_mask, flux_peak >= flux_top)
    linemasks = linemasks[~bad_mask]

    # Exclude lines with EW equal to zero
    rejected_by_zero_ew = (linemasks['ew'] == 0)
    linemasks = linemasks[~rejected_by_zero_ew]

    # Exclude lines that may be affected by tellurics
    rejected_by_telluric_line = (linemasks['telluric_wave_peak'] != 0)
    linemasks = linemasks[~rejected_by_telluric_line]

    if use_ares:
        # Replace the measured equivalent widths by the ones computed by ARES
        old_linemasks = linemasks.copy()
        ### Different rejection parameters (check ARES papers):
        ##   - http://adsabs.harvard.edu/abs/2007A%26A...469..783S
        ##   - http://adsabs.harvard.edu/abs/2015A%26A...577A..67S
        #linemasks = ispec.update_ew_with_ares(normalized_star_spectrum, linemasks, rejt="0.995", tmp_dir=None, verbose=0)
        #linemasks = ispec.update_ew_with_ares(normalized_star_spectrum, linemasks, rejt="3;5764,5766,6047,6052,6068,6076", tmp_dir=None, verbose=0)
        snr = 50
        linemasks = ispec.update_ew_with_ares(normalized_star_spectrum, linemasks, rejt="%s" % (snr), tmp_dir=None, verbose=0)


    #--- Determining abundances by EW of the previously fitted lines ---------------
    # Parameters
    teff = 5777.0
    logg = 4.44
    MH = 0.00
    microturbulence_vel = 1.0

    # Selected model amtosphere and solar abundances
    #model = ispec_dir + "/input/atmospheres/MARCS/modeled_layers_pack.dump"
    model = ispec_dir + "/input/atmospheres/MARCS.GES/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/MARCS.APOGEE/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.APOGEE/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Castelli/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kurucz/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kirby/modeled_layers_pack.dump"

    solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.2007/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2005/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.1998/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Anders.1989/stdatom.dat"

    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)
    # Load SPECTRUM abundances
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)

    # Validate parameters
    if not ispec.valid_atmosphere_target(modeled_layers_pack, teff, logg, MH):
        msg = "The specified effective temperature, gravity (log g) and metallicity [M/H] \
                fall out of theatmospheric models."
        print (msg)

    # Prepare atmosphere model
    atmosphere_layers = ispec.interpolate_atmosphere_layers(modeled_layers_pack, teff, logg, MH, code=code)
    spec_abund, normal_abund, x_over_h, x_over_fe = ispec.determine_abundances(atmosphere_layers, \
            teff, logg, MH, linemasks, solar_abundances, microturbulence_vel = microturbulence_vel, \
            verbose=1, code=code)

    bad = np.isnan(x_over_h)
    fe1 = linemasks['element'] == "Fe 1"
    fe2 = linemasks['element'] == "Fe 2"
    #print "[Fe 1/H]: %.2f" % np.median(x_over_h[np.logical_and(fe1, ~bad)])
    #print "[X/Fe]: %.2f" % np.median(x_over_fe[np.logical_and(fe1, ~bad)])
    #print "[Fe 2/H]: %.2f" % np.median(x_over_h[np.logical_and(fe2, ~bad)])
    #print "[X/Fe]: %.2f" % np.median(x_over_fe[np.logical_and(fe2, ~bad)])



def calculate_theoretical_ew_and_depth(code="spectrum"):
    #--- Calculate theoretical equivalent widths and depths for a linelist ---------
    # Parameters
    teff = 5777.0
    logg = 4.44
    MH = 0.00
    microturbulence_vel = 1.0

    # Selected model amtosphere, linelist and solar abundances
    #model = ispec_dir + "/input/atmospheres/MARCS/modeled_layers_pack.dump"
    model = ispec_dir + "/input/atmospheres/MARCS.GES/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/MARCS.APOGEE/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.APOGEE/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Castelli/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kurucz/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kirby/modeled_layers_pack.dump"

    atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.1100_2400nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_hfs_iso.420_920nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_nohfs_noiso.420_920nm/atomic_lines.tsv"

    solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.2007/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2005/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.1998/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Anders.1989/stdatom.dat"

    isotope_file = ispec_dir + "/input/isotopes/SPECTRUM.lst"

    # Load chemical information and linelist
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file)
    atomic_linelist = atomic_linelist[:100] # Select only the first 100 lines (just to reduce execution time, don't do it in a real analysis)

    isotopes = ispec.read_isotope_data(isotope_file)

    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)

    # Load SPECTRUM abundances
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)

    # Validate parameters
    if not ispec.valid_atmosphere_target(modeled_layers_pack, teff, logg, MH):
        msg = "The specified effective temperature, gravity (log g) and metallicity [M/H] \
                fall out of theatmospheric models."
        print (msg)

    # Enhance alpha elements + CNO abundances following MARCS standard composition
    alpha_enhancement, c_enhancement, n_enhancement, o_enhancement = ispec.determine_abundance_enchancements(MH)
    abundances = ispec.enhance_solar_abundances(solar_abundances, alpha_enhancement, c_enhancement, n_enhancement, o_enhancement)

    # Prepare atmosphere model
    atmosphere_layers = ispec.interpolate_atmosphere_layers(modeled_layers_pack, teff, logg, MH)

    # Synthesis
    #output_wave, output_code, output_ew, output_depth = ispec.calculate_theoretical_ew_and_depth(atmosphere_layers, \
    new_atomic_linelist = ispec.calculate_theoretical_ew_and_depth(atmosphere_layers, \
            teff, logg, MH, \
            atomic_linelist[:10], isotopes, abundances, microturbulence_vel=microturbulence_vel, \
            verbose=1, gui_queue=None, timeout=900)
    ispec.write_atomic_linelist(new_atomic_linelist, linelist_filename="example_linelist.txt")



def analyze(text, number):
    #--- Fake function to be used in the parallelization example -------------------
    multiprocessing.current_process().daemon=False
    import time

    # Print some text and wait for some seconds to finish
    print( "Starting", text)
    time.sleep(2 + number)
    print ("... end of", number)

def paralelize_code():
    number_of_processes = 4
    pool = Pool(number_of_processes)
    #--- Send 5 analyze processes to the pool which will execute 2 in parallel -----
    pool.apply_async(analyze, ["one", 1])
    pool.apply_async(analyze, ["two", 2])
    pool.apply_async(analyze, ["three", 3])
    pool.apply_async(analyze, ["four", 4])
    pool.apply_async(analyze, ["five", 5])
    pool.close()
    pool.join()


def estimate_vmic_from_empirical_relation():
    teff = 5500
    logg = 4.5
    MH = 0.0
    vmic = ispec.estimate_vmic(teff, logg, MH)
    print ("VMIC:", vmic)

def estimate_vmac_from_empirical_relation():
    teff = 5500
    logg = 4.5
    MH = 0.0
    vmac = ispec.estimate_vmac(teff, logg, MH)
    print ("VMAC:", vmac)

def generate_and_plot_YY_isochrone():
    import isochrones
    import matplotlib.pyplot as plt

    logage = 9.409
    age = np.power(10, logage) / 1e9 # Gyrs
    MH = 0.0 # [M/H] (dex)
    isochrone = isochrones.interpolate_isochrone(ispec_dir, age, MH)

    plt.plot(np.power(10, isochrone['logT']),  isochrone['logg'], marker='', ls='-', color="blue", label="[M/H] %.2f, %.2f Gyrs" % (MH, age))
    plt.xlabel("$T_{eff}$ (K)")
    plt.ylabel("$log(g)$ (dex)")
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.show()



def interpolate_atmosphere(code="spectrum"):
    #--- Synthesizing spectrum -----------------------------------------------------
    # Parameters
    teff = 5777.0
    logg = 4.44
    MH = 0.05

    # Selected model amtosphere, linelist and solar abundances
    #model = ispec_dir + "/input/atmospheres/MARCS/modeled_layers_pack.dump"
    model = ispec_dir + "/input/atmospheres/MARCS.GES/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/MARCS.APOGEE/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.APOGEE/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Castelli/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kurucz/modeled_layers_pack.dump"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kirby/modeled_layers_pack.dump"

    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)

    # Validate parameters
    if not ispec.valid_atmosphere_target(modeled_layers_pack, teff, logg, MH):
        msg = "The specified effective temperature, gravity (log g) and metallicity [M/H] \
                fall out of theatmospheric models."
        print (msg)

    # Prepare atmosphere model
    atmosphere_layers = ispec.interpolate_atmosphere_layers(modeled_layers_pack, teff, logg, MH, code=code)
    atmosphere_layers_file = "example_atmosphere.txt"
    atmosphere_layers_file = ispec.write_atmosphere(atmosphere_layers, teff, logg, MH, atmosphere_filename=atmosphere_layers_file, code=code)


if __name__ == '__main__':
                                    
    files=glob.glob('/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0730/reduced/bgoH202207300018.fits')

#    for i in range(0,len(files)):
#    
#        hdu =fits.open(files[i])
#        FIBRE_O=hdu['FIBRE_O'].data
#        WAVE_O=hdu['WAVE_O'].data
#        
#        nord=42
#        x=np.arange(nord)
#        x2=np.arange(nord-7)+5
#        new_wave = np.copy(WAVE_O)
#        for i in range(2048):
#            y = []
#            for j in range(nord-7):
#                j +=5
#                y.append(WAVE_O[j][i])
#    
#            fit = np.polyval(np.polyfit(x2,y,6),x)
#            for ord in range(nord):
#                new_wave[ord][i]=fit[ord]
#        
#        for ord in range(42):
#            wav = WAVE_O[ord]
#            flx = FIBRE_O[ord]
#            snr = np.sqrt(flx)
#            
#            np.savetxt('tmp.dat',np.array([wav/10.,flx,snr]).T)
#            
#            star_spectrum = ispec.read_spectrum('tmp.dat')
#            star_spectrum = resample_spectrum(star_spectrum, np.min(star_spectrum['waveobs']),np.max(star_spectrum['waveobs']))
#            
#            rv = determine_radial_velocity_with_template()
#            
#            plt.plot(wav)
#        plt.show()
    
    star_spectrum = ispec.read_spectrum('../test_out_O.txt')
    plt.plot(star_spectrum['waveobs'],star_spectrum['flux'])
    plt.show()
    star_spectrum['waveobs'] /= 10
    ii=np.where(star_spectrum['waveobs'] > 390)[0]
    star_spectrum = star_spectrum[ii]
    star_spectrum = resample_spectrum(star_spectrum, np.min(star_spectrum['waveobs']),np.max(star_spectrum['waveobs']))
    
    rv = determine_radial_velocity_with_template()

            
