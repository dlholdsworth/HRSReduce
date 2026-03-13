"""
HRSReduce: SALT High Resolution Spectrograph Reduction Pipeline
==========================================================

This script provides the main entry point for reducing SALT HRS data.
It coordinates the full reduction workflow including calibration,
order tracing, spectral extraction, wavelength calibration, and
order merging.

The pipeline performs the following major steps:

    1. Sort raw files by type (science, arc, flat, bias, etc.)
    2. Apply Level-0 detector corrections (overscan, gain, orientation)
    3. Construct master calibration frames (bias, flat)
    4. Trace echelle orders
    5. Correct slit tilt and curvature
    6. Extract spectra
    7. Compute wavelength calibration
    8. Apply blaze correction
    9. Merge spectral orders

Authors
-------
Daniel Holdsworth
daniel.l.holdsworth@gmail.com

Version
-------
1.0 – Initial HRSReduce implementation

License
-------
TBD – Insert project license (e.g. MIT, BSD, GPL).
"""

# -----------------------------------------------------------------------------
# Standard library imports
# -----------------------------------------------------------------------------

import logging
import os
import glob
from os.path import dirname, join
import numpy as np
import arrow

# -----------------------------------------------------------------------------
# Local package imports
# -----------------------------------------------------------------------------

from . import __version__, util

# Utility modules
from hrsreduce.utils.sort_files import SortFiles
from hrsreduce.utils.cr_masking import CosmicRayMasking
from hrsreduce.utils.find_nearest_files import FindNearestFiles

# Level-0 detector corrections
from hrsreduce.L0_Corrections.level0corrections import L0Corrections

# Calibration frame construction
from hrsreduce.master_bias.master_bias import MasterBias, SubtractBias
from hrsreduce.master_flat.master_flat import MasterFlat
from hrsreduce.master_flat.normalisation import FlatNormalisation

# Variance frame generation
from hrsreduce.var_ext.var_ext import VarExts

# Order tracing and rectification
from hrsreduce.order_trace.order_trace import OrderTrace
from hrsreduce.extraction.order_rectification import OrderRectification
from hrsreduce.extraction.slit_correction import SlitCorrection

# Spectral extraction
from hrsreduce.extraction.extraction import SpectralExtraction

# Wavelength calibration
from hrsreduce.wave_cal.wave_cal import WavelengthCalibration

# Final spectral processing
from hrsreduce.order_merge.order_merge import OrderMerge

# -----------------------------------------------------------------------------
# Logger configuration
# -----------------------------------------------------------------------------

# Module-level logger used throughout the pipeline.
logger = logging.getLogger(__name__)


def main(
    night=None,
    modes=None,
    arm=None,
    loc=None,
    base_dir=None,
    input_dir=None,
    output_dir=None,
    plot=False,
    clean=False,
    cal_rvst=False,
    ):
    
    """
    Main entry point for HRSReduce scripts,
    default values can be changed as required if reduce is used as a script
    Finds input directories, and loops over observation night and instrument modes

    Parameters
    ----------
    night : str
        the observation night to reduce, as named in the folder structure.
    modes : str, list[str]
        the instrument modes to use, if None will use all known modes for the current instrument. See instruments for possible options
    arm : str
        the spectrograph arm to reduce.
    loc : str
        the location of the data to reduce.
    base_dir : str, optional
        base data directory that HRSReduce should work in, is prefixed on input_dir and output_dir.
    input_dir : str, optional
        input directory containing raw files. If relative will use base_dir as root.
    output_dir : str, optional
        output directory for intermediary and final results. If relative will use base_dir as root.
    plot : boolean 
        option for plotting or not. Default = False
    clean : boolean
        option for deleting intermediate files. Default = False
    cal_rvst : boolean
        reduce just the RV standard star for the other given inputs.
    """

    # Log the package version at the start of execution.
    logger.info(f"\n\n Version {__version__}\n\n")

    # If a single night string is supplied, derive year / mmdd / full date tokens.
    if isinstance(night, str):
        year = night[0:4]
        mmdd = night[4:8]
        yyyymmdd = year + mmdd
    else:
        logger.error("Night not a string.")

        
    # Map spectrograph arm code to colour name used in directory structure.
    if arm == 'H':
        arm_colour = 'Blu'
    elif arm == 'R':
        arm_colour = 'Red'
    else:
        logger.error("Arm not supported.")
        raise ValueError("Unsupported arm")
        
    # FITS extension name expected after rectification.
    header_ext = 'RECT'

    # Build standard SALT-style input/output paths from location choice.
    if loc == 'work':
        base_dir = "/home/sa/" + str(yyyymmdd) + "/hrs"
        input_dir = "/home/sa/" + str(yyyymmdd) + "/hrs/raw/"
        output_dir = "/home/sa/" + str(yyyymmdd) + "/hrs/product"
    elif loc == 'archive':
        base_dir = "/salt/data/" + str(year) + "/" + str(mmdd) + "/hrs"
        input_dir = "/salt/data/" + str(year) + "/" + str(mmdd) + "/hrs/raw/"
        output_dir = "/salt/data/" + str(year) + "/" + str(mmdd) + "/hrs/reduced/"
    else:
        logging.error("Location unknown. Exiting.")
        return 2
        
    # Local override for desktop testing / development.
    # This currently replaces the loc-based path definitions above.
    base_dir = "/Users/daniel/Desktop/SALT_HRS_DATA/"
    input_dir = arm_colour + "/" + year + "/" + mmdd + "/raw/"
    output_dir = arm_colour + "/" + year + "/" + mmdd + "/reduced/"
    
    # Placeholder return container.
    output = []

    # Normalise scalar inputs into lists for looping.
    if np.isscalar(modes):
        modes = [modes]

    if np.isscalar(arm):
        arm = [arm]

    # Convert input path to absolute path under base_dir.
    input_dir = join(base_dir, input_dir)

    # Test whether the input directory exists before continuing.
    exists = os.path.isdir(input_dir)
    if not exists:
        logger.error(f"Input directory {input_dir} does not exist. Exiting")
        return 2

    # Convert output path to absolute path under base_dir.
    output_dir = join(base_dir, output_dir)

    # Create main output directory if it does not already exist.
    try:
        os.mkdir(output_dir)
    except Exception:
        pass

    # Main reduction loop over observing modes.
    for m in modes:
    
        # Start a log file for this arm/mode/night combination.
        log_file = join(base_dir.format(mode=modes), "logs/%s_%s_%s.log" % (arm_colour, m, yyyymmdd))
        util.start_logging(log_file)
        
        # Find input files and sort them by type.
        files = {}
        nights = {}
        types = ['sci', 'arc', 'lfc', 'flat', 'bias']
        files['bias'], files['flat'], files['arc'], files['lfc'], files['sci'] = SortFiles(input_dir, logger, arm, mode=m, CAL_RVST=cal_rvst)
        
        # Record the nominal night associated with each file type.
        # These may be updated later if fallback calibration files are used.
        nights['bias'] = yyyymmdd
        nights['flat'] = yyyymmdd
        nights['arc'] = yyyymmdd
        nights['lfc'] = yyyymmdd
        nights['sci'] = yyyymmdd
        
        # If no bias frames exist for this night, search nearby nights.
        if not files['bias']:
            logger.warning(
                f"No BIAS files found for instrument: night: {yyyymmdd} in folder: {input_dir} \n    Looking elsewhere...\n"
            )
            # Now search to find a night with the files.
            files['bias'], nights['bias'] = FindNearestFiles('Bias', yyyymmdd, m, base_dir, arm_colour, logger)
            # Create the corresponding output directory if needed.
            output_dir_bias = arm_colour + "/" + nights['bias'][0:4] + "/" + nights['bias'][4:8] + "/reduced/"
            try:
                os.mkdir(str(base_dir + output_dir_bias))
            except Exception:
                pass

            
        # If no flats exist for this night, search nearby nights.
        if not files['flat']:
            logger.warning(
                f"No FLAT files found for instrument: night: {yyyymmdd} in folder: {input_dir} \n    Looking elsewhere...\n"
            )
            
            # Now search to find a night with the files.
            files['flat'], nights['flat'] = FindNearestFiles('Flat', yyyymmdd, m, base_dir, arm_colour, logger)

            # Create the corresponding output directory if needed.
            output_dir_flat = arm_colour + "/" + nights['flat'][0:4] + "/" + nights['flat'][4:8] + "/reduced/"
            try:
                os.mkdir(str(base_dir + output_dir_flat))
            except Exception:
                pass
       
        # If no arcs exist for this night, search nearby nights.
        if not files['arc']:
            logger.warning(
                f"No ARC files found for instrument: night: {yyyymmdd} in folder: {input_dir} \n    Looking elsewhere...\n"
            )
            # Now search to find a night with the files.
            files['arc'], nights['arc'] = FindNearestFiles('Arc', yyyymmdd, m, base_dir, arm_colour, logger)

            # Create the corresponding output directory if needed.
            output_dir_arc = arm_colour + "/" + nights['arc'][0:4] + "/" + nights['arc'][4:8] + "/reduced/"
            try:
                os.mkdir(str(base_dir + output_dir_arc))
            except Exception:
                pass
            
        # Science files are mandatory for reduction output, but pipeline does not abort here.
        if not files['sci']:
            logger.warning(
                f"No SCI files found for instrument: night: {yyyymmdd} in folder: {input_dir} \n"
            )
            pass
            
        # LFC handling is currently disabled / incomplete.
        if not files['lfc']:
            logger.warning(
                f"No LFC files found for instrument: night: {yyyymmdd} in folder: {input_dir} SKIPPING STEP FOR NOW\n"
            )
            pass  # TODO: Fix this when the LFC is on board
            
        '''
        Run through the reduction steps in the following order
            --  Apply level 0 corrections to remove overscan region, flip the red frames and corrects for gain
            --  Calculate the master bias, or read it for the night if already created. This also calculates the read noise
            --  Calculate the master flat, or read it for the night/mode if already created.
            --  Define the orders, or read them from file
            --  Calculate the slit curvature, or read from file
            --  Calculate background scatter
            --  Normalise the Flat
            --  Calculate the slit illumination function, or read from file
            --  Extract the science frame
            --  Calculate the wavelength solution
            --  Blaze correction
            --  Merge orders
        '''
        
        # Apply detector-level corrections (e.g. overscan, gain, orientation).
        files = L0Corrections(files, nights, yyyymmdd, input_dir, output_dir, base_dir, arm).run()

        # Calculate the master bias frame.
        master_bias = MasterBias(files['bias'], input_dir, output_dir, arm_colour, yyyymmdd, plot).create_masterbias()
        
        # Remove intermediate bias frames after master bias creation.
        for redundant in files['bias']:
            try:
                os.remove(redundant)
            except:
                pass
        
        # Subtract the bias from all non-bias frame types.
        # This loop ensures the correct night's bias is used for each frame type
        # when calibrations come from different nights.
        files_out = {}
        for type in ['sci', 'arc', 'lfc', 'flat']:
            files_type = {}
            files_tmp = {}
            files_tmp2 = {}
            tmp_nights = {}
            files_type[str(type)] = files[type]
            tmp_nights['bias'] = nights[type]

            # Use the already-constructed master bias if the nights match.
            if nights[type] == nights['bias']:
                files_out[type] = SubtractBias(master_bias, files_type, base_dir, arm_colour, yyyymmdd, type).subtract()
            else:
                # Otherwise create a new master bias for the relevant night.
                input_dir_tmp = str(base_dir + arm_colour + "/" + nights[type][0:4] + "/" + nights[type][4:8] + "/raw/")
                output_dir_tmp = str(base_dir + arm_colour + "/" + nights[type][0:4] + "/" + nights[type][4:8] + "/reduced/")
                
                files_tmp['bias'], _, _, _, _ = SortFiles(input_dir_tmp, logger, arm, mode=m)
                files_tmp2 = L0Corrections(files_tmp, tmp_nights, nights[type], input_dir_tmp, output_dir_tmp, base_dir, arm).run()
                
                master_bias_tmp = MasterBias(files_tmp2['bias'], input_dir_tmp, output_dir_tmp, arm_colour, nights[type], plot).create_masterbias()
                files_out[type] = SubtractBias(master_bias_tmp, files_type, base_dir, arm_colour, nights[type], type).subtract()

                # Remove intermediate temporary bias frames.
                for redundant in files_tmp2['bias']:
                    try:
                        os.remove(redundant)
                    except:
                        pass
                    
        # Replace original files dictionary with bias-subtracted outputs.
        del files
        files = files_out
        
        # Calculate the master flat frame.
        master_flat = MasterFlat(files['flat'], nights, input_dir, output_dir, base_dir, arm_colour, yyyymmdd, m, plot).create_masterflat()
      
        # Remove intermediate flat files after master flat creation.
        for ff in files['flat']:
            try:
                os.remove(ff)
            except:
                pass
                
        # Run cosmic ray masking on current files.
        _ = CosmicRayMasking(files, arm)
            
        # Find the closest previous Super Flat for order tracing.
        # Restricting to previous frames helps avoid configuration mismatches
        # after instrument interventions such as tank openings.
        super_flat = []
        prev_night = yyyymmdd
        while not super_flat:
            prev_night = arrow.get(prev_night).shift(days=-1).format('YYYYMMDD')
            prev_year = prev_night[0:4]
            prev_mmdd = prev_night[4:8]
            prev_data_location = os.path.join(base_dir, arm_colour + '/????/Super_Flats/')
            super_flats = glob.glob(prev_data_location + m + '_Super_Flat_' + arm[0] + '*.fits')
            s_difference = []
            sup_files = []
            for sfiles in super_flats:
                s_date = (os.path.basename(sfiles)[-13:-5])
                tmp = ((arrow.get(int(yyyymmdd[0:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8])) - arrow.get(int(s_date[0:4]), int(s_date[4:6]), int(s_date[6:8]))).days)
                if tmp > 0:
                    s_difference.append(tmp)
                    sup_files.append(sfiles)
            index_of_closest = (np.abs(s_difference)).argmin()
            if s_difference[index_of_closest] < 365:
                super_flat = sup_files[index_of_closest]
                logger.info(f"Using Super Flat file: {super_flat}")
            else:
                super_flat = super_flats[index_of_closest]
                logger.warning(f"Please update the Super Flat files, currently using potentially outdated file: {super_flat}")
        
        # Trace the echelle orders using the selected Super Flat.
        order_file = OrderTrace(super_flat, nights, base_dir, arm_colour, m, plot).order_trace()
        
        
        # Find the closest previous Super Arc for slit tilt correction.
        super_arc = []
        prev_night = yyyymmdd
        while not super_arc:
            prev_night = arrow.get(prev_night).shift(days=-1).format('YYYYMMDD')
            prev_year = prev_night[0:4]
            prev_mmdd = prev_night[4:8]
            prev_data_location = os.path.join(base_dir, arm_colour + '/????/Super_Arcs/')
            super_arcs = glob.glob(prev_data_location + m + '_Super_Arc_' + arm[0] + '*.fits')
            s_difference = []
            sup_files = []
            for sfiles in super_arcs:
                s_date = (os.path.basename(sfiles)[-13:-5])
                tmp = ((arrow.get(int(yyyymmdd[0:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8])) - arrow.get(int(s_date[0:4]), int(s_date[4:6]), int(s_date[6:8]))).days)
                if tmp > 0:
                    s_difference.append(tmp)
                    sup_files.append(sfiles)
            s_difference = np.array(s_difference)
            ii = np.where(s_difference > 0)[0]
            index_of_closest = (np.abs(s_difference[ii])).argmin()
            if s_difference[index_of_closest] < 365:
                super_arc = sup_files[index_of_closest]
                logger.info(f"Using Super Arc file: {super_arc}")
            else:
                super_arc = super_arcs[index_of_closest]
                logger.warning(f"Please update the Super Arc files, currently using potentially outdated file: {super_arc}")
                
        # Fully process the Super Arc and Super Flat in the correct order.
        logger.info(f"Processing Super arc file: {super_arc}")
        order_file_rect = OrderRectification(super_arc, super_flat, order_file, arm_colour, m, base_dir, super_arc=super_arc).perform()
        SlitCorrection(super_arc, header_ext, order_file_rect, arm[0], m, base_dir, yyyymmdd, plot=plot, super_arc=super_arc).correct()
        VarExts(super_arc, master_bias, master_flat).run()

        logger.info(f"Processing Master Flat file: {master_flat}")

        # Rectify and tilt-correct the master flat once, since this product
        # is reused later in the pipeline.
        _ = OrderRectification(master_flat, master_flat, order_file, arm_colour, m, base_dir, super_arc=super_arc).perform()
        SlitCorrection(master_flat, 'RECT', order_file_rect, arm[0], m, base_dir, yyyymmdd, plot=False, super_arc=super_arc).correct()
        
        # Extract the Super Arc spectrum.
        SpectralExtraction(super_arc, master_flat, super_arc, order_file_rect, arm_colour, m, base_dir).extraction()
        
        # Pre-process arc frames if a Master_Wave does not already exist.
        for arc_file in files['arc']:
            logger.info(f"Processing arc file: {arc_file}")
            cal_type = 'ThAr'

            # Test whether Master_Wave already exists for this arc.
            path = os.path.dirname(arc_file)
            obs_date = (os.path.basename(arc_file)[-17:-5])
            dst = (path + "/" + str(m) + "_Master_Wave_" + str(arm[0]) + str(obs_date) + ".fits")
            master_wave = glob.glob(dst)

            if len(master_wave) == 0:
                VarExts(arc_file, master_bias, master_flat).run()
                _ = OrderRectification(arc_file, master_flat, order_file, arm_colour, m, base_dir, super_arc=super_arc).perform()
                SlitCorrection(arc_file, header_ext, order_file_rect, arm[0], m, base_dir, yyyymmdd, plot=plot, super_arc=super_arc).correct()
            else:
                logger.info("Arc file already processed")

        # Create the normalised flat for blaze correction / extraction support.
        FlatNormalisation(master_flat, order_file_rect).normalise()
        
        # Calculate the blaze using the extraction module on the master flat.
        VarExts(master_flat, master_bias, master_flat).run()
        SpectralExtraction(master_flat, master_flat, files['arc'][0], order_file_rect, arm_colour, m, base_dir).extraction()
        
        # Process any LFC frames: variance, rectification, slit correction, extraction.
        for lfc_file in files['lfc']:
            logger.info(f"Processing LFC file: {lfc_file}")
            VarExts(lfc_file, master_bias, master_flat).run()
            _ = OrderRectification(lfc_file, master_flat, order_file, arm_colour, m, base_dir, super_arc=super_arc).perform()
            SlitCorrection(lfc_file, header_ext, order_file_rect, arm[0], m, base_dir, yyyymmdd, plot=plot, super_arc=super_arc).correct()
            SpectralExtraction(lfc_file, master_flat, files['arc'][0], order_file_rect, arm_colour, m, base_dir).extraction()
 
        # Extract arc frames and compute wavelength solution if needed.
        for arc_file in files['arc']:
            # Test whether Master_Wave already exists for this arc.
            path = os.path.dirname(arc_file)
            obs_date = (os.path.basename(arc_file)[-17:-5])
            dst = (path + "/" + str(m) + "_Master_Wave_" + str(arm[0]) + str(obs_date) + ".fits")
            master_wave = glob.glob(dst)

            if len(master_wave) == 0:
                SpectralExtraction(arc_file, master_flat, arc_file, order_file_rect, arm_colour, m, base_dir).extraction()
                MasterWave = WavelengthCalibration(arc_file, super_arc, arm, m, base_dir, cal_type, plot).execute()
            else:
                MasterWave = master_wave[0]
                logger.info("Arc file already processed")
            
        # Process science frames fully: variance, rectification, slit correction,
        # extraction, and order merging.
        for sci_file in files['sci']:
            logger.info(f"Processing Science file: {sci_file}")
            VarExts(sci_file, master_bias, master_flat).run()
            _ = OrderRectification(sci_file, master_flat, order_file, arm_colour, m, base_dir, super_arc=super_arc).perform()
            SlitCorrection(sci_file, header_ext, order_file_rect, arm[0], m, base_dir, yyyymmdd, plot=plot, super_arc=super_arc).correct()
            SpectralExtraction(sci_file, master_flat, MasterWave, order_file_rect, arm_colour, m, base_dir).extraction()
            OrderMerge(sci_file, master_flat, arm, m, plot=False).execute()
        
        # Optional cleanup of remaining intermediate files.
        if clean:
            for type in ['sci', 'arc', 'lfc', 'flat']:
                for file in files[type]:
                    try:
                        os.remove(file)
                    except:
                        pass

    return output
