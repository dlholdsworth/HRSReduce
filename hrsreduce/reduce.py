# -*- coding: utf-8 -*-
"""
Script for reducing HRS HS  data

Authors
-------
Daniel Holdsworth (daniel.l.holdsworth@gmail.com)

Version
-------
1.0 - Initial HRSReduce

License
--------
...

"""

import logging
import numpy as np
import os
from os.path import dirname, join
from itertools import product
from . import __version__, util
import importlib
import arrow

from hrsreduce.hrs_info import Instrument
from hrsreduce.utils.sort_files import SortFiles
from hrsreduce.utils.cr_masking import CosmicRayMasking
from hrsreduce.utils.find_nearest_files import FindNearestFiles
from hrsreduce.L0_Corrections.level0corrections import L0Corrections
from hrsreduce.master_bias.master_bias import MasterBias, SubtractBias
from hrsreduce.master_flat.master_flat import MasterFlat
from hrsreduce.var_ext.var_ext import VarExts
from hrsreduce.order_trace.order_trace import OrderTrace
#from hrsreduce.extraction.order_rectification import OrderRectification
from hrsreduce.extraction.extraction import SpectralExtraction
from hrsreduce.wave_cal.wave_cal import WavelengthCalibration
from hrsreduce.norm.norm import ContNorm

from .configuration import load_config


logger = logging.getLogger(__name__)

def load_instrument(instrument) -> Instrument:
    """Load an python instrument module

    Parameters
    ----------
    instrument : str
        name of the instrument

    Returns
    -------
    instrument : Instrument
        Instance of the {instrument} class
    """

    if instrument is None:
        instrument = "common"

    fname = ".instruments.%s" % instrument.lower()
    lib = importlib.import_module(fname, package="hrsreduce")
    instrument = getattr(lib, instrument.upper())
    instrument = instrument()

    return instrument

def main(
    night=None,
    modes=None,
    arm=None,
    base_dir=None,
    input_dir=None,
    output_dir=None,
    configuration=None,
    instrument=None,
    allow_calibration_only=False,
    skip_existing=False,
    plot=False,
    ):
    
    r"""
    Main entry point for HRSReduce scripts,
    default values can be changed as required if reduce is used as a script
    Finds input directories, and loops over observation night and instrument modes

    Parameters
    ----------
    night : str
        the observation night to reduce, as named in the folder structure.
    modes : str, list[str], dict[{instrument}:list], None, optional
        the instrument modes to use, if None will use all known modes for the current instrument. See instruments for possible options
    arm : str, list[str]
        the spectrograph arm to reduce, if None will use all known modes for the current instrument
    base_dir : str, optional
        base data directory that HRSReduce should work in, is prefixxed on input_dir and output_dir (default: use settings_pyreduce.json)
    input_dir : str, optional
        input directory containing raw files. If relative will use base_dir as root (default: use settings_pyreduce.json)
    output_dir : str, optional
        output directory for intermediary and final results. If relative will use base_dir as root (default: use settings_pyreduce.json)
    configuration : dict[str:obj], str, list[str], dict[{instrument}:dict,str], optional
        configuration file for the current run, contains parameters for different parts of reduce. Can be a path to a json file, or a dict with configurations for the different instruments. When a list, the order must be the same as instruments (default: settings_{instrument.upper()}.json)
    """

    if night is None or np.isscalar(night):
        night = [night]
        year = night[0].split('-')[0]
        mmdd = night[0].split('-')[1]+night[0].split('-')[2]
        yyyymmdd=year+mmdd
        
    if arm == "H":
        arm_colour = 'Blu'
    elif arm == "R":
        arm_colour = 'Red'
    else:
        print("Arm not supported.")
        exit()

    base_dir = ("/Users/daniel/Desktop/SALT_HRS_DATA/")
    input_dir = arm_colour+"/"+year+"/"+mmdd+"/raw/"
    output_dir = arm_colour+"/"+year+"/"+mmdd+"/reduced/"


    isNone = {
    "modes": modes is None,
    "arm" : arm is None,
    "base_dir": base_dir is None,
    "input_dir": input_dir is None,
    "output_dir": output_dir is None,
    }
    
    output = []

    config = load_config(configuration, instrument, 0)

    if isinstance(instrument, str):
        instrument = load_instrument(instrument)
    info = instrument.info


    # load default settings from settings_hrsreduce.json
    if base_dir is None:
        base_dir = config["hrsreduce"]["base_dir"]
    if input_dir is None:
        input_dir = config["hrsreduce"]["input_dir"]
    if output_dir is None:
        output_dir = config["hrsreduce"]["output_dir"]

    input_dir = join(base_dir, input_dir)
    output_dir = join(base_dir, output_dir)

    try:
        os.mkdir(output_dir)
    except Exception:
        pass
    
    if modes is None:
        modes = info["modes"]
    if np.isscalar(modes):
        modes = [modes]
        
    if arm is None:
        arm = info["arm"]
    if np.isscalar(arm):
        arm = [arm]

    for n, m in product(night, modes):
        log_file = join(base_dir.format(mode=modes),"logs/%s_%s_%s.log" %(arm_colour,m, n))
        
        util.start_logging(log_file)
        # find input files and sort them by type
        files = {}
        nights = {}
        types = ["sci", "arc", "lfc", "flat","bias"]
        
        files["bias"],files["flat"],files["arc"],files["lfc"],files["sci"] = SortFiles(input_dir,logger,arm,mode=m)
        
        #List the nights where the data come from. Is updated below if files are not found in the suggested night.
        nights["bias"] = yyyymmdd
        nights["flat"] = yyyymmdd
        nights["arc"] = yyyymmdd
        nights["lfc"] = yyyymmdd
        nights["sci"] = yyyymmdd
        
        if not files["bias"]:
            logger.warning(
                f"No BIAS files found for instrument: night: %s in folder: %s \n    Looking elsewhere...\n",
                n,
                input_dir,
            )
            #Now search to find a night with the files
            files["bias"], nights["bias"]= FindNearestFiles("Bias",yyyymmdd,m,base_dir,arm_colour,logger)
            print(nights["bias"])
            #Create the output directory if it does not exist
            output_dir_bias = arm_colour+"/"+nights["bias"][0:4]+"/"+nights["bias"][4:8]+"/reduced/"
            try:
                os.mkdir(str(base_dir+output_dir_bias))
            except Exception:
                pass

            
        if not files["flat"]:
            logger.warning(
                f"No FLAT files found for instrument: night: %s in folder: %s \n    Looking elsewhere...\n",
                n,
                input_dir,
            )
            
            #Now search to find a night with the files
            files["flat"], nights["flat"] = FindNearestFiles("Flat",yyyymmdd,m,base_dir,arm_colour,logger)
            #Create the output directory if it does not exist
            output_dir_flat = arm_colour+"/"+nights["flat"][0:4]+"/"+nights["flat"][4:8]+"/reduced/"
            try:
                os.mkdir(str(base_dir+output_dir_flat))
            except Exception:
                pass
       
        if not files["arc"]:
            logger.warning(
                f"No ARC files found for instrument: night: %s in folder: %s \n    Looking elsewhere...\n",
                n,
                input_dir,
            )
            #Now search to find a night with the files
            files["arc"], nights["arc"] = FindNearestFiles("Arc",yyyymmdd,m,base_dir,arm_colour,logger)
            #Create the output directory if it does not exist
            output_dir_arc = arm_colour+"/"+nights["arc"][0:4]+"/"+nights["arc"][4:8]+"/reduced/"
            try:
                os.mkdir(str(base_dir+output_dir_arc))
            except Exception:
                pass
            
        if not files["sci"]:
            logger.warning(
                f"No SCI files found for instrument: night: %s in folder: %s \n",
                n,
                input_dir,
            )
            pass
            
        if not files["lfc"]:
            logger.warning(
                f"No LFC files found for instrument: night: %s in folder: %s SKIPPING STEP FOR NOW\n",
                n,
                input_dir,
            )
            pass #TODO: Fix this when the LFC is on board
            
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
            --  Calculate RVs
            --  Blaze correction
            --  Merge orders
            --  Continuum normalisation
        '''
        
        #Apply the level 0 corrections (gain and overscan)
        files = L0Corrections(files,nights,yyyymmdd,input_dir,output_dir,base_dir,arm).run()

        #Calcualte the master bias
        master_bias = MasterBias(files["bias"],input_dir,output_dir,arm_colour,yyyymmdd,plot).create_masterbias()
        
        #Subtract the bias from all other frames
        #Loop over the files dict for the different types to make sure the correct bias file is subtracted (e.g., if the Flats are from a different night)
        files_out = {}
        for type in types:
            files_type = {}
            files_tmp = {}
            files_tmp2 = {}
            tmp_nights = {}
            files_type[str(type)] = files[type]
            tmp_nights['bias'] = nights[type]
            if nights[type] == nights["bias"]:
                files_out[type] = SubtractBias(master_bias,files_type,base_dir,arm_colour,yyyymmdd,type).subtract()
            else:
                #Create a new master bias for the night of the observations to be reduced
                input_dir_tmp =str(base_dir+arm_colour+"/"+nights[type][0:4]+"/"+nights[type][4:8]+"/raw/")
                output_dir_tmp = str(base_dir+arm_colour+"/"+nights[type][0:4]+"/"+nights[type][4:8]+"/reduced/")
                
                files_tmp["bias"],_,_,_,_= SortFiles(input_dir_tmp,logger,arm,mode=m)
                files_tmp2 = L0Corrections(files_tmp,tmp_nights,nights[type],input_dir_tmp,output_dir_tmp,base_dir,arm).run()
                
                master_bias_tmp = MasterBias(files_tmp2["bias"],input_dir_tmp,output_dir_tmp,arm_colour,nights[type],plot).create_masterbias()
                files_out[type] = SubtractBias(master_bias_tmp,files_type,base_dir,arm_colour,nights[type],type).subtract()
        del files
        files = files_out
        
        #Clean the files of CRs
        _ = CosmicRayMasking(files,arm)
        
        #Remove the intermediate files
        for ff in files["bias"]:
            os.remove(ff)
        
        #Calcualte the master flat
        master_flat = MasterFlat(files["flat"],nights,input_dir,output_dir,base_dir,arm_colour,yyyymmdd,m,plot).create_masterflat()
        
        #Remove the intermediate files
        for ff in files["flat"]:
            os.remove(ff)
            
        #Create the Order file
        order_file = OrderTrace(master_flat,nights,base_dir,arm_colour,m,plot).order_trace()
        
        #Calcualte a blaze using the Spectral Extracton module
        VarExts(master_flat,master_bias,master_flat).run()
        SpectralExtraction(master_flat, master_flat,order_file,arm_colour,m,base_dir).extraction()

        #Calculate the Varience image and extract the frames
        for sci_file in files['sci']:
            VarExts(sci_file,master_bias,master_flat).run()
            #OrderRectification(sci_file,master_flat,order_file,arm_colour,m,base_dir).rectify()
            SpectralExtraction(sci_file, master_flat,order_file,arm_colour,m,base_dir).extraction()
            
        #Calculate the Varience image, extract the frames and calculate the wave solution
        for arc_file in files['arc']:
            cal_type = 'ThAr'
            VarExts(arc_file,master_bias,master_flat).run()
            SpectralExtraction(arc_file, master_flat,order_file,arm_colour,m,base_dir).extraction()
            WavelengthCalibration(arc_file, arm, m, base_dir,cal_type,plot).execute()

        for sci_file in files['sci']:
            ContNorm(sci_file,files['arc'][0],master_flat).execute()
        

    return output

