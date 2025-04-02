# -*- coding: utf-8 -*-
"""
Script for reducing HRS HS  data

Authors
-------
Daniel Holdsworth (daniel.l.holdsworth@gmail.com)

Version
-------
1.0 - Initial PyReduce

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
from hrsreduce.L0_Corrections.level0corrections import L0Corrections
from hrsreduce.master_bias.master_bias import MasterBias
from hrsreduce.master_flat.master_flat import MasterFlat

from .configuration import load_config

#from . import instruments
#from .instruments.instrument_info import load_instrument

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
    steps="all",
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
    steps : tuple(str), "all", optional
        which steps of the reduction process to perform
        the possible steps are: "bias", "flat", "orders", "norm_flat", "wavecal", "science"
        alternatively set steps to "all", which is equivalent to setting all steps
        Note that the later steps require the previous intermediary products to exist and raise an exception otherwise
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
        
        files["bias"],files["flat"],files["arc"],files["lfc"],files["sci"] = instrument.sort_files_2(input_dir,n,mode=m,**config["instrument"])
        
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
            files["bias"], nights["bias"]= instrument.find_nearest_files("Bias",yyyymmdd,m,base_dir,arm_colour,**config["instrument"])

            
        if not files["flat"]:
            logger.warning(
                f"No FLAT files found for instrument: night: %s in folder: %s \n    Looking elsewhere...\n",
                n,
                input_dir,
            )
            
            #Now search to find a night with the files
            files["flat"], nights["flat"] = instrument.find_nearest_files("Flat",yyyymmdd,m,base_dir,arm_colour,**config["instrument"])
 
        if not files["arc"]:
            logger.warning(
                f"No ARC files found for instrument: night: %s in folder: %s \n    Looking elsewhere...\n",
                n,
                input_dir,
            )
            #Now search to find a night with the files
            files["arc"], nights["arc"] = instrument.find_nearest_files("Arc",yyyymmdd,m,base_dir,arm_colour,**config["instrument"])
            
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
            --  Calculate the master flat, or read it for the night/mode if alreadt created.
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
        
        #Apply the level 0 corrections
        files = L0Corrections(files,nights,yyyymmdd,input_dir,output_dir,base_dir,arm).run()
        
        #Calcualte the master bias
        master_bias = MasterBias(files["bias"],input_dir,output_dir,arm_colour,yyyymmdd,plot).create_masterbias()
        
        #Calcualte the master flat
        master_flat = MasterFlat(files["flat"],nights,input_dir,output_dir,arm_colour,plot).create_masterflat()
        

#        files = instrument.sort_files(
#            input_dir,
#            n,
#            mode=m,
#            **config["instrument"],
#            allow_calibration_only=allow_calibration_only,
#        )

#        if len(flat_files) == 0:
#            logger.warning(
#                f"No files found for instrument: %s, night: %s, mode: %s in folder: %s",
#                instrument,
#                n,
#                m,
#                input_dir,
#            )
#            continue
            
#        for k, f in files:
#            logger.info("Settings:")
#            for key, value in k.items():
#                logger.info("%s: %s", key, value)
#            logger.debug("Files:\n%s", f)
#
#            reducer = Reducer(
#                f,
#                output_dir,
#                instrument,
#                m,
#                k.get("night"),
#                config,
#                order_range=order_range,
#                skip_existing=skip_existing,
#            )
#            # try:
#            data = reducer.run_steps(steps=steps)
#            output.append(data)
#            # except Exception as e:
#            #     logger.error("Reduction failed with error message: %s", str(e))
#            #     logger.info("------------")
    return output

#class Step:
#    """Parent class for all steps"""
#
#    def __init__(
#        self, instrument, mode night, output_dir, order_range, **config
#    ):
#        self._dependsOn = []
#        self._loadDependsOn = []
#        #:str: Name of the instrument
#        self.instrument = instrument
#        #:str: Name of the instrument mode
#        self.mode = mode
#        #:str: Name of the observation target
#        self.target = target
#        #:str: Date of the observation (as a string)
#        self.night = night
#        #:tuple(int, int): First and Last(+1) order to process
#        self.order_range = order_range
#        #:bool: Whether to plot the results or the progress of this step
#        self.plot = config.get("plot", False)
#        #:str: Title used in the plots, if any
#        self.plot_title = config.get("plot_title", None)
#        self._output_dir = output_dir
#
#    def run(self, files, *args):  # pragma: no cover
#        """Execute the current step
#
#        This should fail if files are missing or anything else goes wrong.
#        If the user does not want to run this step, they should not specify it in steps.
#
#        Parameters
#        ----------
#        files : list(str)
#            data files required for this step
#
#        Raises
#        ------
#        NotImplementedError
#            needs to be implemented for each step
#        """
#        raise NotImplementedError
#
#    def save(self, *args):  # pragma: no cover
#        """Save the results of this step
#
#        Parameters
#        ----------
#        *args : obj
#            things to save
#
#        Raises
#        ------
#        NotImplementedError
#            Needs to be implemented for each step
#        """
#        raise NotImplementedError
#
#    def load(self):  # pragma: no cover
#        """Load results from a previous execution
#
#        If this raises a FileNotFoundError, run() will be used instead
#        For calibration steps it is preferred however to print a warning
#        and return None. Other modules can then use a default value instead.
#
#        Raises
#        ------
#        NotImplementedError
#            Needs to be implemented for each step
#        """
#        raise NotImplementedError
#
#    @property
#    def dependsOn(self):
#        """list(str): Steps that are required before running this step"""
#        return list(set(self._dependsOn))
#
#    @property
#    def loadDependsOn(self):
#        """list(str): Steps that are required before loading data from this step"""
#        return list(set(self._loadDependsOn))
#
#    @property
#    def output_dir(self):
#        """str: output directory, may contain tags {instrument}, {night}, {target}, {mode}"""
#        return self._output_dir.format(
#            instrument=self.instrument.name.upper(),
#            target=self.target,
#            night=self.night,
#            mode=self.mode,
#        )
#
#    @property
#    def prefix(self):
#        """str: temporary file prefix"""
#        i = self.instrument.name.lower()
#        if self.mode is not None and self.mode != "":
#            m = self.mode.lower()
#            return f"{i}_{m}"
#        else:
#            return i
#
