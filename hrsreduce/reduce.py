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
import glob

from hrsreduce.utils.sort_files import SortFiles
from hrsreduce.utils.cr_masking import CosmicRayMasking
from hrsreduce.utils.find_nearest_files import FindNearestFiles
from hrsreduce.L0_Corrections.level0corrections import L0Corrections
from hrsreduce.master_bias.master_bias import MasterBias, SubtractBias
from hrsreduce.master_flat.master_flat import MasterFlat
from hrsreduce.master_flat.normalisation import FlatNormalisation
from hrsreduce.var_ext.var_ext import VarExts
from hrsreduce.order_trace.order_trace import OrderTrace
from hrsreduce.extraction.order_rectification import OrderRectification
from hrsreduce.extraction.slit_correction import SlitCorrection
from hrsreduce.extraction.extraction import SpectralExtraction
from hrsreduce.wave_cal.wave_cal import WavelengthCalibration
from hrsreduce.order_merge.order_merge import OrderMerge


logger = logging.getLogger(__name__)


def main(
    night=None,
    modes=None,
    arm=None,
    loc=None,
    base_dir=None,
    input_dir=None,
    output_dir=None,
    allow_calibration_only=False,
    skip_existing=False,
    plot=False,
    clean=False,
    ):
    
    """
    Main entry point for HRSReduce scripts,
    default values can be changed as required if reduce is used as a script
    Finds input directories, and loops over observation night and instrument modes

    Parameters
    ----------
    night : str, list[str]
        the observation night to reduce, as named in the folder structure.
    modes : str, list[str]
        the instrument modes to use, if None will use all known modes for the current instrument. See instruments for possible options
    arm : str
        the spectrograph arm to reduce, if None will use all known modes for the current instrument
    base_dir : str, optional
        base data directory that HRSReduce should work in, is prefixed on input_dir and output_dir (default: use settings_pyreduce.json)
    input_dir : str, optional
        input directory containing raw files. If relative will use base_dir as root (default: use settings_pyreduce.json)
    output_dir : str, optional
        output directory for intermediary and final results. If relative will use base_dir as root (default: use settings_pyreduce.json)
    """

    logger.info(f"\n\n Version %s\n\n",__version__)

    if night is None or np.isscalar(night):
        #night = [night]
        year = night[0:4]
        mmdd = night[4:8]
        yyyymmdd=year+mmdd
        
    if arm == "H":
        arm_colour = 'Blu'
    elif arm == "R":
        arm_colour = 'Red'
    else:
        print("Arm not supported.")
        exit()
        
    header_ext = 'RECT'

    if loc =='work':
        base_dir = ("/home/sa/"+str(yyyymmdd)+"/hrs")
        input_dir = ("/home/sa/"+str(yyyymmdd)+"/hrs/raw/")
        output_dir = ("/home/sa/"+str(yyyymmdd)+"/hrs/product")
    elif loc == 'archive':
        base_dir = ("/salt/data/"+str(year)+"/"+str(mmdd)+"/hrs")
        input_dir = ("/salt/data/"+str(year)+"/"+str(mmdd)+"/hrs/raw/")
        output_dir = ("/salt/data/"+str(year)+"/"+str(mmdd)+"/hrs/reduced/")
    else:
        logging.error("Location unknown. Exiting.")
        return 2
        exit()
        
    base_dir = ("/Users/daniel/Desktop/SALT_HRS_DATA/")
    input_dir = arm_colour+"/"+year+"/"+mmdd+"/raw/"
    output_dir = arm_colour+"/"+year+"/"+mmdd+"/reduced/"
    
    output = []

    if np.isscalar(modes):
        modes = [modes]

    if np.isscalar(arm):
        arm = [arm]

    input_dir = join(base_dir, input_dir)
    #Test if input director exists
    exists = os.path.isdir(input_dir)
    if not exists:
        logger.error("Input directory {}does not exist. Exiting".format(input_dir))
        return 2
        exit()
        
    output_dir = join(base_dir, output_dir)

    try:
        os.mkdir(output_dir)
    except Exception:
        pass
        
    for m in modes:
        log_file = join(base_dir.format(mode=modes),"logs/%s_%s_%s.log" %(arm_colour,m, yyyymmdd))
        
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
                yyyymmdd,
                input_dir,
            )
            #Now search to find a night with the files
            files["bias"], nights["bias"]= FindNearestFiles("Bias",yyyymmdd,m,base_dir,arm_colour,logger)
            #Create the output directory if it does not exist
            output_dir_bias = arm_colour+"/"+nights["bias"][0:4]+"/"+nights["bias"][4:8]+"/reduced/"
            try:
                os.mkdir(str(base_dir+output_dir_bias))
            except Exception:
                pass

            
        if not files["flat"]:
            logger.warning(
                f"No FLAT files found for instrument: night: %s in folder: %s \n    Looking elsewhere...\n",
                yyyymmdd,
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
                yyyymmdd,
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
                yyyymmdd,
                input_dir,
            )
            pass
            
        if not files["lfc"]:
            logger.warning(
                f"No LFC files found for instrument: night: %s in folder: %s SKIPPING STEP FOR NOW\n",
                yyyymmdd,
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
            --  Blaze correction
            --  Merge orders
        '''
        
        #Apply the level 0 corrections (gain and overscan)
        files = L0Corrections(files,nights,yyyymmdd,input_dir,output_dir,base_dir,arm).run()

        #Calcualte the master bias
        master_bias = MasterBias(files["bias"],input_dir,output_dir,arm_colour,yyyymmdd,plot).create_masterbias()
        
        #Remove any intermediate bias frames
        for redundant in files["bias"]:
            os.remove(redundant)
        
        #Subtract the bias from all other frames
        #Loop over the files dict for the different types to make sure the correct bias file is subtracted (e.g., if the Flats are from a different night)
        files_out = {}
        for type in ["sci", "arc", "lfc", "flat"]:
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
                
                #Remove any intermediate bias frames
                for redundant in files_tmp2["bias"]:
                    os.remove(redundant)
                    
                
        del files
        files = files_out
        
        #Calcualte the master flat
        master_flat = MasterFlat(files["flat"],nights,input_dir,output_dir,base_dir,arm_colour,yyyymmdd,m,plot).create_masterflat()
        
        #Remove the intermediate files
        for ff in files["flat"]:
            os.remove(ff)
            
        #Clean the files of CRs
        _ = CosmicRayMasking(files,arm)
            
        #Find the appropriate Super Flat file for the order tracing, this must be the closest PREVIOUS frame in case of tank openings etc.
        super_flat = []
        prev_night = yyyymmdd
        while not super_flat:
            prev_night = arrow.get(prev_night).shift(days=-1).format('YYYYMMDD')
            prev_year=prev_night[0:4]
            prev_mmdd=prev_night[4:8]
            prev_data_location = os.path.join(base_dir, arm_colour+'/????/Super_Flats/')
            super_flats = glob.glob(prev_data_location+m+'_Super_Flat_'+arm[0]+'*.fits')
            s_difference = []
            sup_files = []
            for sfiles in super_flats:
                s_date=(os.path.basename(sfiles)[-13:-5])
                tmp = ((arrow.get(int(yyyymmdd[0:4]),int(yyyymmdd[4:6]),int(yyyymmdd[6:8])) - arrow.get(int(s_date[0:4]),int(s_date[4:6]),int(s_date[6:8]))).days)
                if tmp > 0:
                    s_difference.append(tmp)
                    sup_files.append(sfiles)
            index_of_closest = (np.abs(s_difference)).argmin()
            if s_difference[index_of_closest] < 365:
                super_flat = sup_files[index_of_closest]
                logger.info("Using Super Flat file: {}".format(super_flat))
            else:
                super_flat = super_flats[index_of_closest]
                logger.warning("Please update the Super Flat files, currently using potentially outdated file: {}".format(super_flat))
        
        #Create the Order file
        order_file = OrderTrace(super_flat,nights,base_dir,arm_colour,m,plot).order_trace()
        
        
        #Find the appropriate Super Arc file for the Slit Tilt correction, this must be the closest PREVIOUS frame in case of tank openings etc.
        super_arc = []
        prev_night = yyyymmdd
        while not super_arc:
            prev_night = arrow.get(prev_night).shift(days=-1).format('YYYYMMDD')
            prev_year=prev_night[0:4]
            prev_mmdd=prev_night[4:8]
            prev_data_location = os.path.join(base_dir, arm_colour+'/????/Super_Arcs/')
            super_arcs = glob.glob(prev_data_location+m+'_Super_Arc_'+arm[0]+'*.fits')
            s_difference = []
            sup_files = []
            for sfiles in super_arcs:
                s_date=(os.path.basename(sfiles)[-13:-5])
                tmp = ((arrow.get(int(yyyymmdd[0:4]),int(yyyymmdd[4:6]),int(yyyymmdd[6:8])) - arrow.get(int(s_date[0:4]),int(s_date[4:6]),int(s_date[6:8]))).days)
                if tmp > 0:
                    s_difference.append(tmp)
                    sup_files.append(sfiles)
            s_difference = np.array(s_difference)
            ii = np.where(s_difference > 0)[0]
            index_of_closest = (np.abs(s_difference[ii])).argmin()
            if s_difference[index_of_closest] < 365:
                super_arc = sup_files[index_of_closest]
                logger.info("Using Super Arc file: {}".format(super_arc))
            else:
                super_arc = super_arcs[index_of_closest]
                logger.warning("Please update the Super Arc files, currently using potentially outdated file: {}".format(super_arc))
                
        #Fully process the Super Arc and Flat in the right order
        order_file_rect = OrderRectification(super_arc,super_flat,order_file,arm_colour,m,base_dir,super_arc=super_arc).perform()
        SlitCorrection(super_arc,header_ext,order_file_rect,arm[0],m,base_dir,yyyymmdd,plot=plot,super_arc=super_arc).correct()
        VarExts(super_arc,master_bias,master_flat).run()
        #Rectify and tilt correct the master flat as this needs only doing once
        _ = OrderRectification(master_flat,master_flat,order_file,arm_colour,m,base_dir,super_arc=super_arc).perform()
        SlitCorrection(master_flat,'RECT', order_file_rect, arm[0],m, base_dir,yyyymmdd,plot=False,super_arc=super_arc).correct()
        
        SpectralExtraction(super_arc, master_flat,super_arc,order_file_rect,arm_colour,m,base_dir).extraction()
        
        #Calculate the Varience image, extract the frames
        for arc_file in files['arc']:
            cal_type = 'ThAr'
            VarExts(arc_file,master_bias,master_flat).run()
            _ = OrderRectification(arc_file,master_flat,order_file,arm_colour,m,base_dir,super_arc=super_arc).perform()
            SlitCorrection(arc_file,header_ext,order_file_rect,arm[0],m,base_dir,yyyymmdd,plot=plot,super_arc=super_arc).correct()
        
        #Create the normalised flat
        FlatNormalisation(master_flat, order_file_rect).normalise()
        
        #Calcualte a blaze using the Spectral Extracton module
        VarExts(master_flat,master_bias,master_flat).run()
        SpectralExtraction(master_flat, master_flat,files['arc'][0],order_file_rect,arm_colour,m,base_dir).extraction()
        
        #Calculate the Varience image, rectify the orders, perform slit tilt calculation and correction and extract the frames
        for lfc_file in files['lfc']:
            logger.info("Processing file: {}".format(lfc_file))
            VarExts(lfc_file,master_bias,master_flat).run()
            _ = OrderRectification(lfc_file,master_flat,order_file,arm_colour,m,base_dir,super_arc=super_arc).perform()
            header_ext = 'RECT'
            SlitCorrection(lfc_file,header_ext, order_file_rect, arm[0],m, base_dir,yyyymmdd,plot=plot,super_arc=super_arc).correct()
            SpectralExtraction(lfc_file, master_flat,files['arc'][0],order_file_rect,arm_colour,m,base_dir).extraction()
 #           OrderMerge(sci_file,master_flat,arm,plot=False).execute()
 
        #Calculate the extract the frames and calculate the wave solution
        for arc_file in files['arc']:
            SpectralExtraction(arc_file, master_flat,arc_file,order_file_rect,arm_colour,m,base_dir).extraction()
            WavelengthCalibration(arc_file, super_arc, arm, m, base_dir,cal_type,plot).execute()
            
            
        #Calculate the Varience image, rectify the orders, perform slit tilt calculation and correction and extract the frames
        for sci_file in files['sci']:
            logger.info("Processing file: {}".format(sci_file))
            VarExts(sci_file,master_bias,master_flat).run()
            _ = OrderRectification(sci_file,master_flat,order_file,arm_colour,m,base_dir,super_arc=super_arc).perform()
            header_ext = 'RECT'
            SlitCorrection(sci_file,header_ext, order_file_rect, arm[0],m, base_dir,yyyymmdd,plot=plot,super_arc=super_arc).correct()
            SpectralExtraction(sci_file, master_flat,files['arc'][0],order_file_rect,arm_colour,m,base_dir).extraction()
            OrderMerge(sci_file,master_flat,arm,plot=False).execute()
        
        if clean:
            for type in ["sci", "arc", "lfc", "flat"]:
                for file in files[type]:
                    try:
                        os.remove(file)
                    except:
                        pass
    return output

