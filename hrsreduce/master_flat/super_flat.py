import numpy as np
import logging
import glob
from astropy.io import fits
import os
from datetime import date, timedelta

from hrsreduce.utils.frame_stacker import FrameStacker
from hrsreduce.utils.sort_files import SortFiles
from hrsreduce.L0_Corrections.level0corrections import L0Corrections
from hrsreduce.master_bias.master_bias import MasterBias,SubtractBias
from hrsreduce.master_flat.master_flat import MasterFlat


logger = logging.getLogger(__name__)

class SuperFlat():

    def __init__(self,year,base_dir,arm,mode,start,end,plot=False):
    
        self.flat_propid = "CAL_FLAT"
        self.bias_propid = "CAL_BIAS"
        self.year = year
        self.base_dir = base_dir
        self.plot = plot
        self.arm = arm
        if self.arm == "Blu":
            self.sarm = "H"
            self.ax1 = 2074
            self.ax2 = 4102
        else:
            self.sarm = "R"
            self.ax1 = 4122
            self.ax2 = 4112
        self.s_date = start
        self.e_date =end
        self.mode = mode
        if mode == "HS":
            self.modefull = 'HIGH STABILITY'
        if mode == "HR":
            self.modefull = 'HIGH RESOLUTION'
        if mode == "MR":
            self.modefull = 'MEDIUM RESOLUTION'
        if mode == "LR":
            self.modefull = 'LOW RESOLUTION'
        self.tn = '00000000'

        self.low_light_limit = 0.1
        logger.info('Started {}'.format(self.__class__.__name__))
        
    @staticmethod
    def date_range(start_date, end_date):
        return [start_date + timedelta(days=n) for n in range(int((end_date - start_date).days) + 1)]

    def create_superflat(self):
    
        logger.info('Creating Super Flat file for {} arm, {} mode between {} and {}'.format(self.arm, self.mode,self.s_date, self.e_date))
        
        #Find all flat files in the given date range
        files = []
        start = date(int(self.s_date[0:4]), int(self.s_date[4:6]), int(self.s_date[6:8]))
        end = date(int(self.e_date[0:4]), int(self.e_date[4:6]), int(self.e_date[6:8]))
        all_dates = self.date_range(start, end)
        for dates in all_dates:
            yr = dates.strftime("%Y")
            mmdd =dates.strftime("%m%d")
            tmp = (sorted(glob.glob(self.base_dir+self.arm+"/"+yr+"/"+mmdd+"/raw/*.fits")))
            if len(tmp) > 0:
                for t in tmp:
                    files.append(t)
        files = files
        Flat_files = []
        Flat_dirs = []
        Flat_nights = []
        L0_flat = []
        print()
        
        #Search the storage area for the Flat files, when found perform L0 corrections files
        for file in files:
            try:
                with fits.open(file) as hdul:
                    if (hdul[0].header["PROPID"] == self.flat_propid and hdul[0].header["OBSMODE"] == self.modefull):
                        if hdul[0].header["NAXIS1"] == self.ax1 and hdul[0].header["NAXIS2"] == self.ax2:
                            Flat_files.append(file)
                            Flat_dirs.append(os.path.dirname(file))
                            Flat_nights.append(os.path.basename(file)[1:9])
                            input_dir = os.path.dirname(file)+"/"
                            l0_file = {}
                            l0_night = {}
                            l0_file['flat'] = [file]
                            l0_night['flat'] = os.path.basename(file)[1:9]
                            output_dir =os.path.dirname(file)[:-4]+"/reduced/"
                            try:
                                os.mkdir(output_dir)
                            except Exception:
                                pass
                            L0_flat.append(L0Corrections(l0_file,l0_night,self.tn,input_dir,output_dir,self.base_dir,self.sarm).run()['flat'][0])
            except:
                pass
 
        #For each of the nights, need to calcluate a master bias and thus run L0 corrections on those biases first
        single_nights = set(Flat_nights)
        plot = False
        flat_files = []
        for yyyymmdd in single_nights:
            bias_files = {}
            bias_night = {}
            bias_night['bias'] = str(yyyymmdd)
            b_files =[]
            
            year = yyyymmdd[0:4]
            mmdd = yyyymmdd[4:]
            files = glob.glob(self.base_dir+self.arm+"/"+str(self.year)+"/"+mmdd+"/raw/*.fits")
            for file in files:
                try:
                    with fits.open(file) as hdul:
                        if((hdul[0].header["OBSTYPE"] == "Bias" or hdul[0].header["CCDTYPE"] == "Bias") and hdul[0].header["EXPTIME"] == 0.):
                            if hdul[0].header["NAXIS1"] == self.ax1 and hdul[0].header["NAXIS2"] == self.ax2:
                                b_files.append(file)
                                output_dir =os.path.dirname(file)[:-4]+"/reduced/"
                                input_dir = os.path.dirname(file)+"/"
                except:
                    pass
            bias_files['bias'] = b_files
            #Make the output dir
            try:
                os.mkdir(output_dir)
            except Exception:
                pass
            bias_files = L0Corrections(bias_files,bias_night,self.tn,input_dir,output_dir,self.base_dir,self.sarm).run()
            master_bias = MasterBias(bias_files["bias"],input_dir,output_dir,self.arm,yyyymmdd,plot).create_masterbias()
            
            #Find the flat files and subtract the master bias
            flat_dir_files = glob.glob(self.base_dir+self.arm+"/"+str(self.year)+"/"+mmdd+"/reduced/*.fits")

            for file in flat_dir_files:
                file_dict = {}
                with fits.open(file) as hdul:
                    if (hdul[0].header["PROPID"] == self.flat_propid and hdul[0].header["OBSMODE"] == self.modefull):
                        try:
                            noise = hdul[0].header["RONOISE"]
                        except:
                            file_dict['flat'] = [file]
                            flat_files.append(SubtractBias(master_bias,file_dict,self.base_dir,self.arm,yyyymmdd,"flat").subtract()[0])

        #Can now run the master flat code.
        flats = {}
        flats['flat'] = flat_files
        nights = {}
        nights['flat'] = str(self.year+str(self.s_date[4:8]))
        master_flat = MasterFlat(flats["flat"],nights,' ',' ',self.base_dir,self.arm,self.tn,self.mode,plot,super=True).create_masterflat()
