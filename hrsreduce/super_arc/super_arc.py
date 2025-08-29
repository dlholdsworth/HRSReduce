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
from hrsreduce.super_arc.alg import MasterArc


logger = logging.getLogger(__name__)

class SuperArc():

    def __init__(self,year,base_dir,arm,mode,start,end,plot=False):
    
        self.arc_propid = "CAL_ARC"
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
        if self.mode == 'HS':
            self.fullmode = "HIGH STABILITY"
            self.arc_propid = "CAL_STABLE"
            self.I2STAGE = "Reference Fibre"
            self.exp = 30.0
        if self.mode == 'HR':
            self.fullmode = "HIGH RESOLUTION"
            self.arc_propid = "CAL_ARC"
            self.I2STAGE = "Nothing In Beam"
            self.exp = 500.0
        if self.mode == 'MR':
            self.fullmode = "MEDIUM RESOLUTION"
            self.arc_propid = "CAL_ARC"
            self.I2STAGE = "Nothing In Beam"
            self.exp = 400.0
        if self.mode == 'LR':
            self.fullmode = "LOW RESOLUTION"
            self.arc_propid = "CAL_ARC"
            self.I2STAGE = "Nothing In Beam"
            self.exp = 300.0
        self.tn = '00000000'

        self.low_light_limit = 0.1
        logger.info('Started {}'.format(self.__class__.__name__))
        
    @staticmethod
    def date_range(start_date, end_date):
        return [start_date + timedelta(days=n) for n in range(int((end_date - start_date).days) + 1)]

    def create_superarc(self):
        
        #Find all arc files in the given date range
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
        Arc_files = []
        Arc_dirs = []
        Arc_nights = []
        L0_arc = []
        
        #Search the storage area for the Arc files, when found perform L0 corrections files
        for file in files:
            with fits.open(file) as hdul:
                if (hdul[0].header["PROPID"] == self.arc_propid and hdul[0].header["OBSMODE"] == self.fullmode):
                    if (hdul[0].header["I2STAGE"] == self.I2STAGE and hdul[0].header["EXPTIME"] == self.exp):
                        if hdul[0].header["NAXIS1"] == self.ax1 and hdul[0].header["NAXIS2"] == self.ax2:
                            Arc_files.append(file)
                            Arc_dirs.append(os.path.dirname(file))
                            Arc_nights.append(os.path.basename(file)[1:9])
                            input_dir = os.path.dirname(file)+"/"
                            l0_file = {}
                            l0_night = {}
                            l0_file['arc'] = [file]
                            l0_night['arc'] = os.path.basename(file)[1:9]
                            output_dir =os.path.dirname(file)[:-4]+"/reduced/"
                            try:
                                os.mkdir(output_dir)
                            except Exception:
                                pass
                            L0_arc.append(L0Corrections(l0_file,l0_night,self.tn,input_dir,output_dir,self.base_dir,self.sarm).run()['arc'][0])
 
        #For each of the nights, need to calcluate a master bias and thus run L0 corrections on those biases first
        single_nights = sorted(set(Arc_nights))
        plot = False
        arc_files = []
        for yyyymmdd in single_nights:
            bias_files = {}
            bias_night = {}
            bias_night['bias'] = str(yyyymmdd)
            b_files =[]
            
            year = yyyymmdd[0:4]
            mmdd = yyyymmdd[4:]
            files = sorted(glob.glob(self.base_dir+self.arm+"/"+str(self.year)+"/"+mmdd+"/raw/*.fits"))
            for file in files:
                with fits.open(file) as hdul:
                    if((hdul[0].header["OBSTYPE"] == "Bias" or hdul[0].header["CCDTYPE"] == "Bias") and hdul[0].header["EXPTIME"] == 0.):
                        if hdul[0].header["NAXIS1"] == self.ax1 and hdul[0].header["NAXIS2"] == self.ax2:
                            b_files.append(file)
                            output_dir =os.path.dirname(file)[:-4]+"/reduced/"
                            input_dir = os.path.dirname(file)+"/"
            bias_files['bias'] = b_files
            bias_files = L0Corrections(bias_files,bias_night,self.tn,input_dir,output_dir,self.base_dir,self.sarm).run()
            master_bias = MasterBias(bias_files["bias"],input_dir,output_dir,self.arm,yyyymmdd,plot).create_masterbias()
            
            #Find the arc files and subtract the master bias
            arc_dir_files = sorted(glob.glob(self.base_dir+self.arm+"/"+str(self.year)+"/"+mmdd+"/reduced/*.fits"))

            for file in arc_dir_files:
                file_dict = {}
                with fits.open(file) as hdul:
                    if (hdul[0].header["PROPID"] == self.arc_propid and hdul[0].header["OBSMODE"] == self.fullmode):
                        if (hdul[0].header["I2STAGE"] == self.I2STAGE):
                            try:
                                noise = hdul[0].header["RONOISE"]
                            except:
                                file_dict['arc'] = [file]
                                arc_files.append(SubtractBias(master_bias,file_dict,self.base_dir,self.arm,yyyymmdd,"arc").subtract()[0])

        #Can now run the master arc code.
        arcs = {}
        arcs['arc'] = arc_files
        nights = {}
        nights['arc'] = str(self.year+str(self.s_date[4:8]))
        super_arc = MasterArc(arcs["arc"],nights,' ',' ',self.base_dir,self.arm,self.tn,self.mode,plot,super=True).create_masterarc()
