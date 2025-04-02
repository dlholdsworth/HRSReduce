import logging
#import glob
#import numpy as np
#import matplotlib.pyplot as plt
#from astropy.io import fits
#from datetime import datetime
#import pytz
#from scipy.stats import norm
#import os


from hrsreduce.utils.frame_stacker import FrameStacker

logger = logging.getLogger(__name__)


class MasterFlat():

    def __init__(self,files,nights,input_dir,output_dir,arm,plot):
    
        self.propid = "CAL_FLAT"
        self.files = files
        self.nights = nights
        self.out_dir = out_dir
        self.in_dir = in_dir
        self.plot = plot
        self.arm = arm
        if self.arm == "Blu":
            self.sarm = "H"
        else:
            self.sarm = "R"
    
        logger.info('Started {}'.format(self.__class__.__name__))
        
        
        
    def create_masterflat(self):
    
        #TODO: Check if there is a Master file, if not, check if there is a master BIAS for the flat night, then proceed. Check KPF code.
    
        #Test if Master Bias exists
        master_file = glob.glob(out_location+"Master_Flat_"+arm+night+".fits")
        if len(master_file) == 0:

            #Open files to check if they are the correct OBSTYPE (Flat Field)
            Flat_files = []
            if len(all_files) > 0:
                for file in all_files:
                    file_night = file.removeprefix(out_location)[3:11]
                    if(file_night == night):
                        hdu=fits.open(file)
                        if (hdu[0].header["OBSTYPE"] == "Flat field" and hdu[0].header["FIFPORT"] == mode):
                            Flat_files.append(file)
                        hdu.close
            else:
                print ("\n   !!! No files found in {}. Check the arm and night. Exiting.\n".format(path))
            n=len(Flat_files)
            if n <1:
                print ("\n   !!! No Flat files found in {}. Check arm ({}) and night ({}). Exiting.\n".format(path,arm,night))
                exit()
                
            Flat_concat = []
            gain = hdu[0].header["GAIN"]
            
            if arm =="H":
                gain1 = float(gain.split()[0])
                gain2 = float(gain.split()[1])
                gain = gain1
                
            if arm == "R":
                gain1 = float(gain.split()[0])
                gain2 = float(gain.split()[1])
                gain3 = float(gain.split()[2])
                gain4 = float(gain.split()[3])
                gain = gain1
            Flat_files_short = []
            
            jd_mean = []
            PRE_DEW =[]
            PRE_VAC =[]
            TEM_AIR =[]
            TEM_BCAM=[]
            TEM_COLL=[]
            TEM_ECH =[]
            TEM_IOD =[]
            TEM_OB  =[]
            TEM_RCAM=[]
            TEM_RMIR=[]
            TEM_VAC =[]
            
            for file in Flat_files:
                Flat_files_short.append(file.lstrip(out_location))
                hdu=fits.open(file)
                #Perform a test to reject bad files
                if np.std(hdu[0].data) > 200:
                    #Gain correct and subtract master bias
                    flat_data = (hdu[0].data) - master_bias
                    hdu.close
                    #Overwrite the data in the file with the corrected information and write to a new file prefixed with b (for bais corrected) and g (for gain corrected)
                    hdu[0].data = flat_data
                    file_out=str(out_location+"b"+file.removeprefix(out_location))
                    hdu.writeto(file_out,overwrite=True)
                    Flat_concat.append(flat_data)#/flat_mean)
                    jd_mean.append(float(hdu[0].header['JD']))
                    PRE_DEW.append(float(hdu[0].header['PRE-DEW']))
                    PRE_VAC.append(float(hdu[0].header['PRE-VAC']))
                    TEM_AIR.append(float(hdu[0].header['TEM-AIR']))
                    TEM_BCAM.append(float(hdu[0].header['TEM-BCAM']))
                    TEM_COLL.append(float(hdu[0].header['TEM-COLL']))
                    TEM_ECH.append(float(hdu[0].header['TEM-ECH']))
                    TEM_IOD.append(float(hdu[0].header['TEM-IOD']))
                    TEM_OB.append(float(hdu[0].header['TEM-OB']))
                    TEM_RCAM.append(float(hdu[0].header['TEM-RCAM']))
                    TEM_RMIR.append(float(hdu[0].header['TEM-RMIR']))
                    TEM_VAC.append(float(hdu[0].header['TEM-VAC']))
                else:
                #Change the file name to show it is bad
                    os.rename(file, str(path+"Bad_Flat_"+file.removeprefix(out_location)[2:]))
                    hdu.close
            jd_mean = np.array(jd_mean)
            jd_mean = np.mean(jd_mean)
            
            PRE_DEW =np.array(PRE_DEW)
            PRE_VAC =np.array(PRE_VAC)
            TEM_AIR =np.array(TEM_AIR)
            TEM_BCAM=np.array(TEM_BCAM)
            TEM_COLL=np.array(TEM_COLL)
            TEM_ECH =np.array(TEM_ECH)
            TEM_IOD =np.array(TEM_IOD)
            TEM_OB  =np.array(TEM_OB)
            TEM_RCAM=np.array(TEM_RCAM)
            TEM_RMIR=np.array(TEM_RMIR)
            TEM_VAC =np.array(TEM_VAC)
            
            #Create the master flat and write to new FITS file.
            master_flat = np.median(Flat_concat,axis=0)
            master_flat = np.float32(master_flat)
            new_hdu = fits.PrimaryHDU(data=master_flat)
            new_hdu.header.insert(5,('COMMENT',"  FITS (Flexible Image Transport System) format is defined in 'Astronomy"))
            new_hdu.header.insert(6,('COMMENT',"  and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H"))
            new_hdu.header['FIFPORT'] = (str(hdu[0].header['FIFPORT']), "FIF port selection")
            new_hdu.header['OBJECT'] = (str('Master_Flat'), "Object name")
            new_hdu.header['OBSERVAT'] = (str('SALT'), "South African Large Telescope")
            new_hdu.header['SITEELEV'] = (str(1798.), "Site elevation")
            new_hdu.header['SITELAT'] = (str(-32.3795), "Geographic latitude of the observation")
            new_hdu.header['SITELONG'] = (str(20.812), "Geographic longitude of the observation")
            new_hdu.header['AMPSEC'] = (str(hdu[0].header['AMPSEC']),"Amplifier Section")
            new_hdu.header['BIASSEC'] = (str(hdu[0].header['BIASSEC']),"Bias Section")
            new_hdu.header['CCDNAMPS'] = (str(hdu[0].header['CCDNAMPS']), "No. of amplifiers used")
            new_hdu.header['CCDSEC'] = (str(hdu[0].header['CCDSEC']), "CCD Section")
            new_hdu.header['CCDSUM'] = (str(hdu[0].header['CCDSUM']), "On-chip binning")
            new_hdu.header['CCDTYPE'] = (str('Flat'),"Observation type")
            new_hdu.header['DATASEC'] = (str(hdu[0].header['DATASEC']),"Data Section")
            new_hdu.header['DATE-OBS'] = (str(hdu[0].header['DATE-OBS']),"Date of observation")
            new_hdu.header['DETMODE'] = (str(hdu[0].header['DETMODE']),"Detector Mode")
            new_hdu.header['DETNAM'] = (str(hdu[0].header['DETNAM']),"Detector Name")
            new_hdu.header['DETSEC'] = (str(hdu[0].header['DATASEC']), "Detector Section")
            new_hdu.header['DETSER'] = (str(hdu[0].header['DETSER']), "Detector serial number")
            new_hdu.header['DETSIZE'] = (str(hdu[0].header['DETSIZE']), "Detector Size")
            new_hdu.header['DETSWV'] = (str(hdu[0].header['DETSWV']),"Detector software version")
            new_hdu.header['EXPTIME'] = (str(hdu[0].header['EXPTIME']),"Exposure time (s)")
            new_hdu.header['GAIN'] = (str(gain),"CCD gain (photons/ADU)")
            new_hdu.header['GAINSET'] = (str(hdu[0].header['GAINSET']),"Gain Setting")
            new_hdu.header['INSTRUME'] = (str('HRS'),"Instrument name: High Resolution Spectrograph")
            new_hdu.header['NCCDS'] = (str(hdu[0].header['NCCDS']),"Number of CCDs")
            new_hdu.header['NODCOUNT'] = (str(hdu[0].header['NODCOUNT']), "No. of Nod/Shuffles")
            new_hdu.header['NODPER'] = (str(hdu[0].header['NODPER']), "Nod & Shuffle Period (s)")
            new_hdu.header['NODSHUFF'] = (str(hdu[0].header['NODSHUFF']), "Nod & Shuffle enabled?")
            new_hdu.header['OBSMODE'] = (str(hdu[0].header['OBSMODE']),"Observation mode")
            new_hdu.header['OBSTYPE'] = (str('Flat'), "Observation type")
            new_hdu.header['PRESCAN'] = (str(hdu[0].header['PRESCAN']), "Prescan pixels at start/end of line")
            new_hdu.header['ROSPEED'] = (str(hdu[0].header['ROSPEED']),"CCD readout speed (Hz)")
            new_hdu.header['JD_MEAN'] = (str(jd_mean),"Mean JD of input files")
            new_hdu.header['MN_PDEW']= (str(np.mean(PRE_DEW)),"Mean of input files")
            new_hdu.header['MN_PVAC']= (str(np.mean(PRE_VAC)),"Mean of input files")
            new_hdu.header['MN_TAIR']= (str(np.mean(TEM_AIR)),"Mean of input files")
            new_hdu.header['MN_TBCAM']= (str(np.mean(TEM_BCAM)),"Mean of input files")
            new_hdu.header['MN_TCOLL']= (str(np.mean(TEM_COLL)),"Mean of input files")
            new_hdu.header['MN_TECH']= (str(np.mean(TEM_ECH)),"Mean of input files")
            new_hdu.header['MN_TIOD']= (str(np.mean(TEM_IOD)),"Mean of input files")
            new_hdu.header['MN_TOB']= (str(np.mean(TEM_OB)),"Mean of input files")
            new_hdu.header['MN_TRCAM']= (str(np.mean(TEM_RCAM)),"Mean of input files")
            new_hdu.header['MN_TRMIR']= (str(np.mean(TEM_RMIR)),"Mean of input files")
            new_hdu.header['MN_TVAC']= (str(np.mean(TEM_VAC)),"Mean of input files")
            new_hdu.header['GAIN_p'] = (str(gain),"Gain that has been applied")
            DATE_EXT=str(datetime.now(tz=pytz.UTC).strftime("%Y-%m-%d"))
            UTC_EXT = str(datetime.now(tz=pytz.UTC).strftime("%H:%M:%S.%f"))
            new_hdu.header['DATE-EXT'] = (DATE_EXT,'Date file created')
            new_hdu.header['UTC-EXT'] = (UTC_EXT,'Time file created')
            new_hdu.header['N_FILE'] = (str(n),"Number of files combined")
            new_hdu.header['HISTORY'] = ("Files used for Master: "+str(Flat_files_short))
            
            new_hdu.writeto(str(out_location)+"Master_Flat_"+str(arm)+str(night)+".fits",overwrite=True)
        
        else:
            print("\n      +++ Reading Master Flat frame "+master_file[0]+"\n")
            flat_hdu=fits.open(master_file[0])
            master_flat=flat_hdu[0].data
            flat_hdu.close
        
        if Plot == "True":  # pragma: no cover
            title = "Master Flat"
            plt.title(title)
            plt.xlabel("x [pixel]")
            plt.ylabel("y [pixel]")
            bot, top = np.percentile(master_flat, (1, 99))
            plt.imshow(master_flat, vmin=bot, vmax=top, origin="lower")
            plt.show()
            
        return master_flat


    def run(self):
