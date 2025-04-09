import logging
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from datetime import datetime
import pytz
import os


from hrsreduce.utils.frame_stacker import FrameStacker
from hrsreduce.utils.sort_files import SortFiles
from hrsreduce.L0_Corrections.level0corrections import L0Corrections
from hrsreduce.master_bias.master_bias import MasterBias

logger = logging.getLogger(__name__)

class MasterFlat():

    def __init__(self,files,nights,in_dir,out_dir,base_dir,arm,night,mode,plot):
    
        self.propid = "CAL_FLAT"
        self.files = files
        self.nights = nights
        self.tn = night
        self.out_dir = out_dir
        self.in_dir = in_dir
        self.base_dir = base_dir
        self.plot = plot
        self.arm = arm
        self.mode = mode
        if self.arm == "Blu":
            self.sarm = "H"
        else:
            self.sarm = "R"
        self.low_light_limit = 0.1
        logger.info('Started {}'.format(self.__class__.__name__))
        
        
        
    def create_masterflat(self):
        
        #Set the night for the master flat data
        yyyymmdd = str(self.nights['flat'][0:4])+str(self.nights['flat'][4:8])
        self.out_dir = self.base_dir+self.arm+"/"+yyyymmdd[0:4]+"/"+yyyymmdd[4:]+"/reduced/"
    
        #Test if Master Flat exists
        master_file = glob.glob(self.out_dir+self.mode+"_Master_Flat_"+self.sarm+str(yyyymmdd)+".fits")

        if len(master_file) == 0:
            #If there is no MASTER FLAT file we need to create one. But we need to make sure there is a
            #MASTER BIAS frame. If not, the flats were not taken on the same night as the science so need
            #to make a MASTER BIAS that can be subtracted from the FLAT frames.
            

            #Test if Master Bias exists
            flat_bias_loc = self.base_dir+self.arm+"/"+str(self.nights['flat'][0:4])+"/"+str(self.nights['flat'][4:8])
            flat_bias = glob.glob(flat_bias_loc+"/reduced/Master_Bias_"+self.sarm+self.nights['flat']+".fits")

            if len(flat_bias) == 1:
                #Master Bias file has been created
                flat_bias = flat_bias[0]
                pass
            else:
                #Create Master Bias
                logger.info('Need to create MASTER BIAS for night of flats ({})'.format(self.nights['flat']))
                files = {}
                files["bias"],_,_,_,_ = SortFiles(flat_bias_loc+"/raw/",logger,mode=self.mode)
                nts = self.nights
                nts["bias"] = yyyymmdd
                files = L0Corrections(files,self.nights,yyyymmdd, flat_bias_loc+"/raw/",flat_bias_loc+"/reduced/",self.base_dir,self.sarm).run()

                flat_bias = MasterBias(files["bias"],flat_bias_loc+"/reduced/",flat_bias_loc+"/reduced/",self.arm,yyyymmdd,self.plot).create_masterbias()
                
                for ff in files["bias"]:
                    os.remove(ff)
                
            with fits.open(flat_bias) as hdul:
                Master_bias = hdul[0].data
                RONOISE = hdul[0].header["RONOISE"]
            
                
            #Open files to check if they are the PROPID (CAL_FLAT)
            Flat_files = []
            Flat_files_short = []
            for file in self.files:
                with fits.open(file) as hdu:
                    if (hdu[0].header["PROPID"] == "CAL_FLAT"):
                        Flat_files.append(file)
                        gain = hdu[0].header["AVG_GAIN"]

            n=len(Flat_files)
            if n <1:
                logger.error("\n   !!! No Flat files found in {}. Check arm ({}) and night ({}). Exiting.\n".format(self.out_dir,self.arm,yyyymmdd))
                exit()
                
            Flat_concat = []
            Flat_files_short = []
            exptime = []
            
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
                Flat_files_short.append(file.removeprefix(self.out_dir))
                with fits.open(file) as hdu:
                    #Perform a test to reject bad files
                    if np.std(hdu[0].data) > 200:
                        #Subtract master bias
                        flat_data = (hdu[0].data - Master_bias).astype(np.float32)
                        #Overwrite the data in the file with the corrected information and write to a new file prefixed with b (for bais corrected)
                        hdu[0].data = flat_data
                        file_out=str(self.out_dir+"b"+file.removeprefix(self.out_dir))
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
                        exptime.append(float(hdu[0].header['EXPTIME']))
                    else:
                        logger.warning("Flat file {} found not to have enough range. Rejecting it.".format(file))
                        #Change the file name to show it is bad
                        os.rename(file, str(self.out_dir+"Bad_Flat_"+file.removeprefix(out_dir)))
 
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
            
            normalised_frames_data=[]
            
            np_flat_frames = np.array(Flat_concat)
            np_exptimes = np.array(exptime)
            
            for i in range(0,len(np_flat_frames)):
                frame_data = np_flat_frames[i]
                exptime = np_exptimes[i]
                
                logger.debug('Normalizing flat image: i,fitsfile,exp_time = {},{},{}'.format(i,Flat_files_short[i],exptime))
                
                normalised_frames_data.append(frame_data / exptime)       # Separately normalise by EXPTIME.

            normalised_frames_data = np.array(normalised_frames_data)
            
            fs = FrameStacker(normalised_frames_data,2.1,logger)
            flat_avg,flat_var,cnt,flat_unc = fs.compute()
            
            '''
            plt.imshow(flat_avg,origin='lower',vmin=0,vmax=5)
            plt.show()
            
            smooth_lamp_pattern = gaussian_filter(flat_avg, sigma=2.)
            unnormalised_flat = flat_avg / smooth_lamp_pattern
            unnormalised_flat_unc = flat_unc / smooth_lamp_pattern
            
            plt.imshow(unnormalised_flat,origin='lower',vmin=0,vmax=5)
            plt.show()
            
            # Use the low-light pixels as a mask.
            
            np_om_ffi_bool = np.where(flat_avg > self.low_light_limit, True, False)
            np_om_ffi_shape = np.shape(np_om_ffi_bool)
            logger.debug('np_om_ffi_shape = {}'.format(np_om_ffi_shape))

            # Compute mean of unmasked pixels in unnormalised flat.
            unmx = ma.masked_array(unnormalised_flat, mask = ~ np_om_ffi_bool)    # Invert the mask for mask_array operation.
            unnormalised_flat_mean = ma.getdata(unmx.mean()).item()
            logger.debug('unnormalised_flat_mean = {}'.format(unnormalised_flat_mean))

            # Normalize flat.
            flat = unnormalised_flat / unnormalised_flat_mean                     # Normalize the master flat by the mean.
            flat_unc = unnormalised_flat_unc / unnormalised_flat_mean             # Normalize the uncertainties.
            
            plt.imshow(flat,origin='lower',vmin=0,vmax=50)
            plt.show()

            # Less than low-light pixels cannot be reliably adjusted.  Reset below-threshold pixels to have unity flat values.
            flat = np.where(flat_avg < self.low_light_limit, 0.0, flat_avg)
            
            plt.imshow(flat,origin='lower',vmin=0,vmax=5)
            plt.show()
            '''
            
            #Create the master flat and write to new FITS file.
            master_flat = flat_avg.astype(np.float32)
            new_hdu = fits.PrimaryHDU(data=master_flat)
            
            new_hdu.header.insert(6,('COMMENT',"  FITS (Flexible Image Transport System) format is defined in 'Astronomy"))
            new_hdu.header.insert(7,('COMMENT',"  and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H"))
            new_hdu.header['ADCROT'] = (str(hdu[0].header['ADCROT']), "ADC rotation")
            new_hdu.header['ADCSEP'] = (str(hdu[0].header['ADCSEP']), "ADC prism separation")
            new_hdu.header['AIRMASS'] = (str(hdu[0].header['AIRMASS']+' '), "Airmass")
            new_hdu.header['BLOCKID'] = (str(hdu[0].header['BLOCKID']), "Block ID")
            new_hdu.header['BVISITID'] = (str(hdu[0].header['BVISITID']+' '), "Block Visit ID")
            new_hdu.header['CALFILT'] = (str(hdu[0].header['CALFILT']), "Calibration Lamp Filter")
            new_hdu.header['CALND'] = (str(hdu[0].header['CALND']), "Calibration ND setting")
            new_hdu.header['CALSCR'] = (str(hdu[0].header['CALSCR']), "Calibration screen position")
            new_hdu.header['COLPHI'] = (str(hdu[0].header['COLPHI']), "Autocollimator Phi")
            new_hdu.header['COLTHETA'] = (str(hdu[0].header['COLTHETA']), "Autocollimator Theta")
            new_hdu.header['DEC'] = (str(hdu[0].header['DEC']+' '), "Dec of object")
            new_hdu.header['DECPANGL'] = (str(hdu[0].header['DECPANGL']+' '), "RA - focal plane X-axis angle")
            new_hdu.header['ENVDIR'] = (str(hdu[0].header['ENVDIR']),"Wind direction E of N")
            new_hdu.header['ENVHUM'] = (str(hdu[0].header['ENVHUM']), "Relative humidity")
            new_hdu.header['ENVMJD'] = (str(hdu[0].header['ENVMJD']), "Environmental Measurement Time")
            new_hdu.header['ENVWIN'] = (str(hdu[0].header['ENVWIN']),"Wind speed")
            new_hdu.header['EPOCH'] = (str(hdu[0].header['EPOCH']+' '), "Epoch of object RA, Dec")
            new_hdu.header['EQUINOX'] = (str(hdu[0].header['EQUINOX']+' '), "Equinox of object RA, Dec")
            new_hdu.header['FIFCEN'] = (str(hdu[0].header['FIFCEN']),  "FIF centering location")
            new_hdu.header['FIFCOFF'] = (str(hdu[0].header['FIFCOFF']), "FIF centering offset")
            new_hdu.header['FIFPOFF'] = (str(hdu[0].header['FIFPOFF']), "FIF port offset")
            new_hdu.header['FIFPORT'] = (str(hdu[0].header['FIFPORT']), "FIF port selection")
            new_hdu.header['FIFSEP'] = (str(hdu[0].header['FIFSEP']), "FIF fibre separation")
            new_hdu.header['GUIDEC'] = (str(hdu[0].header['GUIDEC']+' '), "Guider Dec")
            new_hdu.header['GUIDER'] = (str(hdu[0].header['GUIDER']+' '), "Name of guider")
            new_hdu.header['GUIEPOCH'] = (str(hdu[0].header['GUIEPOCH']+' '), "Guider epoch")
            new_hdu.header['GUIEQUIN'] = (str(hdu[0].header['GUIEQUIN']+' '), "Guider equinox")
            new_hdu.header['GUIRA'] = (str(hdu[0].header['GUIRA']+' '), "Guider RA")
            new_hdu.header['INSTPORT'] = (str(hdu[0].header['INSTPORT']), "Secondary instrument")
            new_hdu.header['LAMPID'] = (str(hdu[0].header['LAMPID']), "Calibration Lamp")
            new_hdu.header['MBX'] = (str(hdu[0].header['MBX']), "Moving Baffle X position")
            new_hdu.header['MBY'] = (str(hdu[0].header['MBY']), "Moving Baffle Y position")
            new_hdu.header['MOONANG'] = (str(hdu[0].header['MOONANG']+' '), "Angle between Moon and pointing")
            new_hdu.header['NAMPS'] = (str(hdu[0].header['NAMPS']), "Number of amplifiers used")
            new_hdu.header['OBJECT'] = (str('Master_Flat'), "Object name")
            new_hdu.header['OBSERVAT'] = (str('SALT'), "Southern African Large Telescope")
            new_hdu.header['OBSERVER']=  (str(hdu[0].header['OBSERVER']+' '), "SALT Astronomer")
            new_hdu.header['PA']= (str(hdu[0].header['PA']+' '), "Proposed position angle from N")
            new_hdu.header['PAYLTEM']= (str(hdu[0].header['PAYLTEM']), "Payload temperature")
            new_hdu.header['PELLICLE']= (str(hdu[0].header['PELLICLE']), "Pellicle position")
            new_hdu.header['PHOTOMET']= (str(hdu[0].header['PHOTOMET']+' '), "Photometric conditions")
            new_hdu.header['PM-DEC']= (str(hdu[0].header['PM-DEC']+' '), "Proper motion of the source")
            new_hdu.header['PM-RA']= (str(hdu[0].header['PM-RA']+' '), "Proper motion of the source")
            new_hdu.header['PMASKY']= (str(hdu[0].header['PMASKY']), "Pupil Mask Y position")
            new_hdu.header['PROPID']= (str(hdu[0].header['PROPID']), "SALT project ID")
            new_hdu.header['PROPOSER']= (str(hdu[0].header['PROPOSER']), "Name of PI of project")
            new_hdu.header['PUPSTA']= (str(hdu[0].header['PUPSTA']), "Pupil size at start of observation")
            new_hdu.header['RA']= (str(hdu[0].header['RA']+' '), "RA of object")
            new_hdu.header['SEEING']= (str(hdu[0].header['SEEING']+' '), "Seeing")
            new_hdu.header['SITEELEV'] = (str(1798.), "Site elevation")
            new_hdu.header['SITELAT'] = (str(-32.3795), "Geographic latitude of the observation")
            new_hdu.header['SITELONG'] = (str(20.812), "Geographic longitude of the observation")
            new_hdu.header['TELALT']= (str(hdu[0].header['TELALT']), "Telescope altitude")
            new_hdu.header['TELAZ']= (str(hdu[0].header['TELAZ']), "Telescope azimuth")
            new_hdu.header['TELDEC']= (str(hdu[0].header['TELDEC']), "Dec of telescope")
            new_hdu.header['TELDEDOT']= (str(hdu[0].header['TELDEDOT']+' '), "Rate of change of telescope Dec")
            new_hdu.header['TELEPOCH']= (str(hdu[0].header['TELEPOCH']), "Epoch of telescope pointing")
            new_hdu.header['TELEQUIN']= (str(hdu[0].header['TELEQUIN']), "Telescope Equinox")
            new_hdu.header['TELFITS']= (str(hdu[0].header['TELFITS']), "Telescope FITS header version")
            new_hdu.header['TELFOCUS']= (str(hdu[0].header['TELFOCUS']), "Interferometer reading")
            new_hdu.header['TELHA']= (str(hdu[0].header['TELHA']), "HA of telescope")
            new_hdu.header['TELPA']= (str(hdu[0].header['TELPA']+' '), "Position angle of telescope")
            new_hdu.header['TELRA']= (str(hdu[0].header['TELRA']), "RA of telescope")
            new_hdu.header['TELRADOT']= (str(hdu[0].header['TELRADOT']+' '), "Rate of change of telescope RA")
            new_hdu.header['TELTEM']= (str(hdu[0].header['TELTEM']), "Mirror air temperature")
            new_hdu.header['TRANSPAR']= (str(hdu[0].header['TRANSPAR']+' '), "Sky transparency")
            new_hdu.header['TRKPHI']= (str(hdu[0].header['TRKPHI']), "Tracker Phi")
            new_hdu.header['TRKRHO']= (str(hdu[0].header['TRKRHO']), "Tracker Rho")
            new_hdu.header['TRKTHETA']= (str(hdu[0].header['TRKTHETA']), "Tracker Theta")
            new_hdu.header['TRKX']= (str(hdu[0].header['TRKX']), "Tracker X")
            new_hdu.header['TRKY']= (str(hdu[0].header['TRKY']), "Tracker Y")
            new_hdu.header['TRKZ']= (str(hdu[0].header['TRKZ']), "Tracker Z")
            new_hdu.header['CCDNAMPS'] = (str(hdu[0].header['CCDNAMPS']), "No. of amplifiers used")
            new_hdu.header['CCDSEC'] = (str(hdu[0].header['CCDSEC']), "CCD Section")
            new_hdu.header['CCDSUM'] = (str(hdu[0].header['CCDSUM']), "On-chip binning")
            new_hdu.header['CCDTYPE'] = (str('Flat'),"Observation type")
            
            new_hdu.header['DATASEC'] = (str(hdu[0].header['DATASEC']),"Data Section")
            
            new_hdu.header['DATE-OBS'] = (str(hdu[0].header['DATE-OBS']),"Date of observation")
            new_hdu.header['MEAN_JD'] = (str(jd_mean), "Mean of input file JDs")
            new_hdu.header['DETMODE'] = (str(hdu[0].header['DETMODE']),"Detector Mode")
            new_hdu.header['DETNAM'] = (str(hdu[0].header['DETNAM']),"Detector Name")
            new_hdu.header['DETSER'] = (str(hdu[0].header['DETSER']), "Detector serial number")
            new_hdu.header['DETSWV'] = (str(hdu[0].header['DETSWV']),"Detector software version")
            new_hdu.header['EXPTIME'] = (str(hdu[0].header['EXPTIME']),"Exposure time (s)")
            new_hdu.header['GAIN'] = (str(hdu[0].header['AVG_GAIN']),"AVG CCD gain over AMPS (photons/ADU)")
            new_hdu.header['GAINSET'] = (str(hdu[0].header['GAINSET']),"Gain Setting")
            new_hdu.header['INSTRUME'] = (str('HRS'),"Instrument name: High Resolution Spectrograph")
            new_hdu.header['NCCDS'] = (str(hdu[0].header['NCCDS']),"Number of CCDs")
            new_hdu.header['NODCOUNT'] = (str(hdu[0].header['NODCOUNT']), "No. of Nod/Shuffles")
            new_hdu.header['NODPER'] = (str(hdu[0].header['NODPER']), "Nod & Shuffle Period (s)")
            new_hdu.header['NODSHUFF'] = (str(hdu[0].header['NODSHUFF']), "Nod & Shuffle enabled?")
            new_hdu.header['OBSMODE'] = (str(hdu[0].header['OBSMODE']),"Observation mode")
            new_hdu.header['OBSTYPE'] = (str('Flat'), "Observation type")
            new_hdu.header['ROSPEED'] = (str(hdu[0].header['ROSPEED']),"CCD readout speed (Hz)")
            new_hdu.header['RONOISE'] = (str(RONOISE), "Read out noise calculated from the Bias (e-)")

            new_hdu.header['TEM-AIR'] = (str(np.mean(TEM_AIR)),"HRS environment air temperature (K)")
            new_hdu.header['TEM-BCAM'] = (str(np.mean(TEM_BCAM)),"Blue camera temperature (K)")
            new_hdu.header['TEM-COLL'] = (str(np.mean(TEM_COLL)),"Collimator mount temperature (K)")
            new_hdu.header['TEM-ECH'] = (str(np.mean(TEM_ECH)),"Echelle mount temperature (K)")
            new_hdu.header['TEM-IOD'] = (str(np.mean(TEM_IOD)),"Iodine cell heater temperature (K)")
            new_hdu.header['TEM-OB'] = (str(np.mean(TEM_OB)),"Optical bench temperature (K)")
            new_hdu.header['TEM-RCAM'] = (str(np.mean(TEM_RCAM)),"Red camera temperature (K)")
            new_hdu.header['TEM-RMIR'] = (str(np.mean(TEM_RMIR)),"Red pupil mirror cell temperature (K)")
            new_hdu.header['TEM-VAC'] = (str(np.mean(TEM_VAC)),"Vacuum chamber wall temperature (K)")
            new_hdu.header['PRE-DEW'] = (str(np.mean(PRE_DEW)), "Dewar pressure (mbar)")
            new_hdu.header['PRE-VAC'] = (str(np.mean(PRE_VAC)) ,"Vacuum chamber pressure (mbar)")

            DATE_EXT=str(datetime.now(tz=pytz.UTC).strftime("%Y-%m-%d"))
            UTC_EXT = str(datetime.now(tz=pytz.UTC).strftime("%H:%M:%S.%f"))
            new_hdu.header['DATE-EXT'] = (DATE_EXT,'Date file created')
            new_hdu.header['UTC-EXT'] = (UTC_EXT,'Time file created')
            new_hdu.header['N_FILE'] = (str(n),"Number of files combined")
            new_hdu.header['HISTORY'] = ("Files used for Master: "+str(Flat_files_short))
            
            cnt_hdu = fits.ImageHDU(data=cnt.astype(np.int32),name="CNT")
            cnt_hdu.header.insert(8,('COMMENT',"Count of Flat frames used per pixel to calculate MasterFlat"))
            unc_hdu = fits.ImageHDU(data=flat_unc.astype(np.float32),name="UNC")
            unc_hdu.header.insert(8,('COMMENT',"Uncertainty of MasterFlat"))
            
            hdul = fits.HDUList([new_hdu, cnt_hdu, unc_hdu])
                    
            master_file = str(self.out_dir)+self.mode+"_Master_Flat_"+str(self.sarm)+str(yyyymmdd)+".fits"
            hdul.writeto(master_file,overwrite=True)
        
        else:
            logger.info("Reading Master Bias frame "+master_file[0]+"\n")
            master_file = master_file[0]
            
        if self.plot:  # pragma: no cover
            title = "Master Flat Frame"
            plt.title(title)
            plt.xlabel("x [pixel]")
            plt.ylabel("y [pixel]")
            bot, top = np.percentile(master_flat, (1, 99))
            plt.savefig(self.out_dir+self.mode+"_Master_Flat_Frame.png",bbox_inches='tight',dpi=600)
            plt.close()
            
        return master_file

        
