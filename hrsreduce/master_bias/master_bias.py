import logging
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from datetime import datetime
import pytz
from scipy.stats import norm
import os


from hrsreduce.utils.frame_stacker import FrameStacker

logger = logging.getLogger(__name__)


class MasterBias():

    def __init__(self,files,in_dir,out_dir,arm,night,plot):
    
        self.EXPTIME = 0.
        self.propid = "CAL_BIAS"
        self.files = files
        self.out_dir = out_dir
        self.in_dir = in_dir
        self.plot = plot
        self.arm = arm
        if self.arm == "Blu":
            self.sarm = "H"
        else:
            self.sarm = "R"
        self.night = night
    
        logger.info('Started {}'.format(self.__class__.__name__))
        
        
        
    def create_masterbias(self):

        #Test if Master Bias exists
        master_file = glob.glob(self.out_dir+"Master_Bias_"+self.sarm+self.night+".fits")

        if len(master_file) == 0:

            #Open files to check if they are the correct EXPTIME (0) and PROPID (CAL_BIAS)
            Bias_files = []
            Bias_files_short = []
            for file in self.files:
                with fits.open(file) as hdu:
                    if (hdu[0].header["EXPTIME"] == self.EXPTIME and hdu[0].header["PROPID"] == self.propid):
                        Bias_files.append(file)
                        Bias_files_short.append(file.removeprefix(self.out_dir))
                        gain = hdu[0].header["AVG_GAIN"]


            if len(Bias_files) <1:
                logger.error ("\n   !!! No correct Bias files found in night %s for arm %s. Check night and arm. Exiting.\n",(self.night,self.arm))
                exit()
                
            n = len(Bias_files)
            
            #TEST KPF IMPLIMENTATION
            bias_data = []
            jds = []
            TEM_AIR = []
            TEM_BCAM = []
            TEM_COLL = []
            TEM_ECH = []
            TEM_IOD = []
            TEM_OB = []
            TEM_RCAM = []
            TEM_RMIR = []
            TEM_VAC = []
            PRE_DEW = []
            PRE_VAC = []
            OS = []
            
            for file in Bias_files:
                with fits.open(file) as hdu:
                    bias_data.append(hdu[0].data)
                    jds.append(float(hdu[0].header["JD"]))
                    TEM_AIR.append(float(hdu[0].header["TEM-AIR"]))
                    TEM_BCAM.append(float(hdu[0].header["TEM-BCAM"]))
                    TEM_COLL.append(float(hdu[0].header["TEM-COLL"]))
                    TEM_ECH.append(float(hdu[0].header["TEM-ECH"]))
                    TEM_IOD.append(float(hdu[0].header["TEM-IOD"]))
                    TEM_OB.append(float(hdu[0].header["TEM-OB"]))
                    TEM_RCAM.append(float(hdu[0].header["TEM-RCAM"]))
                    TEM_RMIR.append(float(hdu[0].header["TEM-RMIR"]))
                    TEM_VAC.append(float(hdu[0].header["TEM-VAC"]))
                    PRE_DEW.append(float(hdu[0].header["PRE-DEW"]))
                    PRE_VAC.append(float(hdu[0].header["PRE-VAC"]))
                    OS.append(float(hdu[0].header["OS_MEAN"]))
                    
            bias_data = np.array(bias_data)
            fs=FrameStacker(bias_data,2.1,logger)
            avg,var,cnt,unc = fs.compute()
            if self.plot:
                plt.xlabel("x [pixel]")
                plt.ylabel("y [pixel]")
                bot, top = np.percentile(avg, (1, 99))
                plt.imshow(avg, vmin=bot, vmax=top, origin="lower")
                plt.title("Master Bias Frame")
                plt.savefig(self.out_dir+"Master_Bias_Frame.png",bbox_inches='tight',dpi=600)
                plt.close()
            
            #Fit a gaussian to the data to find the read noise (2*sigma)
            (mu, sigma) = norm.fit(avg)

#            nbins=(np.max(avg2)-np.min(avg2)).astype(np.int32)
            
#            # the histogram of the data
#            n, bins, patches = plt.hist(avg2, nbins)
#
#            # add a 'best fit' line
#            y = norm.pdf( bins, mu, sigma)
#            l = plt.plot(bins, y*2000, 'r--', linewidth=2)
#
#            #plot
#            plt.xlabel('Smarts')
#            plt.ylabel('Probability')
#            plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
#            plt.grid(True)
#
#            plt.show()
#
#            avg2=avg.astype(np.int32)
#            plt.hist(avg2)
#            plt.show()
            
            
#            if n == 1:
#                #if there is just one element compare it with itself, not really useful, but it works
#                logger.warning("Only 1 suitable bias frame found for night %s and arm %s",(self.night,self.arm))
#                list1 = list2 = Bias_files
#                n = 2
#            else:
#                list1, list2 = Bias_files[: n // 2], Bias_files[n // 2 :]

#            # Lists of images.
#            n1 = len(list1)
#            n2 = len(list2)
#
#            bias_concat1 = []
#            bias_concat2 = []
#            Bias_files_short = []
#            # Separate images in two groups.
#            for file in list1:
#                Bias_files_short.append(file.lstrip(self.in_dir))
#                with fits.open(file) as hdu:
#                    bias_concat1.append(hdu[0].data.astype(float))
#            bias1 = np.median(bias_concat1, axis=0)
#
#            for file in list2:
#                Bias_files_short.append(file.lstrip(self.in_dir))
#                with fits.open(file) as hdu:
#                    bias_concat2.append(hdu[0].data.astype(float))
#            bias2 = np.median(bias_concat2, axis=0)
#
#            #Plot images side by side
#            if self.plot:
#                f, axarr = plt.subplots(1,2)
#                axarr[0].imshow(bias1, vmin=np.min(bias1), vmax=np.max(bias1), origin="lower")
#                axarr[1].imshow(bias2, vmin=np.min(bias1), vmax=np.max(bias1), origin="lower")
#                plt.show()
#
#        #    # Make sure we know the gain. Since we can have multiple amplifies, take the mean of the possible values
#            hdu = fits.open(list1[0])
#            gain = hdu[0].header["GAIN"]
#            if self.arm =="Blu":
#                gain1 = float(gain.split()[0])
#                gain2 = float(gain.split()[1])
#                gain = gain1
#            if self.arm == "Red":
#                gain1 = float(gain.split()[0])
#                gain2 = float(gain.split()[1])
#                gain3 = float(gain.split()[2])
#                gain4 = float(gain.split()[3])
#                gain = gain1 #np.mean([gain1,gain2,gain3,gain4])
#
#            # Construct normalized sum.
#    #        bias1 *= gain
#    #        bias2 *= gain
#            bias = (((bias1 * n1 + bias2 * n2) / n))
#
#            # Compute noise in difference image by fitting Gaussian to distribution.
#            diff = 0.5 * (bias1 - bias2)
#            if np.min(diff) != np.max(diff):
#                crude = np.ma.median(np.abs(diff))  # estimate of noise
#                hmin = -5.0 * crude
#                hmax = +5.0 * crude
#                bin_size = np.clip(2 / n, 0.5, None)
#                nbins = int((hmax - hmin) / bin_size)
#
#                h, _ = np.histogram(diff, range=(hmin, hmax), bins=nbins)
#                xh = hmin + bin_size * (np.arange(0.0, nbins) + 0.5)
#
#                hfit, par = gaussfit(xh, h)
#                noise = abs(par[2])  # noise in diff, bias
#
#                # Determine where wings of distribution become significantly non-Gaussian.
#                contam = (h - hfit) / np.sqrt(np.clip(hfit, 1, None))
#                imid = np.where(abs(xh) < 2 * noise)
#                consig = np.std(contam[imid])
#
#                smcontam = gaussbroad(xh, contam, 0.1 * noise)
#                igood = np.where(smcontam < 3 * consig)
#                gmin = np.min(xh[igood])
#                gmax = np.max(xh[igood])
#
#                # Find and fix bad pixels.
#                ibad = np.where((diff <= gmin) | (diff >= gmax))
#                nbad = len(ibad[0])
#
#                bias[ibad] = np.clip(bias1[ibad], None, bias2[ibad])
#                bias = np.int16(bias)
#
#                # Compute read noise.
#                biasnoise = noise
#                bgnoise = biasnoise * np.sqrt(n)
#
#                # Print diagnostics.
#                if Plot =="True":
#                    print("change in bias between image sets= ",  np.abs(par[1])," electons")
#                    print("measured background noise per image=", bgnoise)
#                    print("background noise in combined image=", biasnoise)
#                    print("Number of bad pixels fixed %i", nbad)
#
#            else:
#                diff = 0
#                biasnoise = 1.0
#                nbad = 0

            #Write the master bias to file with approraite header info
            new_hdu = fits.PrimaryHDU(data=avg.astype(np.float32))
            
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
            new_hdu.header['OBJECT'] = (str('Master_Bias'), "Object name")
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
            new_hdu.header['CCDTYPE'] = (str('Bias'),"Observation type")
            
            new_hdu.header['DATASEC'] = (str(hdu[0].header['DATASEC']),"Data Section")
            
            new_hdu.header['DATE-OBS'] = (str(hdu[0].header['DATE-OBS']),"Date of observation")
            new_hdu.header['MEAN_JD'] = (str(np.mean(jds)), "Mean of input file JDs")
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
            new_hdu.header['OBSTYPE'] = (str('Bias'), "Observation type")
            new_hdu.header['ROSPEED'] = (str(hdu[0].header['ROSPEED']),"CCD readout speed (Hz)")
            new_hdu.header['RONOISE'] = (str(sigma*2.), "Read out noise calculated from the Bias (e-)")

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
            
            new_hdu.header['OS_MEAN'] = (str(np.mean(OS)) ,"Average overscan region of inputs")
            new_hdu.header['AVG_BIAS'] = (str(np.mean(avg)) ,"Average bias count in Master")

            DATE_EXT=str(datetime.now(tz=pytz.UTC).strftime("%Y-%m-%d"))
            UTC_EXT = str(datetime.now(tz=pytz.UTC).strftime("%H:%M:%S.%f"))
            new_hdu.header['DATE-EXT'] = (DATE_EXT,'Date file created')
            new_hdu.header['UTC-EXT'] = (UTC_EXT,'Time file created')
            new_hdu.header['N_FILE'] = (str(n),"Number of files combined")
            new_hdu.header['HISTORY'] = ("Files used for Master: "+str(Bias_files_short))
            
            cnt_hdu = fits.ImageHDU(data=cnt.astype(np.int32),name="CNT")
            cnt_hdu.header.insert(8,('COMMENT',"Count of Bias frames used per pixel to calculate MasterBias"))
            unc_hdu = fits.ImageHDU(data=unc.astype(np.float32),name="UNC")
            unc_hdu.header.insert(8,('COMMENT',"Uncertainty of MasterBias"))
            
            hdul = fits.HDUList([new_hdu, cnt_hdu, unc_hdu])
                    
            hdul.writeto(str(self.out_dir)+"Master_Bias_"+str(self.sarm)+str(self.night)+".fits",overwrite=True)
            
            master_file = str(self.out_dir)+"Master_Bias_"+str(self.sarm)+str(self.night)+".fits"
            
            hdu.close
        
        else:
            logger.info("Reading Master Bias frame "+master_file[0]+"\n")
            master_file = master_file[0]
        
        return master_file

