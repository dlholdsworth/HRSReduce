import logging
import matplotlib.pyplot as plt

import multiprocessing as mp
import pandas as pd
import numpy as np
import os.path
from astropy.io import fits

import barycorrpy
from astropy.coordinates import SkyCoord
from astropy.time import Time

# Local dependencies
from hrsreduce.extraction.alg import SpectralExtractionAlg

#TODO: Run a test to see if the file has been processed (check FITS extension) to avoid running many times. (Not normally an issue, but testing/development will be slow otherwise)
#TODO: Add in VAR details to get a realistic error


logger = logging.getLogger(__name__)

class SpectralExtraction():

    def __init__(self, sci_frame, flat_frame,arc_frame,order_trace_file, sarm,mode,base_dir):

        self.input_spectrum = sci_frame
        self.input_flat = flat_frame
        self.arc_frame = arc_frame
        self.order_trace_file = order_trace_file
        self.rectification_method = 2 # Normal to the order trace: 0, vertical:1, none: 2
        self.extraction_method = 0 # Optimal extraction: 0, sum: 1, no: 2
        self.arm = sarm
        self.mode = mode
        self.base_dir = base_dir
        self.logger = logger
        self.outlier_rejection = False
        spec_no_bk = sci_frame
        var_ext = 'VAR'
        
        # start a logger
        self.logger.info('Started SpectralExtraction')

        self.order_trace_data = None
        if self.order_trace_file:
            self.order_trace_data = pd.read_csv(self.order_trace_file, header=0, index_col=1)
            df_result_out = []
            for ord in range(self.order_trace_data.shape[0]):
                df_result = {}
                df_result['Coeff0'] = self.order_trace_data['Coeff0'][ord]
                df_result['BottomEdge'] = self.order_trace_data['BottomEdge'][ord]
                df_result['TopEdge'] = self.order_trace_data['TopEdge'][ord]
                df_result['X1'] = self.order_trace_data['X1'][ord]
                df_result['X2'] = self.order_trace_data['X2'][ord]
                df_result_out.append(df_result)
            
            self.order_trace_data = pd.DataFrame(df_result_out)
            poly_degree = self.order_trace_data.shape[1]-5
            origin = [0,0]
            order_trace_header = {'STARTCOL': origin[0], 'STARTROW': origin[1], 'POLY_DEG': poly_degree}

        # Open the data files
        with fits.open(self.input_spectrum) as hdl:
        
            self.spec_header = hdl[0].header
            self.spec_flux = hdl['STRAIGHT'].data
            var_data = hdl['VAR'].data
            data_type = self.spec_header['CCDTYPE']
            
            if data_type == 'Science' or data_type == 'Arc':
                try:
                    test = hdl['FIBRE_P']
                    self.Done_Extraction = True
                except:
                    self.Done_Extraction = False
            elif data_type == 'Flat':
                try:
                    test = hdl['BLAZE_P']
                    self.Done_Extraction = True
                except:
                    self.Done_Extraction = False

        with fits.open(self.input_flat) as hdl:
            self.flat_header = hdl[0].header
            if data_type == 'Science':
                self.flat_data = hdl['NORMALISED'].data
                self.spec_flux = self.spec_flux #/ self.flat_data
            if data_type == 'Arc':
                self.flat_data = hdl['STRAIGHT'].data / hdl['STRAIGHT'].data
            else:
                self.flat_data = hdl['STRAIGHT'].data

            


        try:
            self.alg = SpectralExtractionAlg(self.flat_data,
                                        self.flat_header,
                                        self.spec_flux,
                                        self.spec_header,
                                        self.order_trace_data,
                                        order_trace_header,
                                        self.input_spectrum,
                                        config=None, logger=self.logger,
                                        rectification_method=self.rectification_method,
                                        extraction_method=self.extraction_method,
                                        ccd_index=None,
                                        orderlet_names=None,
                                        total_order_per_ccd=None,
                                        clip_file=None,
                                        do_outlier_rejection = self.outlier_rejection,
                                        outlier_flux=None,
                                        var_data=var_data)
        except Exception as e:
            print(e)
            self.alg = None



    def extraction(self):
        """
        Perform spectral extraction by calling method `extract_spectrum` from SpectralExtractionAlg and create a dataframe to contain the analysis result.

        Returns:
            File name containing extracted results.

        """

        if self.logger:
            self.logger.info("SpectralExtraction: extracting orders ...")

        if self.alg is None:
            if self.logger:
                self.logger.info("SpectralExtraction: no extension data, order trace data or improper header.")
            return None
            
        #If done extraction skip

        if not self.Done_Extraction:
            
            all_o_sets = []
            first_trace_at = []
            s_order = 0

            self.o_set = np.arange(self.order_trace_data.shape[0])
     
            n_ord = int(self.order_trace_data.shape[0] / 2)
            
            for order_name in self.o_set:
                o_set, f_idx = self.get_order_set(s_order)
                all_o_sets.append(o_set)
                first_trace_at.append(f_idx)

            good_result = True

            #Run orders (pairs of fibres) in parallel.
            manager = mp.Manager()
            return_dict = manager.dict()
            processes = [mp.Process(target=self.alg.extract_spectrum, args=(o_set[(2*i):(2*i)+2],i,return_dict, self.input_spectrum)) for i in range(n_ord)]
            for process in processes:
                process.start()
            for process in processes:
                process.join()


            data_df = pd.DataFrame.from_dict(return_dict[0]['spectral_extraction_result'])
            data_P = pd.DataFrame(columns=data_df.columns)
            data_O = pd.DataFrame(columns=data_df.columns)
            for i in range(0,n_ord):
                data_df = pd.concat([data_df,pd.DataFrame.from_dict(return_dict[i]['spectral_extraction_result'])],ignore_index=True)
                tmp = pd.DataFrame.from_dict(return_dict[i]['spectral_extraction_result'])
                data_P = pd.concat([data_P, tmp.iloc[[1]]],axis=0,ignore_index=True)
                data_O = pd.concat([data_O, tmp.iloc[[0]]],axis=0,ignore_index=True)


            good_result = good_result and data_df is not None

            if not good_result and self.logger:
                self.logger.info("SpectralExtraction: no spectrum extracted")
            elif good_result and self.logger:
                self.logger.info("SpectralExtraction: Receipt written")
                self.logger.info("SpectralExtraction: Done for {} orders!".format(int(n_ord*2)))
                
    #            out_file=os.path.splitext(str(os.path.dirname(self.input_spectrum))+"/HRS_E_"+str(os.path.basename(self.input_spectrum)))[0]+'.csv'
    #            data_df.to_csv(out_file)
                
                data_P_np=data_P.to_numpy()
                data_O_np=data_O.to_numpy()
                P_VAR = np.array(data_P_np)
                P_VAR = np.absolute(P_VAR)
                O_VAR = np.array(data_O_np)
                O_VAR = np.absolute(O_VAR)
                
                try:
                    with fits.open(self.arc_frame) as hdul:
                        sciwave_P = hdul['WAVE_P'].data
                        sciwave_O = hdul['WAVE_O'].data
                    wave = True
                except:
                    wave = False

                try:
                    with fits.open(self.input_flat) as hdul:
                        Blaze_P = hdul['BLAZE_P'].data
                        Blaze_O = hdul['BLAZE_O'].data
                    blaze = True
                except:
                    blaze = False
                    

                   
                with fits.open(self.input_spectrum) as hdul:
                    if hdul[0].header["OBJECT"] == "Master_Flat":
                        Ext_ords_P = fits.ImageHDU(data=data_P, name="BLAZE_P")
                    else:
                        Ext_ords_P = fits.ImageHDU(data=data_P, name="FIBRE_P")
                    Ext_ords_P.header["NORDS"] =  ((data_P.shape[0]),"Number of extracted orders")
                    Ext_ords_P.header["E_MTHD"] = ((self.extraction_method),"Extraction Method. 0: optimal, 1: sum")
                    Ext_ords_P.header["R_MTHD"] = ((self.rectification_method), "Rectification Method. 0: Norm, 1: Vert, 2: None")
                    Ext_ords_P.header["FLATFILE"] = (str(os.path.basename(self.input_flat)),"Input flat for extraction")
                    Ext_ords_P.header["ORDFILE"] = (str(os.path.basename(self.order_trace_file)),"Order trace file")
                    hdul.append(Ext_ords_P)
                    if hdul[0].header["OBJECT"] == "Master_Flat":
                        Ext_ords_P_VAR = fits.ImageHDU(data=P_VAR, name="BLAZE_P_VAR")
                    else:
                        Ext_ords_P_VAR = fits.ImageHDU(data=P_VAR, name="FIBRE_P_VAR")
                    hdul.append(Ext_ords_P_VAR)
                    if hdul[0].header["OBJECT"] == "Master_Flat":
                        Ext_ords_O = fits.ImageHDU(data=data_O, name="BLAZE_O")
                    else:
                        Ext_ords_O = fits.ImageHDU(data=data_O, name="FIBRE_O")
                    Ext_ords_O.header["NORDS"] =  ((data_O.shape[0]),"Number of extracted orders")
                    Ext_ords_O.header["E_MTHD"] = ((self.extraction_method),"Extraction Method. 0: optimal, 1: sum")
                    Ext_ords_O.header["R_MTHD"] = ((self.rectification_method), "Rectification Method. 0: Norm, 1: Vert, 2: None")
                    Ext_ords_O.header["FLATFILE"] = (str(os.path.basename(self.input_flat)),"Input flat for extraction")
                    Ext_ords_O.header["ORDFILE"] = (str(os.path.basename(self.order_trace_file)),"Order trace file")
                    hdul.append(Ext_ords_O)
                    if hdul[0].header["OBJECT"] == "Master_Flat":
                        Ext_ords_O_VAR = fits.ImageHDU(data=O_VAR, name="BLAZE_O_VAR")
                    else:
                        Ext_ords_O_VAR = fits.ImageHDU(data=O_VAR, name="FIBRE_O_VAR")
                    hdul.append(Ext_ords_O_VAR)
                    
                    if wave:
                        Ext_wave_P = fits.ImageHDU(data=sciwave_P, name="WAVE_P")
                        hdul.append(Ext_wave_P)
                        Ext_wave_O = fits.ImageHDU(data=sciwave_O, name="WAVE_O")
                        hdul.append(Ext_wave_O)
                        hdul[0].header['MSTRWAVE'] = (str(os.path.basename(self.arc_frame)),"Arc file")
                    
                    if blaze:
                        Ext_ords_P_BLZ = fits.ImageHDU(data=Blaze_P, name="BLAZE_P")
                        hdul.append(Ext_ords_P_BLZ)
                        Ext_ords_O_BLZ = fits.ImageHDU(data=Blaze_O, name="BLAZE_O")
                        hdul.append(Ext_ords_O_BLZ)
                        
                    if hdul[0].header["CCDTYPE"] == 'Science':
                        
                        #Calculate and add the Barycentric correction value
                        obs_date = hdul[0].header["DATE-OBS"]
                        ut = hdul[0].header["TIME-OBS"]
                        if obs_date is not None and ut is not None:
                            obs_date = f"{obs_date}T{ut}"
                        fwmt = hdul[0].header["EXP-MID"]
                        et = hdul[0].header["EXPTIME"]
                        
                        if fwmt > 0.:
                            mid = float(fwmt)/86400.
                        else:
                            mid =  float(float(et)/2./86400.)
        
                        jd = Time(obs_date,scale='utc',format='isot').jd + mid
        
                        lat = -32.3722685109
                        lon = 20.806403441
                        alt = hdul[0].header["SITEELEV"]
                        object = hdul[0].header["OBJECT"]
                        
                        try:
                            BCV =(barycorrpy.get_BC_vel(JDUTC=jd,starname = object, lat=lat, longi=lon, alt=alt, leap_update=True,ephemeris='de430'))

                            BJD = barycorrpy.JDUTC_to_BJDTDB(jd, starname = object, lat=lat, longi=lon, alt=alt)
                            Sub_ms = 'True'
                            
                        except:
                            co = SkyCoord(hdul[0].header["RA"],hdul[0].header["DEC"],unit=(u.hourangle, u.deg))
                            
                            BCV =(barycorrpy.get_BC_vel(JDUTC=jd,ra=co.ra.degree, dec=co.dec.degree, lat=lat, longi=lon, alt=alt, leap_update=True,ephemeris='de430'))

                            BJD = barycorrpy.JDUTC_to_BJDTDB(jd, ra=co.ra.degree, dec=co.dec.degree, lat=lat, longi=lon, alt=alt)
                        
                            Sub_ms = 'False'
                        
                        BARYCORR = BCV[0]/1000.
                        BARYJD = BJD[0]
                        
                        hdul[0].header["BARYRV"] = (str(BARYCORR[0]),"Barycentric RV from barycorrpy")
                        hdul[0].header["BRV_PRE"] = (str(Sub_ms),"BRV precision <3 m/s precision")
                        hdul[0].header["BJD"] = (str(BARYJD[0]),"BJD mid obs (using FWMT if not 0)")
                        
                        
                    hdul.writeto(self.input_spectrum,overwrite='True')
        
        else:
            self.logger.info("SpectralExtraction: Already Extracted")
            
        
    def get_order_set(self, s_order):
        if self.o_set.size > 0:
            e_order = self.o_set.size

            o_set_ary = self.o_set[0:e_order] + s_order
            valid_idx = np.where(o_set_ary >= 0)[0]
            first_idx = valid_idx[0] if valid_idx.size > 0 else -1

            return o_set_ary, first_idx
        else:
            return o_set
