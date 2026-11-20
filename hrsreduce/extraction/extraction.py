import logging
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os.path
from astropy.io import fits

from astropy import units as u

import barycorrpy
from astropy.coordinates import SkyCoord
from astropy.time import Time

# Local dependencies
from hrsreduce.extraction.alg import SpectralExtractionAlg

#TODO: Run a test to see if the file has been processed (check FITS extension) to avoid running many times. (Not normally an issue, but testing/development will be slow otherwise)
#TODO: Add in VAR details to get a realistic error


logger = logging.getLogger(__name__)


MAX_START_OFFSET_S = 30.        # normal TIME-OBS - ENVMJD is +3 to +6 s; stale TIME-OBS is off by -130 s or more
MAX_STALE_OFFSET_S = 12 * 3600.  # a TIME-OBS copied from the previous exposure is from the same night


def observation_start(header, logger=None, max_offset_s=MAX_START_OFFSET_S, max_stale_s=MAX_STALE_OFFSET_S):
    """Return (start, source, offset_s): the UTC start of the exposure as an astropy Time.

    In some raw frames (seen in mid-2018, mostly the Blue arm) DATE-OBS/TIME-OBS, UTC-OBS, LST-OBS and JD are
    copied from the previous exposure. ENVMJD (it holds a JD, despite the name) is written separately and is
    then correct, normally a few seconds before TIME-OBS. A copied TIME-OBS is always EARLIER than ENVMJD (by one
    exposure plus overheads: -131 to -2740 s in 2018). ENVMJD can be stale as well: on 2024-05-14 it stays at one
    value all night, so TIME-OBS is LATER than ENVMJD by hours. Hence:

        offset_s = TIME-OBS - ENVMJD
        |offset_s| <= max_offset_s                  -> TIME-OBS (normal)
        -max_stale_s <= offset_s < -max_offset_s    -> ENVMJD   (TIME-OBS copied from the previous exposure)
        offset_s > max_offset_s                     -> TIME-OBS (ENVMJD stale), source 'TIME-OBS; ENVMJD stale'
        offset_s < -max_stale_s                     -> TIME-OBS (not the same night), source 'TIME-OBS; unconfirmed'

    Every departure from the normal case is logged as a warning, and the source is written to TSTARTSR with
    the offset in TOBSOFF. offset_s is None if either time is missing. An ENVMJD of 0 or less counts as missing.
    """
    t_hdr = None
    date, ut = header.get("DATE-OBS"), header.get("TIME-OBS")
    if date and ut:
        t_hdr = Time(f"{date}T{ut}", scale="utc", format="isot")

    try:
        env = float(header.get("ENVMJD"))
    except (TypeError, ValueError):
        env = None
    if env is not None and env <= 0.:
        env = None
    if env is not None and env < 2400000.:
        env += 2400000.5  # a true MJD, in case SALT ever writes one
    t_env = Time(env, format="jd", scale="utc") if env is not None else None

    if t_hdr is None and t_env is None:
        raise ValueError("No usable start time: DATE-OBS/TIME-OBS and ENVMJD are both missing")
    if t_hdr is None:
        if logger: logger.warning("observation_start: no DATE-OBS/TIME-OBS, using ENVMJD")
        return t_env, "ENVMJD", None
    if t_env is None:
        return t_hdr, "TIME-OBS", None

    offset_s = (t_hdr.jd - t_env.jd) * 86400.
    if abs(offset_s) <= max_offset_s:
        return t_hdr, "TIME-OBS", offset_s
    if -max_stale_s <= offset_s < -max_offset_s:
        if logger:
            logger.warning("observation_start: TIME-OBS %sT%s is %.1f s before ENVMJD %s; TIME-OBS looks copied from "
                           "the previous exposure, using ENVMJD (%s)" % (date, ut, offset_s, header.get("ENVMJD"), t_env.isot))
        return t_env, "ENVMJD", offset_s
    if offset_s > max_offset_s:
        if logger:
            logger.warning("observation_start: TIME-OBS %sT%s is %.1f s after ENVMJD %s; ENVMJD looks stale, "
                           "keeping TIME-OBS" % (date, ut, offset_s, header.get("ENVMJD")))
        return t_hdr, "TIME-OBS; ENVMJD stale", offset_s
    if logger:
        logger.warning("observation_start: TIME-OBS %sT%s is %.1f s before ENVMJD %s, more than %.0f h; keeping "
                       "TIME-OBS, unconfirmed" % (date, ut, offset_s, header.get("ENVMJD"), max_stale_s / 3600.))
    return t_hdr, "TIME-OBS; unconfirmed", offset_s


def frame_mid_jd(header):
    """Mid-exposure JD (UTC) of a frame from its header: the start of observation_start plus half of EXPTIME.
    None when the start time is unknown."""
    try:
        t, _, _ = observation_start(header, None)
    except Exception:
        return None
    try:
        et = float(header.get("EXPTIME", 0.) or 0.)
    except (TypeError, ValueError):
        et = 0.
    return t.jd + et / 2. / 86400.


def arc_time_offset_h(sci_header, arc_file):
    """Science minus arc mid-exposure time in hours (the arc file is a Master Wave, a copy of the arc frame), or None."""
    try:
        arc_hdr = fits.getheader(arc_file)
    except Exception:
        return None
    a, s = frame_mid_jd(arc_hdr), frame_mid_jd(sci_header)
    return None if a is None or s is None else (s - a) * 24.


class SpectralExtraction():

    def __init__(self, sci_frame, flat_frame,arc_frame,order_trace_file, sarm,mode,base_dir,extraction_method=0):

        self.input_spectrum = sci_frame
        self.input_flat = flat_frame
        self.arc_frame = arc_frame
        self.order_trace_file = order_trace_file
        self.rectification_method = 2 # Normal to the order trace: 0, vertical:1, none: 2
        self.extraction_method = extraction_method #0 # Optimal extraction: 0, sum: 1, no: 2
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
            # per-pixel variance in the geometry of the straightened image: RECT_VAR (propagated through the
            # rectification) when present, otherwise the raw-frame VAR
            var_data = hdl['RECT_VAR'].data if 'RECT_VAR' in hdl else hdl['VAR'].data
            data_type = self.spec_header['CCDTYPE']
            
            if data_type == 'Science' or data_type == 'Arc':
                try:
                    test = hdl['FIBRE_U']
                    self.Done_Extraction = True
                except:
                    self.Done_Extraction = False
            elif data_type == 'Flat':
                try:
                    test = hdl['BLAZE_U']
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

            # Extract the orders (pairs of fibres) one after the other. Up to 1.3.7 this used a multiprocessing pool,
            # but each task pickled the whole extraction object, so ran 4-5 times slower than this loop, and
            # could hang for good in pool.join()
            
            return_dict = {}
            for i in range(n_ord):
                _, return_dict[i], _ = self.alg.extract_spectrum(o_set[(2*i):(2*i)+2], i, {}, self.input_spectrum)

            data_df = pd.DataFrame.from_dict(return_dict[0]['spectral_extraction_result'],dtype=float)
            data_U = pd.DataFrame(columns=data_df.columns,dtype=float)
            data_L = pd.DataFrame(columns=data_df.columns,dtype=float)
            var_U = pd.DataFrame(columns=data_df.columns,dtype=float)
            var_L = pd.DataFrame(columns=data_df.columns,dtype=float)
            have_var = all('spectral_extraction_variance' in return_dict[i] for i in range(0,n_ord))
            
            for i in range(0,n_ord):
                data_df = pd.concat([data_df,pd.DataFrame.from_dict(return_dict[i]['spectral_extraction_result'],dtype=float)],ignore_index=True)
                tmp = pd.DataFrame.from_dict(return_dict[i]['spectral_extraction_result'],dtype=float)
                data_U = pd.concat([data_U, tmp.iloc[[1]]],axis=0,ignore_index=True)
                data_L = pd.concat([data_L, tmp.iloc[[0]]],axis=0,ignore_index=True)
                if have_var:
                    tmpv = pd.DataFrame.from_dict(return_dict[i]['spectral_extraction_variance'],dtype=float)
                    var_U = pd.concat([var_U, tmpv.iloc[[1]]],axis=0,ignore_index=True)
                    var_L = pd.concat([var_L, tmpv.iloc[[0]]],axis=0,ignore_index=True)
                

            good_result = good_result and data_df is not None

            if not good_result and self.logger:
                self.logger.info("SpectralExtraction: no spectrum extracted")
            elif good_result and self.logger:
                self.logger.info("SpectralExtraction: Receipt written")
                self.logger.info("SpectralExtraction: Done for {} orders!".format(int(n_ord*2)))
                
                data_U_np=data_U.to_numpy()
                data_L_np=data_L.to_numpy()
                # variance of the extracted flux, propagated from the per-pixel variance of the straightened image through
                # the extraction weights; the earlier placeholder |flux| remains only as a fallback when no variance came back
                U_VAR = var_U.to_numpy() if have_var else None
                L_VAR = var_L.to_numpy() if have_var else None
                if U_VAR is None or L_VAR is None or not np.isfinite(U_VAR).any() or not np.isfinite(L_VAR).any():
                    if self.logger:
                        self.logger.warning("SpectralExtraction: no propagated variance available, writing |flux| as the variance")
                    U_VAR = np.absolute(np.array(data_U_np))
                    L_VAR = np.absolute(np.array(data_L_np))
                
                try:
                    with fits.open(self.arc_frame) as hdul:
                        sciwave_U = hdul['WAVE_U'].data
                        sciwave_L = hdul['WAVE_L'].data
                    wave = True
                except:
                    wave = False

                try:
                    with fits.open(self.input_flat) as hdul:
                        Blaze_U = hdul['BLAZE_U'].data
                        Blaze_L = hdul['BLAZE_L'].data
                    blaze = True
                except:
                    blaze = False
                    

                   
                with fits.open(self.input_spectrum) as hdul:
                    if hdul[0].header["OBJECT"] == "Master_Flat":
                        Ext_ords_U = fits.ImageHDU(data=data_U, name="BLAZE_U")
                    else:
                        Ext_ords_U = fits.ImageHDU(data=data_U, name="FIBRE_U")
                    Ext_ords_U.header["NORDS"] =  ((data_U.shape[0]),"Number of extracted orders")
                    Ext_ords_U.header["E_MTHD"] = ((self.extraction_method),"Extraction Method. 0: optimal, 1: sum")
                    Ext_ords_U.header["R_MTHD"] = ((self.rectification_method), "Rectification Method. 0: Norm, 1: Vert, 2: None")
                    Ext_ords_U.header["FLATFILE"] = (str(os.path.basename(self.input_flat)),"Input flat for extraction")
                    Ext_ords_U.header["ORDFILE"] = (str(os.path.basename(self.order_trace_file)),"Order trace file")
                    hdul.append(Ext_ords_U)
                    if hdul[0].header["OBJECT"] == "Master_Flat":
                        Ext_ords_U_VAR = fits.ImageHDU(data=U_VAR, name="BLAZE_U_VAR")
                    else:
                        Ext_ords_U_VAR = fits.ImageHDU(data=U_VAR, name="FIBRE_U_VAR")
                    hdul.append(Ext_ords_U_VAR)
                    if hdul[0].header["OBJECT"] == "Master_Flat":
                        Ext_ords_L = fits.ImageHDU(data=data_L, name="BLAZE_L")
                    else:
                        Ext_ords_L = fits.ImageHDU(data=data_L, name="FIBRE_L")
                    Ext_ords_L.header["NORDS"] =  ((data_L.shape[0]),"Number of extracted orders")
                    Ext_ords_L.header["E_MTHD"] = ((self.extraction_method),"Extraction Method. 0: optimal, 1: sum")
                    Ext_ords_L.header["R_MTHD"] = ((self.rectification_method), "Rectification Method. 0: Norm, 1: Vert, 2: None")
                    Ext_ords_L.header["FLATFILE"] = (str(os.path.basename(self.input_flat)),"Input flat for extraction")
                    Ext_ords_L.header["ORDFILE"] = (str(os.path.basename(self.order_trace_file)),"Order trace file")
                    hdul.append(Ext_ords_L)
                    if hdul[0].header["OBJECT"] == "Master_Flat":
                        Ext_ords_L_VAR = fits.ImageHDU(data=L_VAR, name="BLAZE_L_VAR")
                    else:
                        Ext_ords_L_VAR = fits.ImageHDU(data=L_VAR, name="FIBRE_L_VAR")
                    hdul.append(Ext_ords_L_VAR)
                    
                    if wave:
                        Ext_wave_U = fits.ImageHDU(data=sciwave_U, name="WAVE_U")
                        hdul.append(Ext_wave_U)
                        Ext_wave_L = fits.ImageHDU(data=sciwave_L, name="WAVE_L")
                        hdul.append(Ext_wave_L)
                        hdul[0].header['MSTRWAVE'] = (str(os.path.basename(self.arc_frame)),"Arc file")
                        if hdul[0].header.get("CCDTYPE") == 'Science':
                            dt = arc_time_offset_h(hdul[0].header, self.arc_frame)
                            hdul[0].header['ARCDT'] = (round(dt, 4) if dt is not None else 'nan',
                                                       "[h] science minus arc mid-exposure time")
                    
                    if blaze:
                        Ext_ords_U_BLZ = fits.ImageHDU(data=Blaze_U, name="BLAZE_U")
                        hdul.append(Ext_ords_U_BLZ)
                        Ext_ords_L_BLZ = fits.ImageHDU(data=Blaze_L, name="BLAZE_L")
                        hdul.append(Ext_ords_L_BLZ)
                        
                    if hdul[0].header["CCDTYPE"] == 'Science':
                        #Calculate and add the Barycentric correction value (never stops the run; see barycentric)
                        self.barycentric(hdul[0].header)

                    hdul.writeto(self.input_spectrum,overwrite='True')
        
        else:
            self.logger.info("SpectralExtraction: Already Extracted")


    def barycentric(self, hdr):
        """Write BARYRV, BRV_PRE, BJD, BARYSRC and the start-time cards of a science frame into hdr. Never raises.

        The target is resolved by name (SIMBAD, through barycorrpy); if that fails, the header RA/DEC (J2000) are used.
        If those are missing or unusable too, or the start time cannot be found, BARYRV and BJD are written as nan and
        BARYSRC says why, so that the frame is still reduced. TELRA/TELDEC are deliberately not used (the telescope
        pointing can be far from the target).
        """
        name = os.path.basename(self.input_spectrum)

        def write(bcv, bjd, sub_ms, src):
            hdr["BARYRV"] = (str(bcv), "Barycentric RV from barycorrpy")
            hdr["BRV_PRE"] = (str(sub_ms), "BRV precision below 3 m s^-1")
            hdr["BJD"] = (str(bjd), "BJD mid obs (using FWMT if not 0)")
            hdr["BARYSRC"] = (src, "Coordinates used for BARYRV and BJD")

        # Start time from DATE-OBS/TIME-OBS, or from ENVMJD when TIME-OBS is stale
        try:
            t_start, t_src, t_off = observation_start(hdr, self.logger)
        except Exception as e:
            self.logger.warning(f"SpectralExtraction: no start time for {name} ({e}); BARYRV and BJD set to nan")
            write('nan', 'nan', 'False', 'none: no start time')
            return
        hdr["TSTART"] = (t_start.isot, "UTC start used for BJD and BARYRV")
        hdr["TSTARTSR"] = (t_src, "Source of TSTART (see TOBSOFF)")
        if t_off is not None:
            hdr["TOBSOFF"] = (round(t_off, 1), "TIME-OBS minus ENVMJD (s)")

        try:
            fwmt = float(hdr.get("EXP-MID", 0.) or 0.)
        except (TypeError, ValueError):
            fwmt = 0.
        et = float(hdr.get("EXPTIME", 0.) or 0.)
        mid = fwmt / 86400. if fwmt > 0. else et / 2. / 86400.
        jd = t_start.jd + mid

        lat = -32.3722685109
        lon = 20.806403441
        alt = hdr["SITEELEV"]
        target = hdr.get("OBJECT", "")

        try:
            BCV = barycorrpy.get_BC_vel(JDUTC=jd, starname=target, lat=lat, longi=lon, alt=alt, leap_update=True, ephemeris='de430')
            BJD = barycorrpy.JDUTC_to_BJDTDB(jd, starname=target, lat=lat, longi=lon, alt=alt)
            write(BCV[0][0] / 1000., BJD[0][0], 'True', 'SIMBAD')
            return
        except Exception:
            pass
        try:
            co = SkyCoord(hdr["RA"], hdr["DEC"], unit=(u.hourangle, u.deg))
            BCV = barycorrpy.get_BC_vel(JDUTC=jd, ra=co.ra.degree, dec=co.dec.degree, lat=lat, longi=lon, alt=alt, leap_update=True, ephemeris='de430')
            BJD = barycorrpy.JDUTC_to_BJDTDB(jd, ra=co.ra.degree, dec=co.dec.degree, lat=lat, longi=lon, alt=alt)
            write(BCV[0][0] / 1000., BJD[0][0], 'False', 'header RA/DEC')
        except Exception as e:
            self.logger.warning(f"SpectralExtraction: {target!r} not resolved and no usable RA/DEC in {name} "
                                f"({type(e).__name__}: {e}); BARYRV and BJD set to nan")
            write('nan', 'nan', 'False', 'none: no usable coordinates')


    def get_order_set(self, s_order):
        if self.o_set.size > 0:
            e_order = self.o_set.size

            o_set_ary = self.o_set[0:e_order] + s_order
            valid_idx = np.where(o_set_ary >= 0)[0]
            first_idx = valid_idx[0] if valid_idx.size > 0 else -1

            return o_set_ary, first_idx
        else:
            return o_set
