import logging
import glob
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import math
import numpy as np
import pandas as pd

from .alg import OrderTraceAlg
from .reference_trace import trace as reference_trace, have_reference, reference_file

logger = logging.getLogger(__name__)

class OrderTrace():
    """
    Trace the echelle orders of an HRS master flat.

    The orders are traced with the reference tracer (`reference_trace`): the night's flat is measured against
    a stored reference of the arm and mode (labelled traces, cross-dispersion profiles, widths), the same code
    for both arms and all modes. When there is no reference for the arm and mode, when the reference tracer
    raises an error, or when it finds fewer than half the orderlets (flag FAILED), the cluster tracer
    (`OrderTraceAlg`: pixel clusters, cleaning, merging, HRS-specific cuts, widths) is used instead.

    The output is a CSV file with the polynomial coefficients and extraction widths of every orderlet, the same
    columns from either tracer. The reference tracer also writes a per-orderlet table (`*_diag.csv`) and the
    header cards TRCREF, TRCDY, TRCCC, TRCNFB, TRCNANC and TRCFLAG on the master flat (TRCFLAG = CLUSTER when
    the cluster tracer was used). A diagnostic plot can also be written.

    Parameters
    ----------
    MFlat : str
        Path to the master flat FITS file.
    nights : dict
        Dictionary giving the observing night associated with each file group.
    base_dir : str
        Base reduction directory.
    arm : str
        Arm label used in the directory structure, typically "Blu" or "Red".
    mode : str
        Observing mode, e.g. "LR", "MR", "HR", or "HS".
    plot : bool
        If True, save a diagnostic plot of the traced orders.

    Attributes
    ----------
    MFlat : str
        Path to the input master flat file.
    nights : dict
        Observing-night mapping for the current data set.
    base_dir : str
        Base reduction directory.
    arm : str
        Full arm label used in the directory tree.
    mode : str
        Observing mode.
    plot : bool
        Flag controlling diagnostic plot output.
    poly_degree : int
        Polynomial degree used for the order-trace model.
    sarm : str
        Short arm label used internally ("H" or "R").
    cols_to_reset : list or None
        Optional detector columns to exclude during tracing.
    rows_to_reset : list or None
        Optional detector rows to exclude during tracing.
    do_post : bool
        Flag controlling post-processing of trace widths.
    logger : logging.Logger
        Logger used for status and debug messages.
    orderlet_gap_pixels : int
        Desired minimum gap between neighbouring traced orderlets.
    flat_data : numpy.ndarray
        Master flat image used for tracing, optionally background-subtracted.
    alg : OrderTraceAlg or None
        The cluster tracer, set up only when it is used.
    """

    def __init__(self,MFlat,nights,base_dir,arm,mode,plot):
    
        self.MFlat = MFlat
        self.nights = nights
        self.base_dir = base_dir
        self.arm = arm
        self.mode = mode
        self.plot = plot
        self.poly_degree = 5
        if self.arm == "Blu":
            self.sarm = "H"
        else:
            self.sarm = "R"
        logger.info('Started {}'.format(self.__class__.__name__))
        self.cols_to_reset = None
        self.rows_to_reset = None
        self.do_post = True
        self.logger=logger
        self.orderlet_gap_pixels = 2
        
        #Open the flat file
        with fits.open(self.MFlat) as hdu:
            self.flat_data = hdu[0].data
            # Night whose geometry this flat shows: a Super Flat's first night (SF_START).
            # Plain master flats have no SF_START and fall back to the observing night.
            self.flat_start = str(hdu[0].header.get('SF_START', '')).strip()

        if self.sarm == "R":
            self.flat_data[:,4020:] = 0.
        self.alg = None             # the cluster tracer, set up only when the reference tracer cannot be used
    
    
    
    def order_trace(self):
        """
        Create or load the traced-order table for the master flat.

        The output directory is that of the flat's observing night. If the order-trace CSV file exists, it is
        returned. Otherwise the orders are traced with the reference tracer, or with the cluster tracer when the
        reference tracer cannot be used (see the class description), and the table is written.

        Returns
        -------
        str
            Path to the existing or newly created order-trace CSV file.
        """
        #Set the night for the master order data based on the flat location
        yyyymmdd = str(self.nights['flat'][0:4])+str(self.nights['flat'][4:8])
        self.out_dir = self.base_dir+self.arm+"/"+yyyymmdd[0:4]+"/"+yyyymmdd[4:]+"/reduced/"
        out_file = self.out_dir+self.mode+"_Orders_"+self.sarm+yyyymmdd+".csv"

        #Test if Order file exists
        order_file = glob.glob(out_file)
        if len(order_file) > 0:
            logger.info('OrderTrace: Reading predetermined orders')
            return order_file[0]

        df = self.trace_with_reference(yyyymmdd)
        if df is None:
            df = self.trace_with_clusters(yyyymmdd)
            self.write_cards({'TRCFLAG': ('CLUSTER', 'orders traced by the cluster tracer')})

        #Save to file
        df.to_csv(out_file)
        self.logger.info("OrderTrace: Receipt written")
        self.logger.info("OrderTrace: Done!\n")
        return str(out_file)

    def trace_with_reference(self, yyyymmdd):
        """
        Trace the orders with the reference tracer.

        Returns the order table (pipeline CSV columns), or None when the reference tracer cannot be used: no
        reference for the arm and mode, an error, or flag FAILED. Writes the per-orderlet table
        (<mode>_Orders_<arm><night>_diag.csv) and the TRC* header cards on the master flat.
        """
        if not have_reference(self.sarm, self.mode):
            self.logger.warning(f"OrderTrace: no {self.mode} reference for the {self.arm} arm; using the cluster tracer")
            return None
        self.logger.info("OrderTrace: tracing against the reference...")
        try:
            table, diag, info = reference_trace(self.flat_data, self.sarm, self.mode)
        except Exception as e:
            self.logger.error(f"OrderTrace: the reference tracer failed ({type(e).__name__}: {e}); using the cluster tracer")
            return None
        flag = info['flag']
        if flag == 'FAILED':
            self.logger.error(f"OrderTrace: the reference tracer found only {info['frac_good'] * 100:.0f} % of the orderlets "
                              f"(flag FAILED); using the cluster tracer")
            return None
        self.logger.info(f"OrderTrace: offset {info['offset']:+.2f} px at the chip centre; orderlets measured "
                         f"{info['n_measured']}, blended {info['n_blended']}, anchored {info['n_anchored']}, "
                         f"without light {info['n_reference']}; flag {flag}")
        if flag == 'AMBIGUOUS':
            self.logger.warning(f"OrderTrace: another labelling of the orders fits almost as well "
                                f"(ratio {info['mismatch_ratio']:.1f}); check the trace")
        elif flag == 'FALLBACK':
            dark = diag.orderlet[diag.source == 'reference'].tolist()
            self.logger.warning(f"OrderTrace: orderlets without usable light {dark}: their traces are the reference "
                                f"moved by the night's geometry")
        diag.to_csv(self.out_dir+self.mode+"_Orders_"+self.sarm+yyyymmdd+"_diag.csv", index=False)
        self.write_cards({'TRCREF': (os.path.basename(reference_file(self.sarm, self.mode)), 'order-trace reference'),
                          'TRCDY': (round(float(info['offset']), 3), '[px] trace offset from reference, chip centre'),
                          'TRCCC': (round(float(info['cc_min']), 3) if np.isfinite(info['cc_min']) else -1.,
                                    'lowest correlation of the coarse offset'),
                          'TRCNFB': (int(info['n_reference']), 'orderlets without usable light'),
                          'TRCNANC': (int(info['n_anchored']), 'orderlets mostly on the reference shape'),
                          'TRCFLAG': (flag, 'order trace: OK, FALLBACK, AMBIGUOUS, CLUSTER')})
        if self.plot:
            self.plot_trace(table)
        return table

    def write_cards(self, cards):
        """Write header cards {key: (value, comment)} into the primary header of the master flat."""
        try:
            with fits.open(self.MFlat, mode='update') as hdu:
                for k, v in cards.items():
                    hdu[0].header[k] = v
        except Exception as e:
            self.logger.warning(f"OrderTrace: could not write the trace cards to {self.MFlat} ({type(e).__name__}: {e})")

    def plot_trace(self, table):
        """Order centres and extraction edges over the flat."""
        plt.figure(figsize=(10, 10))
        plt.title(f"OrderTrace: {self.mode} {self.arm}, {len(table)} orderlets")
        plt.imshow(self.flat_data, origin='lower', vmin=0, vmax=np.nanpercentile(self.flat_data, 99), aspect='auto', cmap='gray')
        for _, r in table.iterrows():
            x = np.arange(int(r.X1), int(r.X2) + 1)
            y = np.polynomial.polynomial.polyval(x, [r[f'Coeff{i}'] for i in range(6)])
            plt.plot(x, y, 'g', lw=0.3)
            plt.plot(x, y - r.BottomEdge, 'r', lw=0.3)
            plt.plot(x, y + r.TopEdge, 'b', lw=0.3)
        plt.savefig(self.out_dir+self.mode+"_Order_Trace.png", bbox_inches='tight', dpi=300)
        plt.close()

    def trace_with_clusters(self, yyyymmdd):
        """
        Trace the orders with the cluster tracer (OrderTraceAlg), the tracer of versions up to 1.3.5:
            1. locating candidate order pixels in the master flat
            2. grouping pixels into connected clusters
            3. cleaning noisy clusters and removing border artefacts
            4. merging compatible cluster fragments into full orders
            5. applying HRS-specific filtering to reject spurious orders
            6. estimating upper and lower order widths
            7. optionally post-processing the widths
            8. matching fibre-pair order lengths
        Returns the order table.
        """
        self.alg = OrderTraceAlg(self.flat_data, self.mode, self.sarm, poly_degree=self.poly_degree, logger=self.logger)
        # 1) Locate cluster
        if self.logger:
            self.logger.info("OrderTrace: locating clusters...")
            #self.logger.warning("OrderTrace: locating cluster...")
        cluster_xy = self.alg.locate_clusters(self.rows_to_reset, self.cols_to_reset)
        
#            if self.plot:
#                plt.title("OrderTrace: locating cluster result")
#                plt.imshow(self.flat_data,origin='lower',vmin=0,vmax=10)
#                plt.plot(cluster_xy['x'],cluster_xy['y'],'.')
#                plt.show()
        
        # 2) assign cluster id and do basic cleaning
        if self.logger:
            self.logger.info("OrderTrace: assigning cluster id and cleaning...")
            #self.logger.warning("OrderTrace: assigning cluster id and cleaning...")
        x, y, index = self.alg.form_clusters(cluster_xy['x'], cluster_xy['y'])

#            if self.plot:
#                plt.title("OrderTrace: cluster id and cleaning")
#                plt.imshow(self.flat_data,origin='lower',vmin=0,vmax=10)
#                uniq =np.unique(index)
#                for i in uniq:
#                    ii = np.where(index == i)[0]
#                    plt.plot(x[ii],y[ii],'.')
#                plt.show()

        # 3) advanced cleaning and border cleaning
        if self.logger:
            self.logger.info("OrderTrace: advanced cleaning...")
            #self.logger.warning("OrderTrace: advanced cleaning...")
 
        new_x, new_y, new_index, all_status = self.alg.advanced_cluster_cleaning_handler(index, x, y)
        new_x, new_y, new_index = self.alg.clean_clusters_on_borders(new_x, new_y, new_index)
        
#            if self.plot:
#                plt.title("OrderTrace: advanced and boarder and cleaning")
#                uniq = np.unique(new_index)
#                plt.imshow(self.flat_data,origin='lower',vmin=0,vmax=10)
#                for i in uniq:
#                    ii = np.where(new_index == i)[0]
#                    plt.plot(new_x[ii],new_y[ii],'.')
#                plt.show()


        # 5) Merge cluster
        if self.logger:
            self.logger.info("OrderTrace: merging cluster...")
            #self.logger.warning("OrderTrace: merging cluster...")
        c_x, c_y, c_index = self.alg.merge_clusters_and_clean(new_index, new_x, new_y)
#            
#            if self.plot:
#                plt.title("OrderTrace: merged clusters")
#                uniq = np.unique(c_index)
#                plt.imshow(self.flat_data,origin='lower',vmin=0,vmax=10)
#                for i in uniq:
#                    ii = np.where(c_index == i)[0]
#                    plt.plot(c_x[ii],c_y[ii],'.')
#                plt.show()
        
        # Clean the merged clusters to the expected number for arm / mode
        self.logger.info("OrderTrace: cleaning spurious orders...")
        flat_date = self.flat_start if self.flat_start else yyyymmdd
        HRS_x, HRS_y, HRS_index = self.alg.HRS_clean(c_x, c_y, c_index, yyyymmdd, flat_date)
        
        # 6) Find width
        if self.logger:
            self.logger.info("OrderTrace: finding widths...")
            #self.logger.warning("OrderTrace: finding width...")
        all_widths, cluster_coeffs = self.alg.find_all_cluster_widths(HRS_index, HRS_x, HRS_y, power_for_width_estimation=3)
        
        if self.plot:
            uniq = np.unique(HRS_index)
            plt.title("OrderTrace: Widths\nNumber ords="+str(len(uniq)))
            plt.imshow(self.flat_data,origin='lower',vmin=0,vmax=500,aspect='auto')

            count=0
            for i in uniq:
                s_x = int(cluster_coeffs[i, self.poly_degree + 1])
                e_x = int(cluster_coeffs[i, self.poly_degree + 2] + 1)
                x=np.arange(s_x,e_x)
                ord_cen=np.polyval(cluster_coeffs[i,0:self.poly_degree+1],x)

                plt.plot(x,ord_cen,'g',lw=0.1)
                plt.plot(x,ord_cen-all_widths[i-1]['bottom_edge'],'r',lw=0.1)
                plt.plot(x,ord_cen+all_widths[i-1]['top_edge'],'b',lw=0.1)
                count=count+1
                
            plt.savefig(self.out_dir+self.mode+"_Order_Trace.png",bbox_inches='tight',dpi=900)
            plt.close()

        
        # 7) post processing
        if self.do_post:
            if self.logger:
                self.logger.info('OrderTrace: post processing...')

            post_coeffs, post_widths = self.alg.convert_for_post_process(cluster_coeffs, all_widths)
            _, all_widths = self.alg.post_process(post_coeffs, post_widths, orderlet_gap=self.orderlet_gap_pixels)
            
            
        # 8) Fixing order properties based on the ensamble. This is for the bluest 5 orders and 3 reddest orders
        # Fit 3rd order polynomials to the 'good' orders and use the parameters to fix the other 7 orders
        # Re-order the orders based on the intercpt value (coeff5)
#            
#            for i in range(5):
#                coeffs = cluster_coeffs[:,i]
#                x=np.arange(len(coeffs))
#                pars = np.polyfit(x[5:-3],coeffs[5:-3],3)
#
#                for j in range(5):
#                    fit = np.polyval(pars,j)
#                    cluster_coeffs[j,i] = fit
#                for j in range(3):
#                    fit = np.polyval(pars,len(coeffs)-j-1)
#                    cluster_coeffs[len(coeffs)-j-1,i] = fit
#                coeffs = cluster_coeffs[:,i]
#                
        if self.plot:
            uniq = np.unique(HRS_index)
            plt.title("OrderTrace: Widths\nNumber ords="+str(len(uniq)))
            plt.imshow(self.flat_data,origin='lower',vmin=0,vmax=10)

            count=0
            for i in uniq:
                s_x = int(cluster_coeffs[i, self.poly_degree + 1])
                e_x = int(cluster_coeffs[i, self.poly_degree + 2] + 1)
                x=np.arange(s_x,e_x)
                ord_cen=np.polyval(cluster_coeffs[i,0:self.poly_degree+1],x)

                plt.plot(x,ord_cen,'g',lw=0.3)
                plt.plot(x,ord_cen-all_widths[i-1]['bottom_edge'],'r',lw=0.3)
                plt.plot(x,ord_cen+all_widths[i-1]['top_edge'],'b',lw=0.3)
                count=count+1

            plt.savefig(self.out_dir+self.mode+"_Order_Trace.png",bbox_inches='tight',dpi=900)
            plt.close()
        
        starts = cluster_coeffs[:,5]
        srt_idx = sorted(range(len(starts)),key=starts.__getitem__)
        sorted_cls = cluster_coeffs[srt_idx]
        all_widths = [all_widths[i-1] for i in srt_idx[1:]]
        del cluster_coeffs
        cluster_coeffs = sorted_cls

        # 9) HRS orders are in pairs, so correct the order lenghts to be the same for the fibre pairs
        if self.logger:
            self.logger.info("OrderTrace: correcting order lengths...")
        max_cluster_no = np.amax(HRS_index)
        cluster_set = list(range(1, max_cluster_no+1))
        img_edge= self.flat_data.shape[1]-1

        for n in range(1,max_cluster_no,2):

            s_x_L = int(cluster_coeffs[n, self.poly_degree + 1])
            s_x_U = int(cluster_coeffs[n+1, self.poly_degree + 1])

            if s_x_L != s_x_U:
                if s_x_L < s_x_U:
                    cluster_coeffs[n+1, self.poly_degree + 1] = cluster_coeffs[n, self.poly_degree + 1]
                else:
                    cluster_coeffs[n+1, self.poly_degree + 1] = s_x_L
            if int(cluster_coeffs[n+1, self.poly_degree + 2])/ img_edge > 0.95:
                cluster_coeffs[n+1, self.poly_degree + 2] = img_edge
                cluster_coeffs[n, self.poly_degree + 2] = img_edge
            else:
                cluster_coeffs[n, self.poly_degree + 2] = cluster_coeffs[n+1, self.poly_degree + 2]
                
                

        # 10) convert result to dataframe
        if self.logger:
            self.logger.info("OrderTrace: writing cluster into dataframe...")

        df = self.alg.write_cluster_info_to_dataframe(all_widths, cluster_coeffs)
        
        assert(isinstance(df, pd.DataFrame))
        return df
