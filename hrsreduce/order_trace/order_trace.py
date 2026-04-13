import logging
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import math
import numpy as np
import pandas as pd

from .alg import OrderTraceAlg

logger = logging.getLogger(__name__)

class OrderTrace():
    """
    Run the full HRS order-tracing workflow on a master flat frame.

    This class provides the high-level wrapper around `OrderTraceAlg` for
    tracing echelle orders in an HRS master flat. It loads the master flat,
    optionally removes large-scale background structure, detects candidate
    order pixels, groups them into clusters, cleans and merges those clusters,
    estimates order widths, applies HRS-specific corrections, and writes the
    final order-trace table to disk.

    The tracing process is arm- and mode-dependent, and the final output is a
    CSV file describing the polynomial coefficients and extraction widths of
    each traced order. A diagnostic plot and a NumPy archive of traced-order
    pixels can also be written.

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
    alg : OrderTraceAlg
        Lower-level tracing algorithm instance.
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
            
        # 0) Remove the background
        if self.sarm == 'R':
            if self.logger:
                self.logger.info("OrderTrace: Removing background from flat...")
            
            def _robust_polyfit_1d(x, y, deg, n_iter=6, clip_sigma=4.0):
                x = np.asarray(x, float)
                y = np.asarray(y, float)
                m = np.isfinite(x) & np.isfinite(y)
                x, y = x[m], y[m]
                if x.size < deg + 2:
                    return None

                keep = np.ones_like(y, dtype=bool)
                for _ in range(n_iter):
                    if keep.sum() < deg + 2:
                        break
                    c = np.polyfit(x[keep], y[keep], deg)
                    r = y - np.polyval(c, x)

                    med = np.median(r[keep])
                    mad = np.median(np.abs(r[keep] - med))
                    sig = 1.4826 * mad if mad > 0 else np.std(r[keep])
                    if sig <= 0:
                        break

                    new_keep = np.abs(r - med) < clip_sigma * sig
                    if new_keep.sum() == keep.sum():
                        keep = new_keep
                        break
                    keep = new_keep

                if keep.sum() < deg + 2:
                    return None
                return np.polyfit(x[keep], y[keep], deg)
            
            def estimate_background_minima_per_column(
                img,
                y_bin=64,          # bins along y; minima taken within each bin
                deg_y=3,           # poly degree vs y per column
                deg_x=5,           # poly degree vs x for coefficient trends
                robust_iters=6,
                clip_sigma=4.0,
                min_quantile=None, # if set (e.g. 0.01), use that quantile instead of absolute min
            ):
                """
                For every x pixel (each column):
                  - split column into y bins
                  - take MINIMUM (or very-low quantile) in each y bin
                  - fit poly(y) to those minima
                Then:
                  - fit poly(x) to each y coefficient => smooth 2D background surface.

                Returns: bg, img, info
                """

                if img.ndim != 2:
                    raise ValueError(f"Expected 2D image in HDU {ext}, got {img.shape}")

                ny, nx = img.shape
                y_bin = int(y_bin)
                if y_bin < 1:
                    raise ValueError("y_bin must be >= 1")

                # y bin centers
                y_edges = np.arange(0, ny + y_bin, y_bin)
                y_centers = 0.5 * (y_edges[:-1] + y_edges[1:] - 1)

                # Per-column fitted coeffs (nx, deg_y+1)
                coeff_matrix = np.full((nx, deg_y + 1), np.nan, float)

                # Store minima samples for a few columns if you want diagnostics
                # (keeping all columns is big; you can still if you want)
                for x in range(nx):
                    col = img[:, x]
                    zmins = np.full_like(y_centers, np.nan, dtype=float)

                    for j in range(len(y_centers)):
                        y0, y1 = int(y_edges[j]), int(min(y_edges[j + 1], ny))
                        chunk = col[y0:y1]
                        chunk = chunk[np.isfinite(chunk)]
                        if chunk.size < 5:
                            continue

                        if min_quantile is None:
                            zmins[j] = np.min(chunk)
                        else:
                            zmins[j] = np.quantile(chunk, float(min_quantile))

                    ok = np.isfinite(zmins)
                    if ok.sum() < deg_y + 2:
                        continue

                    c = _robust_polyfit_1d(
                        y_centers[ok], zmins[ok],
                        deg=deg_y,
                        n_iter=robust_iters,
                        clip_sigma=clip_sigma
                    )
                    if c is None:
                        continue
                    coeff_matrix[x, :] = c

                # Fill missing columns by interpolation per coefficient
                x_full = np.arange(nx, dtype=float)
                for k in range(deg_y + 1):
                    v = coeff_matrix[:, k]
                    ok = np.isfinite(v)
                    if ok.sum() < 2:
                        raise RuntimeError(
                            f"Coefficient {k} has too few valid columns ({ok.sum()}). "
                            f"Try larger y_bin, lower deg_y, or use min_quantile=0.01."
                        )
                    coeff_matrix[~ok, k] = np.interp(x_full[~ok], x_full[ok], v[ok])

                # Fit polynomial in x for each coefficient (to smooth across columns)
                coeff_polys_x = []
                for k in range(deg_y + 1):
                    v = coeff_matrix[:, k]
                    cx = _robust_polyfit_1d(
                        x_full, v,
                        deg=deg_x,
                        n_iter=robust_iters,
                        clip_sigma=clip_sigma
                    )
                    if cx is None:
                        raise RuntimeError(f"Failed poly(x) fit for coefficient {k}. Try smaller deg_x.")
                    coeff_polys_x.append(cx)

                # Evaluate background surface: compute y-coeffs at each x, then eval in y
                y_full = np.arange(ny, dtype=float)
                Vy = np.vander(y_full, N=deg_y + 1, increasing=False)  # (ny, deg_y+1)

                coeffs_y_at_x = np.vstack([np.polyval(cx, x_full) for cx in coeff_polys_x]).T  # (nx, deg_y+1)
                bg = Vy @ coeffs_y_at_x.T  # (ny, nx)

                info = {
                    "shape": (ny, nx),
                    "y_bin": y_bin,
                    "deg_y": deg_y,
                    "deg_x": deg_x,
                    "min_quantile": min_quantile,
                    "coeff_matrix_columns": coeff_matrix,
                    "coeff_polys_x": coeff_polys_x,
                    "y_centers": y_centers,
                }
                
                return bg, img, info
            
            def rm_bkg(data):
                    
                bg, img, info = estimate_background_minima_per_column(
                    data,
                    y_bin=64,
                    deg_y=4,
                    deg_x=5,
                    min_quantile=None,   # or 0.01
                )
                
                bkg_removed = (img - bg).astype(np.float64)
                
                return bkg_removed

            self.flat_data = rm_bkg(self.flat_data)
        
        
        self.alg = OrderTraceAlg(self.flat_data,self.mode, self.sarm, poly_degree=self.poly_degree, logger=self.logger)
    
    
    
    def order_trace(self):
        """
        Create or load the traced-order table for the master flat.

        This method determines the output directory from the flat-field observing
        night and checks whether an order-trace CSV file already exists. If so,
        that file is returned immediately. Otherwise, the full order-tracing
        workflow is executed.

        The tracing sequence consists of:
            1. locating candidate order pixels in the master flat
            2. grouping pixels into connected clusters
            3. cleaning noisy clusters and removing border artefacts
            4. merging compatible cluster fragments into full orders
            5. applying HRS-specific filtering to reject spurious orders
            6. estimating upper and lower order widths
            7. optionally post-processing the widths
            8. smoothing edge-order polynomial properties using ensemble fits
            9. matching fibre-pair order lengths
           10. writing the final trace solution to CSV

        In addition to the CSV trace table, the method also writes a NumPy archive
        containing the traced order pixels. If plotting is enabled, a diagnostic
        image showing the order centres and widths is saved.

        Returns
        -------
        str
            Path to the existing or newly created order-trace CSV file.
        """
    
        #Set the night for the master order data based on the flat location
        yyyymmdd = str(self.nights['flat'][0:4])+str(self.nights['flat'][4:8])
        self.out_dir = self.base_dir+self.arm+"/"+yyyymmdd[0:4]+"/"+yyyymmdd[4:]+"/reduced/"
    
        #Test if Order file exists
        order_file = glob.glob(self.out_dir+self.mode+"_Orders_"+self.sarm+yyyymmdd+".csv")
        
        if len(order_file) == 0:
            #If there is no ORDER file we need to create one.
            
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
            HRS_x, HRS_y, HRS_index = self.alg.HRS_clean(c_x, c_y, c_index,yyyymmdd)
            
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

                    plt.plot(x,ord_cen,'g')
                    plt.plot(x,ord_cen-all_widths[i-1]['bottom_edge'],'r')
                    plt.plot(x,ord_cen+all_widths[i-1]['top_edge'],'b')
                    count=count+1
                    
                plt.savefig(self.out_dir+self.mode+"_Order_Trace.png",bbox_inches='tight',dpi=600)
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
            
            for i in range(5):
                coeffs = cluster_coeffs[:,i]
                x=np.arange(len(coeffs))
                pars = np.polyfit(x[5:-3],coeffs[5:-3],3)

                for j in range(5):
                    fit = np.polyval(pars,j)
                    cluster_coeffs[j,i] = fit
                for j in range(3):
                    fit = np.polyval(pars,len(coeffs)-j-1)
                    cluster_coeffs[len(coeffs)-j-1,i] = fit
                coeffs = cluster_coeffs[:,i]
                
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

                    plt.plot(x,ord_cen,'g')
                    plt.plot(x,ord_cen-all_widths[i-1]['bottom_edge'],'r')
                    plt.plot(x,ord_cen+all_widths[i-1]['top_edge'],'b')
                    count=count+1
                plt.savefig(self.out_dir+self.mode+"_Order_Trace.png",bbox_inches='tight',dpi=600)
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
            
            #Save to file
            df.to_csv(self.out_dir+self.mode+"_Orders_"+self.sarm+yyyymmdd+".csv")
            self.logger.info("OrderTrace: Receipt written")
            self.logger.info("OrderTrace: Done!\n")
            
            #Save the pixel locations of the orders too, for background subtraction
            np.savez(self.out_dir+self.mode+"_Orders_"+self.sarm+yyyymmdd+".npz", orders=c_index,x_pix=c_x,y_pix=c_y)

            return str(self.out_dir+self.mode+"_Orders_"+self.sarm+yyyymmdd+".csv")
            
        else:
            logger.info('OrderTrace: Reading predetermined orders')
            
            return order_file[0]
