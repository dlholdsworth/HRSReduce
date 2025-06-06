import numpy as np
import os.path
from astropy.io import fits
import logging
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
from scipy.optimize import least_squares
import glob
import multiprocessing as mp
logger = logging.getLogger(__name__)

class SlitCorrection():

    def __init__(self, sci_frame, ext_name,order_file,arm,mode,base_dir,yyyymmdd,plot=False,super_arc=None):
        
        self.sci_frame = sci_frame
        self.ext_name = ext_name
        self.order_file = order_file
        self.sarm = arm
        self.super_arc = super_arc
        if arm == 'H':
            self.arm_col = 'Blu'
        if arm == 'R':
            self.arm_col = 'Red'
        self.mode = mode
        self.base_dir = base_dir
        self.logger = logger
        self.plot = plot
        self.yyyymmdd = yyyymmdd
        
        # start a logger
        self.logger.info('Started SlitCorrection')
        
        # Open the data files
        with fits.open(self.sci_frame) as hdul:
            self.spec_header = hdul[0].header
            self.flux = hdul[self.ext_name].data
        self.straight = self.flux.copy()

        if self.order_file:
            self.order_trace_data = pd.read_csv(self.order_file, header=0, index_col=0)
      
    def gaussian(self,x, A, mu, sig):
        """
        A: height
        mu: offset from central line
        sig: standard deviation
        """
        return A * np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))
        
    def lorentzian(self,x, A, x0, mu):
        """
        A: height
        x0: offset from central line
        mu: width of lorentzian
        """
        return A * mu / ((x - x0) ** 2 + 0.25 * mu ** 2)
        
    def fit_curvature_single_order(self,peaks, tilt, shear):
        try:
            middle = np.median(tilt)
            sigma = np.percentile(tilt, (32, 68))
            sigma = middle - sigma[0], sigma[1] - middle
            mask = (tilt >= middle - 5 * sigma[0]) & (tilt <= middle + 5 * sigma[1])
            peaks, tilt, shear = peaks[mask], tilt[mask], shear[mask]

            coef_tilt = np.zeros(1 + 1)
            res = least_squares(
                lambda coef: np.polyval(coef, peaks) - tilt,
                x0=coef_tilt,
                loss="arctan",
            )
            coef_tilt = res.x

            coef_shear = np.zeros(1 + 1)
            res = least_squares(
                lambda coef: np.polyval(coef, peaks) - shear,
                x0=coef_shear,
                loss="arctan",
            )
            coef_shear = res.x

        except:
            print(
                "Could not fit the curvature of this order. Using no curvature instead"
            )
            coef_tilt = np.zeros(1 + 1)
            coef_shear = np.zeros(1 + 1)

        return coef_tilt, coef_shear, peaks
        
    def fit(self, peaks, tilt, shear,nord):
        mode="1D"
        if mode == "1D":
            coef_tilt = np.zeros((nord, 1 + 1))
            coef_shear = np.zeros((nord, 1 + 1))
            for j in range(nord):
                coef_tilt[j], coef_shear[j], _ = self.fit_curvature_single_order(
                    peaks[j], tilt[j], shear[j]
                )
        elif self.mode == "2D":
            x = np.concatenate(peaks)
            y = [np.full(len(p), i) for i, p in enumerate(peaks)]
            y = np.concatenate(y)
            z = np.concatenate(tilt)
            coef_tilt = polyfit2d(x, y, z, degree=self.fit_degree, loss="arctan")

            z = np.concatenate(shear)
            coef_shear = polyfit2d(x, y, z, degree=self.fit_degree, loss="arctan")

        return coef_tilt, coef_shear
        
    def eval(self,peaks, order, coef_tilt, coef_shear):
        mode="1D"
        if mode == "1D":
            tilt = np.zeros(peaks.shape)
            shear = np.zeros(peaks.shape)
            for i in np.unique(order):
                idx = order == i
                tilt[idx] = np.polyval(coef_tilt[i], peaks[idx])
                shear[idx] = np.polyval(coef_shear[i], peaks[idx])
        elif mode == "2D":
            tilt = polyval2d(peaks, order, coef_tilt)
            shear = polyval2d(peaks, order, coef_shear)
        return tilt, shear



    def make_index(self,ymin, ymax, xmin, xmax, zero=0):
        """Create an index (numpy style) that will select part of an image with changing position but fixed height

        The user is responsible for making sure the height is constant, otherwise it will still work, but the subsection will not have the desired format

        Parameters
        ----------
        ymin : array[ncol](int)
            lower y border
        ymax : array[ncol](int)
            upper y border
        xmin : int
            leftmost column
        xmax : int
            rightmost colum
        zero : bool, optional
            if True count y array from 0 instead of xmin (default: False)

        Returns
        -------
        index : tuple(array[height, width], array[height, width])
            numpy index for the selection of a subsection of an image
        """

        # in x: the rows between ymin and ymax
        # in y: the column, but n times to match the x index
        ymin = np.asarray(ymin, dtype=int)
        ymax = np.asarray(ymax, dtype=int)
        xmin = int(xmin)
        xmax = int(xmax)

        if zero:
            zero = xmin

        index_x = np.array(
            [np.arange(ymin[col], ymax[col] + 1) for col in range(xmin - zero, xmax - zero)]
        )
        index_y = np.array(
            [
                np.full(ymax[col] - ymin[col] + 1, col)
                for col in range(xmin - zero, xmax - zero)
            ]
        )
        index = index_x.T, index_y.T + zero

        return index
        
    def determine_curvature_single_line(self,original, peak, ycen, ycen_int, xwd):
        """
        Fit the curvature of a single peak in the spectrum
        This is achieved by fitting a model, that consists of gaussians
        in spectrum direction, that are shifted by the curvature in each row.
        Parameters
        ----------
        original : array of shape (nrows, ncols)
            whole input image
        peak : int
            column position of the peak
        ycen : array of shape (ncols,)
            row center of the order of the peak
        xwd : 2 tuple
            extraction width above and below the order center to use
        Returns
        -------
        tilt : float
            first order curvature
        shear : float
            second order curvature
        """
        _, ncol = original.shape

        window_width = 9.
        curv_degree = 1
        # look at +- width pixels around the line
        # Extract short horizontal strip for each row in extraction width
        # Then fit a gaussian to each row, to find the center of the line
        x = peak + np.arange(-window_width, window_width + 1)
        x = x[(x >= 0) & (x < ncol)]
        xmin, xmax = int(x[0]), int(x[-1] + 1)

        # Look above and below the line center
        y = np.arange(-xwd[0], xwd[1] + 1)[:, None] - ycen[xmin:xmax][None, :]

        x = x[None, :]
        idx = self.make_index(ycen_int - xwd[0], ycen_int + xwd[1], xmin, xmax)
        img = original[idx].astype('float64')
        img_compressed = np.ma.compressed(img)

        img -= np.percentile(img_compressed, 1)
        img /= np.percentile(img_compressed, 99)
        img = np.ma.clip(img, 0, 1)

        sl = np.ma.mean(img, axis=1)
        sl = sl[:, None]

        peak_func = {"gaussian": self.gaussian, "lorentzian": self.lorentzian}
        peak_func = peak_func["gaussian"]

        def model(coef):
            A, middle, sig, *curv = coef
            mu = middle + shift(curv)
            mod = peak_func(x, A, mu, sig)
            mod *= sl
            return (mod - img).ravel()

        def model_compressed(coef):
            return np.ma.compressed(model(coef))

        A = np.nanpercentile(img_compressed, 95)
        sig = (xmax - xmin) / 4  # TODO
        if curv_degree == 1:
            shift = lambda curv: curv[0] * y
        elif curv_degree == 2:
            shift = lambda curv: (curv[0] + curv[1] * y) * y
        else:
            raise ValueError("Only curvature degrees 1 and 2 are supported")
        x0 = [A, peak, sig] + [0] * curv_degree

        res = least_squares(
            model_compressed, x0=x0, method="lm", loss="linear", f_scale=0.1
        )


        if curv_degree == 1:
            tilt, shear = res.x[3], 0
        elif curv_degree == 2:
            tilt, shear = res.x[3], res.x[4]
        else:
            tilt, shear = 0, 0

        model = model(res.x).reshape(img.shape) + img
        vmin = 0
        vmax = np.max(model)

        y = y.ravel()
        x = res.x[1] - xmin + (tilt + shear * y) * y
        y += xwd[0]

        return tilt, shear
        
    def find_peaks(self,vec, cr):
        # This should probably be the same as in the wavelength calibration
        max_vec=np.max(vec)
        threshold = 2.
        peak_width = 2.
        window_width = 15.
        #vec -= np.ma.median(vec)
        height = np.median(vec)
        height=0.001
        #vec = np.ma.filled(vec, 0)

        #height = np.percentile(vec, 90) * threshold
        peaks, _ = signal.find_peaks(
            vec/max_vec, height=height, width=peak_width, distance=window_width
        )

        # Remove peaks at the edge
        peaks = peaks[
            (peaks >= window_width + 1)
            & (peaks < len(vec) - window_width - 1)
        ]
        # Remove the offset, due to vec being a subset of extracted
        peaks += cr[0]
        return vec, peaks

        
    def determine_curvature_all_lines(self,original, extracted,column_range,extraction_width):
        n_col = original.shape[1]
        n_ord =extracted.shape[0]
        # Store data from all orders
        all_peaks = []
        all_tilt = []
        all_shear = []
        plot_vec = []

        for j in tqdm(range(n_ord), desc="Order",leave=False):
            cr = column_range[j]
            xwd = extraction_width[j]
            ycen = np.zeros(cr[1])+self.order_trace_data['Coeff0'][j]
            ycen_int = ycen.astype(int)
            ycen -= ycen_int

            # Find peaks
            vec = extracted[j,cr[0] : cr[1]].astype('float64')
            vec, peaks = self.find_peaks(vec, cr)

            npeaks = len(peaks)
            

            # Determine curvature for each line seperately
            tilt = np.zeros(npeaks)
            shear = np.zeros(npeaks)
            mask = np.full(npeaks, True)
            for ipeak, peak in tqdm(
                enumerate(peaks), total=len(peaks), desc="Peak", leave=False
            ):
                try:
                    tilt[ipeak], shear[ipeak] = self.determine_curvature_single_line(
                        original, peak, ycen, ycen_int, xwd
                    )
                except RuntimeError:  # pragma: no cover
                    mask[ipeak] = False

            # Store results
            all_peaks += [peaks[mask]]
            all_tilt += [tilt[mask]]
            all_shear += [shear[mask]]
            plot_vec += [vec]
        return all_peaks, all_tilt, all_shear, plot_vec
        
    def plot_comparison(self,original, tilt, shear, peaks,extraction_width,nord,column_range,savefile):  # pragma: no cover
        _, ncol = original.shape
        output = np.zeros((np.sum(extraction_width) + nord, ncol),dtype=np.int16)
        pos = [0]
        x = np.arange(ncol)
        for i in range(nord):
            ycen = np.zeros([ncol])+self.order_trace_data['Coeff0'][i]
            yb = ycen - extraction_width[i, 0]
            yt = ycen + extraction_width[i, 1]
            xl, xr = column_range[i]
            index = self.make_index(yb, yt, xl, xr)
            yl = pos[i]
            yr = pos[i] + index[0].shape[0]
            output[yl:yr, xl:xr] = original[index]
            pos += [yr]

        vmin, vmax = np.percentile(output[output != 0], (5, 95))
        plt.imshow(output, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")

        for i in range(nord):
            for p in peaks[i]:
                ew = extraction_width[i]
                x = np.zeros(ew[0] + ew[1] + 1)
                y = np.arange(-ew[0], ew[1] + 1)
                for j, yt in enumerate(y):
                    x[j] = p + yt * tilt[i, p] + yt ** 2 * shear[i, p]
                y += pos[i] + ew[0]
                plt.plot(x, y, "r")

        locs = np.sum(extraction_width, axis=1) + 1
        locs = np.array([0, *np.cumsum(locs)[:-1]])
        locs[:-1] += (np.diff(locs) * 0.5).astype(int)
        locs[-1] += ((output.shape[0] - locs[-1]) * 0.5).astype(int)

        plt.yticks(locs, range(len(locs)))

        plt.xlabel("x [pixel]")
        plt.ylabel("order")
        plt.tight_layout()
        plt.savefig(savefile,dpi=600)
        plt.close()
        
    def correct_for_curvature(self,img_order, tilt, shear, xwd):
  
        mask = ~np.ma.getmaskarray(img_order)
        xt = np.arange(img_order.shape[1])
        
        for y, yt in zip(range(int(xwd[0]) + int(xwd[1])), range(-int(xwd[0]), int(xwd[1]))):
            xi = xt + yt * tilt + yt ** 2 * shear

            img_order[y] = np.interp(
                xi, xt[mask[y]], img_order[y][mask[y]], left=0, right=0
            )

        xt = np.arange(img_order.shape[0])
        for x in range(img_order.shape[1]):
            img_order[:, x] = np.interp(
                xt, xt[mask[:, x]], img_order[:, x][mask[:, x]], left=0, right=0
            )

        return img_order


    def calculate(self):
    
        #Initially test to see if the super arc already has a correction file
        curve_file = os.path.dirname(self.super_arc)+"/"+os.path.splitext(os.path.basename(self.super_arc))[0]+"_Curvature.npz"
        #tiltfile = glob.glob(self.base_dir+self.arm_col+"/"+self.yyyymmdd[0:4]+"/"+self.yyyymmdd[4:8]+"/reduced/Curvature_"+self.sarm+self.yyyymmdd+".npz")
        tiltfile = glob.glob(curve_file)
        if len(tiltfile) == 0:
        
            n_ord = len(self.order_trace_data)
            n_col = self.flux.shape[1]
            
            extracted = np.zeros((n_ord,n_col))
            x = np.arange(n_col)
            mask = np.full(self.flux.shape, True)
            
            # Add mask as defined by column ranges
            mask1 = np.full((n_ord, n_col), True)
            for i in range(n_ord):
                mask1[i, 0 : n_col] = False
            extracted = np.ma.array(extracted, mask=mask1)
            
            column_range = np.zeros((n_ord,2),dtype=np.int16)
            extraction_width = np.zeros((n_ord,2),dtype=np.int16)
            
            for i in range(n_ord):
                x_left_lim = 0
                x_right_lim = self.flux.shape[1]
                column_range[i][0] = int(x_left_lim)
                column_range[i][1] = int(x_right_lim)
                
                extraction_width[i][0] =self.order_trace_data['BottomEdge'][i]
                extraction_width[i][1] =self.order_trace_data['TopEdge'][i]
                
                ycen = np.zeros([x_right_lim])+self.order_trace_data['Coeff0'][i]
                upper,lower =self.order_trace_data['TopEdge'][i],self.order_trace_data['BottomEdge'][i]
                yt,yb = ycen + upper, ycen-lower
                
                index = self.make_index(yb, yt, x_left_lim, x_right_lim)
                
                mask[index] = False
                
                extracted_ord = np.sum(self.flux[index],axis=0)
                
                extracted[i,0 : n_col] = extracted_ord

            peaks, tilt, shear, vec = self.determine_curvature_all_lines(
                    self.flux, extracted, column_range,extraction_width)
                    
            coef_tilt, coef_shear = self.fit(peaks, tilt, shear,n_ord)

            iorder,ipeaks = np.indices(extracted.shape)
            
            tilt, shear = self.eval(ipeaks, iorder, coef_tilt, coef_shear)

            if self.plot:
                savefile = (self.base_dir+self.arm_col+"/"+self.yyyymmdd[0:4]+"/"+self.yyyymmdd[4:8]+"/reduced/Curvature_fits.png")
                self.plot_comparison(self.flux, tilt, shear, peaks,extraction_width,n_ord,column_range,savefile)
                
            np.savez(curve_file, tilt=tilt, shear=shear)
        else:
            previous_data = np.load(tiltfile[0],allow_pickle=True)
            tilt = previous_data['tilt']
            shear = previous_data['shear']
            
        return tilt,shear
        
    
    def run_correction(self,i):
        x_right_lim = self.flux.shape[1]
        x_left_lim = 0

        ycen = np.zeros([x_right_lim],dtype=np.int16)+self.order_trace_data['Coeff0'][i]
        yb, yt = ycen - self.order_trace_data['BottomEdge'][i], ycen + self.order_trace_data['TopEdge'][i]
        index = self.make_index(yb, yt, x_left_lim, x_right_lim)
        #Do the actual correction
        img_order = self.correct_for_curvature(
                        self.straight[index],
                        self.tilt[i, x_left_lim:x_right_lim],
                        self.shear[i, x_left_lim:x_right_lim],
                        [self.order_trace_data['BottomEdge'][i],self.order_trace_data['TopEdge'][i]])

        if self.plot:

            f, axarr = plt.subplots(2,figsize=(15,7),sharex=True)
            f.suptitle("Order "+str(i))
            axarr[0].imshow(self.flux[index],origin='lower',aspect='auto')
            axarr[1].imshow(img_order,origin='lower',aspect='auto')
            axarr[0].set_title("Input order")
            axarr[1].set_title("Corrected order")
            plt.tight_layout()
            plt.xlabel("Pixel")
            base=os.path.basename(self.sci_frame)
            save_dir = os.path.dirname(self.sci_frame)+"/"+os.path.splitext(base)[0]+"_tilt_correction/"
            try:
                os.mkdir(save_dir)
            except:
                pass
            plt.savefig(
            '{}/Order_{}.png'.format(save_dir,i),
            dpi=500
        )
            plt.close()

        #self.straight[index] = img_order
        
        return index,img_order
            
    
        
    def correct(self):
    
        #Initially, test to see if there is an extension with the name STRAIGHT which already holds the straighened spectrum
        try:
            hdul=fits.open(self.sci_frame)
            test = hdul['STRAIGHT'].data
            Corrected = True
        except:
            Corrected = False
            
        if not Corrected:
            
            self.tilt,self.shear = self.calculate()
            
            n_ord = len(self.order_trace_data)
            column_range = np.zeros((n_ord,2),dtype=np.int16)
            #Loop over all orders to correct
            if self.tilt is not None and self.shear is not None:
                pool=mp.Pool(mp.cpu_count())
                workers = [pool.apply_async(self.run_correction, args=(i,)) for i in range(n_ord)]
                idxs = []
                imgs = []
                for w in workers:
                    idxs.append(w.get()[0])
                    imgs.append(w.get()[1])
                
                for i in range(n_ord):
                    self.straight[idxs[i]] = imgs[i]

            
                with fits.open(self.sci_frame) as hdul:
                    straight_img = fits.ImageHDU(data=self.straight, name="STRAIGHT")
                    hdul.append(straight_img)
                    hdul.writeto(self.sci_frame,overwrite='True')
                
            


    
