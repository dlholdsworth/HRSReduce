import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
import scipy.interpolate as inter
import pandas as pd
import alphashape
import shapely
from math import ceil
from scipy import linalg
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import make_smoothing_spline

import hrsreduce.norm.continuum_normalization as pyreduce_cont_norm

class ContNormAlg:
    """
    Continuum normalization module algorithm. Purpose is to measure and remove variability in blaze
    of stellar spectrum.

    Attributes:
        config_param(ConfigHandler): Instance representing pull from config file.
    """

    def __init__(self,logger = None):
        """Initializes continuum normalization algorithm.

        Args:
            mask_array_provided (boolean): Whether or not mask array is provided.
            method (str): Method of continuum normalization within the following:
                'Spline','AFS','Polynomial'.
            continuum_guess_provided (boolean): If initial guess of continuum normalization
                is provided.
            plot_results (boolean): Whether to plot results
            n_iter (int): Number of iterations
            n_order (int): Number of order in polynomial fit
            ffrac (float): Percentile above which is considered as continuum
            a (int): Radius of AFS circle (1/a)
            d (float): Window width in AFS
            config (configparser.ConfigParser, optional): Config context. Defaults to None.
            logger (logging.Lobber, optional): Instance of logging.Logger. Defaults to None.
        """

        self.mask_array_provided=False
        self.method='pyreduce'
        self.continuum_guess_provided=True
        self.plot_results=False
        self.n_iter=5
        self.n_order=8
        self.ffrac=0.0098
        self.med_window=15
        self.std_window=15
        self.a=6
        self.d=0.2
        self.edge_clip=0
        self.output_dir='/Users/daniel/Documents/Work/SALT_Pipeline/HRSReduce/hrsreduce/norm/test'
        self.logger = logger

    def lowess_ag(self, x, y,n_iter=3):
        """Lowess smoother: Robust locally weighted regression.
        The lowess function fits a nonparametric regression curve to a scatterplot.
        The arrays x and y contain an equal number of elements; each pair
        (x[i], y[i]) defines a data point in the scatterplot. The function returns
        the estimated (smooth) values of y.The smoothing span is given by f. A larger
        value for f will result in a smoother curve. The number of robustifying
        iterations is given by iter. The function will run faster with a smaller number of iterations.

        Args:
            x (np.array): X-data points.
            y (np.array): Y-data points.
            n_iter (int): Number of iterations. Defaults to 3.

        Returns:
            yest (np.array): Smoothed values of Y.
        """
        n = len(x)
        r = int(ceil(self.d * n))
        h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]#identify nearest neighbours
        w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
        w = (1 - w ** 3) ** 3
        yest = np.zeros(n)
        delta = np.ones(n)
        for iteration in range(self.n_iter):
            for i in range(n):
                weights = delta * w[:, i]
                b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
                A = np.array([[np.sum(weights), np.sum(weights * x)],
                            [np.sum(weights * x), np.sum(weights * x * x)]])
                beta = linalg.solve(A, b)
                yest[i] = beta[0] + beta[1] * x[i]

            residuals = y - yest
            s = np.median(np.abs(residuals))
            delta = np.clip(residuals / (6.0 * s), -1, 1)
            delta = (1 - delta ** 2) ** 2

        return yest

    def AFS (self,order, q=0.95):
        """Algorithm for alpha-shape fitting to spectrum.

        Args:
            order (np.array): Order of spectrum to remove blaze function. Array of n x 2 shape,
                where n is number of pixels. Each row is wavelength and intensity at each pixel.
            q (float, optional): Refers to upper q quantile within each window to be used
                to fit a local polynomial model. Defaults to 0.95.

        Returns:
            order["intens"].values/y_final (np.array): Normalized flux data.
            y_final (np.array): Smoothed/alpha-shape-fitted flux data.
        """
        # Change the column names and format of the dataset.
        order.columns=["wv","intens"]

        # n records the number of pixels.
        n=order.shape[0]
        # ref is a pandas series recording wavelength
        ref=order["wv"]
        # Variable u is the parameter u in the step 1 of AFS algorithm. It scales the intensity vector.
        u=(ref.max()-ref.min())/10/order["intens"].max()
        order["intens"] = order["intens"]*u

        # Let alpha be 1/6 of the wavelength range of the whole order.
        alpha= (order["wv"].max()-order["wv"].min())/self.a

        # This chunk of code detects loops in the boundary of the alpha shape.
        # Ususally only one loop(polygon). Variable loop is a list.
        # The indices of the k-th loop are recorded in the k-th element of variable loop.
        loops=[]
        # Variable points is a list that represents all the sample point (lambda_i,y_i)
        points=[(order["wv"][i],order["intens"][i]) for i in range(order.shape[0])]
        alpha_shape = alphashape.alphashape(points, 1/alpha)

        # Input Variables:
        # polygon: shapely polygon object
        # return Variable:
        # variable indices is a list recording the indices of the vertices in the polygon
        def find_vertices(polygon):
            coordinates=list(polygon.exterior.coords)
            return [ref[ref==coordinates[i][0]].index[0] for i in range(len(coordinates))]

        # if alpha_shape is just a polygon, there is only one loop
        # if alpha_shape is a multi-polygon, we interate it and find all the loops.
        if (isinstance(alpha_shape,shapely.geometry.polygon.Polygon)):
            temp= find_vertices(alpha_shape)
            loops.append(temp)

        else:
            for polygon in alpha_shape:
                temp= find_vertices(polygon)
                loops.append(temp)

        # Use the loops to get the set W_alpha.
        # Variable Wa is a vector recording the indices of points in W_alpha.
        Wa=[0]
        for loop in loops:
            temp=loop
            temp=loop[:-1]
            temp=[i for i in temp if (i<n-1)]
            max_k=max(temp)
            min_k=min(temp)
            len_k=len(temp)
            as_k=temp
            if((as_k[0] == min_k and as_k[len_k-1] == max_k)==False):
                    index_max= as_k.index(max_k)
                    index_min= as_k.index(min_k)
                    if (index_min < index_max):
                        as_k =as_k[index_min:(index_max+1)]
                    else:
                        as_k= as_k[index_min:]+as_k[0:(index_max+1)]

            Wa=Wa+as_k
        Wa.sort()
        Wa=Wa[1:]

        # AS is an n by 2 matrix recording tilde(AS_alpha). Each row is the wavelength and intensity of one pixel.
        AS=order.copy()
        for i in range(n-1):
            indices=[m for m,v in enumerate(Wa) if v > i]
            if(len(indices)!=0):
                index=indices[0]
                a= Wa[index-1]
                b= Wa[index]
                AS["intens"][i]= AS["intens"][a]+(AS["intens"][b]-AS["intens"][a])*((AS["wv"][i]-AS["wv"][a])/(AS["wv"][b]-AS["wv"][a]))
            else:
                break

        # Run a local polynomial on tilde(AS_alpha), as described in step 3 of the AFS algorithm.
        x=AS["wv"].values
        y=AS["intens"].values

        B1 = self.lowess_ag(x, y)

        select= order["intens"].values/B1

        order["select"]=select
        # Make indices in Wa to the format of small windows.
        # Each row of the variable window is a pair of neighboring indices in Wa.
        window= np.column_stack((Wa[0:len(Wa)-1],Wa[1:]))

        # This chunk of code select the top q quantile of points in each window.
        # The point indices are recorded in variable index, which is S_alpha, q in step 4
        # of the AFS algorithm.
        index=[0]
        for i in range(window.shape[0]):
            loc_window= window[i,]
            temp = order.loc[loc_window[0]:loc_window[1]]
            index_i= temp[temp["select"] >= np.quantile(temp["select"],q)].index
            index=index+list(index_i)
        index=np.unique(index[1:])
        index=np.sort(index)

        # Run Loess for the last time
        x_2=order.iloc[index]["wv"].values
        y_2=order.iloc[index]["intens"].values
        y_final = self.lowess_ag(x_2,y_2)
        # Return the blaze-removed spectrum.

        y_final = InterpolatedUnivariateSpline(x_2,y_final, k=2)(x)
        return order["intens"].values/y_final,y_final


    def spline_fit(self,x,y,window):
        """Perform spline fit.

        Args:
            x (np.array): X-data to be splined (wavelength).
            y (np.array): Y-data to be splined (flux).
            window (float): Window value for breakpoint's number of samples.

        Returns:
            ss (LSQUnivariateSpline): Spline-fit of data.
        """
        breakpoint = np.linspace(np.min(x),np.max(x),int((np.max(x)-np.min(x))/window))
        ss = inter.LSQUnivariateSpline(x,y,breakpoint[1:-1])
        return ss
        
    def flatspec_spline(self,x,rawspec,weight):
        """Performs spline fit for specified number of iterations.

        Args:
            x (np.array): Wavelength data.
            rawspec (np.array): Flux data.
            weight (np.array): Weighting for fitting.

        Returns:
            normspec (np.array): Normalized flux data.
            yfit (np.array): Spline fit.
        """
        pos = np.where((np.isnan(rawspec)==False) & (np.isnan(weight)==False))[0]

        ss = self.spline_fit(x[pos],rawspec[pos],5.)
        yfit = ss(x)

        for i in range(self.n_iter):
            normspec = rawspec / yfit

            pos = np.where((normspec >= self.ffrac) & (yfit > 0))[0]#& (normspec <= 2.)

            ss = self.spline_fit(x[pos],rawspec[pos],5.)
            yfit = ss(x)

        normspec = rawspec / yfit

        return normspec,yfit

    def flatspec(self,x,rawspec,weight):
        """Performs polynomial fitting for specified number of iterations.

        Args:
            x (np.array): Wavelength data.
            rawspec (np.array): Flux data.
            weight (np.array): Weighting for fitting.

        Returns:
            normspec (np.array): Normalized flux data.
            yfit (np.array): Polynomial fit.
        """
        pos = np.where((np.isnan(rawspec)==False) & (np.isnan(weight)==False))[0]

        coef = np.polyfit(x[pos],rawspec[pos],self.n_order)
        poly = np.poly1d(coef)
        yfit = poly(x)

        for i in range(self.n_iter):
            normspec = rawspec / yfit

            pos = np.where((normspec >= self.ffrac))[0]#& (normspec <= 2.)
            #print('order',order)
            coef = np.polyfit(x[pos],rawspec[pos],self.n_order)
            poly = np.poly1d(coef)
            yfit = poly(x)

        normspec = rawspec / yfit

        return normspec,yfit


    def RunningSTD(self,flux):
        """Generates running standard deviation array of flux input.

        Args:
            flux (np.array): Array of flux data.
            
        Returns:
            flux_std (np.array): Array of running standard deviations.
        """
        flux_std = np.ones_like(flux)
        for i in range(len(flux)):
            flux_std[i] = np.nanstd(flux[np.max([0,i-self.std_window]):np.min([len(flux),i+self.std_window])])
        return flux_std


    def RunningMedian(self,flux):
        """Generates running median array of flux input.

        Args:
            flux (np.array): Array of flux data.

        Returns:
            flux_median (np.array): Array of running median fluxes.
        """
        flux_median = np.ones_like(flux)
        for i in range(len(flux)):
            flux_median[i] = np.nanmedian(flux[np.max([0,i-self.med_window]):np.min([len(flux),i+self.med_window])])
        return flux_median


    def continuum_combined(self, wav, data, normalized, weight = None, mask_array = None, continuum_guess = None):
        """Runs continuum normalization according to specified method.

        Args:
            wav (np.array): Wavelength data (sciwav)
            data (np.array): Flux data (sciflux)
            normalized (np.array): Zeros-array placeholder for normalized result.
            weight (np.array): Weighting for fitting. Defaults to none.
            mask_array (np.array, optional): Mask array. Defaults to None.
            continuum_guess (np.array, optional): Continuum guess. Defaults to None.
            method (str, optional): Preferred continuum normalization method. Defaults to 'AFS'.
        """
        if self.mask_array_provided == False:#no outlier or masked lines were provided, we perform outlier rejection ourselves
            mask_array = np.zeros(np.shape(data),'i')

        #lets remove outliers with sigma clipping
        if self.method != 'pyreduce':
            for i in range(np.shape(wav)[0]):
                for i_iter in range(self.n_iter):
                    flux_med = np.nanmedian(data[i,:])
                    data[i,:] =data[i,:]/flux_med

                    flux_median = self.RunningMedian(data[i,:])

                    flux_std = self.RunningSTD(data[i,:])
                    bad = np.where((data[i,:]-flux_median)>3*flux_std)
                    mask_array[i,bad[0]] = 1

                    data[i,bad[0]] = np.nan
                    good = np.where(mask_array[i,:] == 0)[0]

                if self.method == 'Spline':
                    wav[i,bad[0]] = np.nan
                    data[i,bad[0]] = np.nan
                    weight[i,bad[0]] = np.nan
                    normalized[i,:],trend = self.flatspec_spline(wav[i,:],data[i,:],weight[i,:])
                    _,trend = self.flatspec_spline(wav[i,:],continuum_guess[i,:],weight[i,:])
                    normalized[i,:] = data[i,:] / trend

                if self.method == 'Polynomial':
                    wav[i,bad[0]] = np.nan
                    data[i,bad[0]] = np.nan
                    weight[i,bad[0]] = np.nan
                    normalized[i,:],trend = self.flatspec(wav[i,:],data[i,:],weight[i,:])
                    _,trend = self.flatspec(wav[i,:],continuum_guess[i,:],weight[i,:])
                    normalized[i,:] = data[i,:] / trend

                if self.method == 'AFS':
                    dataframe = pd.DataFrame({'wav': np.array(wav[i,good],'d'), 'flux': np.array(data[i,good],'d')}, columns=['wav','flux'])
                    normalized[i,good],trend_= self.AFS(dataframe)
                    trend_ =trend_ /np.max(trend_)*np.percentile(data[i,good],self.ffrac*100)
                    trend = np.zeros_like(normalized[i,:])
                    trend[good] = trend_

        else:
            sm_blz = continuum_guess
            for i in range(wav.shape[0]):
#                spl = make_smoothing_spline(wav[i], continuum_guess[i], lam=1)
#                sm_blz[i]=spl(wav[i])
                sm_blz[i] = self.lowess_ag(wav[i],continuum_guess[i])
                
            s_spec, s_wave, s_blaze,s_sigma = pyreduce_cont_norm.splice_orders(data,wav,sm_blz,weight,scaling=False)

            plt.show()
            trend, full_wave, full_spec, full_cont = pyreduce_cont_norm.continuum_normalize(np.ma.array(s_spec), np.ma.array(s_wave), np.ma.array(s_blaze), np.ma.array(s_sigma), iterations=self.n_iter, scale_vert=1,smooth_initial=3e4, smooth_final=5e6, plot=False, plot_title=None)
            normalized = data.copy()
            for i in range(wav.shape[0]):
                normalized[i] = normalized[i]/trend[i]

                
        if self.plot_results == True:
            #gs = gridspec.GridSpec(2,1 , height_ratios=[1.,1.])
            fig, (ax, ax1) = plt.subplots(2,1, sharex=True,figsize=(12,8))
            plt.subplots_adjust(bottom = 0.15, left = 0.15)
            #ax = plt.subplot(gs[0])
            #ax1 = plt.subplot(gs[1])
            fig.subplots_adjust(hspace=1)
            ax.plot(wav[i],data[i], color = 'blue')
            ax.plot(wav[i],continuum_guess[i], color = 'red')
            ax.plot(wav[i],trend[i],color = 'orange')
            ax1.plot(wav[i],normalized[i],color = 'blue')
            plt.show()
            plt.savefig(self.output_dir+'/'+str(i)+'_'+self.method+'.png')
            plt.close('all')
            
        return normalized,full_wave,full_spec,full_cont,trend
            

    def run_cont_norm(self,wav_i,data_i,flat_i):
        """Prepare and run entire continuum normalization algorithm.

        Args:
            wav_i (np.array): Wavelength data.
            data_i (np.array): Flux data.
        Returns:
            normalized (np.array): Normalized flux data.
        """
        #this is neid set up right now
        #need to set up to get orderlets data
        # wav_i = file_obj[7].data
        # data_i = file_obj[1].data

        #weight = np.ones_like(data_i[:,self.edge_clip:-self.edge_clip])
        
        weight = np.sqrt(data_i)
        
        
        normalized = np.ones_like(data_i)
        mask_array = np.zeros(np.shape(data_i),'i')
        
#        for i in range(42):
#            #plt.plot(s_wave[i],s_spec[i])
#            #plt.plot(s_wave[i],s_blaze[i])
#            #plt.plot(s_wave[i],sm_blz[i])
#            y = self.lowess_ag(wav_i[i],flat_i[i])
#            plt.plot(wav_i[i], flat_i[i])
#            plt.plot(wav_i[i],y,'--')
#        plt.show()
        
        normalized,full_wave,full_spec,full_cont,cont = self.continuum_combined(wav_i, data_i, normalized, continuum_guess = flat_i, weight = weight)

        for i in range(42):
            plt.plot(wav_i[i],data_i[i]/cont[i])
        plt.show()

        return normalized,full_wave,full_spec,full_cont
