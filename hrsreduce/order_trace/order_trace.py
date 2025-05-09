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
        
        
        self.alg = OrderTraceAlg(self.flat_data,self.mode, self.sarm, poly_degree=self.poly_degree, logger=self.logger)
    
    
    
    def order_trace(self):
    
        #Set the night for the master order data based on the flat location
        yyyymmdd = str(self.nights['flat'][0:4])+str(self.nights['flat'][4:8])
        self.out_dir = self.base_dir+self.arm+"/"+yyyymmdd[0:4]+"/"+yyyymmdd[4:]+"/reduced/"
    
        #Test if Order file exists
        order_file = glob.glob(self.out_dir+self.mode+"_Orders_"+self.sarm+yyyymmdd+".csv")
        
        if len(order_file) == 0:
            #If there is no ORDER file we need to create one.
            
            # 1) Locate cluster
            if self.logger:
                self.logger.info("OrderTrace: locating cluster...")
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
            HRS_x, HRS_y, HRS_index = self.alg.HRS_clean(c_x, c_y, c_index)
            
            # 6) Find width
            if self.logger:
                self.logger.info("OrderTrace: finding widths...")
                #self.logger.warning("OrderTrace: finding width...")
            all_widths, cluster_coeffs = self.alg.find_all_cluster_widths(HRS_index, HRS_x, HRS_y, power_for_width_estimation=3)
            
            if self.plot:
                uniq = np.unique(HRS_index)
                plt.title("OrderTrace: Widths\nNumber ords="+str(len(uniq)))
                plt.imshow(self.flat_data,origin='lower',vmin=0,vmax=500)

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
                plt.imshow(self.flat_data,origin='lower',vmin=0,vmax=500)

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
                plt.show()
            
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

                s_x_O = int(cluster_coeffs[n, self.poly_degree + 1])
                s_x_P = int(cluster_coeffs[n+1, self.poly_degree + 1])

                if s_x_O != s_x_P:
                    if s_x_O < s_x_P:
                        cluster_coeffs[n+1, self.poly_degree + 1] = cluster_coeffs[n, self.poly_degree + 1]
                    else:
                        cluster_coeffs[n+1, self.poly_degree + 1] = s_x_O
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
