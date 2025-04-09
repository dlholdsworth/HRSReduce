import logging
import glob
from astropy.io import fits
import matplotlib.pyplot as plt

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
        
        self.alg = OrderTraceAlg(self.flat_data,poly_degree=self.poly_degree,logger=self.logger)
    
    
    
    def order_trace(self):
    
        #Set the night for the master order data bsed on the flat location
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
            
            # 2) assign cluster id and do basic cleaning
            if self.logger:
                self.logger.info("OrderTrace: assigning cluster id and cleaning...")
                #self.logger.warning("OrderTrace: assigning cluster id and cleaning...")
            x, y, index = self.alg.form_clusters(cluster_xy['x'], cluster_xy['y'])

            # 3) advanced cleaning and border cleaning
            if self.logger:
                self.logger.info("OrderTrace: advanced cleaning...")
                #self.logger.warning("OrderTrace: advanced cleaning...")
            new_x, new_y, new_index, all_status = self.alg.advanced_cluster_cleaning_handler(index, x, y)

#            uniq = np.unique(new_index)
#            plt.imshow(self.flat_data,origin='lower',vmin=0,vmax=10)
#            for i in uniq:
#                ii = np.where(new_index == i)[0]
#                plt.plot(new_x[ii],new_y[ii],'.')
#            plt.show()
            new_x, new_y, new_index = self.alg.clean_clusters_on_borders(new_x, new_y, new_index)
            
#            uniq = np.unique(new_index)
#            plt.imshow(self.flat_data,origin='lower',vmin=0,vmax=10)
#            for i in uniq:
#                ii = np.where(new_index == i)[0]
#                plt.plot(new_x[ii],new_y[ii],'.')
#            plt.show()


            # 5) Merge cluster
            if self.logger:
                self.logger.info("OrderTrace: merging cluster...")
                #self.logger.warning("OrderTrace: merging cluster...")
            c_x, c_y, c_index = self.alg.merge_clusters_and_clean(new_index, new_x, new_y)
            
#            uniq = np.unique(c_index)
#            plt.imshow(self.flat_data,origin='lower',vmin=0,vmax=10)
#            for i in uniq:
#                ii = np.where(c_index == i)[0]
#                plt.plot(c_x[ii],c_y[ii],'.')
#            plt.show()
            
            # 6) Find width
            if self.logger:
                self.logger.info("OrderTrace: finding widths...")
                #self.logger.warning("OrderTrace: finding width...")
            all_widths, cluster_coeffs = self.alg.find_all_cluster_widths(c_index, c_x, c_y, power_for_width_estimation=3)

            uniq = np.unique(c_index)
            plt.imshow(self.flat_data,origin='lower',vmin=0,vmax=10)
            for i in uniq:
                count=0
                s_x = int(cluster_coeffs[i, self.poly_degree + 1])
                e_x = int(cluster_coeffs[i, self.poly_degree + 2] + 1)
                x=np.arange(s_x,e_x)
                ord_cen=np.polyval(cluster_coeffs[i,0:self.poly_degree+1],x)

                plt.plot(x,ord_cen,'g')
                plt.plot(x,ord_cen-all_widths[count]['bottom_edge'],'r')
                plt.plot(x,ord_cen+all_widths[count]['top_edge'],'b')
            plt.show()
            
            # 7) post processing
            if self.do_post:
                if self.logger:
                    self.logger.info('OrderTrace: post processing...')

                post_coeffs, post_widths = self.alg.convert_for_post_process(cluster_coeffs, all_widths)
                _, all_widths = self.alg.post_process(post_coeffs, post_widths, orderlet_gap=self.orderlet_gap_pixels)
                
            # 7.5 HRS Specific
            if self.logger:
                self.logger.info("OrderTrace: correcting order lengths...")
            #HRS orders are in pairs, so correct the order lenghts to be the same for the fibre pairs
            max_cluster_no = np.amax(c_index)
            cluster_set = list(range(1, max_cluster_no+1))
            img_edge= self.flat_data.shape[1]

            for n in range(1,max_cluster_no,2):

                s_x_O = int(cluster_coeffs[n, self.poly_degree + 1])
                s_x_P = int(cluster_coeffs[n+1, self.poly_degree + 1])

                if s_x_O != s_x_P:
                    if s_x_O > s_x_P:
                        cluster_coeffs[n+1, self.poly_degree + 1] = cluster_coeffs[n, self.poly_degree + 1]
                    else:
                        s_x_O = s_x_P
                if int(cluster_coeffs[n+1, self.poly_degree + 2])/ img_edge > 0.95:
                    cluster_coeffs[n+1, self.poly_degree + 2] = img_edge
                    cluster_coeffs[n, self.poly_degree + 2] = img_edge
                else:
                    cluster_coeffs[n, self.poly_degree + 2] = cluster_coeffs[n+1, self.poly_degree + 2]
                    
                                 

                    

            # 8) convert result to dataframe
            if self.logger:
                self.logger.info("OrderTrace: writing cluster into dataframe...")

            df = self.alg.write_cluster_info_to_dataframe(all_widths, cluster_coeffs)
            assert(isinstance(df, pd.DataFrame))
            
            #Save to file
            df.to_csv(self.out_dir+self.mode+"_Orders_"+self.arm+yyyymmdd+".csv")
            self.logger.info("OrderTrace: Receipt written")
            self.logger.info("OrderTrace: Done!")

            return str(self.out_dir+self.mode+"_Orders_"+self.arm+yyyymmdd+".csv")
