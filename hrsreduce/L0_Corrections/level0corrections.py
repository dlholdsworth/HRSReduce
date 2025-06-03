import numpy as np
import glob
import logging
from astropy.io import fits
import re
import matplotlib.pyplot as plt
import os

logger = logging.getLogger(__name__)

class L0Corrections():

    """
    Description:
        This class applyies the level 0 corrections to HRS data. This included the removal of the overscan region, gain correction (inc. for multiport readout) and flipping of Red frames so that the orders run bottom --> top , blue --> red.

    Arguments:


    Attributes:

    """

    def __init__(self,files,nights,targ_night,in_dir,out_dir,base_dir,arm):
    
        self.files=files
        self.nights = nights
        self.tn = targ_night
        self.out_dir = out_dir
        self.in_dir = in_dir
        self.base_dir = base_dir
        self.arm = arm[0]
        if self.arm == "H":
            self.arm_col = "Blu"
            self.naxis1 = 2074
            self.naxis2 = 4102
        elif self.arm == "R":
            self.arm_col = "Red"
            self.naxis1 = 4122
            self.naxis2 = 4112
        logger.info('Started {}'.format(self.__class__.__name__))
    
    def oscan(self,file,night):
    
        '''
        This removes the overscan region, while adding the average as a new header keyword ['OS_AVG']
        '''

        with fits.open(file) as hdu:
            hdr = hdu[0].header
            namps = hdr["NAMPS"]
            if self.arm == "H":
                if namps == 1:
                    overscan_region = hdr["BIASSEC"]
                    data_region = hdr["DATASEC"].strip()
                    delimiters = ":", ",", "[","]"
                    regex_pattern = '|'.join(map(re.escape, delimiters))
                    xy=re.split(regex_pattern, overscan_region)
                    x1=int(xy[1])-1
                    x2=int(xy[2])
                    y1=int(xy[3])-1
                    y2=int(xy[4])
                    
                    mean_os = np.mean(hdu[0].data[y1:y2,x1:x2])
                    hdr['OS_MEAN'] = (mean_os,'Mean of overscan region')
                    
                    xy=re.split(regex_pattern, data_region)
                    x1=int(xy[1])-1
                    x2=int(xy[2])
                    y1=int(xy[3])-1
                    y2=int(xy[4])
                    trimmed = hdu[0].data[y1:y2,x1:x2]
                    hdu[0].data=trimmed
                    
                    hdr["DATASEC"] = str("[1:2048,1:4102]")
                    
                    if night == self.tn:
                        file_out=str(self.out_dir+"o"+file.removeprefix(self.in_dir))
                    else:
                        in_dir = file[:-18]
                        file_out=str(self.base_dir+self.arm_col+"/"+night[0:4]+"/"+night[4:8]+"/reduced/o"+file.removeprefix(in_dir))
                        
                    hdu.writeto(file_out,overwrite=True)
                    
                    return file_out
                    
                elif namps == 2:
                    overscan_region = hdr["BIASSEC"].strip()
                    delimiters = ":", ",", "[","]"
                    regex_pattern = '|'.join(map(re.escape, delimiters))
                    xy=re.split(regex_pattern, overscan_region)
                    
                    #Amp 1
                    bx11=int(xy[1])-1
                    bx12=int(xy[2])
                    by11=int(xy[3])-1
                    by12=int(xy[4])
                    mean_os_1 = np.mean(hdu[0].data[by11:by12,bx11:bx12])
                    #Amp 2
                    bx21=int(xy[7])-1
                    bx22=int(xy[8])
                    by21=int(xy[9])-1
                    by22=int(xy[10])
                    mean_os_2 = np.mean(hdu[0].data[by21:by22,bx21:bx22])
                    
                    #Add the means to the header
                    info = str(str(mean_os_1)+" "+str(mean_os_2))
                    hdr['OS_MEAN'] = (info,'Mean of overscan regions')
                    
                    data_region = hdr["DATASEC"].strip()
                    delimiters = ":", ",", "[","]"
                    regex_pattern = '|'.join(map(re.escape, delimiters))
                    xy=re.split(regex_pattern, data_region)
                    
                    #Amp 1
                    x11=int(xy[1])-1
                    x12=int(xy[2])
                    y11=int(xy[3])-1
                    y12=int(xy[4])
                    data_1 = (hdu[0].data[y11:y12,x11:x12])
                    #Amp 2
                    x21=int(xy[7])-1
                    x22=int(xy[8])
                    y21=int(xy[9])-1
                    y22=int(xy[10])
                    data_2 = (hdu[0].data[y21:y22,x21:x22])

                    #Create an array to hold the data combined and populate it
                    all_data = np.zeros([data_1.shape[0],data_1.shape[1]+data_2.shape[1]],dtype=float)
                    all_data[:,0:data_1.shape[1]]=data_1
                    all_data[:,data_1.shape[1]:data_1.shape[1]+data_2.shape[1]] = data_2
                    
                    area1 = "[1:1024,1:4102]"
                    area2 = "[1025:2048,1:4102]"
                    
                    hdr['DATASEC1'] = (area1,'DATASEC for Amp1')
                    hdr['DATASEC2'] = (area2,'DATASEC for Amp2')
                    hdr['DATASEC'] = str("[1:2048,1:4102]")
                    hdu[0].data = all_data
                    
                    if night == self.tn:
                        file_out=str(self.out_dir+"o"+file.removeprefix(self.in_dir))
                    else:
                        in_dir = file[:-18]
                        file_out=str(self.base_dir+self.arm_col+"/"+night[0:4]+"/"+night[4:8]+"/reduced/o"+file.removeprefix(in_dir))
                    hdu.writeto(file_out,overwrite=True)
                    
                    return file_out
                    
                else:
                    logger.error("Reduction failed in %s as Number of Amplifiers not known", self.__class__.__name__)
                    raise ValueError("Reduction failed in "+self.__class__.__name__+" as Number of Amplifiers not known (given "+str(namps)+" not in [1,2])")
            
            elif self.arm == "R":
                if namps == 1:
                    overscan_region = hdr["BIASSEC"]
                    data_region = hdr["DATASEC"].strip()
                    delimiters = ":", ",", "[","]"
                    regex_pattern = '|'.join(map(re.escape, delimiters))
                    xy=re.split(regex_pattern, overscan_region)
                    x1=int(xy[1])-1
                    x2=int(xy[2])
                    y1=int(xy[3])-1
                    y2=int(xy[4])
                    
                    mean_os = np.mean(hdu[0].data[y1:y2,x1:x2])
                    hdr['OS_MEAN'] = (mean_os,'Mean of overscan region')
                    
                    xy=re.split(regex_pattern, data_region)
                    x1=int(xy[1])-1
                    x2=int(xy[2])
                    y1=int(xy[3])-1
                    y2=int(xy[4])
                    
                    trimmed = hdu[0].data[y1:y2,x1:x2]
                    hdu[0].data = trimmed
                    
                    hdr["DATASEC"] = str("[1:4096,1:4112]")
                    
                    if night == self.tn:
                        file_out=str(self.out_dir+"o"+file.removeprefix(self.in_dir))
                    else:
                        in_dir = file[:-18]
                        file_out=str(self.base_dir+self.arm_col+"/"+night[0:4]+"/"+night[4:8]+"/reduced/o"+file.removeprefix(in_dir))
                    hdu.writeto(file_out,overwrite=True)
                    
                    return file_out
                
                elif namps == 4:
                    overscan_region = hdr["BIASSEC"].strip()
                    delimiters = ":", ",", "[","]"
                    regex_pattern = '|'.join(map(re.escape, delimiters))
                    xy=re.split(regex_pattern, overscan_region)
                    
                    '''
                    This 4 port readout is presumed to follow the following pattern
                    _________________
                    |       |       |
                    | AMP2  | AMP4  |
                    |_______|_______|
                    |       |       |
                    | AMP1  | AMP3  |
                    |_______|_______|
                    '''
                    
                    
                    #Amp 1
                    bx11=int(xy[1])-1
                    bx12=int(xy[2])
                    by11=int(xy[3])-1
                    by12=2056
                    mean_os_1 = np.mean(hdu[0].data[by11:by12,bx11:bx12])
                    #Amp 2
                    bx21=int(xy[1])-1
                    bx22=int(xy[2])
                    by21=2056
                    by22=int(xy[4])
                    mean_os_2 = np.mean(hdu[0].data[by21:by22,bx21:bx22])
                    #Amp 3
                    bx31=int(xy[7])-1
                    bx32=int(xy[8])
                    by31=int(xy[9])-1
                    by32=2056
                    mean_os_3 = np.mean(hdu[0].data[by31:by32,bx31:bx32])
                    #Amp 4
                    bx41=int(xy[7])-1
                    bx42=int(xy[8])
                    by41=2056
                    by42=int(xy[10])
                    mean_os_4 = np.mean(hdu[0].data[by41:by42,bx41:bx42])

                    
                    #Add the means to the header
                    info = str(str(mean_os_1)+" "+str(mean_os_2)+" "+str(mean_os_3)+" "+str(mean_os_4))
                    hdr['OS_MEAN'] = (info,'Mean of overscan regions')
                    
                    data_region = hdr["DATASEC"].strip()
                    delimiters = ":", ",", "[","]"
                    regex_pattern = '|'.join(map(re.escape, delimiters))
                    xy=re.split(regex_pattern, data_region)

                    #Amp 1
                    x11=int(xy[1])-1
                    x12=int(xy[2])
                    y11=int(xy[3])-1
                    y12=2056
                    data_1 = (hdu[0].data[y11:y12,x11:x12])
#                    plt.imshow(data_1,origin='lower',aspect='auto',vmin=950,vmax=1000)
#                    plt.show()
                    #Amp 2
                    x21=int(xy[1])-1
                    x22=int(xy[2])
                    y21=2056
                    y22=int(xy[4])
                    data_2 = (hdu[0].data[y21:y22,x21:x22])
#                    plt.imshow(data_2,origin='lower',aspect='auto',vmin=950,vmax=1000)
#                    plt.show()
                    #Amp 3
                    x31=int(xy[7])-1
                    x32=int(xy[8])
                    y31=int(xy[9])-1
                    y32=2056
                    data_3 = (hdu[0].data[y31:y32,x31:x32])
#                    plt.imshow(data_3,origin='lower',aspect='auto',vmin=950,vmax=1000)
#                    plt.show()
                    #Amp4
                    x41=int(xy[7])-1
                    x42=int(xy[8])
                    y41=2056
                    y42=int(xy[10])
                    data_4 = (hdu[0].data[y41:y42,x41:x42])
#                    plt.imshow(data_4,origin='lower',aspect='auto',vmin=950,vmax=1000)
#                    plt.show()

                    #Create an array to hold the data combined and populate it
                    all_data = np.zeros([data_1.shape[0]+data_3.shape[0],data_1.shape[1]+data_2.shape[1]])

                    all_data[0:data_1.shape[0],0:data_1.shape[1]]=data_1
                    all_data[0:data_2.shape[0],data_1.shape[1]:data_1.shape[1]+data_2.shape[1]]=data_3
                    all_data[data_1.shape[0]:data_1.shape[0]+data_3.shape[0],0:data_3.shape[1]]=data_2
                    all_data[data_1.shape[0]:data_1.shape[0]+data_3.shape[0],data_3.shape[1]:data_3.shape[1]+data_4.shape[1]] = data_4
                    
                    area1 = "[1:2048,1:2056]"
                    area2 = "[1:2048,2057:4112]"
                    area3 = "[2049:4096,1:2056]"
                    area4 = "[2049:4096,2057:4112]"
                    
                    hdr['DATASEC1'] = (area1,'DATASEC for Amp1')
                    hdr['DATASEC2'] = (area2,'DATASEC for Amp2')
                    hdr['DATASEC3'] = (area3,'DATASEC for Amp3')
                    hdr['DATASEC4'] = (area4,'DATASEC for Amp4')
                    
                    hdr["DATASEC"] = str("[1:4096,1:4112]")

                    hdu[0].data = all_data
                    
                    if night == self.tn:
                        file_out=str(self.out_dir+"o"+file.removeprefix(self.in_dir))
                    else:
                        in_dir = file[:-18]
                        file_out=str(self.base_dir+self.arm_col+"/"+night[0:4]+"/"+night[4:8]+"/reduced/o"+file.removeprefix(in_dir))
                        
                    hdu.writeto(file_out,overwrite=True)
                    
                    return file_out
                else:
                    logger.error("Reduction failed in %s as Number of Amplifiers not known", self.__class__.__name__)
                    raise ValueError("Reduction failed in "+self.__class__.__name__+" as Number of Amplifiers not known (given "+str(namps)+" not in [1,4])")
    
    
    
    def gain(self, file,night):
    
        '''
        This corrects for the gain converting to adu to photoelectrons, dealing with the relavent amplifiers if needed.
        It also flips the Red arm data
        '''
        
        with fits.open(file) as hdu:
        
            hdr=hdu[0].header
            namps = hdr["NAMPS"]
            if self.arm == "H":
                if namps == 1:
                    gain = hdr["GAIN"]
                    gain = float(gain.split()[0])
                    corrected = np.float32(hdu[0].data * gain)
                    hdu[0].data = corrected
                    
                    if night == self.tn:
                        file_out=str(self.out_dir+"g"+file.removeprefix(self.out_dir))
                    else:
                        in_dir = file[:-19]
                        file_out=str(self.base_dir+self.arm_col+"/"+night[0:4]+"/"+night[4:8]+"/reduced/g"+file.removeprefix(in_dir))
                        
                    #Update the overscan mean
                    os = float(hdr["OS_MEAN"])
                    hdr['AVG_GAIN'] = (gain,'Average gain for all amps')
                    hdr['OS_MEAN'] = str("%.4f" % (os * gain))
                    hdu.writeto(file_out,overwrite=True)
                    return file_out
                    
                elif namps == 2:
                
                    gain = hdr["GAIN"]
                    gain1 = float(gain.split()[0])
                    gain2 = float(gain.split()[1])
                    
                    #Apply the gians to the correct parts of the new image
                    datasec1 = hdr["DATASEC1"]
                    datasec2 = hdr["DATASEC2"]
                   
                    delimiters = ":", ",", "[","]"
                    regex_pattern = '|'.join(map(re.escape, delimiters))
                    
                    #Amp 1
                    xy=re.split(regex_pattern, datasec1)
                    x11=int(xy[1])-1
                    x12=int(xy[2])
                    y11=int(xy[3])-1
                    y12=int(xy[4])
                    hdu[0].data[y11:y12,x11:x12] = (hdu[0].data[y11:y12,x11:x12]) * gain1
                    
                    #Amp2
                    xy=re.split(regex_pattern, datasec2)
                    x21=int(xy[1])-1
                    x22=int(xy[2])
                    y21=int(xy[3])-1
                    y22=int(xy[4])
                    hdu[0].data[y21:y22,x21:x22] = (hdu[0].data[y21:y22,x21:x22]) * gain2
                    
                    hdu[0].data= np.float32(hdu[0].data)

                    hdr['AVG_GAIN'] = ((gain1+gain2)/2.,'Average gain for all amps')
                    
                    #Update the overscan mean
                    os1 = str("%.4f" % (float(hdr["OS_MEAN"].split()[0])*gain1))
                    os2 = str("%.4f" % (float(hdr["OS_MEAN"].split()[1])*gain2))
                    info = str(os1+" "+os2)
                    hdr['OS_MEAN'] = (info)
                    
                    
                    if night == self.tn:
                        file_out=str(self.out_dir+"g"+file.removeprefix(self.out_dir))
                    else:
                        in_dir = file[:-19]
                        file_out=str(self.base_dir+self.arm_col+"/"+night[0:4]+"/"+night[4:8]+"/reduced/g"+file.removeprefix(in_dir))
                        
                    hdu.writeto(file_out,overwrite=True)
                    
                    return file_out
                
                else:
                    logger.error("Reduction failed in %s as Number of Amplifiers not known", self.__class__.__name__)
                    raise ValueError("Reduction failed in "+self.__class__.__name__+" as Number of Amplifiers not known (given "+str(namps)+" not in [1,2])")
            
            
            if self.arm == "R":
                if namps == 1:
                    gain = hdr["GAIN"]
                    gain = float(gain.split()[0])
                    corrected = np.float32(hdu[0].data * gain)
                    
                    #Flip the Red frame so orders run from bottom to top, blue to red
                    hdu[0].data = corrected[::-1,::]
                    
                    if night == self.tn:
                        file_out=str(self.out_dir+"g"+file.removeprefix(self.out_dir))
                    else:
                        in_dir = file[:-19]
                        file_out=str(self.base_dir+self.arm_col+"/"+night[0:4]+"/"+night[4:8]+"/reduced/g"+file.removeprefix(in_dir))
                                        #Update the overscan mean
                    
                    os = float(hdr["OS_MEAN"])
                    hdr['OS_MEAN'] = str("%.4f" % (os * gain))
                    
                    hdr['AVG_GAIN'] = (gain,'Average gain for all amps')
                    hdu.writeto(file_out,overwrite=True)
                    
                    return file_out
                    
                elif namps == 4:
                
                    gain = hdr["GAIN"]
                    gain1 = float(gain.split()[0])
                    gain2 = float(gain.split()[1])
                    gain3 = float(gain.split()[2])
                    gain4 = float(gain.split()[3])
                    
                    #Apply the gians to the correct parts of the new image
                    datasec1 = hdr["DATASEC1"]
                    datasec2 = hdr["DATASEC2"]
                    datasec3 = hdr["DATASEC3"]
                    datasec4 = hdr["DATASEC4"]
                   
                    delimiters = ":", ",", "[","]"
                    regex_pattern = '|'.join(map(re.escape, delimiters))
                    
                    #Amp 1
                    xy=re.split(regex_pattern, datasec1)
                    x11=int(xy[1])-1
                    x12=int(xy[2])
                    y11=int(xy[3])-1
                    y12=int(xy[4])
                    hdu[0].data[y11:y12,x11:x12] = (hdu[0].data[y11:y12,x11:x12]) * gain1
                    
                    #Amp2
                    xy=re.split(regex_pattern, datasec2)
                    x21=int(xy[1])-1
                    x22=int(xy[2])
                    y21=int(xy[3])-1
                    y22=int(xy[4])
                    hdu[0].data[y21:y22,x21:x22] = (hdu[0].data[y21:y22,x21:x22]) * gain2
                    
                    #Amp3
                    xy=re.split(regex_pattern, datasec3)
                    x31=int(xy[1])-1
                    x32=int(xy[2])
                    y31=int(xy[3])-1
                    y32=int(xy[4])
                    hdu[0].data[y31:y32,x31:x32] = (hdu[0].data[y31:y32,x31:x32]) * gain3
                    
                    #Amp4
                    xy=re.split(regex_pattern, datasec4)
                    x41=int(xy[1])-1
                    x42=int(xy[2])
                    y41=int(xy[3])-1
                    y42=int(xy[4])
                    hdu[0].data[y41:y42,x41:x42] = (hdu[0].data[y41:y42,x41:x42]) * gain4
                    
                    #Flip the Red frame so orders run from bottom to top, blue to red
                    hdu[0].data = np.float32(hdu[0].data[::-1,::])
                    
                    hdr['AVG_GAIN'] = ((gain1+gain2+gain3+gain4)/4.,'Average gain for all amps')
                    
                    #Update overscan mean
                    os1 = str("%.4f" % (float(hdr["OS_MEAN"].split()[0])*gain1))
                    os2 = str("%.4f" % (float(hdr["OS_MEAN"].split()[1])*gain2))
                    os3 = str("%.4f" % (float(hdr["OS_MEAN"].split()[2])*gain3))
                    os4 = str("%.4f" % (float(hdr["OS_MEAN"].split()[3])*gain4))
                    info = str(os1+" "+os2+" "+os3+" "+os4)
                    hdr['OS_MEAN'] = (info)
                    
                    if night == self.tn:
                        file_out=str(self.out_dir+"g"+file.removeprefix(self.out_dir))
                    else:
                        in_dir = file[:-19]
                        file_out=str(self.base_dir+self.arm_col+"/"+night[0:4]+"/"+night[4:8]+"/reduced/g"+file.removeprefix(in_dir))
                        
                    hdu.writeto(file_out,overwrite=True)
                    
                    return file_out
                
                else:
                    logger.error("Reduction failed in %s as Number of Amplifiers not known", self.__class__.__name__)
                    raise ValueError("Reduction failed in "+self.__class__.__name__+" as Number of Amplifiers not known (given "+str(namps)+" not in [1,4])")

    def run(self):
    
        #First check to see if this has been run by checking for intermediate files
        for type in self.files.keys():
            for i,file in enumerate(self.files[type]):
                processed_file = glob.glob(str(self.out_dir+"go"+file.removeprefix(self.in_dir)))
                if len(processed_file) == 1:
                    self.files[type][i]=processed_file[0]
                    continue
                else:
                    file_o = self.oscan(file,self.nights[type])
                    file = self.gain(file_o,self.nights[type])
                    
                    os.remove(file_o)

                    self.files[type][i]=file
        
        return self.files
                    
                    
                


