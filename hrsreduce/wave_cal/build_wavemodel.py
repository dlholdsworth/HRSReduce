from astropy import units as u
from astropy import modeling as mod
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import logging

logger = logging.getLogger(__name__)

def sind(x):
    """Return the sin of x where x is in degrees"""
    if isinstance(x, np.ndarray):
        return np.sin(np.pi * x / 180.0)
    return np.sin(np.radians(x))


def cosd(x):
    """Return the cos of x where x is in degrees"""
    if isinstance(x, np.ndarray):
        return np.cos(np.pi * x / 180.0)
    return np.cos(np.radians(x))



class BuildWaveModel():
    def __init__(self, arm, mode,ord_lens):
        
        self.arm = arm
        self.mode = mode
        self.logger = logger
        self.ord_lens =ord_lens
        
        # start a logger
        self.logger.info('Started BuildWaveModel')
        
    def execute(self):
    
        if self.mode == "HS" or self.mode == "HR":
            self.xpos = -0.025

        if self.arm == "R" and self.mode == "MR":
            self.xpos = 1.325
        elif self.arm == "R" and self.mode == "LR":
            self.xpos = -0.825
            
        elif self.arm == "H" and self.mode == "MR":
            self.xpos = 1.55
        elif self.arm == "H" and self.mode == "LR":
            self.xpos = -0.3
            
        if self.arm == "R":
            camera_name = 'hrdet'
            ords=np.zeros((2,33))
            for i in range(33):
                ords[0][i]=int(i)
                ords[1][i]=85-i
        else:
            camera_name = 'hbdet'
            ords=np.zeros((2,42))
            for i in range(42):
                ords[0][i]=i
                ords[1][i]=125-i
        
    
        rough_wls = []
        for order in range(ords.shape[1]):
            n_order = ords[1][order]

            xarr = np.arange(self.ord_lens[order])

            hrs_model = HRSModel(camera_name=camera_name, order=n_order)
            hrs_model.detector.xpos = self.xpos
            warr = hrs_model.get_wavelength(xarr) * u.mm
            warr = warr.to(u.angstrom).value
            rough_wls.append(warr)

        rough_wls = np.array(rough_wls)
        return rough_wls,ords[1]
            
class HRSModel ():

    """HRSModel is a class that describes the High Resolution Specotrgraph  on SALT
    """

    def __init__(self, grating_name='hrs', camera_name='hrdet', slit=2.0,
                 order=83, gamma=None, xbin=1, ybin=1, xpos=0.00, ypos=0.00):

        # set up the parts of the grating
        self.grating_name = grating_name

        # set the telescope
        self.set_telescope('SALT')

        # set the collimator
        self.set_collimator('hrs')

        # set the camera
        self.set_camera(camera_name)

        # set the detector
        self.set_detector(
            camera_name,
            xbin=xbin,
            ybin=ybin,
            xpos=xpos,
            ypos=ypos)

        # set up the grating
        self.set_grating(self.grating_name, order=order)

        # set up the slit
        self.set_slit(slit)

        # set up the grating angle
        if gamma is not None:
            self.gamma = gamma

    def alpha(self, da=0.00):
        """Return the value of alpha for the spectrograph"""
        return self.grating.blaze + self.gamma

    def beta(self, db=0):
        """Return the value of beta for the spectrograph

           Beta_o=(1+fA)*(camang)-gratang+beta_o
        """
        return self.grating.blaze - self.gamma + db
    def n_index():
        return 1.0000
        
    @staticmethod
    def gratingequation(sigma, order, sign, alpha, beta, gamma=0, nd=n_index):
        """Apply the grating equation to determine the wavelength
           w = sigma/m cos (gamma) * n_ind *(sin alpha +- sin beta)

           returns wavelength in mm
        """
        angle = cosd(gamma) * nd() * (sind(alpha) + sign * sind(beta))
        return sigma / order * angle

        
    def calc_wavelength(self, alpha, beta, gamma=0.0, nd=n_index):
        """Apply the grating equation to determine the wavelength
           returns wavelength in mm
        """
        w = self.gratingequation(self.grating.sigma, self.grating.order, self.grating.sign, alpha, beta, gamma=gamma, nd=nd)
        return w


    def get_wavelength(self, xarr, gamma=0.0):
        """For a given spectrograph configuration, return the wavelength coordinate
           associated with a pixel coordinate.

           xarr: 1-D Array of pixel coordinates
           gamma: Value of gamma for the row being analyzed

           returns an array of wavelengths in mm
        """
        d = self.detector.xbin * self.detector.pix_size * \
            (xarr - self.detector.get_xpixcenter())
        dbeta = -np.degrees(np.arctan(d / self.camera.focallength))
        return self.calc_wavelength(
            self.alpha(), (-self.beta() + dbeta), gamma=gamma)

    def set_telescope(self, name='SALT'):
        if name == 'SALT':
            self.telescope = Optics(name=name, focallength=46200.0)
        else:
            raise SpectrographError('%s is not a supported Telescope' % name)
    def set_collimator(self, name='hrs', focallength=2000.0):
        if name == 'hrs':
            self.collimator = Optics(name=name, focallength=focallength)
        else:
            msg = '{0} is not a supported collimator'.format(name)
            raise SpectrographError(msg)
            
    def set_camera(self, name='hrdet', focallength=None):
        if name == 'hrdet':
            self.camera = Optics(name=name, focallength=402.26)
            self.gamma = 2.43
        elif name == 'hbdet':
            self.camera = Optics(name=name, focallength=333.6)
            self.gamma = 2.00
        else:
            raise SpectrographError('%s is not a supported camera' % name)

    def set_detector(
            self, name='hrdet', geom=None, xbin=1, ybin=1, xpos=0, ypos=0):
        if name == 'hrdet':
            ccd = CCD(name='hrdet', xpix=4122, ypix=4112,
                      pix_size=0.015, xpos=0.00, ypos=0.00)
            self.detector = Detector(name=name, ccd=[ccd], xbin=xbin,
                                     ybin=ybin, xpos=xpos, ypos=ypos)
        elif name == 'hbdet':
            ccd = CCD(name='hrdet', xpix=2100, ypix=4112,
                      pix_size=0.015, xpos=0.00, ypos=0.00)
            self.detector = Detector(name=name, ccd=[ccd], xbin=xbin,
                                     ybin=ybin, xpos=xpos, ypos=ypos)
        else:
            raise SpectrographError('%s is not a supported detector' % name)
            
    def set_grating(self, name=None, order=83):
        if name == 'hrs':
            self.grating = Grating(name='hrs', spacing=41.59, blaze=76.0,
                                   order=order)
            self.set_order(order)
        elif name == 'red beam':
            self.grating = Grating(name='red beam', spacing=855, blaze=0,
                                   order=1)
            self.alpha_angle = 17.5
            self.set_order(1)
        elif name == 'blue beam':
            self.grating = Grating(
                name='blue beam',
                spacing=1850,
                blaze=0,
                order=1)
            self.alpha = 24.6
            self.set_order(1)
        else:
            raise SpectrographError('%s is not a supported grating' % name)

    def set_grating(self, name=None, order=83):
        if name == 'hrs':
            self.grating = Grating(name='hrs', spacing=41.59, blaze=76.0,
                                   order=order)
            self.set_order(order)
        elif name == 'red beam':
            self.grating = Grating(name='red beam', spacing=855, blaze=0,
                                   order=1)
            self.alpha_angle = 17.5
            self.set_order(1)
        elif name == 'blue beam':
            self.grating = Grating(
                name='blue beam',
                spacing=1850,
                blaze=0,
                order=1)
            self.alpha = 24.6
            self.set_order(1)
        else:
            raise SpectrographError('%s is not a supported grating' % name)

    def set_order(self, order):
        self.order = order
        self.grating.order = order
        
    def set_slit(self, slitang=2.2):
        self.slit = Slit(name='Fiber', phi=slitang)
        self.slit.width = self.slit.calc_width(self.telescope.focallength)
   
   
class Optics:

    """A class that describing optics.  All dimensions should in mm.  This assumes all optics
       can be desribed by a diameter and focal length. zpos is the distance in mm that the center
       of the optic is from the primary mirror.   focas is the offset from that position
    """

    def __init__(self, name='', diameter=100, focallength=100, width=100,
                 zpos=0, focus=0):
        # define the variables that describe the opticsg
        self.diameter = diameter
        self.focallength = focallength
        self.width = width
        self.name = name
        # set distances of the optics
        self.zpos = zpos
        self.focus = focus

class CCD:

    """Defines a CCD by x and y position, size, and pixel size.  The x and y position are
       set such that they are zero relative to the detector position.  This assumes that
       the x and y positions are in the center of the pixels and that the ccd is symmetric.

       pix_size is in mm
    """

    def __init__(self, name='', height=0, width=0, xpos=0, ypos=0,
                 pix_size=0.015, xpix=2048, ypix=2048):
        # set the variables
        self.xpos = xpos
        self.ypos = ypos
        self.pix_size = pix_size
        self.xpix = xpix
        self.ypix = ypix
        self.height = self.set_height(height)
        self.width = self.set_width(width)

    def set_height(self, h):
        """If the height is less than the number of pixels, then the height is
           given by the number of pixels
        """
        min = self.ypix * self.pix_size
        return max(h, min)

    def set_width(self, w):
        """If the width  is less than the number of pixels, then the width is
           given by the number of pixels
        """
        min = self.xpix * self.pix_size
        return max(w, min)


class Detector(CCD):

    """A class that describing the Detector.  It inherets from the CCD class as there could be
       multiple ccds at each position.

       name--Name of the detector
       ccd--a CCD class or list describing the CCDs in the detecfor
       xpos--Offset of the x center of the ccd from the central ray in mm
       ypos--Offset of the y center of the ccd from the central ray in mm
       zpos--Offset of the z center of the ccd from the central ray in mm
       xbin--ccd binning in x-direction
       ybin--ccd binning in y-direction
       plate_scale--plate scale in mm/"
    """

    def __init__(self, name='', ccd=CCD(), zpos=0, xpos=0, ypos=0, xbin=2, ybin=2, plate_scale=0.224):

        # Set the detector up as a list of CCDs.
        self.detector = []
        self.pix_size = None
        if isinstance(ccd, CCD):
            self.detector = [ccd]
            self.pix_size = ccd.pix_size
        elif isinstance(ccd, list):
            for c in ccd:
                if isinstance(c, CCD):
                    self.detector.append(c)
                    if self.pix_size:
                        self.pix_size = min(self.pix_size, c.pix_size)
                    else:
                        self.pix_size = c.pix_size
        else:
            return

        self.nccd = len(self.detector)

        # set up the zero points for the detector
        self.name = name
        self.zpos = zpos
        self.xpos = xpos
        self.ypos = ypos
        self.xbin = xbin
        self.ybin = ybin
        self.plate_scale = plate_scale
        self.pix_scale = self.plate_scale / self.pix_size

        # check to make sure that the ccds don't overlap
        self.real = self.check_ccds()

        # determine the max width and height for the detector
        self.width = self.find_width()

        # determine the max width and height for the detector
        self.height = self.find_height()

    def check_ccds(self):
        """Check to make sure none of the ccds overlap"""
        if self.nccd <= 1:
            return True

        # loop over each ccd and check to see if any of the ccd
        # overlaps with the coordinates of another ccd
        for i in range(self.nccd):
            ax1, ax2, ay1, ay2 = self.detector[i].find_corners()
            for j in range(i + 1, self.nccd):
                bx1, bx2, by1, by2 = self.detector[j].find_corners()
                if ax1 <= bx1 < ax2 or ax1 < bx2 < ax2:
                    if ay1 <= by1 < ay2 or ay1 < by2 < ay2:
                        return False

        return True
        
    def find_width(self):
        """Loop over all the ccds in detector and find the width"""
        width = 0
        # return zero if no detector
        if self.nccd < 1:
            return width

        # handle a single detector
        width = self.detector[0].width
        if self.nccd == 1:
            return width

        # Loop over multipe CCDs to find the width
        ax1, ax2, ay1, ay2 = self.detector[0].find_corners()
        xmin = min(ax1, ax2)
        xmax = max(ax1, ax2)
        for ccd in self.detector[1:]:
            ax1, ax2, ay1, ay2 = ccd.find_corners()
            xmin = min(xmin, ax1, ax2)
            xmax = max(xmax, ax1, ax2)
        return xmax - xmin
        
    def find_height(self):
        """Loop over all the ccds in detector and find the height"""
        height = 0
        # return zero if no detector
        if self.nccd < 1:
            return height

        # handle a single detector
        height = self.detector[0].height
        if self.nccd == 1:
            return height

        # Loop over multipe CCDs to find the width
        ax1, ax2, ay1, ay2 = self.detector[0].find_corners()
        ymin = min(ay1, ay2)
        ymax = max(ay1, ay2)
        for ccd in self.detector[1:]:
            ax1, ax2, ay1, ay2 = ccd.find_corners()
            ymin = min(ymin, ay1, ay2)
            ymax = max(ymax, ay1, ay2)
        height = ymax - ymin
        return height

    def get_xpixcenter(self):
        """Return the xpixel center based on the x and y position"""
        return int((0.5 * self.find_width() - self.xpos) / self.pix_size / self.xbin)
        
        
class Grating:

    """A class that describing gratings.  Sigma should be in lines/mm and the
       units of the dimensions should be mm.
    """

    def __init__(self, name='', spacing=600, order=1, height=100, width=100,
                 thickness=100, blaze=0, type='transmission'):
        # define the variables that describe the grating
        self.order = order
        self.height = height
        self.width = width
        self.thickness = thickness
        self.sigma = 1.0 / spacing
        self.blaze = blaze
        self.name = name
        self.type = type
        # set the sign for the grating equation
        self.sign = 1
        if self.type == 'transmission':
            self.sign = -1

class Slit:

    """A class that describing the slit.  Only assuming a single slit.  All sizes are in mm.
       All positions assume the center of the slit. Phi is in arcseconds
    """

    def __init__(self, name='', height=100, width=100, zpos=0, xpos=0, ypos=0, phi=1):

        # define the variables that describe the grating
        self.height = height
        self.width = width
        self.name = name
        self.zpos = zpos
        self.xpos = xpos
        self.ypos = ypos
        self.phi = phi

    def calc_width(self, ftel):
        """Calculate the width assuming ftel*phi(rad).

           returns the slit width in mm
        """
        return ftel * np.radians(self.phi / 3600.0)
