# -*- coding: utf-8 -*-
"""
Handles instrument specific info for the HRS spectrograph

Mostly reading data from the header
"""
import logging
import os.path
import re

import numpy as np
from astropy.io import fits
from dateutil import parser
from astropy.time import Time

from .common import InstrumentWithModes, getter,NightFilter,InstrumentFilter#DLH , observation_date_to_night
from .filters import Filter

logger = logging.getLogger(__name__)


class HRS(InstrumentWithModes):
    def __init__(self):
        super().__init__()
        # The date is a combination of the two values
        kw = f"{{{self.info['date']}}}T{{{self.info['universal_time']}}}"
        self.filters["night"].keyword = kw
    
        
    def add_header_info(self, header, mode, **kwargs):
        """read data from header and add it as REDUCE keyword back to the header"""
        # "Normal" stuff is handled by the general version, specific changes to values happen here
        # alternatively you can implement all of it here, whatever works
        
        header = super().add_header_info(header, mode, **kwargs)
        info = self.load_info()
        get = getter(header, info, mode)
        
        obs_date = get("date")
        ut = get("universal_time")
        if obs_date is not None and ut is not None:
            obs_date = f"{obs_date}T{ut}"

        return header

    def get_wavecal_filename(self, header, mode, **kwargs):
        """Get the filename of the wavelength calibration config file"""
        fname = "/Users/daniel/Documents/Work/SALT_Pipeline/PyReduce-HRS/datasets/HRS/reduced/hrs_hs.linelist.npz".format(instrument="hrs",mode=mode)
        return fname
        
    def observation_date_to_night(observation_date):
        """Convert an observation timestamp into the date of the observation night
        Nights start at 12am and end at 12 am the next day

        Parameters
        ----------
        observation_date : datetime
            timestamp of the observation

        Returns
        -------
        night : datetime.date
            night of the observation
        """
        if observation_date == "":
            return None
    
        observation_date = parser.parse(observation_date)
        oneday = datetime.timedelta(days=1)

        if observation_date.hour < 12:
            observation_date -= oneday
        print("DLH******",observation_date.date())
        return observation_date.date()


#    def get_mask_filename(self, mode, **kwargs):
#        i = self.name.lower()HRh
#        m = mode.lower()
#        fname = f"mask_{i}_{m}.fits.gz"
#        cwd = os.path.dirname(__file__)
#        fname = os.path.join(cwd, "..", "masks", fname)
#        return fname

