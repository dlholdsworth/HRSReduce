# -*- coding: utf-8 -*-
"""
Simple usage example for HRSReduce
Loads a HRS dataset, and runs the full extraction
"""

import os.path
import hrsreduce
from hrsreduce import configuration,reduce

#import pyreduce
#from pyreduce import datasets

if __name__ == '__main__':

    # define parameters
    night = "2024-11-09"
    mode = "MR"
    arm = "H"
    steps = (
         "bias",
         "flat",
         "orders",
         "norm_flat",
         "wavecal",
         "curvature",
         "science",
         "continuum",
         "finalize",
    )

    # some basic settings
    # Expected Folder Structure: base_dir/arm/year/mmdd/raw/*.fits

    config = hrsreduce.configuration.get_configuration_for_instrument(plot=1)

    hrsreduce.reduce.main(
        night,
        mode,
        arm,
        steps,
        configuration=config,
        instrument="HRS",
        plot=True
    )
