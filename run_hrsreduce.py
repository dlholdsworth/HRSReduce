# -*- coding: utf-8 -*-
"""
Simple usage example for HRSReduce
Loads a HRS dataset, and runs the full extraction
"""

import os.path
import hrsreduce
from hrsreduce import reduce
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="run_hrsreduce.py",
                                     description="----------------- HRSReduce Pipeline -----------------",
                                     epilog="run_hrsreduce.py is the call script to invoke the full HRSReduce pipeline\n"
                                            "(D.L.Holdsworth). \n")
    parser.add_argument('-d', '--date', nargs='+',
                        help="Date for which to reduce data, can be many strings of CCYYMMDD, e.g., 20140101 20140102 [no comma or quotes]")
    parser.add_argument('-m','--modes', nargs='+',default ='ALL',
                        help="A list of strings with the instrument modes to reduce. Default = 'ALL', options HR MR LR or combinations thereof [no comma or quotes].")
    parser.add_argument('-l','--location', type=str, default='work',
                        help="Location of the data to reduce. Default = 'work', can be 'archive'")
    parser.add_argument('-c','--clean', type=str, default=True,
                        help="Toggle True/False to clean the intermediate files. Default True")
    parser.add_argument('-rv','--rvst', type=str, default=False,
                        help="Toggle True/False to run the reductions on just RV standard stars. Default False")
    parser.add_argument('-pid','--propid', type=str, default=None,
                        help="Provide a PROPID to reduce just those data. Default None")
                        
    args = parser.parse_args()
    
    nights = args.date
    modes = args.modes
    loc = args.location
    cln = args.clean
    rv = args.rvst
    propid = args.propid
    arms = ['Blue','Red']
    
    if modes[0] == 'ALL':
        modes = ["HR","MR","LR"]
        
    for night in nights:
        for mode in modes:
            #Run the code for BLUE data
            exit_code = hrsreduce.reduce.main(
                night,
                mode,
                "H",
                loc,
                clean=cln,
                cal_rvst=rv,
                propid=propid
            )
            #Run the code for RED data
            exit_code = hrsreduce.reduce.main(
                night,
                mode,
                "R",
                loc,
                clean=cln,
                cal_rvst=rv,
                propid=propid
            )
