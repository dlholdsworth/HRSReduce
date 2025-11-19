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
                        
    args = parser.parse_args()
    
    nights = args.date
    mode = args.modes
    loc = args.location
    arms = ['Blue','Red']
    
    if mode[0] == 'ALL':
        mode = ["HR"] #Update to ["HR","MR","LR"]
        
    for night in nights:
        mode = mode
        #Run the code for BLUE data
        exit_code = hrsreduce.reduce.main(
            night,
            mode,
            "H",
            loc
        )
        #Run the code for RED data
        exit_code = hrsreduce.reduce.main(
            night,
            mode,
            "R",
            loc
        )
