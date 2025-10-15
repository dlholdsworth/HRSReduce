import os
import sys
import matplotlib
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

from astropy.io import fits
from astropy.io import ascii
import numpy as np
import glob
import barycorrpy
import scipy.constants as conts
from astropy.time import Time

from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u

files=sorted(glob.glob('./mbgph?202*u1wm.fits'))

for fname in files:

    #Open the header
    hdu = fits.open(fname)
    header = hdu[0].header
    
    #Extract info for calculating BC and BJD
    
    obs_date = header["DATE-OBS"]
    ut = header["TIME-OBS"]
    
    arm = header["DETNAM"]
    target = header[""]
    if obs_date is not None and ut is not None:
        obs_date = f"{obs_date}T{ut}"
    fwmt = header["EXP-MID"]
    et = header["EXPTIME"]

    if fwmt > 0.:
        mid = float(fwmt)/86400.
    else:
        mid =  float(float(et)/2./86400.)
    
    jd = Time(obs_date,scale='utc',format='isot').jd + mid
    
    lat = -32.3722685109
    lon = 20.806403441
    alt = header["SITEELEV"]
    object = header["OBJECT"]
    
    #object=object.replace(" first fiber","")
    
    co = SkyCoord(header["RA"],header["DEC"],unit=(u.hourangle, u.deg))
    print("DLH",co.dec.degree)
    print("DLH",co.ra.degree)


    BCV =(barycorrpy.get_BC_vel(JDUTC=jd,ra=co.ra.degree, dec=co.dec.degree, lat=lat, longi=lon, alt=alt, leap_update=True,ephemeris='de430'))
    
    print("B CORR", BCV[0])

    BJD = barycorrpy.JDUTC_to_BJDTDB(jd, ra=co.ra.degree, dec=co.dec.degree, lat=lat, longi=lon, alt=alt)

    #Extract the spectrum (in Å)
    spec = hdu[0].data

    specaxis = str(1)
    flux = spec.flatten()
    wave_step = header['CDELT%s' % (specaxis)]
    wave_base = header['CRVAL%s' % (specaxis)]
    reference_pixel = header['CRPIX%s' % (specaxis)]
    xconv = lambda v: ((v-reference_pixel+1)*wave_step+wave_base)
    waveobs = xconv(np.arange(len(flux)))

    
    #Apply the BC
    wave_corr = ((waveobs)*(1.0+(BCV[0]/conts.c)))
    
    if arm == 'HRDET':
        out_arm = "R"
    if arm == 'HBDET':
        out_arm = "B"

    object = object.replace(" ", "_")
    out_fname = (str(object)+"_"+(str(BJD[0])).strip("[]") + "_" + str(out_arm)+"_BC.dat")
        
    ascii.write([waveobs, flux, np.sqrt(np.abs(flux))], out_fname, overwrite=True, names=['WAVE', 'FLUX', 'ERR'], format='tab')

    print(fname,str(BJD[0]).strip("[]"),str(BCV[0]/1000.).strip("[]"))#BCV/1000.)
