import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy import constants as const
import scipy.constants as conts
from astropy.modeling import models, fitting
import math
import glob
import dlh_RV_calc
from astroquery.simbad import Simbad
import os
import arrow

custom_simbad = Simbad()
custom_simbad.add_votable_fields('rv_value')
#Simbad.list_votable_fields()[["name", "description"]]
custom_simbad.add_votable_fields('sptype')

fig, axs = plt.subplots(3, 1, figsize=(10, 6),sharex=True)


arm='H'
files =sorted(glob.glob("/Users/daniel/Desktop/SALT_HRS_DATA/Blu/????/????/reduced/*product.fits"))
rvs_H = []
rv_errs_H = []
BJDs_H = []

rvs_R = []
rv_errs_R = []
BJDs_R = []

rvs_HR =[]
rv_errs_HR = []
BJDs_HR = []

for sci in files:
    with fits.open(sci) as hdu:
        if hdu[0].header["PROPID"] == "CAL_RVST":
            cal_frame =hdu[0].header["MSTRWAVE"]
            cal_date = cal_frame[4:12]
            targ_date =hdu[0].header["DATE-OBS"]
            targ_date = str(targ_date[0:4])+str(targ_date[5:7])+str(targ_date[8:10])
            cal_diff = ((arrow.get(int(cal_date[0:4]),int(cal_date[4:6]),int(cal_date[6:8])) - arrow.get(int(targ_date[0:4]),int(targ_date[4:6]),int(targ_date[6:8]))).days)
            #Find the equivalent red frame
            f_name = os.path.basename(sci)
            f_name = f_name[1:]
            files_R =sorted(glob.glob("/Users/daniel/Desktop/SALT_HRS_DATA/Red/????/????/reduced/*"+f_name))
            if len(files_R) > 0:
                print("FILE= ",sci)
                spectrum = hdu['FIBRE_O'].data
                waves = hdu['WAVE_O'].data
                blaze = hdu['BLAZE_O'].data
                header=hdu[0].header
            
                bary_corr = float(header['BARYRV'])*1000.
                BJD = float(header['BJD'])
            
                result = custom_simbad.query_object(header['OBJECT'])
                known_rv = (result["RV_VALUE"][0])*1000.
                sp = result['SP_TYPE'][0][0:2]
                
                
                if sp[0] == 'F':
                    mask_name = "F9_espresso.txt"
                if sp[0] == 'G':
                    sp_available = np.array([2,8])
                    diff = np.abs(sp_available - float(sp[1]))
                    ii=np.where(diff == np.min(diff))[0]
                    mask_name = "G"+str(sp_available[ii][0])+"_espresso.txt"
                if sp[0] == 'K':
                    sp_available = np.array([2,6])
                    diff = np.abs(sp_available - float(sp[1]))
                    ii=np.where(diff == np.min(diff))[0]
                    mask_name = "K"+str(sp_available[ii][0])+"_espresso.txt"
                
                #mask_name = "F9_espresso.txt"
            
                good_ords=[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]

                rv_H,rv_err_H=dlh_RV_calc.execute(header,waves,spectrum,blaze,known_rv,bary_corr,BJD, good_ords,mask_name)
            
                BJD -=2450000.0
                BJDs_H.append(BJD)
   
                rvs_H.append(known_rv - rv_H)
                rv_errs_H.append(rv_err_H)
            
                print("\n\nTarget: ",header['OBJECT'], "\nBlue ΔRV",known_rv - rv_H,"±",rv_err_H)
            
                axs[0].errorbar(BJD,(known_rv - rv_H),rv_err_H,color='b',fmt='o')
                
                with fits.open(files_R[0]) as hdu_R:
                    spectrum = hdu_R['FIBRE_O'].data
                    waves = hdu_R['WAVE_O'].data
                    blaze = hdu_R['BLAZE_O'].data
                    header=hdu_R[0].header
            
                    bary_corr = float(header['BARYRV'])*1000.
                    BJD = float(header['BJD'])
            
                    result = custom_simbad.query_object(header['OBJECT'])
                    known_rv = (result["RV_VALUE"][0])*1000.
            
                    good_ords=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]

                    rv_R,rv_err_R=dlh_RV_calc.execute(header,waves,spectrum,blaze,known_rv,bary_corr,BJD, good_ords,mask_name)
            
                    BJD -=2450000.0
                    BJDs_R.append(BJD)
            
                    rvs_R.append(known_rv - rv_R)
                    rv_errs_R.append(rv_err_R)
                    
    #                both_rv[1]=known_rv - rv_R
    #                both_err[1] = rv_err_R
                    
    #                rvs_HR.append(np.average(both_rv,weights = 1/both_err))
    #                chunk_std = np.std(rvs_HR, ddof=1)
    #                chunk_err = chunk_std / np.sqrt(len(rvs_HR))
    #                chunk_err = np.sqrt(np.sum(both_err^2))
    #                rv_errs_HR.append(chunk_err)
            
                    print("Red ΔRV",known_rv - rv_R,"±",rv_err_R)
            
                    axs[1].errorbar(BJD,(known_rv - rv_R),rv_err_R,color='r',fmt='o')
                    #axs[3].errorbar(cal_diff,(known_rv - rv_R),rv_err_R,color='r',fmt='.')
                    #axs[3].errorbar(cal_diff,(known_rv - rv_H),rv_err_H,color='b',fmt='.')
        
rvs_H = np.array(rvs_H)
rv_errs_H = np.array(rv_errs_H)
BJDs_H = np.array(BJDs_H)

chunk_std = np.std(rvs_H, ddof=1)
chunk_err = chunk_std / np.sqrt(rvs_H.size)  # standard error of the mean

rv_mean = np.average(rvs_H,weights=(1/rv_errs_H),returned=True)
mn_rv = rv_mean[0]

axs[0].hlines(mn_rv,np.min(BJDs_H),np.max(BJDs_H),'b',lw=0.7)
axs[0].hlines(mn_rv-chunk_err,np.min(BJDs_H),np.max(BJDs_H),'b',ls='--',lw=0.7)
axs[0].hlines(mn_rv+chunk_err,np.min(BJDs_H),np.max(BJDs_H),'b',ls='--',lw=0.7)
axs[0].hlines(0,np.min(BJDs_H),np.max(BJDs_H),'k',lw=0.7)
axs[0].set_ylabel("Blue ΔRV (m/s)")


axs[0].text(0.5,0.9, "HR mode, Blue arm Mean = "+str(np.round(mn_rv,decimals=2)) + "±"+str(np.round(chunk_err,decimals=2))+" m/s",
            color='blue', fontsize=10,ha='center',va='top',transform=axs[0].transAxes)
            

##plt.ylim(np.max(np.abs(rvs))*(-1.5),np.max(np.abs(rvs))*1.5)
##
#
#arm='R'
#files =sorted(glob.glob("/Users/daniel/Desktop/SALT_HRS_DATA/Red/????/????/reduced/*product.fits"))
#
#rvs = []
#rv_errs = []
#BJDs = []
#
#for sci in files:
#    with fits.open(sci) as hdu:
#        if hdu[0].header["PROPID"] == "CAL_RVST":
#            spectrum = hdu['FIBRE_O'].data
#            waves = hdu['WAVE_O'].data
#            blaze = hdu['BLAZE_O'].data
#            header=hdu[0].header
#            
#            bary_corr = float(header['BARYRV'])*1000.
#            BJD = float(header['BJD'])
#            
#            result = custom_simbad.query_object(header['OBJECT'])
#            known_rv = (result["RV_VALUE"][0])*1000.
#            
#            if arm == 'R':
#                good_ords=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
#
#            rv,rv_err=dlh_RV_calc.execute(header,waves,spectrum,blaze,known_rv,bary_corr,BJD, good_ords)
#            
#            BJD -=2450000.0
#            BJDs.append(BJD)
#            
#            rvs.append(known_rv - rv)
#            rv_errs.append(rv_err)
#            
#            print("\n\n Target: ",header['OBJECT'], "\nΔRV",known_rv - rv,"±",rv_err)
#            
#            axs[1].errorbar(BJD,(known_rv - rv),rv_err,color='r',fmt='o')
#        
#        
rvs_R = np.array(rvs_R)
rv_errs_R = np.array(rv_errs_R)
BJDs_R = np.array(BJDs_R)

chunk_std = np.std(rvs_R, ddof=1)
chunk_err = chunk_std / np.sqrt(rvs_R.size)  # standard error of the mean

rv_mean = np.average(rvs_R,weights=(1/rv_errs_R),returned=True)
mn_rv = rv_mean[0]

axs[1].hlines(mn_rv,np.min(BJDs_R),np.max(BJDs_R),'r',lw=0.7)
axs[1].hlines(mn_rv-chunk_err,np.min(BJDs_R),np.max(BJDs_R),'r',ls='--',lw=0.7)
axs[1].hlines(mn_rv+chunk_err,np.min(BJDs_R),np.max(BJDs_R),'r',ls='--',lw=0.7)
axs[1].hlines(0,np.min(BJDs_R),np.max(BJDs_R),'k',lw=0.7)
axs[1].set_ylabel("Red ΔRV (m/s)")

axs[1].text(0.5,0.9, "HR mode, Red arm Mean = "+str(np.round(mn_rv,decimals=2)) + "±"+str(np.round(chunk_err,decimals=2))+" m/s",
            color='red', fontsize=10,ha='center',va='top',transform=axs[1].transAxes)
            



for epoch in range(len(BJDs_H)):
    both_rv = np.zeros(2)
    both_err = np.zeros(2)
    both_rv[0],both_rv[1] = rvs_H[epoch],rvs_R[epoch]
    both_err[0],both_err[1] =rv_errs_H[epoch],rv_errs_R[epoch]
    
    rv_combined = np.average(both_rv,weights=1./both_err)
#    chunk_std = np.std(rv_combined, ddof=1)
#    chunk_err = chunk_std / np.sqrt(2)
    
    axs[2].errorbar(BJDs_R[epoch],rv_combined,chunk_err,color='k',fmt='o')
    rvs_HR.append(rv_combined)
    rv_errs_HR.append(np.sqrt(both_err[0]**2+both_err[1]**2))

rv_errs_HR = np.array(rv_errs_HR)
rvs_HR = np.array(rvs_HR)
chunk_std = np.std(rvs_HR)
chunk_err = chunk_std / np.sqrt(rvs_HR.size)  # standard error of the mean

rv_mean = np.average(rvs_HR,weights=(1/rv_errs_HR),returned=True)
mn_rv = rv_mean[0]

axs[2].hlines(mn_rv,np.min(BJDs_R),np.max(BJDs_R),'k',lw=0.7)
axs[2].hlines(mn_rv-chunk_err,np.min(BJDs_R),np.max(BJDs_R),'k',ls='--',lw=0.7)
axs[2].hlines(mn_rv+chunk_err,np.min(BJDs_R),np.max(BJDs_R),'k',ls='--',lw=0.7)
axs[2].hlines(0,np.min(BJDs_R),np.max(BJDs_R),'k',lw=0.7)
axs[2].set_ylabel("ΔRV (m/s)")

axs[2].text(0.5,0.9, "HR mode = "+str(np.round(mn_rv,decimals=2)) + "±"+str(np.round(chunk_err,decimals=2))+" m/s",
            color='black', fontsize=10,ha='center',va='top',transform=axs[2].transAxes)
            
axs[2].set_xlabel("BJD-2450000 (days)")

axs[0].set_ylim(-750,750)
axs[1].set_ylim(-750,750)
axs[2].set_ylim(-750,750)
#axs[3].set_ylim(-750,750)
#axs[3].set_xlim(-7,7)

#axs[3].set_ylabel("ΔRV (m/s)")
#axs[3].set_xlabel("Days since Arc Frame")

print("Blue error range:", np.min(rv_errs_H),np.max(rv_errs_H),np.average(rv_errs_H))
print("Red error range:", np.min(rv_errs_R),np.max(rv_errs_R),np.average(rv_errs_R))
print("Combined error range:", np.min(rv_errs_HR),np.max(rv_errs_HR),np.average(rv_errs_HR))
print("Number of spectra analysed:", len(rv_errs_HR))

plt.savefig("RV_stability.png",bbox_inches='tight',dpi=900)
plt.show()
