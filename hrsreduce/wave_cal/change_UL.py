from astropy.io import fits

file = "HR_Super_Arc_R_Reference.fits"

with fits.open(file,mode='update') as hdu:

    new_data_U = hdu['FIBRE_P'].data
    ext_u = fits.ImageHDU(data=new_data_U,name="FIBRE_U")
    ext_u.header["NORDS"] = ((33),"Number of extracted orders")
    ext_u.header["E_MTHD"] = ((0), "Extraction Method. 0: optimal, 1: sum")
    ext_u.header["R_MTHD"] = ((2), "Rectification Method. 0: Norm, 1: Vert, 2: None")
    ext_u.header["FLATFILE"] = (('HR_Master_Flat_R20220729.fits'), "Input flat for extraction")
    ext_u.header["ORDFILE"] = (('HR_Super_Arc_R20220701_Orders_Rect.csv'),"Order trace file")
    hdu.append(ext_u)
    
    new_VAR_U = hdu['FIBRE_P_VAR'].data
    ext_u = fits.ImageHDU(data=new_VAR_U,name="FIBRE_U_VAR")
    hdu.append(ext_u)
    
    new_data_L = hdu['FIBRE_O'].data
    ext_l = fits.ImageHDU(data=new_data_L,name="FIBRE_L")
    ext_l.header["NORDS"] = ((33),"Number of extracted orders")
    ext_l.header["E_MTHD"] = ((0), "Extraction Method. 0: optimal, 1: sum")
    ext_l.header["R_MTHD"] = ((2), "Rectification Method. 0: Norm, 1: Vert, 2: None")
    ext_l.header["FLATFILE"] = (('HR_Master_Flat_R20220729.fits'), "Input flat for extraction")
    ext_l.header["ORDFILE"] = (('HR_Super_Arc_R20220701_Orders_Rect.csv'),"Order trace file")
    hdu.append(ext_l)
    
    new_VAR_L = hdu['FIBRE_O_VAR'].data
    ext_l = fits.ImageHDU(data=new_VAR_L,name="FIBRE_L_VAR")
    hdu.append(ext_l)
    
    new_BLZ_U = hdu['BLAZE_P'].data
    ext_u = fits.ImageHDU(data=new_BLZ_U,name="BLAZE_U")
    hdu.append(ext_u)
    
    new_BLZ_L = hdu['BLAZE_O'].data
    ext_l = fits.ImageHDU(data=new_BLZ_L,name="BLAZE_L")
    hdu.append(ext_l)
    
    hdu.pop('FIBRE_P')
    hdu.pop('FIBRE_P_VAR')
    hdu.pop('BLAZE_P')
    
    hdu.pop('FIBRE_O')
    hdu.pop('FIBRE_O_VAR')
    hdu.pop('BLAZE_O')
    
    hdu.writeto(file,overwrite='True')
    
    
    
