import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

arm = 'R'

if arm == 'H':
    file = 'Master_Bias_H20220729.fits'
    flat = 'HR_Master_Flat_H20220729.fits'
    with fits.open(file) as hdu:
        image = hdu[0].data
        
    with fits.open(flat) as hdu:
        flat = hdu[0].data

    image /=image
    image -= 1

    image[3509:,538:540] = 1
    image[1430:,484:498] = 1
    image[:,853:854] = 1
    BPM = image
    plt.imshow(image*flat,origin='lower',vmin=0,vmax=10)

    new_hdu = fits.PrimaryHDU(data=BPM.astype(np.float32))
    new_hdu.header.insert(6,('COMMENT',"  FITS (Flexible Image Transport System) format is defined in 'Astronomy"))
    new_hdu.header.insert(7,('COMMENT',"  and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H"))
    new_hdu.header.insert(8,('COMMENT',"Bad Pixel Mask"))

    hdul = fits.HDUList([new_hdu])
    hdul.writeto("BPM_H.fits",overwrite=True)
    
if arm == 'R':
    file = 'Master_Bias_R20220729.fits'
    flat = 'HR_Master_Flat_R20220729.fits'
    with fits.open(file) as hdu:
        image = hdu[0].data
        
    with fits.open(flat) as hdu:
        flat = hdu[0].data

    image /=image
    image -= 1

#    image[3509:,538:540] = 1
#    image[1430:,484:498] = 1
#    image[:,853:854] = 1
    BPM = image
    plt.imshow(image*flat,origin='lower',vmin=0,vmax=10)

    new_hdu = fits.PrimaryHDU(data=BPM.astype(np.float32))
    new_hdu.header.insert(6,('COMMENT',"  FITS (Flexible Image Transport System) format is defined in 'Astronomy"))
    new_hdu.header.insert(7,('COMMENT',"  and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H"))
    new_hdu.header.insert(8,('COMMENT',"Bad Pixel Mask"))

    hdul = fits.HDUList([new_hdu])
    hdul.writeto("BPM_R.fits",overwrite=True)
