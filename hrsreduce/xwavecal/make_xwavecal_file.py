from astropy.io import fits
import numpy as np

from astropy.table import Table

file = 'HR_Master_Wave_H202511270036.fits'

with fits.open(file) as hdul:
    u_data = hdul['FIBRE_U'].data
    l_data = hdul['FIBRE_L'].data
    u_blaz = hdul['BLAZE_U'].data
    l_blaz = hdul['BLAZE_L'].data
    im = hdul[0].data
    hdr = hdul[0].header
   
count = 0
count2 = 0

id = []
flux = []
blaz_flux = []
stderr = []
blz_err = []
pixel = []
fibre = []
ref_id = []
wavelength = []
for ord in range(84):
    id.append(ord)

for ord in range(41,-1,-1):
    ii=np.where(u_data[ord]<0)[0]
    u_data[ord][ii]=1e-12
    
    ii=np.where(l_data[ord]<0)[0]
    l_data[ord][ii]=1e-12
    
    ii=np.where(u_blaz[ord]<0)[0]
    u_blaz[ord][ii]=1e-12
    
    ii=np.where(l_blaz[ord]<0)[0]
    l_blaz[ord][ii]=1e-12
    
    tmp = np.zeros(2048)
    tmp[:len(u_data[ord])] = u_data[ord]
    flux.append(tmp)
    
    tmp = np.zeros(2048)
    tmp[:len(u_data[ord])] = np.sqrt(u_data[ord])
    stderr.append(tmp)
    
    blaz_flux.append(u_data[ord]/u_blaz[ord])
    blz_err.append(np.sqrt(u_data[ord]/u_blaz[ord]))
    
    tmp = np.zeros(2048)
    tmp[:len(l_data[ord])] = l_data[ord]
    flux.append(tmp)
    
    tmp = np.zeros(2048)
    tmp[:len(l_data[ord])] = np.sqrt(l_data[ord])
    stderr.append(tmp)
    
    blaz_flux.append(l_data[ord]/l_blaz[ord])
    blz_err.append(np.sqrt(l_data[ord]/l_blaz[ord]))
        
    pixel.append(np.arange(2048))
    pixel.append(np.arange(2048))
    fibre.append(1)
    fibre.append(2)
    ref_id.append(count)
    ref_id.append(count)
    wavelength.append(np.arange(2048)*0.)
    wavelength.append(np.arange(2048)*0.)
    
    count +=1

flux = np.nan_to_num(flux)
blaz_flux = np.nan_to_num(blaz_flux)

col0 = fits.Column(name='id', format='K', array=np.array(id))
col1 = fits.Column(name='flux', format='PD', array=np.array(flux))
col2 = fits.Column(name='stderr', format='PD', array=np.array(stderr))
col3 = fits.Column(name='pixel', format='PK', array=pixel)
col4 = fits.Column(name='fiber', format='K', array=np.array(fibre))
col5 = fits.Column(name='ref_id', format='K', array=np.array(ref_id))
col6 = fits.Column(name='wavelength', format='PD', array=wavelength)

coldefs = fits.ColDefs([col0, col1, col2, col3, col4, col5, col6])

table_hdu1 = fits.BinTableHDU.from_columns(coldefs,name='SPECBOX')

col0 = fits.Column(name='id', format='K', array=np.array(id))
col1 = fits.Column(name='flux', format='PD', array=blaz_flux)
col2 = fits.Column(name='stderr', format='PD', array=blz_err)
col3 = fits.Column(name='pixel', format='PK', array=pixel)
col4 = fits.Column(name='fiber', format='K', array=np.array(fibre))
col5 = fits.Column(name='ref_id', format='K', array=np.array(ref_id))
col6 = fits.Column(name='wavelength', format='PD', array=wavelength)


coldefs = fits.ColDefs([col0, col1, col2, col3, col4, col5, col6])

table_hdu2 = fits.BinTableHDU.from_columns(coldefs,name='BLZCORR')

date = hdr['DATE-OBS']
time = hdr['TIME-OBS']
datetime=date+"T"+time

hdr['OBJECTS'] = (str("thar&thar&none"), "DUMMY setup for xwavecal")
hdr['DATETIME'] = (datetime, "DUMMY setup for xwavecal")
hdr['RONOISE'] = (float(hdr['RONOISE']), "DUMMY setup for xwavecal")

hdu_out = fits.PrimaryHDU(data=im,header=hdr)

hdul_out = fits.HDUList([hdu_out, table_hdu1,table_hdu2])

spec = Table(hdul_out['SPECBOX'].data)
print(spec.info())

hdul_out.writeto('xwavecal_test.fits', overwrite=True)



from astropy.io import fits
import numpy as np

# --------------------------------------------------------
# Configuration
# --------------------------------------------------------
FILE = "HR_Master_Wave_H202511270036.fits"
MAX_PIX = 2048   # fixed-length column size

# --------------------------------------------------------
# Helper: pad list of arrays to shape (Nrows, MAX_PIX)
# --------------------------------------------------------
def pad_array_list(arr_list, maxlen=MAX_PIX, dtype=float):
    out = np.zeros((len(arr_list), maxlen), dtype=dtype)
    for i, arr in enumerate(arr_list):
        n = min(len(arr), maxlen)
        out[i, :n] = arr[:n]
    return out

# --------------------------------------------------------
# Read FITS input
# --------------------------------------------------------
with fits.open(FILE) as hdul:
    u_data = hdul["FIBRE_U"].data
    l_data = hdul["FIBRE_L"].data
    u_blaz = hdul["BLAZE_U"].data
    l_blaz = hdul["BLAZE_L"].data
    im = hdul[0].data
    hdr = hdul[0].header

# --------------------------------------------------------
# Build table data lists (84 rows total)
# --------------------------------------------------------
id_list = []        # order number (0–83)
flux_list = []
stderr_list = []
pixel_list = []
fiber_list = []
refid_list = []
wavelength_list = []

row_index = 0

for ord in range(41,-1,-1):

    ii=np.where(u_data[ord]<=0)[0]
    u_data[ord][ii]=1e-12
    
    ii=np.where(l_data[ord]<=0)[0]
    l_data[ord][ii]=1e-12
    
    ii=np.where(u_blaz[ord]<=0)[0]
    u_blaz[ord][ii]=1e-12
    
    ii=np.where(l_blaz[ord]<=0)[0]
    l_blaz[ord][ii]=1e-12
    
    # Fibre U
    fU = u_data[ord]
    id_list.append(ord)
    flux_list.append(fU)
    stderr_list.append(np.sqrt(fU))
    pixel_list.append(np.arange(len(fU), dtype=int))
    fiber_list.append(1)
    refid_list.append(row_index)
    wavelength_list.append(np.zeros(len(fU)))
    row_index += 1

    # Fibre L
    fL = l_data[ord]
    id_list.append(ord)
    flux_list.append(fL)
    stderr_list.append(np.sqrt(fL))
    pixel_list.append(np.arange(len(fL), dtype=int))
    fiber_list.append(2)
    refid_list.append(row_index - 1)   # same reference as U fibre
    wavelength_list.append(np.zeros(len(fL)))

# --------------------------------------------------------
# Convert variable-length rows → fixed-length padded arrays
# --------------------------------------------------------
flux_arr       = pad_array_list(flux_list,       MAX_PIX)
stderr_arr     = pad_array_list(stderr_list,     MAX_PIX)
pixel_arr      = pad_array_list(pixel_list,      MAX_PIX, dtype=int)
wavelength_arr = pad_array_list(wavelength_list, MAX_PIX)

id_arr    = np.array(id_list,    dtype=np.int64)
fiber_arr = np.array(fiber_list, dtype=np.int64)
ref_arr   = np.array(refid_list, dtype=np.int64)

flux_arr = np.nan_to_num(flux_arr,nan=1e-12)
stderr_arr = np.nan_to_num(stderr_arr,nan=1e-12) 
#blaz_flux = np.nan_to_num(blaz_flux)

# --------------------------------------------------------
# Build FITS table (fixed length columns)
# --------------------------------------------------------
col_id     = fits.Column(name="id",         format="K",            array=id)
col_flux   = fits.Column(name="flux",       format=f"{MAX_PIX}D",  array=flux_arr)
col_stderr = fits.Column(name="stderr",     format=f"{MAX_PIX}D",  array=stderr_arr)
col_pixel  = fits.Column(name="pixel",      format=f"{MAX_PIX}K",  array=pixel_arr)
col_fiber  = fits.Column(name="fiber",      format="K",            array=fiber_arr)
col_ref    = fits.Column(name="ref_id",     format="K",            array=ref_arr)
col_wave   = fits.Column(name="wavelength", format=f"{MAX_PIX}D",  array=wavelength_arr)

coldefs = fits.ColDefs([col_id, col_flux, col_stderr, col_pixel, col_fiber, col_ref, col_wave])
table_hdu1 = fits.BinTableHDU.from_columns(coldefs,name='SPECBOX')


# --------------------------------------------------------
# Build table data lists (84 rows total)
# --------------------------------------------------------
id_list = []        # order number (0–83)
flux_list = []
stderr_list = []
pixel_list = []
fiber_list = []
refid_list = []
wavelength_list = []

row_index = 0

for ord in range(41,-1,-1):

    ii=np.where(u_data[ord]<0)[0]
    u_data[ord][ii]=1e-12
    
    ii=np.where(l_data[ord]<0)[0]
    l_data[ord][ii]=1e-12
    
    ii=np.where(u_blaz[ord]<0)[0]
    u_blaz[ord][ii]=1e-12
    
    ii=np.where(l_blaz[ord]<0)[0]
    l_blaz[ord][ii]=1e-12
    
    # Fibre U
    fU = u_data[ord]/u_blaz[ord]
    id_list.append(ord)
    flux_list.append(fU)
    stderr_list.append(np.sqrt(fU))
    pixel_list.append(np.arange(len(fU), dtype=int))
    fiber_list.append(1)
    refid_list.append(row_index)
    wavelength_list.append(np.zeros(len(fU)))
    row_index += 1

    # Fibre L
    fL = l_data[ord]/l_blaz[ord]
    id_list.append(ord)
    flux_list.append(fL)
    stderr_list.append(np.sqrt(fL))
    pixel_list.append(np.arange(len(fL), dtype=int))
    fiber_list.append(2)
    refid_list.append(row_index - 1)   # same reference as U fibre
    wavelength_list.append(np.zeros(len(fL)))

# --------------------------------------------------------
# Convert variable-length rows → fixed-length padded arrays
# --------------------------------------------------------
flux_arr       = pad_array_list(flux_list,       MAX_PIX)
stderr_arr     = pad_array_list(stderr_list,     MAX_PIX)
pixel_arr      = pad_array_list(pixel_list,      MAX_PIX, dtype=int)
wavelength_arr = pad_array_list(wavelength_list, MAX_PIX)

id_arr    = np.array(id_list,    dtype=np.int64)
fiber_arr = np.array(fiber_list, dtype=np.int64)
ref_arr   = np.array(refid_list, dtype=np.int64)

flux_arr = np.nan_to_num(flux_arr)

# --------------------------------------------------------
# Build FITS table (fixed length columns)
# --------------------------------------------------------
col_id     = fits.Column(name="id",         format="K",            array=id)
col_flux   = fits.Column(name="flux",       format=f"{MAX_PIX}D",  array=flux_arr)
col_stderr = fits.Column(name="stderr",     format=f"{MAX_PIX}D",  array=stderr_arr)
col_pixel  = fits.Column(name="pixel",      format=f"{MAX_PIX}K",  array=pixel_arr)
col_fiber  = fits.Column(name="fiber",      format="K",            array=fiber_arr)
col_ref    = fits.Column(name="ref_id",     format="K",            array=ref_arr)
col_wave   = fits.Column(name="wavelength", format=f"{MAX_PIX}D",  array=wavelength_arr)

coldefs = fits.ColDefs([col_id, col_flux, col_stderr, col_pixel, col_fiber, col_ref, col_wave])
table_hdu2 = fits.BinTableHDU.from_columns(coldefs,name='BLZCORR')

# --------------------------------------------------------
# Output FITS file
# --------------------------------------------------------

hdr['OBJECTS'] = (str("thar&thar&none"), "DUMMY setup for xwavecal")
hdr['DATETIME'] = (datetime, "DUMMY setup for xwavecal")
hdr['RONOISE'] = (float(hdr['RONOISE']), "DUMMY setup for xwavecal")

primary_hdu = fits.PrimaryHDU(data=im, header=hdr)
hdul_out = fits.HDUList([primary_hdu, table_hdu1])
hdul_out.writeto("xwavecal_test.fits", overwrite=True)

print("Wrote xwavecal_test.fits")
