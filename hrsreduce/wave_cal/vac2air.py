import numpy as np
import matplotlib.pyplot as plt
def vac2air(wl_vac):
    """
    Convert vacuum wavelengths to wavelengths in air
    Author: Nikolai Piskunov
    """
    wl_air = wl_vac
    ii = np.where(wl_vac > 2e3)

    sigma2 = (1e4 / wl_vac[ii]) ** 2  # Compute wavenumbers squared
    fact = (
        1e0
        + 8.34254e-5
        + 2.406147e-2 / (130e0 - sigma2)
        + 1.5998e-4 / (38.9e0 - sigma2)
    )
    wl_air[ii] = wl_vac[ii] / fact  # Convert to air wavelength
    return wl_air


data_NIST = np.loadtxt("./Intermediate_files/Th_linelist_NIST_air.list", usecols=(0),unpack=True)

Redman = np.loadtxt("Intermediate_files/Redman_line_list.list",usecols=(2),unpack=True)

#intent = intent.astype(int)
air_Redman = vac2air(Redman)

#out_data = np.array([air_data,intent])

#np.savetxt("New_Th_linelist_air.list",out_data.T,fmt='%16.8f %5i')

#plt.vlines(air_data,0,intent)

data = np.loadtxt("thar_list_orig.txt",usecols=0,unpack=True)
plt.vlines(data,0,100,'r')
plt.vlines(data_NIST,0,50,'b')
plt.xlim(3700,9000)

plt.show()
