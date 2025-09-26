import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

file = '/Users/daniel/Documents/Work/Teaching/Simon_Ebo/Spectra/HD189995/20220717/mbgphH202207170035_1we.fits'

M_hdu=fits.open(file)
X=[1142,      1148,      1124,      1165,      1170,      1177,      1185,
1193,      1201 ,     1203 ,     1219  ,    1228,      1237,      1247,
1256,      1267 ,     1276 ,     1287 ,     1299,      1309,      1321,
1333,      1345 ,     1357 ,     1370,      1383,      1397,      1411,
1426 ,     1441 ,     1455 ,     1472,      1489,      1506,      1524,
1542 ,     1561 ,     1579]

W_0= [3.826851141083100E+03,  3.858801677339700E+03 , 3.892522326408300E+03,
3.924207345743100E+03,  3.957927994811700E+03,  3.992135424903600E+03,
4.026873888839100E+03,  4.062231892258800E+03,  4.098253687983000E+03,
4.135072034472600E+03,  4.172155897884000E+03,  4.210124817701400E+03,
4.248801782643600E+03,  4.288186792710600E+03,  4.328279847902400E+03,
4.369125201039300E+03,  4.410811357761900E+03,  4.453205559609300E+03,
4.496484817862700E+03,  4.540560626881500E+03,  4.585565745126600E+03,
4.631411666957400E+03,  4.678231150834800E+03,  4.725979943938500E+03,
4.774702299088800E+03,  4.824442469105999E+03 , 4.875200453990100E+03,
4.927064759381700E+03,  4.980035385280800E+03,  5.034156584507700E+03,
5.089516862703001E+03,  5.145983461405800E+03,  5.203777644717600E+03,
5.262899412638400E+03,  5.323348765168200E+03,  5.385169955127300E+03,
5.448495740976600E+03,  5.513326122716099E+03   ]

wave_step  =         0.0442528203
    
file = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0717/reduced/bgoH202207170035.fits'
flat = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0717/reduced/HR_Master_Flat_H20220717.fits'

hdu=fits.open(file)
O_wave = hdu['WAVE_O'].data
O_spec = hdu['FIBRE_O'].data
hdu.close

hdu=fits.open(flat)
O_blaze =hdu['BLAZE_O'].data
P_blaze = hdu['BLAZE_P'].data
hdu.close


for order in range(0,42):
    
    new_wave = np.arange(np.min(O_wave[order]),np.max(O_wave[order]),0.035)
    new_spec = np.interp(new_wave,O_wave[order], O_spec[order])
    
    #plt.plot(O_wave[order],O_spec[order],'b')
    #plt.plot(O_wave[order],O_blaze[order],'r')
    plt.plot(O_wave[order],O_spec[order]/O_blaze[order])
    plt.plot(O_wave[order],O_spec[order]/np.sqrt(O_spec[order]))
    
    #plt.plot(new_wave,new_spec)
    
#for M_order in range(0,1):
#    
#    xconv = lambda v: ((v)*wave_step+W_0[M_order])
#    MIDAS_wave = xconv(np.arange(X[M_order]))
#    new_spec_M = np.interp(new_wave,MIDAS_wave, M_hdu[M_order].data)
  
#plt.plot(MIDAS_wave, M_hdu[M_order].data,'xr')


#plt.plot(new_wave,new_spec-new_spec_M)
plt.show()

#for i in range(len(MIDAS_wave)-1):
#    plt.plot(i,MIDAS_wave[i+1]-MIDAS_wave[i],'ro')

#for order in range(42):
#    for i in range(len(O_wave[order])-1):
#        plt.plot(O_wave[order][i],O_wave[order][i+1]-O_wave[order][i],'bo')
#    
#plt.show()
    
