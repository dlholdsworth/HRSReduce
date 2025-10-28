import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from scipy.special import binom
from mpl_toolkits.mplot3d import axes3d, Axes3D
from astropy.io import fits

def evaluate_solution(pos, order, solution, dimensionality="2D"):
    """
    Evaluate the 1d or 2d wavelength solution at the given pixel positions and orders

    Parameters
    ----------
    pos : array
        pixel position on the detector (i.e. x axis)
    order : array
        order of each point
    solution : array of shape (nord, ndegree) or (degree_x, degree_y)
        polynomial coefficients. For mode=1D, one set of coefficients per order.
        For mode=2D, the first dimension is for the positions and the second for the orders
    mode : str, optional
        Wether to interpret the solution as 1D or 2D polynomials, by default "1D"

    Returns
    -------
    result: array
        Evaluated polynomial

    Raises
    ------
    ValueError
        If pos and order have different shapes, or mode is of the wrong value
    """
    if not np.array_equal(np.shape(pos), np.shape(order)):
        raise ValueError("pos and order must have the same shape")

#    if step_mode:
#        return evaluate_step_solution(pos, order, solution)

    if dimensionality == "1D":
        result = np.zeros(pos.shape)
        for i in np.unique(order):
            select = order == i
            result[select] = np.polyval(solution[int(i)], pos[select])
    elif dimensionality == "2D":
        result = np.polynomial.polynomial.polyval2d(pos, order, solution)
    else:
        raise ValueError(
            f"Parameter 'mode' not understood, expected '1D' or '2D' but got {self.dimensionality}"
        )
    return result
    
def make_wave(wave_solution, nord, ncol, plot=False):
    """Expand polynomial wavelength solution into full image

    Parameters
    ----------
    wave_solution : array of shape(degree,)
        polynomial coefficients of wavelength solution
    plot : bool, optional
        wether to plot the solution (default: False)

    Returns
    -------
    wave_img : array of shape (nord, ncol)
        wavelength solution for each point in the spectrum
    """

    y, x = np.indices((nord, ncol))
    wave_img = evaluate_solution(x, y, wave_solution)

    return wave_img
        
def polyscale2d(coeff, scale_x, scale_y, copy=True):
    if copy:
        coeff = np.copy(coeff)
    idx = _get_coeff_idx(coeff)
    for k, (i, j) in enumerate(idx):
        coeff[i, j] /= scale_x ** i * scale_y ** j
    return coeff

def polyshift2d(coeff, offset_x, offset_y, copy=True):
    if copy:
        coeff = np.copy(coeff)
    idx = _get_coeff_idx(coeff)
    # Copy coeff because it changes during the loop
    coeff2 = np.copy(coeff)
    for k, m in idx:
        not_the_same = ~((idx[:, 0] == k) & (idx[:, 1] == m))
        above = (idx[:, 0] >= k) & (idx[:, 1] >= m) & not_the_same
        for i, j in idx[above]:
            b = binom(i, k) * binom(j, m)
            sign = (-1) ** ((i - k) + (j - m))
            offset = offset_x ** (i - k) * offset_y ** (j - m)
            coeff[k, m] += sign * b * coeff2[i, j] * offset
    return coeff
    
def _scale(x, y):
    # Normalize x and y to avoid huge numbers
    # Mean 0, Variation 1
    offset_x, offset_y = np.mean(x), np.mean(y)
    norm_x, norm_y = np.std(x), np.std(y)
    if norm_x == 0:
        norm_x = 1
    if norm_y == 0:
        norm_y = 1
    x = (x - offset_x) / norm_x
    y = (y - offset_y) / norm_y
    return x, y, (norm_x, norm_y), (offset_x, offset_y)

def _unscale(x, y, norm, offset):
    x = x * norm[0] + offset[0]
    y = y * norm[1] + offset[1]
    return x, y
    
def _get_coeff_idx(coeff):
    idx = np.indices(coeff.shape)
    idx = idx.T.swapaxes(0, 1).reshape((-1, 2))
    # degree = coeff.shape
    # idx = [[i, j] for i, j in product(range(degree[0]), range(degree[1]))]
    # idx = np.asarray(idx)
    return idx
    
def polyvander2d(x, y, degree):
    # A = np.array([x ** i * y ** j for i, j in idx], dtype=float).T
    A = np.polynomial.polynomial.polyvander2d(x, y, degree)
    return A

def plot2d(x, y, z, coeff, title=None):
    # regular grid covering the domain of the data
    if x.size > 500:
        choice = np.random.choice(x.size, size=500, replace=False)
    else:
        choice = slice(None, None, None)
    x, y, z = x[choice], y[choice], z[choice]
    X, Y = np.meshgrid(
        np.linspace(np.min(x), np.max(x), 20), np.linspace(np.min(y), np.max(y), 20)
    )
    Z = np.polynomial.polynomial.polyval2d(X, Y, coeff)
    fig = plt.figure()
    #ax = fig.gca(projection="3d")
    ax = Axes3D(fig)
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(x, y, z, c="r", s=50)
    plt.xlabel("X")
    plt.ylabel("Y")
    ax.set_zlabel("Z")
    if title is not None:
        plt.title(title)
    # ax.axis("equal")
    # ax.axis("tight")
    plt.show()

def polyfit2d(
    x, y, z, degree=1, max_degree=None, scale=True, plot=False, plot_title=None
):
    """A simple 2D plynomial fit to data x, y, z
    The polynomial can be evaluated with numpy.polynomial.polynomial.polyval2d

    Parameters
    ----------
    x : array[n]
        x coordinates
    y : array[n]
        y coordinates
    z : array[n]
        data values
    degree : int, optional
        degree of the polynomial fit (default: 1)
    max_degree : {int, None}, optional
        if given the maximum combined degree of the coefficients is limited to this value
    scale : bool, optional
        Wether to scale the input arrays x and y to mean 0 and variance 1, to avoid numerical overflows.
        Especially useful at higher degrees. (default: True)
    plot : bool, optional
        wether to plot the fitted surface and data (slow) (default: False)

    Returns
    -------
    coeff : array[degree+1, degree+1]
        the polynomial coefficients in numpy 2d format, i.e. coeff[i, j] for x**i * y**j
    """
    # Flatten input
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()

    # Removed masked values
    mask = ~(np.ma.getmask(z) | np.ma.getmask(x) | np.ma.getmask(y))
    x, y, z = x[mask].ravel(), y[mask].ravel(), z[mask].ravel()

    if scale:
        x, y, norm, offset = _scale(x, y)

    # Create combinations of degree of x and y
    # usually: [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), ....]
    if np.isscalar(degree):
        degree = (int(degree), int(degree))
    assert len(degree) == 2, "Only 2D polynomials can be fitted"
    degree = [int(degree[0]), int(degree[1])]
    # idx = [[i, j] for i, j in product(range(degree[0] + 1), range(degree[1] + 1))]
    coeff = np.zeros((degree[0] + 1, degree[1] + 1))
    idx = _get_coeff_idx(coeff)

    # Calculate elements 1, x, y, x*y, x**2, y**2, ...
    A = polyvander2d(x, y, degree)

    # We only want the combinations with maximum order COMBINED power
    if max_degree is not None:
        mask = idx[:, 0] + idx[:, 1] <= int(max_degree)
        idx = idx[mask]
        A = A[:, mask]

    # Do least squares fit
    C, *_ = lstsq(A, z)

    # Reorder coefficients into numpy compatible 2d array
    for k, (i, j) in enumerate(idx):
        coeff[i, j] = C[k]

    # # Backup copy of coeff
    if scale:
        coeff = polyscale2d(coeff, *norm, copy=False)
        coeff = polyshift2d(coeff, *offset, copy=False)

    if plot:  # pragma: no cover
        if scale:
            x, y = _unscale(x, y, norm, offset)
        plot2d(x, y, z, coeff, title='Title')
        
    return coeff


lines = np.load('HR_R_linelist_P.npy',allow_pickle=True).item()

m_pix = []
m_ord = []
m_wave = []

for ord in range(len(lines)):
    for line in range(len(lines[ord]['line_positions'])):
    
        m_pix.append(lines[ord]['line_positions'][line])
        m_wave.append(lines[ord]['known_wavelengths_air'][line])
        m_ord.append(ord)
    

wave_solution = polyfit2d(m_pix, m_ord, m_wave, degree=[6,6], plot=False)

wave_img = make_wave(wave_solution,33,4096)

lines_P = np.load('HR_R_linelist_P.npy',allow_pickle=True).item()
lines_O = np.load('HR_R_linelist_O.npy',allow_pickle=True).item()

m_pix_P = []
m_ord_P = []
m_wave_P = []
m_pix_O = []
m_ord_O = []
m_wave_O = []

for ord in range(len(lines_P)):
    for line in range(len(lines_P[ord]['line_positions'])):
    
        m_pix_P.append(lines_P[ord]['line_positions'][line])
        m_wave_P.append(lines_P[ord]['known_wavelengths_air'][line])
        m_ord_P.append(ord)
        
for ord in range(len(lines_O)):
    for line in range(len(lines_O[ord]['line_positions'])):

        m_pix_O.append(lines_O[ord]['line_positions'][line])
        m_wave_O.append(lines_O[ord]['known_wavelengths_air'][line])
        m_ord_O.append(ord)
        
plt.plot(m_pix_P,m_wave_P,'o')

plt.plot(m_pix_O,m_wave_O,'rx')

lines_O = np.load('./Intermediate_files/HR_R_linelist_O_TEST.npy',allow_pickle=True).item()
m_pix_O = []
m_ord_O = []
m_wave_O = []

for ord in range(len(lines_O)):
    for line in range(len(lines_O[ord]['line_positions'])):

        m_pix_O.append(lines_O[ord]['line_positions'][line])
        m_wave_O.append(lines_O[ord]['known_wavelengths_air'][line])
        m_ord_O.append(ord)
        
plt.plot(m_pix_O,m_wave_O,'gx',)
plt.show()
wave_solution_P = polyfit2d(m_pix, m_ord, m_wave, degree=[6,6], plot=False)

wave_img_P = make_wave(wave_solution_P,42,2048)

ref_file = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/Super_Arcs/HR_Super_Arc_H20220701.fits'
ref_file = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2022/0717/reduced/bgoH202207170027.fits"
#ref_file = '/Users/daniel/Desktop/bgoR202510110034.fits'
#ref_file = '/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2025/1011/reduced/bgoH202510110034.fits'

hdu=fits.open(ref_file)
P_Fibre = hdu['FIBRE_P'].data
P_Wave = hdu['WAVE_P'].data
O_Fibre = hdu['FIBRE_O'].data
O_Wave = hdu['WAVE_O'].data

with fits.open('thar_best.fits') as hdu1:
    header = hdu1[0].header
    known_spec = (hdu1[0].data)

    specaxis = str(1)
    flux = known_spec
    wave_step = header['CDELT%s' % (specaxis)]
    wave_base = header['CRVAL%s' % (specaxis)]
    reference_pixel = header['CRPIX%s' % (specaxis)]
    xconv = lambda v: ((v-reference_pixel+1)*wave_step+wave_base)
    known_waveobs = xconv(np.arange(len(flux)))
    
#plt.plot(known_waveobs,known_spec,'k')

th = np.loadtxt("./thar_list_orig.txt",usecols=(0),unpack=True)

for ord in range(42):
    ii=np.where(np.logical_and(known_waveobs > np.min(P_Wave[ord]), known_waveobs < np.max(O_Wave[ord])))[0]
    plt.plot(known_waveobs[ii],known_spec[ii]/np.max(known_spec[ii]),'k')
#    plt.plot(wave_img_P[ord],P_Fibre[ord]/np.nanmax(P_Fibre[ord]))

#    plt.plot(P_Wave[ord],P_Fibre[ord]/np.nanmax(P_Fibre[ord]))
#    plt.plot(O_Wave[ord],O_Fibre[ord]/np.nanmax(O_Fibre[ord]))
    plt.plot(hdu[94].data[ord],P_Fibre[ord]/np.nanmax(P_Fibre[ord]),':')
    plt.plot(hdu[98].data[ord],P_Fibre[ord]/np.nanmax(P_Fibre[ord]))
    
#    ii=np.where(np.logical_and(th > np.min(P_Wave[ord]), th < np.max(O_Wave[ord])))[0]
#    plt.vlines(th[ii],0,0.1,'r')
    #plt.title(str(ord))
    plt.vlines(lines[ord]['known_wavelengths_air'],0,0.5,'r')

plt.show()
    
#for ord in range(42):
##    plt.plot(wave_img_O[ord],O_Fibre[ord])
#    plt.plot(O_Wave[ord],O_Fibre[ord])
#plt.show()

#for ord in range(42):
#    for i in range(2047):
#        plt.plot(wave_img[ord][i],wave_img[ord][i+1]-wave_img[ord][i],'.')
#plt.show()
