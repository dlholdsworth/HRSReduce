import numpy as np
import scipy
from scipy.ndimage.filters import median_filter
from astropy.io import fits
from scipy.linalg import solve_banded #lstsq, solve
from itertools import chain
import matplotlib.pyplot as plt

import logging

logger = logging.getLogger(__name__)

class OrderMerge():

    def __init__(self,spec,wave,cont,sigma,scaling=False, plot=False):
    
        self.spec = spec
        self.wave = wave
        self.cont = cont
        self.sigma = sigma
        self.scaling = scaling
        self.plot = plot
        self.logger = logger
        
        # start a logger
        self.logger.info('Started OrderMerge')
        
    def middle(self,
        f,
        param,
        x=None,
        iterations=40,
        eps=0.001,
        poly=False,
        weight=1,
        lambda2=-1,
        mn=None,
        mx=None,
    ):
        """
        middle tries to fit a smooth curve that is located
        along the "middle" of 1D data array f. Filter size "filter"
        together with the total number of iterations determine
        the smoothness and the quality of the fit. The total
        number of iterations can be controlled by limiting the
        maximum number of iterations (iter) and/or by setting
        the convergence criterion for the fit (eps)
        04-Nov-2000 N.Piskunov wrote.
        09-Nov-2011 NP added weights and 2nd derivative constraint as LAM2

        Parameters
        ----------
        f : Callable
            Function to fit
        filter : int
            Smoothing parameter of the optimal filter (or polynomial degree of poly is True)
        iter : int
            maximum number of iterations [def: 40]
        eps : float
            convergence level [def: 0.001]
        mn : float
            minimum function values to be considered [def: min(f)]
        mx : float
            maximum function values to be considered [def: max(f)]
        lam2 : float
            constraint on 2nd derivative
        weight : array(float)
            vector of weights.
        """
        mn = mn if mn is not None else np.min(f)
        mx = mx if mx is not None else np.max(f)

        f = np.asarray(f)

        if x is None:
            xx = np.linspace(-1, 1, num=f.size)
        else:
            xx = np.asarray(x)

        if poly:
            j = (f >= mn) & (f <= mx)
            n = np.count_nonzero(j)
            if n <= round(param):
                return f

            fmin = np.min(f[j]) - 1
            fmax = np.max(f[j]) + 1
            ff = (f[j] - fmin) / (fmax - fmin)
            ff_old = ff
        else:
            fmin = np.min(f) - 1
            fmax = np.max(f) + 1
            ff = (f - fmin) / (fmax - fmin)
            ff_old = ff
            n = len(f)

        for _ in range(iterations):
            if poly:
                param = round(param)
                if param > 0:
                    t = median_filter(np.polyval(np.polyfit(xx, ff, param), xx), 3)
                    tmp = np.polyval(np.polyfit(xx, (t - ff) ** 2, param), xx)
                else:
                    t = np.tile(np.polyfit(xx, ff, param), len(f))
                    tmp = np.tile(np.polyfit(xx, (t - ff) ** 2, param), len(f))
            else:
                t = median_filter(self.opt_filter(ff, param, weight=weight, lambda2=lambda2), 3)
                tmp = self.opt_filter(
                    weight * (t - ff) ** 2, param, weight=weight, lambda2=lambda2
                )

            dev = np.sqrt(np.clip(tmp, 0, None))
            ff = np.clip(t - dev, ff, t + dev)
            dev2 = np.max(weight * np.abs(ff - ff_old))
            ff_old = ff

            # print(dev2)
            if dev2 <= eps:
                break

        if poly:
            xx = np.linspace(-1, 1, len(f))
            if param > 0:
                t = median_filter(np.polyval(np.polyfit(xx, ff, param), xx), 3)
            else:
                t = np.tile(np.polyfit(xx, ff, param), len(f))

        return t * (fmax - fmin) + fmin
        
    @staticmethod
    def opt_filter(y, par, par1=None, weight=None, lambda2=-1, maxiter=100):
        """
        Optimal filtering of 1D and 2D arrays.
        Uses tridiag in 1D case and sprsin and linbcg in 2D case.
        Written by N.Piskunov 8-May-2000

        Parameters
        ----------
        f : array
            1d or 2d array
        xwidth : int
            filter width (for 2d array width in x direction (1st index)
        ywidth : int
            (for 2d array only) filter width in y direction (2nd index) if ywidth is missing for 2d array, it set equal to xwidth
        weight : array(float)
            an array of the same size(s) as f containing values between 0 and 1
        lambda1: float
            regularization parameter
        maxiter : int
            maximum number of iteration for filtering of 2d array
        """

        y = np.asarray(y)

        if y.ndim not in [1, 2]:
            raise ValueError("Input y must have 1 or 2 dimensions")

        if par < 1:
            par = 1

        # 1D case
        if y.ndim == 1 or (y.ndim == 2 and (y.shape[0] == 1 or y.shape[1] == 1)):
            y = y.ravel()
            n = y.size

            if weight is None:
                weight = np.ones(n)
            elif np.isscalar(weight):
                weight = np.full(n, weight)
            else:
                weight = weight[:n]

            if lambda2 > 0:
                # Apply regularization lambda
                aij = np.zeros((5, n))
                # 2nd lower subdiagonal
                aij[0, 2:] = lambda2
                # Lower subdiagonal
                aij[1, 1] = -par - 2 * lambda2
                aij[1, 2:-1] = -par - 4 * lambda2
                aij[1, -1] = -par - 2 * lambda2
                # Main diagonal
                aij[2, 0] = weight[0] + par + lambda2
                aij[2, 1] = weight[1] + 2 * par + 5 * lambda2
                aij[2, 2:-2] = weight[2:-2] + 2 * par + 6 * lambda2
                aij[2, -2] = weight[-2] + 2 * par + 5 * lambda2
                aij[2, -1] = weight[-1] + par + lambda2
                # Upper subdiagonal
                aij[3, 0] = -par - 2 * lambda2
                aij[3, 1:-2] = -par - 4 * lambda2
                aij[3, -2] = -par - 2 * lambda2
                # 2nd lower subdiagonal
                aij[4, 0:-2] = lambda2
                # RHS
                b = weight * y

                f = solve_banded((2, 2), aij, b)
            else:
                a = np.full(n, -abs(par))
                b = np.copy(weight) + abs(par)
                b[1:-1] += abs(par)
                aba = np.array([a, b, a])

                f = solve_banded((1, 1), aba, weight * y)

            return f
        else:
            # 2D case
            if par1 is None:
                par1 = par
            if par == 0 and par1 == 0:
                raise ValueError("xwidth and ywidth can't both be 0")
            n = y.size
            nx, ny = y.shape

            lam_x = abs(par)
            lam_y = abs(par1)

            n = nx * ny
            ndiag = 2 * nx + 1
            aij = np.zeros((n, ndiag))
            aij[nx, 0] = weight[0, 0] + lam_x + lam_y
            aij[nx, 1:nx] = weight[0, 1:nx] + 2 * lam_x + lam_y
            aij[nx, nx : n - nx] = weight[1 : ny - 1] + 2 * (lam_x + lam_y)
            aij[nx, n - nx : n - 1] = weight[ny - 1, 0 : nx - 1] + 2 * lam_x + lam_y
            aij[nx, n - 1] = weight[ny - 1, nx - 1] + lam_x + lam_y

            aij[nx - 1, 1:n] = -lam_x
            aij[nx + 1, 0 : n - 1] = -lam_x

            ind = np.arrange(ny - 1) * nx + nx + nx * n
            aij[ind - 1] = aij[ind - 1] - lam_x
            aij[ind] = aij[ind] - lam_x

            ind = np.arrange(ny - 1) * nx + nx
            aij[nx + 1, ind - 1] = 0
            aij[nx - 1, ind] = 0

            aij[0, nx:n] = -lam_y
            aij[ndiag - 1, 0 : n - nx] = -lam_y

            rhs = f * weight

            model = solve_banded((nx, nx), aij, rhs)
            model = np.reshape(model, (ny, nx))
            return model

        
    @staticmethod
    def bezier_interp(x_old, y_old, x_new):
        """
        Bezier interpolation, based on the scipy methods

        This mostly sanitizes the input by removing masked values and duplicate entries
        Note that in case of duplicate entries (in x_old) the results are not well defined as only one of the entries is used and the other is discarded

        Parameters
        ----------
        x_old : array[n]
            old x values
        y_old : array[n]
            old y values
        x_new : array[m]
            new x values

        Returns
        -------
        y_new : array[m]
            new y values
        """

        # Handle masked arrays
        if np.ma.is_masked(x_old):
            x_old = np.ma.compressed(x_old)
            y_old = np.ma.compressed(y_old)

        # avoid duplicate entries in x
        assert x_old.size == y_old.size
        x_old, index = np.unique(x_old, return_index=True)
        y_old = y_old[index]

        knots, coef, order = scipy.interpolate.splrep(x_old, y_old, s=0)
        y_new = scipy.interpolate.BSpline(knots, coef, order)(x_new)
        return y_new
    
    def splice_orders(self):
        """
        Splice orders together so that they form a continous spectrum
        This is achieved by linearly combining the overlaping regions
        From PyReduce

        Parameters
        ----------
        spec : array[nord, ncol]
            Spectrum to splice, with seperate orders
        wave : array[nord, ncol]
            Wavelength solution for each point
        cont : array[nord, ncol]
            Continuum, blaze function will do fine as well
        sigm : array[nord, ncol]
            Errors on the spectrum
        scaling : bool, optional
            If true, the spectrum/continuum will be scaled to 1 (default: False)
        plot : bool, optional
            If true, will plot the spliced spectrum (default: False)

        Raises
        ------
        NotImplementedError
            If neighbouring orders dont overlap

        Returns
        -------
        spec, wave, cont, sigm : array[nord, ncol]
            spliced spectrum
        """
        
        plot_title = 'Order Merging'
        nord, _ = self.spec.shape  # Number of sp. orders, Order length in pixels

        if self.cont is None:
            cont = np.ones_like(self.spec)

        # Just to be extra safe that they are all the same
        mask = (
            np.ma.getmaskarray(self.spec)
            | (np.ma.getdata(self.spec) == 0)
            | (np.ma.getdata(self.cont) == 0)
        )
        spec = np.ma.masked_array(self.spec, mask=mask)
        wave = np.ma.masked_array(np.ma.getdata(self.wave), mask=mask)
        cont = np.ma.masked_array(np.ma.getdata(self.cont), mask=mask)
        sigm = np.ma.masked_array(np.ma.getdata(self.sigma), mask=mask)

        if self.scaling:
            # Scale everything to roughly the same size, around spec/blaze = 1
            scale = np.ma.median(spec / cont, axis=1)
            cont *= scale[:, None]

        if self.plot:  # pragma: no cover
            plt.subplot(411)
            if plot_title is not None:
                plt.suptitle(plot_title)
            plt.title("Before")
            for i in range(spec.shape[0]):
                plt.plot(wave[i], spec[i] / cont[i])
            plt.ylim([0, 2])

            plt.subplot(412)
            plt.title("Before Error")
            for i in range(spec.shape[0]):
                plt.plot(wave[i], sigm[i] / cont[i])
            plt.ylim((0, np.ma.median(sigm[i] / cont[i]) * 2))

        # Order with largest signal, everything is scaled relative to this order
        iord0 = np.argmax(np.ma.median(spec / cont, axis=1))

        # Loop from iord0 outwards, first to the top, then to the bottom
        tmp0 = chain(range(iord0, 0, -1), range(iord0, nord - 1))
        tmp1 = chain(range(iord0 - 1, -1, -1), range(iord0 + 1, nord))

        # Looping over order pairs
        for iord0, iord1 in zip(tmp0, tmp1):
            # Get data for current order
            # Note that those are just references to parts of the original data
            # any changes will also affect spec, wave, cont, and sigm
            s0, s1 = spec[iord0], spec[iord1]
            w0, w1 = wave[iord0], wave[iord1]
            c0, c1 = cont[iord0], cont[iord1]
            u0, u1 = sigm[iord0], sigm[iord1]

            # Calculate Overlap
            i0 = np.ma.where((w0 >= np.ma.min(w1)) & (w0 <= np.ma.max(w1)))
            i1 = np.ma.where((w1 >= np.ma.min(w0)) & (w1 <= np.ma.max(w0)))

            # Orders overlap
            if i0[0].size > 0 and i1[0].size > 0:
                # Interpolate the overlapping region onto the wavelength grid of the other order
                tmpS0 = self.bezier_interp(w1, s1, w0[i0])
                tmpB0 = self.bezier_interp(w1, c1, w0[i0])
                tmpU0 = self.bezier_interp(w1, u1, w0[i0])

                tmpS1 = self.bezier_interp(w0, s0, w1[i1])
                tmpB1 = self.bezier_interp(w0, c0, w1[i1])
                tmpU1 = self.bezier_interp(w0, u0, w1[i1])

                # Combine the two orders weighted by the relative error
                wgt0 = np.ma.vstack([c0[i0].data / u0[i0].data, tmpB0 / tmpU0]) ** 2
                wgt1 = np.ma.vstack([c1[i1].data / u1[i1].data, tmpB1 / tmpU1]) ** 2

                s0[i0], utmp = np.ma.average(
                    np.ma.vstack([s0[i0], tmpS0]), axis=0, weights=wgt0, returned=True
                )
                c0[i0] = np.ma.average([c0[i0], tmpB0], axis=0, weights=wgt0)
                u0[i0] = c0[i0] * utmp ** -0.5

                s1[i1], utmp = np.ma.average(
                    np.ma.vstack([s1[i1], tmpS1]), axis=0, weights=wgt1, returned=True
                )
                c1[i1] = np.ma.average([c1[i1], tmpB1], axis=0, weights=wgt1)
                u1[i1] = c1[i1] * utmp ** -0.5
            else:  # pragma: no cover
                # TODO: Orders dont overlap
                continue

        if self.plot:  # pragma: no cover
            plt.subplot(413)
            plt.title("After")
            for i in range(nord):
                plt.plot(wave[i], spec[i] / cont[i], label="order=%i" % i)

            plt.subplot(414)
            plt.title("Error")
            for i in range(nord):
                plt.plot(wave[i], sigm[i] / cont[i], label="order=%i" % i)
            plt.ylim((0, np.ma.median(sigm[i] / cont[i]) * 2))
            plt.show()
            
        smooth_initial = 1e5
            
        # Create new equispaced wavelength grid
        tmp = wave.compressed()
        wmin = np.min(tmp)
        wmax = np.max(tmp)
        dwave = np.abs(tmp[tmp.size // 2] - tmp[tmp.size // 2 - 1]) * 0.5
        nwave = np.ceil((wmax - wmin) / dwave) + 1
        new_wave = np.linspace(wmin, wmax, int(nwave), endpoint=True)

        # Combine all orders into one big spectrum, sorted by wavelength
        wsort, j, index = np.unique(tmp, return_index=True, return_inverse=True)
        sB = (spec / cont).compressed()[j]

        # Get initial weights for each point
        weight = self.middle(sB, 0.5, x=wsort - wmin)
        weight = weight / self.middle(weight, 3 * smooth_initial) + np.concatenate(
            ([0], 2 * weight[1:-1] - weight[0:-2] - weight[2:], [0])
        )
        weight = np.clip(weight, 0, None)
        # TODO for some reason the interpolation messes up, use linear instead for now
        # weight = self.safe_interpolation(wsort, weight, new_wave)
        weight = np.interp(new_wave, wsort, weight)
        weight /= np.max(weight)

        # Interpolate Spectrum onto the new grid
        # ssB = self.safe_interpolation(wsort, sB, new_wave)
        ssB = np.interp(new_wave, wsort, sB)
        # Keep the scale of the continuum
        bbb = self.middle(cont.compressed()[j], 1)
        
      
        plt.plot(new_wave,ssB+5)
 
        for ord in range(42):
            plt.plot(wave[ord],spec[ord]/cont[ord])

        plt.show()

        return new_wave,ssB


