"""Reference order tracer for HRS master flats: one code path for both arms and all modes.

    from hrsreduce.order_trace.reference_trace import trace, have_reference
    table, diag, info = trace(img, 'H' or 'R', 'HR')       # img: the master flat, raw orientation

Each arm and mode has a reference (references/<MODE>_<H|R>_reference.npz next to this module): the labelled traces
(pair, fibre) of a deep stack of nightly master flats from one stable stretch of nights, the cross-dispersion profile
of every orderlet on a 0.1 px grid per 256 columns, the extraction widths, column profiles for the coarse offset and
the prior widths of the per-orderlet correction (references/README.md). The tracer measures the night against it:

  1. Coarse offset: high-passed column profiles in 8 x windows x 3 y bands, cross-correlated with the reference;
     plane dy = a + b x + c y, the start of the map below.
  2. Centres: every 32 columns, the pixels within +-H (half the gap to the nearest trace) of each predicted orderlet
     are fitted with  A P(u - d) + p + q u,  P the reference profile of that orderlet; Gauss-Newton in d, clipping.
     Errors from the covariance scaled by the reduced chi2 of the block (empirical).
  3. The geometry of the night: a smooth map of the whole chip (Legendre in x times powers of y, 11 terms, TERMS),
     fitted to the good blocks of all orderlets, twice; the reference traces moved by it are the predicted traces.
  4. Each orderlet: one regularised fit, the reference shape plus a degree-5 Legendre correction whose terms have
     priors as wide as the real night-to-night changes (prior_tau in the reference): bright orderlets follow their
     data, faint ones the reference shape, with no switch. 'source' only describes how many terms the data set
     (measured >= 5.5, blended, anchored < 2.5; reference: no usable blocks).
  5. Widths from the reference; X1 = 0; X2 the last column where the aperture is on the chip and usable, made equal
     within a pair and set to the chip edge above 95 % of it (as OrderTrace).
  Self-check: the labelling is compared with the other correlation peaks and with starts one and two orderlet
  spacings away, by how well the profiles fit (a slip by one orderlet fits each fibre with the other fibre's profile).

Output table: the pipeline's trace CSV columns (Coeff0..5 ascending in raw x, BottomEdge, TopEdge, X1, X2),
bottom to top. diag: per orderlet pair, fibre, source, blocks, coverage, rms, fit error, offset.
Flag (info['flag']): FAILED when fewer than half the orderlets are found; AMBIGUOUS when another labelling fits
almost as well; FALLBACK when an orderlet has no usable light ('reference'); else OK. Anchored orderlets are counted
(info['n_anchored']) but do not set a flag.
"""
import os
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import median_filter

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, 'references')
BPM = {'H': os.path.join(HERE, '..', 'utils', 'BPM_H.fits'), 'R': os.path.join(HERE, '..', 'utils', 'BPM_R.fits')}
USABLE_X = {'H': None, 'R': 4019}       # last usable column (Red: columns from 4020 are not used)
STEP = 0.1          # profile grid (px)
GROUP = 256         # columns per profile group
BLK = 32            # columns per measured block
N_WIN = 8           # x windows for the coarse offset
WIN_HALF = 40       # half width of each window (columns)
N_BAND = 3          # y bands per window
MAXLAG = 80         # coarse offset search (px)
DEG = 5             # trace polynomial degree
P = np.polynomial.polynomial.polyval


def prepare(img, sarm):
    """mask (bad-pixel mask, non-finite pixels, unused columns) and the masked image (masked pixels set to 0)"""
    img = np.array(img, dtype=float)
    mask = ~np.isfinite(img)
    bpm = fits.getdata(BPM[sarm]) != 0
    if bpm.shape == img.shape:
        mask |= bpm
    usable_x = USABLE_X[sarm] if USABLE_X[sarm] is not None else img.shape[1] - 1
    mask[:, usable_x + 1:] = True
    img[mask] = 0.
    return img, mask, usable_x


GOOD_SIG = 0.3        # px: a block centre is used when its error is below this
MIN_ANCHOR = 4        # good blocks for an orderlet to count as found in the labelling check
_REF = {}


def reference_file(sarm, mode):
    return os.path.join(REF_DIR, f'{mode}_{sarm}_reference.npz')


def have_reference(sarm, mode):
    return os.path.isfile(reference_file(sarm, mode))


def reference(sarm, mode):
    key = (sarm, mode)
    if key not in _REF:
        z = np.load(reference_file(sarm, mode))
        _REF[key] = {k: z[k] for k in z.files}
        r = _REF[key]; nom = r['nominal']
        r['nom'] = np.flatnonzero(nom)
        r['dP'] = np.gradient(r['prof'].astype(float), STEP, axis=2)
    return _REF[key]


# ------------------------------------------------------------------ 1. coarse offset
def _xcorr(r, p):
    n = r.size; rr = r[MAXLAG:n - MAXLAG]
    lags = np.arange(-MAXLAG, MAXLAG + 1)
    seg = np.array([p[MAXLAG + l:n - MAXLAG + l] for l in lags])
    cc = seg @ rr / np.sqrt(np.dot(rr, rr) * np.einsum('ij,ij->i', seg, seg) + 1e-30)
    i = int(np.argmax(cc)); f = 0.
    if 0 < i < cc.size - 1:
        a, b, c = cc[i - 1], cc[i], cc[i + 1]
        if a - 2 * b + c != 0:
            f = 0.5 * (a - c) / (a - 2 * b + c)
    return lags[i] + f, cc[i], cc[np.abs(lags - lags[i]) > 8].max(), cc


def column_profiles(img, mask, col_x):
    out = []
    for c in col_x:
        blk = np.where(mask[:, c - WIN_HALF:c + WIN_HALF], np.nan, img[:, c - WIN_HALF:c + WIN_HALF])
        p = np.nan_to_num(np.nanmedian(blk, axis=1))
        out.append(p - median_filter(p, 151))
    return np.array(out)


def coarse_plane(img, mask, ref):
    ny = img.shape[0]
    prof = column_profiles(img, mask, ref['col_x'])
    edges = np.linspace(0, ny, N_BAND + 1).astype(int)
    pts, curves = [], []
    for i, x in enumerate(ref['col_x']):
        for b in range(N_BAND):
            lo, hi = edges[b], edges[b + 1]
            dy, cc, cc2, curve = _xcorr(ref['col_prof'][i, lo:hi].astype(float), prof[i, lo:hi])
            pts.append((x, 0.5 * (lo + hi), dy, cc, cc2)); curves.append(curve)
    pts = np.array(pts)
    good = (pts[:, 3] > 0.5) & (pts[:, 4] < 0.9 * pts[:, 3])
    sol = np.array([np.median(pts[:, 2]), 0., 0.])
    if good.sum() >= 4:
        for _ in range(3):
            A = np.column_stack([np.ones(good.sum()), pts[good, 0], pts[good, 1]])
            sol = np.linalg.lstsq(A, pts[good, 2], rcond=None)[0]
            r = pts[:, 2] - (sol[0] + sol[1] * pts[:, 0] + sol[2] * pts[:, 1])
            s = 1.4826 * np.median(np.abs(r[good])) + 0.1
            good = good & (np.abs(r) < 4 * s)
    # candidate offsets for the self-check: peaks of the summed correlation curve
    summed = np.sum(curves, axis=0); lags = np.arange(-MAXLAG, MAXLAG + 1)
    peaks = [lags[i] for i in np.argsort(summed)[::-1]]
    cand = []
    for l in peaks:
        if all(abs(l - c) > 8 for c in cand):
            cand.append(l)
        if len(cand) == 3:
            break
    return sol, dict(n_good=int(good.sum()), n_win=len(pts), cc_min=float(pts[good, 3].min()) if good.any() else np.nan,
                     ratio_max=float(np.max(pts[good, 4] / pts[good, 3])) if good.any() else np.nan, candidates=cand)


# The geometry of a night relative to the reference: a smooth map of the whole chip, Legendre polynomials in x
# (order j) times powers of y (i), evaluated along every reference trace. The terms are those the per-orderlet
# deviations from a plane show to be shared over the chip (prior.csv): offset and tilt quadratic in y, curvature
# linear in y, higher orders constant. The plane of the coarse offset (a + b x + c y) is its lowest part.
TERMS = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0), (4, 0), (5, 0))
TAU_GLOBAL = 2.0      # px: weak prior on every term of the map, only to keep it bounded where few orderlets are found


def basis(x, y, nx, ny):
    hx, hy = 0.5 * (nx - 1), 0.5 * (ny - 1)
    L = np.polynomial.legendre.legvander((np.asarray(x, float) - hx) / hx, DEG)
    v = (np.asarray(y, float) - hy) / hy
    return np.column_stack([L[:, j] * v ** i for j, i in TERMS])


def distortion(g, x, y, nx, ny):
    return basis(x, y, nx, ny) @ g


def model_from_plane(plane, nx, ny):
    """a + b x + c y as a map: (a + b hx + c hy) L0 + b hx L1(u) + c hy v"""
    a, b, c = plane; hx, hy = 0.5 * (nx - 1), 0.5 * (ny - 1)
    g = np.zeros(len(TERMS))
    g[TERMS.index((0, 0))] = a + b * hx + c * hy
    g[TERMS.index((1, 0))] = b * hx
    g[TERMS.index((0, 1))] = c * hy
    return g


def predict(ref, g, nx, ny, coeffs=None):
    """the reference traces moved by the map g, refitted with degree 5 (exact to about 0.001 px)"""
    coeffs = ref['coeffs'][ref['nom']] if coeffs is None else coeffs
    xs = np.linspace(0, nx - 1, 129); out = []
    for c in coeffs:
        y0 = P(xs, c); y = y0 + distortion(g, xs, y0, nx, ny)
        p = np.polynomial.Polynomial.fit(xs, y, DEG, domain=[0, nx - 1]).convert().coef
        q = np.zeros(DEG + 1); q[:p.size] = p; out.append(q)
    return np.array(out)


# ------------------------------------------------------------------ 2. centres
def fit_block(u, v, P_, dP_, grid, dmax):
    """v = A P(u - d) + p + q u; weights 1 / (|v| + 5 % of the block peak); error scaled by the reduced chi2."""
    w = 1. / (np.abs(v) + 0.05 * max(np.abs(v).max(), 1e-12))
    keep = np.ones(u.size, bool); d = 0.
    for it in range(8):
        Pu = np.interp(u - d, grid, P_); dPu = np.interp(u - d, grid, dP_)
        X = np.column_stack([Pu, -dPu, np.ones_like(u), u])
        Xk, wk = X[keep], w[keep]
        try:
            C = np.linalg.inv((Xk.T * wk) @ Xk)
        except np.linalg.LinAlgError:
            return None
        coef = C @ ((Xk.T * wk) @ v[keep])
        A = coef[0]
        if not np.isfinite(A) or A <= 0:
            return None
        step = coef[1] / A
        d += np.clip(step, -1.0, 1.0)
        if abs(d) > dmax:
            return None
        r = (v - X @ coef) * np.sqrt(w)
        s = 1.4826 * np.median(np.abs(r[keep])) + 1e-12
        newkeep = np.abs(r) < 5 * s
        if abs(step) < 2e-3 and np.array_equal(newkeep, keep):
            break
        keep = newkeep
    n = keep.sum()
    chi2 = np.sum((r[keep]) ** 2) / max(n - 4, 1)
    # profile mismatch: weighted power not explained by the model, over the power of the orderlet's light
    sig = A * Pu
    mis = np.sum(w[keep] * (v[keep] - X[keep] @ coef) ** 2) / max(np.sum(w[keep] * sig[keep] ** 2), 1e-30)
    return d, np.sqrt(C[1, 1] * chi2) / A, A, mis


def measure(img, mask, ref, C, usable_x, step=1, orderlets=None):
    """block centres of every (or the given) nominal orderlet: dict i -> array (x, d, sig, A, mismatch)"""
    ny = img.shape[0]
    grid, prof, dP, half = ref['grid'], ref['prof'], ref['dP'], ref['half'][ref['nom']]
    out = {}
    for i in (range(len(C)) if orderlets is None else orderlets):
        h = int(half[i]); res = []
        for b0 in range(0, usable_x + 1, BLK * step):
            cols = np.arange(b0, min(b0 + BLK, usable_x + 1))
            yc = P(cols, C[i])
            if yc.max() + h + 1 >= ny or yc.min() - h < 0:
                continue
            off = np.arange(-h, h + 2)
            rows = np.floor(yc).astype(int)[:, None] + off[None, :]
            u = rows - yc[:, None]
            ok = ~mask[rows, cols[:, None]] & (np.abs(u) <= h)
            if ok.sum() < 0.5 * ok.size:
                continue
            g = min(int(cols.mean()) // GROUP, prof.shape[1] - 1)
            f = fit_block(u[ok], img[rows, cols[:, None]][ok], prof[i, g].astype(float), dP[i, g], grid, 0.6 * h)
            if f is not None:
                res.append((cols.mean(),) + f)
        out[i] = np.array(res, float).reshape(-1, 5)
    return out


def good_blocks(r, h):
    return np.isfinite(r[:, 2]) & (r[:, 2] < GOOD_SIG) & (np.abs(r[:, 1]) < 0.6 * h)


# ------------------------------------------------------------------ 3. refinement of the map
def refine(ref, C, blocks, nx, ny):
    """change of the map from the good blocks of all orderlets (weighted, 4-sigma clipping, weak prior)"""
    xs, ys, ds, ws = [], [], [], []
    half = ref['half'][ref['nom']]
    for i, r in blocks.items():
        g = good_blocks(r, half[i])
        xs.append(r[g, 0]); ys.append(P(r[g, 0], C[i])); ds.append(r[g, 1]); ws.append(1. / r[g, 2] ** 2)
    x, y, d, w = (np.concatenate(v) for v in (xs, ys, ds, ws))
    if x.size < 3 * len(TERMS):
        return np.zeros(len(TERMS))
    B = basis(x, y, nx, ny); lam = np.full(len(TERMS), 1. / TAU_GLOBAL ** 2)
    keep = np.ones(x.size, bool)
    for _ in range(5):
        A = (B[keep].T * w[keep]) @ B[keep] + np.diag(lam)
        sol = np.linalg.solve(A, (B[keep].T * w[keep]) @ d[keep])
        r = (d - B @ sol) * np.sqrt(w)
        s = 1.4826 * np.median(np.abs(r[keep])) + 1e-6
        new = np.abs(r) < 4 * s
        if np.array_equal(new, keep):
            break
        keep = new
    return sol


# ------------------------------------------------------------------ 4. per orderlet
def on_chip_range(c, bottom, top, ny, usable_x):
    xs = np.arange(0, usable_x + 1)
    y = P(xs, c)
    on = (y - bottom >= 0) & (y + top <= ny - 1)
    return xs[on] if on.any() else xs[:0]


DEFAULT_TAU = (1.0, 0.5, 0.05, 0.05, 0.05, 0.05)   # px; used only when the reference carries no prior


def legendre_fit(x, d, s, nx, tau=None):
    """d(x) = sum_j c_j L_j(u), u = (x - h) / h over the chip, weights 1 / s^2, 4-sigma clipping on the scatter.
    tau: prior widths of c_j (c_j ~ N(0, tau_j^2)), None for an unregularised fit. Returns c, covariance, kept."""
    h = 0.5 * (nx - 1); u = (np.asarray(x, float) - h) / h; w = 1. / np.asarray(s, float) ** 2
    V = np.polynomial.legendre.legvander(u, DEG)
    lam = np.zeros(DEG + 1) if tau is None else 1. / np.asarray(tau, float) ** 2
    keep = np.ones(u.size, bool)
    for _ in range(6):
        A = (V[keep].T * w[keep]) @ V[keep] + np.diag(lam)
        cov = np.linalg.inv(A); c = cov @ ((V[keep].T * w[keep]) @ np.asarray(d, float)[keep])
        if u.size == 0:
            break
        res = (d - V @ c) * np.sqrt(w)
        sc = 1.4826 * np.median(np.abs(res[keep])) + 1e-6
        new = np.abs(res) < 4 * sc
        if np.array_equal(new, keep) or new.sum() <= DEG:
            break
        keep = new
    return c, cov, keep


def fit_orderlet(r, h, c_pred, xr, nx, tau):
    """One regularised fit for every orderlet, no switch: the block offsets from the predicted trace (reference
    shape + map) are fitted with a degree-5 Legendre series whose terms have priors as wide as the real changes
    from night to night (prior_tau in the reference). A bright orderlet follows its own data, a faint one stays
    close to the reference shape, and one without light keeps it. Returns (source, coefficients, rms, expected max
    error, offset, effective number of terms set by the data); source only describes that number."""
    g = good_blocks(r, h)
    x, d, s = r[g, 0], r[g, 1], r[g, 2]
    if xr.size == 0 or x.size == 0:
        return 'reference', np.array(c_pred, float), np.nan, np.nan, 0., 0.
    c, cov, keep = legendre_fit(x, d, s, nx, tau)
    hh = 0.5 * (nx - 1)
    V = np.polynomial.legendre.legvander((x[keep] - hh) / hh, DEG)
    edf = float(np.trace(cov @ ((V.T / s[keep] ** 2) @ V)))
    Vg = np.polynomial.legendre.legvander((xr[::8] - hh) / hh, DEG)
    err = float(np.sqrt(np.einsum('ij,jk,ik->i', Vg, cov, Vg)).max())
    corr = np.polynomial.Legendre(c, domain=[0, nx - 1]).convert(kind=np.polynomial.Polynomial).coef
    out = np.array(c_pred, float).copy(); out[:corr.size] += corr
    rms = float(np.sqrt(np.mean((d[keep] - V @ c) ** 2)))
    src = 'measured' if edf >= DEG + 0.5 else ('anchored' if edf < 2.5 else 'blended')
    return src, out, rms, err, float(np.median(d)), edf


# ------------------------------------------------------------------ the tracer
def trace(img, sarm, mode, force_offset=None):
    ref = reference(sarm, mode)
    img, mask, usable_x = prepare(img, sarm)
    ny, nx = img.shape
    plane, info = coarse_plane(img, mask, ref)
    if force_offset is not None:                       # tests only: start from a given offset
        plane = np.array([force_offset, 0., 0.])
    half = ref['half'][ref['nom']]
    nnom = len(ref['nom'])

    def frac_good(blocks):
        return np.mean([good_blocks(r, half[i]).sum() >= MIN_ANCHOR for i, r in blocks.items()])

    # Self-check of the labelling. A start one orderlet off also gives good blocks (on the neighbouring orderlet), so
    # the labellings are compared by how well the profiles fit: the two fibres of a pair have different profiles, and
    # a slipped labelling leaves about 100 times more of the light unexplained. Candidates: the coarse offset, the
    # other correlation peaks, and the start moved by one and two typical orderlet spacings either way; each is
    # refined once on a subset of orderlets and blocks.
    sub = list(range(0, nnom, 3))
    yc = np.sort(P(nx // 2, ref['coeffs'][ref['nom']].T))
    spacing = float(np.median(np.diff(yc)))

    def evaluate(g0):
        C0 = predict(ref, g0, nx, ny); b = measure(img, mask, ref, C0, usable_x, step=4, orderlets=sub)
        g1 = g0 + refine(ref, C0, b, nx, ny)
        b = measure(img, mask, ref, predict(ref, g1, nx, ny), usable_x, step=4, orderlets=sub)
        mis = np.concatenate([r[good_blocks(r, half[i]), 4] for i, r in b.items()])
        return g1, frac_good(b), float(np.median(mis)) if mis.size else np.inf

    # Two maps are the same labelling when they put the traces within 0.3 of an orderlet spacing of each other (one
    # refinement on a subset leaves them a few px apart; a slip is a whole spacing or more).
    probe = ref['coeffs'][ref['nom']][[0, nnom // 2, nnom - 1]]
    px = np.array([0., nx // 2, nx - 1]); py = np.array([P(px, c) for c in probe]).ravel(); px = np.tile(px, len(probe))
    def same(g, q):
        return np.abs(distortion(g - q, px, py, nx, ny)).max() < 0.3 * spacing

    starts = [plane] + [np.array([c, plane[1], plane[2]]) for c in info['candidates']] + \
             [np.array([plane[0] + m * spacing, plane[1], plane[2]]) for m in (-2, -1, 1, 2)]
    starts = [model_from_plane(p, nx, ny) for p in starts]
    tried = []
    for g0 in starts:
        if any(same(g0, t[0]) for t in tried):
            continue
        tried.append(evaluate(g0))
    # A wrong labelling always loses orderlets: one slipped by a whole pair keeps the fibres matched (mismatch only a few
    # times worse), but not where the pair spacing changes across the chip. So only the labellings whose good fraction
    # is within 0.1 of the best compete, and of those the one whose profiles fit best is taken. A partial solution
    # (half the orderlets) can fit the few it keeps well, but never competes.
    fg_best = max(t[1] for t in tried)
    contenders = sorted([t for t in tried if t[1] >= fg_best - 0.1], key=lambda t: t[2])
    g, fg, mis = contenders[0]
    others = [t[2] for t in contenders[1:] if not same(t[0], g)]
    centre = lambda q: float(distortion(q, [nx // 2], [ny // 2], nx, ny)[0])
    info.update(labelling_tried=[(round(centre(t[0]), 2), round(t[1], 2), round(t[2], 5)) for t in tried],
                mismatch=mis, mismatch_ratio=(min(others) / mis) if others and mis > 0 else np.inf,
                offset_scan=not same(g, starts[0]))
    # the map on all orderlets and blocks, twice
    for _ in range(2):
        C = predict(ref, g, nx, ny)
        blocks = measure(img, mask, ref, C, usable_x)
        g = g + refine(ref, C, blocks, nx, ny)
    C = predict(ref, g, nx, ny)
    blocks = measure(img, mask, ref, C, usable_x)

    rows, diag = [], []
    tau = ref['prior_tau'] if 'prior_tau' in ref else DEFAULT_TAU
    for i in range(nnom):
        k = ref['nom'][i]
        xr = on_chip_range(C[i], ref['bottom_edge'][i], ref['top_edge'][i], ny, usable_x)
        src, coef, rms, err, off, edf = fit_orderlet(blocks[i], half[i], C[i], xr, nx, tau)
        xr2 = on_chip_range(coef, ref['bottom_edge'][i], ref['top_edge'][i], ny, usable_x)
        x2 = int(xr2.max()) if xr2.size else 0
        gb = good_blocks(blocks[i], half[i])
        rows.append(list(coef) + [ref['bottom_edge'][i], ref['top_edge'][i], 0, x2])
        diag.append(dict(orderlet=i, pair=int(ref['pair'][k]), fibre=int(ref['fibre'][k]), source=src,
                         n_blocks=len(blocks[i]), n_good=int(gb.sum()), rms=rms, fit_err=err, offset=off, edf=edf,
                         signal=float(np.median(blocks[i][gb, 3])) if gb.any() else 0.))
    table = pd.DataFrame(rows, columns=[f'Coeff{j}' for j in range(DEG + 1)] + ['BottomEdge', 'TopEdge', 'X1', 'X2'])
    # X2 equal within a pair, chip edge above 95 % (as OrderTrace)
    edge = nx - 1
    for n in range(0, nnom - 1, 2):
        if table.at[n + 1, 'X2'] / edge > 0.95:
            table.at[n, 'X2'] = table.at[n + 1, 'X2'] = edge
        else:
            table.at[n, 'X2'] = table.at[n + 1, 'X2']
    diag = pd.DataFrame(diag)
    # FALLBACK only when an orderlet has no usable light (its trace is the prediction alone). Faint orderlets that lean
    # on the reference shape ('anchored') are normal for the faintest bluest ones and are only counted (n_anchored).
    n_fb = int((diag.source == 'reference').sum())
    info.update(pred=C, blocks=blocks, model=g, offset=centre(g))     # map, and its value at the chip centre
    info.update(n_measured=int((diag.source == 'measured').sum()), n_blended=int((diag.source == 'blended').sum()),
                n_anchored=int((diag.source == 'anchored').sum()),
                n_reference=int((diag.source == 'reference').sum()), frac_good=float(fg),
                flag='FAILED' if fg < 0.5 else ('AMBIGUOUS' if info['mismatch_ratio'] < 2 else ('FALLBACK' if n_fb else 'OK')))
    return table, diag, info
