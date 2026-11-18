"""Trace-masked scattered-light model.

Used by the pipeline in two places:
  * master flats   : hrsreduce.master_flat.master_flat.FlatBackground      (frame='flat')
  * science frames : hrsreduce.extraction.order_rectification              (frame='science')

The order trace says where the two fibres of every order are, so the light between them can be sampled without
touching the fibres.  One sample ("node") is taken per inter-fibre gap and per block of columns:

  * wide gap (the trace apertures leave >= `wide_min` clear pixels): the median of the middle of the clear zone;
  * narrow gap (the bluest orders, where the apertures nearly touch): the lowest 3-pixel mean near the gap centre;
  * the strips just outside the first and the last traced fibre are sampled the same way ("edge" nodes).

Gaps that are filled by the wings of the fibres (mostly the gap between the two fibres of one order) sit above
their neighbours.  A narrow node that lies above a wide one next to it is dropped, and so is any node that stands
above both of its neighbours, so the surface follows the lower envelope of the nodes.  Across the orders the surface is an
interpolation through the lightly smoothed nodes of each column block; along the dispersion it is
median-filtered and smoothed over a few blocks.  It is local on purpose: a bright patch at one frame edge cannot
influence the rest of the frame.  Outside the outermost nodes it is held constant.

Two variants (fit_background(..., frame=...)):
  'flat'    : as described above; for flats and other high-signal frames with both fibres lit.
  'science' : noise-safe, for single science frames.  A lower envelope sinks into the noise of a faint frame, so
              here every gap sample carries its standard error, a gap is rejected only when it is significantly
              and consistently above its neighbours (wings of the lit fibre), nothing is replaced by a minimum, and
              the smoothing is wider.  The trace is trusted when the frame is too faint to check it.

Nothing here raises on a poor calibration: if the trace is missing, unreadable, or does not match the frame, the
order positions are taken from the frame itself and the result is flagged (info['method'], info['flags']).

"""
import warnings
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.signal import find_peaks, savgol_coeffs, savgol_filter

__all__ = ['fit_background', 'read_trace', 'header_cards', 'order_shares']
MAD = lambda q: 1.4826 * np.nanmedian(np.abs(q - np.nanmedian(q)))


# ----------------------------------------------------------------------------------------------
def read_trace(trace):
    """Order trace as a dict of arrays.  `trace` is the pipeline's CSV (index, Coeff0..5, BottomEdge, TopEdge,
    X1, X2), or an array / DataFrame with those ten columns.  Returns None if it cannot be read."""
    try:
        if isinstance(trace, str):
            t = np.genfromtxt(trace, delimiter=',', skip_header=1)[:, 1:]
        else:
            t = np.asarray(getattr(trace, 'values', trace), float)
        t = np.atleast_2d(t)
        if t.shape[1] < 10 or t.shape[0] < 4 or not np.isfinite(t[:, :10]).all():
            return None
        return dict(coef=t[:, 0:6], lo=t[:, 6], hi=t[:, 7], x1=t[:, 8], x2=t[:, 9])
    except Exception:
        return None


def _centres(coef, x):
    """centres of every orderlet at the columns x: (n_orderlets, len(x))"""
    return coef @ np.vander(np.atleast_1d(np.asarray(x, float)), coef.shape[1], increasing=True).T


def _trace_shift(img, tr, search=12, one_fibre=False):
    """Offset (pixels, along the cross-dispersion axis) between the trace and the frame, from the light summed at
    the trace centres as the trace is slid over the frame.  Returns (shift, contrast); contrast is the light at
    the shifted trace centres over the light midway between well-separated fibres (large when the trace fits the
    frame, about 1 when it does not).  one_fibre: use only the brighter of the two fibres (science frames, where
    the other fibre is dark and its trace centres would otherwise catch the wings of the lit one)."""
    ny, nx = img.shape; shifts = np.arange(-search, search + 1); best = []; contrast = []
    for xc in (0.25 * nx, 0.5 * nx, 0.75 * nx):
        ok = (tr['x1'] <= xc) & (tr['x2'] >= xc)
        if ok.sum() < 4: continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); prof = np.nanmedian(img[:, int(xc) - 8:int(xc) + 9], axis=1)
        prof = np.nan_to_num(prof); prof = prof / np.maximum(maximum_filter1d(prof, 151), 1e-9)   # every order counts the same
        cf = np.sort(_centres(tr['coef'][ok], [xc])[:, 0]); c = np.rint(cf).astype(int)
        if one_fibre:
            sc = [np.array([np.sum(prof[np.clip(c[p::2] + s, 0, ny - 1)]) for s in shifts]) for p in (0, 1)]
            p = 0 if sc[0].max() >= sc[1].max() else 1; c = c[p::2]
        score = np.array([np.sum(prof[np.clip(c + s, 0, ny - 1)]) for s in shifts])
        k = int(np.argmax(score)); s = float(shifts[k])
        if 0 < k < len(shifts) - 1:                                                      # parabolic refinement
            a, b, d = score[k - 1], score[k], score[k + 1]
            if (a - 2 * b + d) < 0: s += 0.5 * (a - d) / (a - 2 * b + d)
        best.append(s)
        on = np.median(prof[np.clip(c + int(shifts[k]), 0, ny - 1)])
        wide = np.diff(cf) > 40; mid = np.rint(0.5 * (cf[:-1] + cf[1:])[wide]).astype(int) + int(shifts[k])
        off = np.median(prof[np.clip(mid, 0, ny - 1)]) if wide.sum() >= 3 else np.min(score) / len(c)
        contrast.append(on / max(off, 1e-3))
    if not best: return 0.0, 0.0
    return float(np.median(best)), float(np.median(contrast))


# ------------------------------------------------------------------------------------------------ nodes
def _run3(p):
    """3-pixel running mean that keeps NaN where any of the three is NaN"""
    out = np.full(len(p), np.nan)
    if len(p) >= 3: out[1:-1] = (p[:-2] + p[1:-1] + p[2:]) / 3.0
    return out


def _gap_profile(img, xs, cl, cu, want_sigma=False):
    """Median cross-dispersion profile of the gap between the centre lines cl and cu (given on the columns xs),
    straightened along the line midway between them.  Returns (offsets from the mid line, profile, mean mid
    line, separation) or None; with want_sigma also the standard error of the profile at every offset, from the
    scatter of the pixels along the columns."""
    ny = img.shape[0]; sep = float(np.mean(cu - cl))
    if not np.isfinite(sep) or sep < 6: return None
    mid = 0.5 * (cl + cu); half = int(sep // 2); off = np.arange(-half, half + 1)
    rows = np.rint(mid).astype(int)[:, None] + off[None, :]
    good = (rows >= 2) & (rows < ny - 2)
    vals = np.where(good, img[np.clip(rows, 0, ny - 1), xs[:, None]], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); prof = np.nanmedian(vals, axis=0)
    prof[np.mean(np.isfinite(vals), axis=0) < 0.5] = np.nan
    if want_sigma:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sig = 1.2533 * 1.4826 * np.nanmedian(np.abs(vals - prof[None, :]), axis=0) / np.sqrt(np.maximum(np.isfinite(vals).sum(axis=0), 1))
        return off, prof, float(np.mean(mid)), sep, sig
    return off, prof, float(np.mean(mid)), sep


def _nodes_from_trace(img, tr, dy, xbin, margin, wide_min, science=False):
    """One node per gap and column block.  Wide gaps: median of the middle of the clear zone.  Narrow gaps: the
    3-pixel mean at the offset where the gap is lowest; that offset is found once per gap, on the median profile
    of all column blocks, so that picking a minimum does not bias the faint flats low.
    science: each node also gets its standard error (ninth element), and the per-gap profile stack is not
    normalised by the neighbouring peak (in a faint science frame that peak is noise)."""
    ny, nx = img.shape; nb = max(4, nx // xbin); e = np.linspace(0, nx, nb + 1).astype(int); recs = []
    for b in range(nb):
        xs = np.arange(e[b], e[b + 1]); xc = xs.mean()
        C = _centres(tr['coef'], xs) + dy
        idx = np.argsort(C.mean(axis=1)); idx = idx[(tr['x1'][idx] <= xc) & (tr['x2'][idx] >= xc)]
        if len(idx) < 4: continue
        pairs = [(C[a], C[c], tr['hi'][a], tr['lo'][c], 0, (a, c)) for a, c in zip(idx[:-1], idx[1:])]
        a0, a1, a2 = idx[:3]; z0, z1, z2 = idx[-1], idx[-2], idx[-3]
        pairs.insert(0, (C[a0] - (C[a2] - C[a1]), C[a0], tr['hi'][a1], tr['lo'][a0], -1, (-1, a0)))
        pairs.append((C[z0], C[z0] + (C[z1] - C[z2]), tr['hi'][z0], tr['lo'][z1], +1, (z0, -1)))
        for g, (cl, cu, h, l, edge, key) in enumerate(pairs):
            p = _gap_profile(img, xs, cl, cu, want_sigma=science)
            if p is None: continue
            off, prof, midm, sep = p[:4]
            ends = ([prof[:3]] if edge >= 0 else []) + ([prof[-3:]] if edge <= 0 else [])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore'); peak = float(np.nanmax(np.concatenate(ends)))
            o_lo = -sep / 2 + h + margin; o_hi = sep / 2 - l - margin
            recs.append(dict(b=b, xc=xc, g=g, key=key, off=off, prof=prof, mid=midm, sep=sep, peak=peak, edge=edge,
                             o_c=0.5 * (o_lo + o_hi), clear=o_hi - o_lo, sig=p[4] if science else None,
                             ap_lo=o_lo - margin, ap_hi=o_hi + margin))
    bykey = {}
    for r in recs: bykey.setdefault(r['key'], []).append(r)
    out = []
    for key, rr in bykey.items():
        H = max(len(r['off']) // 2 for r in rr); stack = np.full((len(rr), 2 * H + 1), np.nan)
        for i, r in enumerate(rr):
            h = len(r['off']) // 2
            if science: stack[i, H - h:H + h + 1] = r['prof']
            else: stack[i, H - h:H + h + 1] = r['prof'] / max(abs(r['peak']), 1e-9) if np.isfinite(r['peak']) else np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); med = np.nanmedian(stack, axis=0)
        med[np.mean(np.isfinite(stack), axis=0) < 0.3] = np.nan; r3m = _run3(med)
        kid = (key[0] + 1) * 1000 + key[1] + 1
        for r in rr:
            off, prof = r['off'], r['prof']; fin = np.isfinite(prof)
            if r['clear'] >= wide_min:
                sel = (np.abs(off - r['o_c']) <= 0.3 * r['clear']) & fin
                if sel.sum() < 3: continue
                if science:
                    out.append((r['b'], r['xc'], r['mid'] + float(np.mean(off[sel])), float(np.median(prof[sel])), r['peak'], 'wide', r['g'], kid,
                                float(1.2533 * np.sqrt(np.nanmean(r['sig'][sel] ** 2) / sel.sum()))))
                    continue
                out.append((r['b'], r['xc'], r['mid'] + float(np.mean(off[sel])), float(np.median(prof[sel])), r['peak'], 'wide', r['g'], kid))
                continue
            r3 = _run3(prof); sel = (np.abs(off - r['o_c']) <= max(2.0, 0.2 * r['sep'])) & np.isfinite(r3)
            if not sel.any(): continue
            cand = np.flatnonzero(sel); ref = r3m[off[cand] + H]
            k = cand[int(np.nanargmin(ref))] if np.isfinite(ref).any() else cand[int(np.argmin(r3[cand]))]
            if science:
                # Is the gap structured (wings of a lit fibre), judged on the stack of all column blocks?  If it is,
                # the lowest point is the floor, as in a flat.  If it is not (faint frame), picking a minimum would
                # only pick noise: take the plain mean of everything between the two aperture edges instead.
                s_stack = 1.2533 * np.sqrt(np.nanmean(r['sig'][cand] ** 2) / 3.0) / np.sqrt(len(rr))
                # (the range of a dozen noise values is already about 3 sigma, so the bar is set at 6)
                structured = np.isfinite(ref).sum() >= 2 and (np.nanmax(ref) - np.nanmin(ref)) > 6.0 * s_stack
                zone = (off >= r['ap_lo']) & (off <= r['ap_hi']) & fin & np.isfinite(r['sig'])
                kind = 'edge' if r['edge'] else 'narrow'
                if structured or zone.sum() < 3:
                    if not structured: k = cand[int(np.argmin(np.abs(off[cand] - r['o_c'])))]
                    out.append((r['b'], r['xc'], r['mid'] + float(off[k]), float(r3[k]), r['peak'], kind, r['g'], kid,
                                float(np.sqrt(np.nansum(r['sig'][k - 1:k + 2] ** 2)) / 3.0)))
                else:
                    out.append((r['b'], r['xc'], r['mid'] + float(np.mean(off[zone])), float(np.mean(prof[zone])), r['peak'], kind, r['g'], kid,
                                float(np.sqrt(np.sum(r['sig'][zone] ** 2)) / zone.sum())))
                continue
            out.append((r['b'], r['xc'], r['mid'] + float(off[k]), float(r3[k]), r['peak'], 'edge' if r['edge'] else 'narrow', r['g'], kid))
    return out, 0.5 * (e[:-1] + e[1:])


def _nodes_from_image(img, xbin, depth=0.4):
    """Fallback without a trace: the gaps are the deep valleys of the cross-dispersion profile of each column
    block (a valley must rise by `depth` of the local order peak on both sides, so the structure inside a sliced
    fibre image is not taken for a gap)."""
    ny, nx = img.shape; nb = max(4, nx // xbin); e = np.linspace(0, nx, nb + 1).astype(int); out = []
    for b in range(nb):
        xc = 0.5 * (e[b] + e[b + 1]); w = min(8, (e[b + 1] - e[b]) // 2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); prof = np.nanmedian(img[:, int(xc) - w:int(xc) + w + 1], axis=1)
        p = np.nan_to_num(median_filter(prof, 5, mode='nearest')); top = np.maximum(maximum_filter1d(p, 201), 1e-9)
        pn = p / top
        vl, _ = find_peaks(-pn, prominence=depth, distance=8)
        vl = vl[(vl > 4) & (vl < ny - 5)]
        if len(vl) < 6: continue
        r3 = _run3(prof)
        for g, j in enumerate(vl):
            lo = j
            while lo > 2 and pn[lo - 1] <= pn[j] + 0.03: lo -= 1
            hi = j
            while hi < ny - 3 and pn[hi + 1] <= pn[j] + 0.03: hi += 1
            if hi - lo + 1 >= 7:
                q = int(0.2 * (hi - lo)); seg = prof[lo + q:hi - q + 1]
                if np.isfinite(seg).sum() < 3: continue
                out.append((b, xc, 0.5 * (lo + hi), float(np.nanmedian(seg)), float(top[j]), 'wide', g, -1))
            else:
                seg = r3[max(j - 2, 1):j + 3]
                if not np.isfinite(seg).any(): continue
                k = max(j - 2, 1) + int(np.nanargmin(seg))
                out.append((b, xc, float(k), float(r3[k]), float(top[j]), 'narrow', g, -1))
    return out, 0.5 * (e[:-1] + e[1:])


# ------------------------------------------------------------------------------------------------ surface
def _block_profile(y, v, kind, ygrid, neighbour_ratio):
    """Background across the orders in one column block, from its nodes.  Returns (values on ygrid, kept mask).

    1. A narrow or edge node that lies above a wide node next to it is dropped: where clean wide gaps exist, the
       narrow gap between the two fibres of an order is filled by their wings.  (A narrow node below its wide
       neighbour is kept: a "wide" gap can hold an order the trace missed.)
    2. A node that stands above both remaining neighbours by `neighbour_ratio` is dropped (lower envelope).
    3. Where narrow gaps follow one another (the bluest orders), each is capped at the mean of its two neighbours:
       one of the two kinds of gap (inside a fibre pair, between orders) is a little higher than the other, and
       the lower one is the floor.  The outermost node, if narrow or edge, is capped at its neighbour.
    4. Light 3-point smoothing, then (PCHIP) interpolation, held constant outside the nodes."""
    o = np.argsort(y); y, v, kind = y[o], v[o], kind[o]; n = len(y); wide = kind == 'wide'
    big = np.inf; wv = np.where(wide & np.isfinite(v), v, big)
    floor = np.minimum(np.r_[big, wv[:-1]], np.r_[wv[1:], big])            # lowest wide neighbour, if any
    keep = np.isfinite(v) & ~(~wide & (v > floor))
    for _ in range(3):
        i = np.flatnonzero(keep)
        if len(i) < 4: break
        vv = v[i]; prev = np.r_[vv[1], vv[:-1]]; nxt = np.r_[vv[1:], vv[-2]]
        tol = 2 * MAD(np.diff(vv)) / np.sqrt(2) if len(vv) > 8 else 0.0
        bad = vv > neighbour_ratio * np.maximum(np.maximum(prev, nxt), 0.0) + tol
        if not bad.any(): break
        keep[i[bad]] = False
    i = np.flatnonzero(keep); back = np.zeros(n, bool); back[o[i]] = True
    if len(i) < 4: return None, back
    yk, vk = y[i], v[i].copy(); nar = ~wide[i]
    cap = nar[1:-1] & nar[:-2] & nar[2:]
    vk[1:-1] = np.where(cap, np.minimum(vk[1:-1], 0.5 * (vk[:-2] + vk[2:])), vk[1:-1])
    if nar[0]: vk[0] = min(vk[0], vk[1])
    if nar[-1]: vk[-1] = min(vk[-1], vk[-2])
    vs = vk.copy(); vs[1:-1] = 0.25 * vk[:-2] + 0.5 * vk[1:-1] + 0.25 * vk[2:]
    yk, u = np.unique(yk, return_index=True); vs = vs[u]
    if len(yk) < 4: return None, back
    return PchipInterpolator(yk, vs)(np.clip(ygrid, yk[0], yk[-1])), back


def _winged_gaps(N, vs, ss, held, nsig, rel=0.10):
    """Science frames: narrow gaps that are filled by the wings of a lit fibre, judged once per gap over the whole
    dispersion (a property of the gap, so all column blocks vote).  A gap is winged when it stands above the mean
    of its two neighbours by more than nsig times the error of that excess AND by more than `rel` of the level of
    the neighbours.  Of two adjacent winged gaps only the worse is taken out.  In a faint frame nothing is
    significant, nothing is dropped, and the surface is an unbiased smoothing of all gaps.  Returns a node mask."""
    n = len(N['v']); drop = np.zeros(n, bool)
    look = {(int(b), int(g)): i for i, (b, g) in enumerate(zip(N['bin'], N['gap'])) if not held[i] and np.isfinite(vs[i])}
    exc = {}
    for (b, g), i in look.items():
        j, k = look.get((b, g - 1)), look.get((b, g + 1))
        if j is None or k is None: continue
        m = 0.5 * (vs[j] + vs[k]); exc.setdefault(int(N['kid'][i]), []).append((vs[i] - m, m, ss[i] ** 2 + 0.25 * (ss[j] ** 2 + ss[k] ** 2), int(g)))
    cand = []
    narrow_gap = {int(k): np.mean(N['kind'][N['kid'] == k] != 'wide') > 0.5 for k in np.unique(N['kid'])}
    for kid, q in exc.items():
        if kid < 0 or len(q) < 8 or not narrow_gap.get(kid, False): continue       # a wide gap is the clean reference
        q = np.array(q); e = np.median(q[:, 0]); err = 1.2533 * np.sqrt(np.mean(q[:, 2]) / len(q)); lvl = np.median(q[:, 1])
        err = max(err, 1.2533 * MAD(q[:, 0]) / np.sqrt(len(q)))             # never trust the formal error below the scatter
        if e > nsig * err and e > rel * abs(lvl): cand.append((e / err, kid, int(np.median(q[:, 3]))))
    taken = set()
    for _, kid, g in sorted(cand, reverse=True):                              # worst first; never two neighbours
        if (g - 1) in taken or (g + 1) in taken: continue
        taken.add(g); drop |= (N['kid'] == kid)
    return drop


def _block_profile_science(y, v, sig, kind, ygrid, neighbour_ratio, nsig, y_smooth=100.0):
    """Science-frame counterpart of _block_profile.  Same four steps, but every decision that would push the
    surface down needs the excess to be significant (nsig times its error), nothing is ever replaced by a
    minimum, and the smoothing across the orders is wider (y_smooth pixels).  With pure noise in the nodes it reduces to
    smoothing and interpolation, so it does not sink below the true level as a lower envelope would.
    Returns (values on ygrid, kept mask)."""
    o = np.argsort(y); y, v, sig, kind = y[o], v[o], sig[o], kind[o]; n = len(y); wide = kind == 'wide'
    fin = np.isfinite(v) & np.isfinite(sig); big = np.inf
    wv = np.where(wide & fin, v, big); ws = np.where(wide & fin, sig, 0.0)
    lv, rv = np.r_[big, wv[:-1]], np.r_[wv[1:], big]; floor = np.minimum(lv, rv); fs = np.where(lv <= rv, np.r_[0.0, ws[:-1]], np.r_[ws[1:], 0.0])
    with np.errstate(invalid='ignore'):
        keep = fin & ~(~wide & np.isfinite(floor) & (v - floor > nsig * np.sqrt(sig ** 2 + fs ** 2)))
    for _ in range(3):
        i = np.flatnonzero(keep)
        if len(i) < 4: break
        vv, ss = v[i], sig[i]; pv, nv = np.r_[vv[1], vv[:-1]], np.r_[vv[1:], vv[-2]]; ps, ns = np.r_[ss[1], ss[:-1]], np.r_[ss[1:], ss[-2]]
        hi = np.maximum(pv, nv); hs = np.where(pv >= nv, ps, ns); exc = vv - hi
        bad = (exc > nsig * np.sqrt(ss ** 2 + hs ** 2)) & (exc > (neighbour_ratio - 1) * np.maximum(hi, 0.0))
        if not bad.any(): break
        keep[i[bad]] = False
    i = np.flatnonzero(keep); back = np.zeros(n, bool); back[o[i]] = True
    if len(i) < 4: return None, back
    yk, vk, sk = y[i], v[i].copy(), sig[i]; nar = ~wide[i]
    m = 0.5 * (vk[:-2] + vk[2:]); em = np.sqrt(sk[1:-1] ** 2 + 0.25 * (sk[:-2] ** 2 + sk[2:] ** 2))
    cap = nar[1:-1] & nar[:-2] & nar[2:] & (vk[1:-1] - m > nsig * em)
    vk[1:-1] = np.where(cap, m, vk[1:-1])
    if nar[0] and vk[0] - vk[1] > nsig * np.hypot(sk[0], sk[1]): vk[0] = vk[1]
    if nar[-1] and vk[-1] - vk[-2] > nsig * np.hypot(sk[-1], sk[-2]): vk[-1] = vk[-2]
    # smoothing across the orders: local straight-line fit around every node, Gaussian in pixels (not in gap
    # number: the gaps are 25 px apart at one end of the frame and 130 px at the other), weighted by the node
    # errors.  A straight line, not a mean, so the surface is not pulled inwards at the first and last orders.
    d = yk[None, :] - yk[:, None]; w = np.exp(-0.5 * (d / y_smooth) ** 2) / np.maximum(sk[None, :], 1e-6) ** 2
    s0, s1, s2 = w.sum(1), (w * d).sum(1), (w * d * d).sum(1); t0, t1 = (w * vk[None, :]).sum(1), (w * d * vk[None, :]).sum(1)
    den = s0 * s2 - s1 ** 2
    vs = np.where(den > 1e-9 * s0 * s2, (s2 * t0 - s1 * t1) / np.where(den != 0, den, 1.0), t0 / s0)
    yk, u = np.unique(yk, return_index=True); vs = vs[u]
    if len(yk) < 4: return None, back
    return PchipInterpolator(yk, vs)(np.clip(ygrid, yk[0], yk[-1])), back


def _smooth_along_gaps(N, held, x_median, x_window):
    """Node values smoothed along the dispersion, gap by gap (median over x_median blocks, then a quadratic
    Savitzky-Golay filter over x_window blocks).  Done before the lower envelope is taken across the orders, so
    that in a faint flat the envelope follows the floor and not the noise of single nodes."""
    vs = N['v'].copy()
    for kid in np.unique(N['kid']):
        if kid < 0: continue                         # nodes found without a trace cannot be followed along x
        i = np.flatnonzero(N['kid'] == kid); i = i[np.argsort(N['bin'][i])]
        v = N['v'][i]; ok = np.isfinite(v) & ~held[i]
        if ok.sum() < 5: continue
        b = N['bin'][i].astype(float); m = np.interp(b, b[ok], v[ok])
        if x_median > 1: m = median_filter(m, x_median, mode='nearest')
        win = min(x_window, len(m) if len(m) % 2 else len(m) - 1)
        vs[i] = savgol_filter(m, win, 2, mode='interp') if win >= 5 else m
    return vs


def fit_background(img, trace=None, bpm=None, xbin=32, margin=2.0, wide_min=7.0, neighbour_ratio=1.25,
                   x_median=None, x_window=None, ygrid_step=8, max_shift=12, holdout_frac=0.0, seed=1,
                   frame='flat', nsig=3.0, y_smooth=100.0):
    """Scattered-light surface of a frame.

    frame : 'flat' (default) -- high-signal frames with both fibres lit (flats, stacked calibrations): the surface
        is the LOWER ENVELOPE of the gap samples.
        'science' -- single, possibly faint frames with one fibre lit: noise-safe.  Each gap sample carries its
        standard error; a gap is taken out only when it stands above its neighbours significantly (nsig) and
        consistently along the whole order (wings of the lit fibre); nothing is replaced by a minimum; the
        smoothing is wider.  On pure noise it is an unbiased smoothing, where the lower envelope would sink.
    nsig : significance (in standard errors) needed in a science frame before a gap sample is rejected.
    y_smooth : science frames only; smoothing length across the orders, in pixels.

    img : 2-D frame, orders running along axis 1.
    trace : order-trace CSV path, array or DataFrame; None or unusable -> order positions from the frame itself.
    bpm : optional boolean bad-pixel mask (True = bad).
    xbin : width of the column blocks in which the gaps are sampled.
    margin : extra pixels kept clear outside each trace aperture.
    wide_min : clear width (pixels) from which a gap is sampled by its median instead of its minimum.
    neighbour_ratio : a node is dropped when it exceeds both neighbours by this factor.
    x_median, x_window : along the dispersion the surface is median-filtered over x_median column blocks, then
        smoothed with a quadratic Savitzky-Golay filter over x_window blocks (default 3 and 7 for a flat, 5 and 21
        for a science frame).
    holdout_frac : for testing only; this fraction of the nodes is left out of the fit and reported back.

    Returns (bkg, info).  bkg has the shape of img.  info holds the method, flags, the trace shift applied, node
    counts, the node scatter (absolute, and as a percentage of the order peak), and the node table
    (info['nodes'], a dict of arrays)."""
    img = np.array(img, dtype=float, copy=True); ny, nx = img.shape; flags = []
    science = str(frame).lower().startswith('sci')
    if x_median is None: x_median = 5 if science else 3
    if x_window is None: x_window = 21 if science else 7
    if bpm is not None and np.shape(bpm) == img.shape: img[np.asarray(bpm, bool)] = np.nan
    tr = read_trace(trace) if trace is not None else None
    method = 'trace'; dy = 0.0; contrast = np.nan
    if tr is None:
        method = 'image'; flags.append('NOTRACE')
    elif science:
        # The trace is used as it is and the frame is never searched for orders: the pipeline extracts the
        # spectrum with this same trace, and with one flat-topped fibre lit a shift cannot be measured to better
        # than several pixels.  The contrast is only reported (flag when the frame is too faint to show orders).
        _, contrast = _trace_shift(img, tr, max_shift, one_fibre=True); dy = 0.0
        if contrast < 4.0: flags.append('TRACEUNCHECKED')
    else:
        dy, contrast = _trace_shift(img, tr, max_shift)
        if contrast < 4.0:                       # the trace does not line up with the orders at any shift
            method = 'image'; flags.append('TRACEMISMATCH'); dy = 0.0
        elif abs(dy) >= max_shift - 0.5:
            method = 'image'; flags.append('TRACESHIFT>MAX'); dy = 0.0
        elif abs(dy) < 3.0:                      # within the reach of the node search; leave the trace alone
            dy = 0.0
        else:
            flags.append('TRACESHIFTED')
    if method == 'trace':
        raw, xc = _nodes_from_trace(img, tr, dy, xbin, margin, wide_min, science=science)
        if len(raw) < 50: method = 'image'; flags.append('FEWNODES')
    if method == 'image':
        raw, xc = _nodes_from_image(img, xbin)
    nb = len(xc)

    def constant(level, n, N=None):              # nothing to fit: a flat level, flagged, never an exception
        flags.append('NOFIT')
        return np.full((ny, nx), level), dict(method='constant', flags=flags, shift=dy, contrast=contrast, n_nodes=n,
                                               n_used=0, node_rms=np.nan, node_scatter_pct=np.nan, bkg_min=level,
                                               bkg_median=level, bkg_max=level, nodes=N)
    if len(raw) < 20:
        return constant(float(np.nanpercentile(img, 2)) if np.isfinite(img).any() else 0.0, len(raw))
    N = dict(bin=np.array([r[0] for r in raw]), x=np.array([r[1] for r in raw]), y=np.array([r[2] for r in raw]),
             v=np.array([r[3] for r in raw]), peak=np.array([r[4] for r in raw]), kind=np.array([r[5] for r in raw]),
             gap=np.array([r[6] for r in raw]), kid=np.array([r[7] for r in raw]))
    n = len(N['v']); held = np.zeros(n, bool)
    if holdout_frac > 0: held = np.random.default_rng(seed).random(n) < holdout_frac
    ygrid = np.arange(0, ny + ygrid_step, ygrid_step, dtype=float); S = np.full((nb, len(ygrid)), np.nan)
    used = np.zeros(n, bool)
    vs = _smooth_along_gaps(N, held, x_median, x_window)
    if science:
        # node errors: measured where the trace gave them, else from the scatter of the nodes about their smooth run
        sg = np.array([r[8] if len(r) > 8 else np.nan for r in raw], float)
        fallback = MAD((N['v'] - vs)[np.isfinite(vs)]) if np.isfinite(vs).any() else 1.0
        sg = np.where(np.isfinite(sg) & (sg > 0), sg, fallback); N['sig'] = sg
        win = min(x_window, nb if nb % 2 else nb - 1)
        ss = sg * (np.sqrt(np.sum(savgol_coeffs(win, 2) ** 2)) if win >= 5 else 1.0)       # error after the smoothing along x
        winged = _winged_gaps(N, vs, ss, held, nsig); N['winged'] = winged
    for b in range(nb):
        if science:
            i = np.flatnonzero((N['bin'] == b) & ~held & ~winged)
            if len(i) < 6: continue
            prof, k = _block_profile_science(N['y'][i], vs[i], ss[i], N['kind'][i], ygrid, neighbour_ratio, nsig, y_smooth)
        else:
            i = np.flatnonzero((N['bin'] == b) & ~held)
            if len(i) < 6: continue
            prof, k = _block_profile(N['y'][i], vs[i], N['kind'][i], ygrid, neighbour_ratio)
        used[i] = k
        if prof is not None: S[b] = prof
    okb = np.isfinite(S).all(axis=1)
    if okb.sum() < 4:
        return constant(float(np.nanmedian(N['v'])), n, N)
    if (~okb).any():                             # column blocks without a profile: interpolate along the dispersion
        for j in range(S.shape[1]): S[~okb, j] = np.interp(xc[~okb], xc[okb], S[okb, j])
        flags.append('BINSFILLED')
    if x_median > 1: S = median_filter(S, size=(x_median, 1), mode='nearest')        # one bad block cannot pull the surface
    win = min(x_window, nb if nb % 2 else nb - 1)
    if win >= 5: S = savgol_filter(S, win, 2, axis=0, mode='interp')
    xx = np.clip(np.arange(nx, dtype=float), xc[0], xc[-1]); jx = np.clip(np.searchsorted(xc, xx, side='right') - 1, 0, nb - 2)
    wx = ((xx - xc[jx]) / (xc[jx + 1] - xc[jx]))[:, None]
    Bx = (1 - wx) * S[jx] + wx * S[jx + 1]                                           # (nx, n_ygrid)
    yy = np.arange(ny, dtype=float); j = np.clip(np.searchsorted(ygrid, yy, side='right') - 1, 0, len(ygrid) - 2)
    w = ((yy - ygrid[j]) / ygrid_step)[:, None]
    bkg = (1 - w) * Bx.T[j] + w * Bx.T[j + 1]
    xi = np.clip(np.rint(N['x']).astype(int), 0, nx - 1); yi = np.clip(np.rint(N['y']).astype(int), 0, ny - 1)
    N['fit'] = bkg[yi, xi]; N['used'] = used; N['held'] = held
    r = (N['v'] - N['fit'])[used]; rp = (100 * (N['v'] - N['fit']) / N['peak'])[used]
    info = dict(method=method, flags=flags, shift=float(dy), contrast=float(contrast), n_nodes=int(n),
                n_used=int(used.sum()), node_rms=float(MAD(r)) if used.any() else np.nan,
                node_scatter_pct=float(MAD(rp[np.isfinite(rp)])) if used.any() else np.nan, bkg_min=float(bkg.min()),
                bkg_median=float(np.median(bkg)), bkg_max=float(bkg.max()), nodes=N)
    info['frame'] = 'science' if science else 'flat'
    if science:                                  # a science frame may sit below zero (bias level); that is not an error
        info['node_sigma'] = float(np.nanmedian(ss[used])) if used.any() else np.nan
        info['n_winged_gaps'] = int(len(np.unique(N['kid'][winged]))) if winged.any() else 0
    elif bkg.min() < -5 * max(info['node_rms'], 1e-9): flags.append('NEGATIVE')
    return bkg, info


def header_cards(info):
    """FITS cards that record what was done, for the product header."""
    return [('BKGTYPE', info.get('frame', 'flat'), 'scattered-light model variant: flat or science'),
            ('BKGMETH', info['method'], 'scattered light from: trace, image or constant'),
            ('BKGNODE', info['n_used'], 'inter-order nodes used in the fit'),
            ('BKGRMS', round(float(info.get('node_rms', np.nan)), 5), 'node scatter about the surface'),
            ('BKGSHIFT', round(float(info.get('shift', 0.0)), 2), 'trace shift applied (pixels)'),
            ('BKGFLAG', ','.join(info['flags']) or 'OK', 'scattered-light model flags')]


def order_shares(img, bkg, trace, half=100):
    """Share of the scattered light in the light of every orderlet: sum of `bkg` over the trace aperture divided by
    the sum of `img` (the frame BEFORE the subtraction) over the same aperture, in the central 2*half columns of
    the orderlet.  Returns an array in the row order of the trace (NaN where it cannot be measured), or None."""
    tr = read_trace(trace) if trace is not None else None
    if tr is None: return None
    ny, nx = np.shape(img); out = np.full(len(tr['lo']), np.nan)
    for k in range(len(out)):
        xm = int(0.5 * (max(tr['x1'][k], 0) + min(tr['x2'][k], nx - 1))); xs = np.arange(max(xm - half, 0), min(xm + half, nx))
        c = _centres(tr['coef'][k:k + 1], xs)[0]; top = bot = 0.0
        for x, cc in zip(xs, c):
            lo = int(round(cc - tr['lo'][k])); hi = int(round(cc + tr['hi'][k])) + 1
            if lo < 0 or hi > ny: continue
            top += float(np.nansum(bkg[lo:hi, x])); bot += float(np.nansum(img[lo:hi, x]))
        if bot > 0: out[k] = top / bot
    return out


if __name__ == '__main__':
    import argparse
    from astropy.io import fits
    ap = argparse.ArgumentParser(description='Trace-masked scattered-light model of one frame')
    ap.add_argument('frame'); ap.add_argument('trace', nargs='?'); ap.add_argument('--out')
    ap.add_argument('--science', action='store_true', help='noise-safe variant for single science frames')
    a = ap.parse_args()
    data = fits.getdata(a.frame).astype(float); bkg, info = fit_background(data, a.trace, frame='science' if a.science else 'flat')
    print({k: v for k, v in info.items() if k != 'nodes'})
    if a.out:
        hdr = fits.Header()
        for c in header_cards(info): hdr[c[0]] = (c[1], c[2])
        fits.PrimaryHDU(bkg.astype(np.float32), hdr).writeto(a.out, overwrite=True)
