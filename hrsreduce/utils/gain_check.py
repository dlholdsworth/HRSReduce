"""Gain check on the flats of a master flat.

MasterFlat measures the detector gain from the flats it stacks, as a check of the gain L0Corrections applied (from
hrsreduce/L0_Corrections/gain_table.csv). Nothing in the data is changed: the result goes into GAIN* cards of the
master flat, and a measured gain that differs from the applied one by more than DEV_LIMIT and by more than DEV_SIGMA
times its uncertainty is flagged (GAINFLAG DEVIATES and a warning), the sign that gain_table.csv needs a new row.
The check never stops a reduction.

The frames come bias-subtracted and in electrons with the applied gain g_a, so the photon variance is m / k with
k = g_true / g_a (k = 1 when the applied gain is right) and the measured gain is k g_a. Each set of at least
MIN_FRAMES flats of one night and exposure time is one estimate:
  1. the frames are scaled to a common lamp level (median of their brightest 10 % of pixels);
  2. pixels more than 8 sigma from the median frame in any frame (cosmic rays, defects) are left out;
  3. large-scale differences between the frames (lamp colour drift, the shutter pattern of short exposures) are
     removed by dividing each frame by its ratio to the mean frame, summed over BLOCK x BLOCK pixel blocks and
     interpolated between the block centres (a flux-weighted ratio adds no variance that matters; block medians do);
  4. per-pixel mean m and variance v across the frames, in NBINS bins of m between LO and HI ADU; the mean variance
     of a bin leaves out only values beyond the 1e-6 tail of a chi-square (a sigma clip would bias it low);
  5. a weighted fit v = c + m / k + a m^2 over the bins, with c measured in the same frames from the pixels between
     the orders (mean 0-20 ADU) as v - m / k. sqrt(a) is the frame-to-frame illumination change (ILL).
A set is used when it has at least MIN_BINS bins and ILL below ILL_LIMIT; the gain is the weighted mean of the sets.
Measured this way on the regular calibrations of 2023-2026 (HR, MR, LR, HS) one set scatters by 0.7-1 % and the
night-to-night scatter of the per-night medians is 0.2-0.3 %.

A multi-amplifier frame is measured per amplifier (DATASEC1, DATASEC2, ... of the L0 frame; the red frames are
flipped by L0Corrections, so their y ranges are flipped here too).
"""
import os
import re

import numpy as np
from astropy.io import fits

MIN_FRAMES = 3        # flats per set
EDGE = 40             # px left out at each edge of an amplifier region
LO, HI = 300., 3000.  # ADU range of the per-pixel mean used in the fit (well within the linear range)
NBINS = 7
MIN_BIN_PIX = 5000    # pixels per bin
MIN_BINS = 4
BLOCK = 128           # px, blocks of the ratio maps
STRIP = 1024          # rows processed at a time (multiple of BLOCK; bounds the memory of large stacks)
ILL_LIMIT = 0.003     # frame-to-frame illumination term above which a set is not used
DEV_LIMIT = 0.02      # |measured / applied - 1| above which the gain is flagged, if also more than DEV_SIGMA sigma
DEV_SIGMA = 3.0       # (one set of 3-5 flats scatters by 0.5-1 %; the steps of the HRS gain seen so far are 5-20 %)


def _chi2_mean(x, nu, p=1e-6):
    """Mean of per-pixel sample variances with nu degrees of freedom, leaving out values above the 1 - p quantile of
    a chi-square distribution with that mean (pixels with excess variance). A symmetric sigma clip would cut the
    long upper tail of the chi-square and bias the mean low (by ~0.7 % for 3 frames, i.e. the gain high)."""
    from scipy.stats import chi2
    x = x[np.isfinite(x)]
    if not x.size:
        return np.nan
    t = chi2.ppf(1 - p, nu) / nu
    mu = float(np.mean(x))
    for _ in range(10):
        new = float(np.mean(x[x < t * mu])) if mu > 0 else mu
        if abs(new - mu) <= 1e-7 * abs(mu):
            break
        mu = new
    return mu


def _flux_map(f, ref, ok, den_min):
    """sum(f) / sum(ref) over BLOCK x BLOCK blocks (ok pixels), linearly interpolated between the block centres."""
    from scipy.ndimage import map_coordinates
    ny, nx = f.shape
    by, bx = max(1, ny // BLOCK), max(1, nx // BLOCK)
    cut = lambda a: a[:by * BLOCK, :bx * BLOCK].reshape(by, BLOCK, bx, BLOCK).sum(axis=(1, 3), dtype=np.float64)
    num, den = cut(np.where(ok, f, 0.0)), cut(np.where(ok, ref, 0.0))
    with np.errstate(all='ignore'):
        r = np.where(den > den_min, num / np.where(den > 0, den, 1), np.nan)
    fill = np.nanmedian(r) if np.isfinite(r).any() else 1.0
    r = np.where(np.isfinite(r), r, fill)
    yy = np.clip((np.arange(ny) + 0.5) / BLOCK - 0.5, 0, by - 1)
    xx = np.clip((np.arange(nx) + 0.5) / BLOCK - 0.5, 0, bx - 1)
    Y, X = np.meshgrid(yy, xx, indexing='ij')
    return map_coordinates(r, [Y, X], order=1, mode='nearest').astype(np.float32)


def _section(sec, ny, flip):
    """(row slice, column slice) of a FITS section '[x1:x2,y1:y2]' of the L0 frame; rows flipped for the red."""
    v = [int(t) for t in re.findall(r'\d+', str(sec))]
    if len(v) != 4:
        return None
    x1, x2, y1, y2 = v
    if flip:
        y1, y2 = ny - y2 + 1, ny - y1 + 1
    return slice(y1 - 1, y2), slice(x1 - 1, x2)


def _set_gain(frames, idx, rows, cols, ga, rn2):
    """Gain estimate of the frames idx (in electrons with applied gain ga) over one amplifier region. Only one strip
    of rows of the set is copied at a time."""
    rows = slice(rows.start + EDGE, rows.stop - EDGE)
    cols = slice(cols.start + EDGE, cols.stop - EDGE)
    if rows.stop - rows.start < BLOCK or cols.stop - cols.start < BLOCK:
        return dict(reason='region too small')
    lo, hi = LO * ga, HI * ga                    # the ADU limits in the electrons of the frames
    edges = np.geomspace(lo, hi, NBINS + 1)
    lvl = []
    for i in idx:
        f = frames[i, rows, cols]
        lvl.append(np.median(f[f > np.percentile(f, 90)]))
    lvl = np.array(lvl, dtype=np.float64)
    if not np.all(np.isfinite(lvl)) or np.any(lvl <= 0):
        return dict(reason='no light')
    scale = (lvl / lvl.mean()).astype(np.float32)
    vbin = [[] for _ in range(NBINS)]
    msum, nbin = np.zeros(NBINS), np.zeros(NBINS, dtype=np.int64)
    dark_v, dark_m = [], []
    for y0 in range(rows.start, rows.stop, STRIP):
        Fn = frames[idx, y0:min(y0 + STRIP, rows.stop), cols].astype(np.float32) / scale[:, None, None]
        med = np.median(Fn, axis=0)
        pred = np.sqrt(np.clip(med, 0, None) + rn2)      # photon + read noise for k ~ 1
        bad = np.any(np.abs(Fn - med) > 8 * pred, axis=0)
        mean0 = Fn.mean(axis=0)
        ok = (mean0 > 100 * ga) & ~bad
        for i in range(len(Fn)):
            Fn[i] /= _flux_map(Fn[i], mean0, ok, 2e4 * ga)
        m = Fn.mean(axis=0)
        v = Fn.var(axis=0, ddof=1)
        for j in range(NBINS):
            sel = ~bad & (m >= edges[j]) & (m < edges[j + 1])
            vbin[j].append(v[sel]); msum[j] += m[sel].sum(); nbin[j] += sel.sum()
        dk = ~bad & (m > -5 * ga) & (m < 20 * ga)
        dark_v.append(v[dk]); dark_m.append(m[dk])
    keep = nbin >= MIN_BIN_PIX
    if keep.sum() < MIN_BINS:
        return dict(reason=f'{int(keep.sum())} bins with light')
    ms = (msum / np.maximum(nbin, 1))[keep]
    nu = len(idx) - 1
    vs = np.array([_chi2_mean(np.concatenate(vbin[j]), nu) for j in np.flatnonzero(keep)])
    ns = nbin[keep]
    dark_v, dark_m = np.concatenate(dark_v), np.clip(np.concatenate(dark_m), 0, None)
    sw = np.sqrt(ns) / vs                      # the error of a bin variance is ~ v sqrt(2 / (n (N - 1)))
    A = np.vstack([ms, ms ** 2]).T
    k = 1.0
    for _ in range(3):                         # the zero point needs k; k needs the zero point
        c = (_chi2_mean(dark_v, nu) - float(np.mean(dark_m)) / k) if dark_v.size > 20000 else rn2
        y = vs - c
        (b, a), *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        if not np.isfinite(b) or b <= 0:
            return dict(reason='fit failed')
        k = 1.0 / b
    res = (y - A @ np.array([b, a])) * sw
    dof = max(len(ms) - 2, 1)
    cov = np.linalg.pinv((A * sw[:, None]).T @ (A * sw[:, None])) * max(float(res @ res) / dof, 1.0)
    err_k = np.sqrt(max(cov[0, 0], 0.0)) / b ** 2
    return dict(gain=k * ga, err=err_k * ga, ill=float(np.sqrt(max(a, 0.0))), zero=float(c), nbins=int(len(ms)))


def _night(file, hdr):
    name = os.path.basename(str(file))
    digits = re.search(r'[HR](\d{8})\d{4}', name)
    return digits.group(1) if digits else str(hdr.get('DATE-OBS', '')).replace('-', '')[:8]


def check_gain(frames, files, exptimes, read_noise, arm, logger=None):
    """Measure the gain from the flats of a master flat. Returns {card: (value, comment)} for the master flat header.

    frames: the bias-subtracted flats in electrons (n, ny, nx), as stacked; files and exptimes: one per frame;
    read_noise: in electrons; arm: 'Blu'/'Red' (or 'H'/'R'). Never raises.
    """
    cards = {}
    try:
        hdrs = [fits.getheader(f) for f in files]
        ga = np.array([float(h.get('AVG_GAIN', np.nan)) for h in hdrs])
        used = [str(h.get('GAINUSED', h.get('AVG_GAIN', ''))).split() for h in hdrs]
        namps = int(str(hdrs[0].get('NAMPS', 1)).strip() or 1)
        flip = str(arm).strip().upper()[:1] == 'R'
        ny = frames.shape[1]
        if namps > 1:
            regions = [_section(hdrs[0].get(f'DATASEC{i}'), ny, flip) for i in range(1, namps + 1)]
            if any(r is None for r in regions):
                cards['GAINFLAG'] = ('SKIPPED: no DATASECn', 'Gain check of the flats')
                return cards
        else:
            regions = [(slice(0, frames.shape[1]), slice(0, frames.shape[2]))]
        groups = {}
        for i, (f, h) in enumerate(zip(files, hdrs)):
            groups.setdefault((_night(f, h), round(float(exptimes[i]), 2), ' '.join(used[i])), []).append(i)
        sets = {key: idx for key, idx in groups.items() if len(idx) >= MIN_FRAMES}
        if not sets:
            cards['GAINFLAG'] = (f'NO SET: <{MIN_FRAMES} flats per exptime', 'Gain check of the flats')
            return cards
        rn2 = float(read_noise) ** 2
        results, reasons, nfr = [[] for _ in regions], [], 0
        for (night, et, _), idx in sorted(sets.items()):
            g_a = [float(u) for u in used[idx[0]]] if len(used[idx[0]]) >= namps else [ga[idx[0]]] * namps
            nfr += len(idx)
            for a, (rows, cols) in enumerate(regions):
                r = _set_gain(frames, idx, rows, cols, g_a[a], rn2)
                r.update(night=night, exptime=et, n=len(idx), applied=g_a[a])
                if 'gain' in r and r['ill'] >= ILL_LIMIT:
                    r['reason'] = f"illumination {100 * r['ill']:.2f} %"
                if 'reason' in r:
                    reasons.append(f"{night} {et:g}s: {r['reason']}")
                else:
                    results[a].append(r)
                if logger:
                    if 'reason' in r:
                        logger.info("Gain check %s %s %g s x%d%s: not used (%s)", arm, night, et, len(idx),
                                    f' amp {a + 1}' if namps > 1 else '', r['reason'])
                    else:
                        logger.info("Gain check %s %s %g s x%d%s: %.3f +- %.3f e-/ADU (applied %.3f), illumination %.2f %%",
                                    arm, night, et, len(idx), f' amp {a + 1}' if namps > 1 else '', r['gain'], r['err'],
                                    g_a[a], 100 * r['ill'])
        cards['GAINNSET'] = (len(sets), 'Flat sets in the gain check')
        cards['GAINNFR'] = (nfr, 'Flats in the gain check')
        if not any(results):
            cards['GAINFLAG'] = (('REJECTED: ' + reasons[0])[:60], 'Gain check of the flats')
            return cards
        flag, worst, worst_sig = 'OK', 0.0, 0.0
        for a, res in enumerate(results):
            sfx = str(a + 1) if namps > 1 else 'MEAS'
            if not res:
                cards[f'GAIN{sfx}'] = ('', 'Measured gain (e- per ADU): no usable set')
                flag = 'PARTIAL'
                continue
            g = np.array([r['gain'] for r in res]); e = np.array([r['err'] for r in res])
            w = 1.0 / np.maximum(e, 1e-6) ** 2
            gm = float(np.sum(w * g) / np.sum(w))
            em = float(max(1.0 / np.sqrt(np.sum(w)), (np.std(g, ddof=1) / np.sqrt(len(g))) if len(g) > 1 else 0.0))
            applied = float(np.mean([r['applied'] for r in res]))
            dev = gm / applied - 1.0
            cards[f'GAIN{sfx}'] = (round(gm, 4), 'Measured gain from the flats (e- per ADU)')
            cards[f'GAINERR{"" if namps == 1 else a + 1}'] = (round(em, 4), 'Uncertainty of the measured gain')
            cards[f'GAINDEV{"" if namps == 1 else a + 1}'] = (round(100 * dev, 2), 'Measured vs applied gain (%)')
            cards[f'GAINILL{"" if namps == 1 else a + 1}'] = (round(100 * float(np.median([r['ill'] for r in res])), 3),
                                                             'Frame-to-frame illumination term (%)')
            sig = abs(dev) / max(em / applied, 1e-6)
            if abs(dev) > DEV_LIMIT and sig > DEV_SIGMA and abs(dev) > abs(worst):
                worst, worst_sig = dev, sig
        if worst:
            flag = 'DEVIATES'
            if logger:
                logger.warning("Gain check %s: the flats give a gain %+.1f %% (%.1f sigma) from the applied one (limit %.0f %% "
                               "and %.0f sigma); gain_table.csv may need a new row", arm, 100 * worst, worst_sig,
                               100 * DEV_LIMIT, DEV_SIGMA)
        cards['GAINFLAG'] = (flag, 'Gain check of the flats')
    except Exception as e:                     # a quality check must never stop the reduction
        cards = {'GAINFLAG': (f'FAILED: {type(e).__name__}'[:60], 'Gain check of the flats')}
        if logger:
            logger.warning("Gain check of the flats failed (%s: %s); master flat made without it", type(e).__name__, e)
    return cards
