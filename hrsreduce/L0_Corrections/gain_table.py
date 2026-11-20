"""Detector gains of HRS (e-/ADU) for L0Corrections, read from gain_table.csv next to this file.

The GAIN cards in the raw HRS headers are not the gains of the detectors: measured in 2026 from a photon-transfer
sequence and from the nightly flats, the blue gain is 2.11 and the red 2.95 e-/ADU, while the headers give 2.406 and
2.301. L0Corrections therefore takes the gains from gain_table.csv, and the header values only for a readout the
table does not cover.

gain_table.csv (lines starting with # are comments) has one row per year, arm, readout and amplifier:
    year      the year the row applies to
    start     MMDD from which the row applies in that year (0101: the whole year). A change during a year gets a
              second row for that year with the date of the change.
    arm       Blu or Red
    namps     number of amplifiers of the readout: 1, or 2 (blue) and 4 (red) for the multi-amplifier readouts
    amp       the amplifier, 1..namps, in the order of the header GAIN values and of DATASEC1, DATASEC2, ...
    rospeed   readout speed of the row, header ROSPEED in Hz (400000 for the HRS calibrations and science; some frames
              are read out at 1000000, with other header gains; * for any)
    binning   on-chip binning of the row, header CCDSUM as 1x1, 2x2, ... (* for any)
    gain      e-/ADU
    gain_err  its uncertainty
    status    measured (from that year's calibrations) or adopted (from the nearest measured year)
    n_nights  nights behind a measured value
    n_sets    flat sets behind a measured value
    note      free text

Lookup for a frame of night YYYYMMDD: the row of that year with the latest start on or before the night. A year
without rows takes the most recent earlier row (before the first year, the earliest row), so the last measured gain
carries on until the table says otherwise. A readout without any rows (another number of amplifiers, readout speed
or binning than those measured) keeps the header GAIN values. A frame without ROSPEED or CCDSUM is taken to be
400000 Hz and 1x1. Products made with a different gain are made again on the next run, because
their gain no longer matches (see hrsreduce.utils.provenance).
"""
import csv
import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gain_table.csv')
_ARMS = {'H': 'Blu', 'B': 'Blu', 'BLU': 'Blu', 'BLUE': 'Blu', 'R': 'Red', 'RED': 'Red'}


def arm_name(arm):
    """'Blu' or 'Red' for any of H, Blu, R, Red (any case)."""
    return _ARMS.get(str(arm).strip().upper(), str(arm).strip())


@lru_cache(maxsize=4)
def _read(path, mtime):
    with open(path) as fh:
        lines = [line for line in fh if line.strip() and not line.lstrip().startswith('#')]
    rows = []
    for r in csv.DictReader(lines):
        try:
            rows.append(dict(year=int(r['year']), start=int(str(r.get('start') or '0101').strip() or 101),
                             arm=arm_name(r['arm']), namps=int(r['namps']), amp=int(r['amp']),
                             rospeed=_speed(r.get('rospeed') or '*'), binning=_binning(r.get('binning') or '*'),
                             gain=float(r['gain']), status=str(r.get('status') or '').strip()))
        except (KeyError, TypeError, ValueError):
            logger.warning("gain_table.csv: row %s cannot be read; it is not used", dict(r))
    return rows


def _speed(v):
    """readout speed as an integer number of Hz ('*' stays '*')"""
    v = str(v).strip()
    return '*' if v in ('*', '') else int(round(float(v)))


def _binning(v):
    """'1 1', '1x1' -> '1x1' ('*' stays '*')"""
    v = str(v).strip().lower().replace('x', ' ')
    return '*' if v in ('*', '') else 'x'.join(v.split())


def read_table(path=TABLE):
    """The rows of the gain table ([] when the file is missing or cannot be read)."""
    try:
        return _read(path, os.path.getmtime(path))
    except OSError as e:
        logger.warning("Gain table %s cannot be read (%s); the header GAIN values are used", path, e)
        return []


def header_gains(hdr):
    """The GAIN values of a raw header, one per amplifier ([] when missing or unreadable)."""
    try:
        return [float(v) for v in str(hdr.get('GAIN', '')).split()]
    except ValueError:
        return []


def gains_for(hdr, arm, night, namps=1, path=TABLE):
    """Gains (e-/ADU) of the `namps` amplifiers of a raw frame of `arm` taken on `night` (YYYYMMDD), and their source.

    Returns (gains, source): gains is a list of length namps; source is a short text for the GAINSRC card, e.g.
    'table 2026 measured', 'table 2026 (no 2027 row)' or 'header (no table row for NAMPS=4)'.
    """
    namps = int(namps)
    arm = arm_name(arm)
    try:
        speed = _speed(hdr.get('ROSPEED') or 400000)
    except ValueError:
        speed = 400000
    binning = _binning(hdr.get('CCDSUM') or '1 1')
    hg = header_gains(hdr)
    rows = [r for r in read_table(path) if r['arm'] == arm and r['namps'] == namps
            and r['rospeed'] in ('*', speed) and r['binning'] in ('*', binning)]
    try:
        when = int(str(night)[:8])
    except ValueError:
        when = None
    if rows and when is not None:
        picked = []
        for amp in range(1, namps + 1):
            ra = [r for r in rows if r['amp'] == amp]
            if not ra:
                break
            key = lambda r: r['year'] * 10000 + r['start']
            before = [r for r in ra if key(r) <= when]
            picked.append(max(before, key=key) if before else min(ra, key=key))
        if len(picked) == namps:
            year = when // 10000
            r0 = picked[0]
            source = f"table {r0['year']} {r0['status']}".strip() if r0['year'] == year else f"table {r0['year']} (no {year} row)"
            return [r['gain'] for r in picked], source
    if len(hg) >= namps:
        why = f'no table row for {arm} NAMPS={namps} {speed / 1e3:g} kHz {binning}'
        return hg[:namps], f'header ({why})'
    logger.warning("No gain for %s NAMPS=%d on %s in the table or the header; a gain of 1 is used", arm, namps, night)
    return [1.0] * namps, 'none (no table row, no header GAIN)'


def gains_card(gains):
    """The GAINUSED card value: the applied gains, one per amplifier."""
    return ' '.join('%.4f' % g for g in gains)
