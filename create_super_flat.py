import logging
import os
import glob
import shutil
import traceback
from astropy.io import fits
import hrsreduce

from hrsreduce.master_flat.super_flat import SuperFlat

"""
Super Flat date ranges from the order-position measurement using HR flats 2014 to 
June 2026 with a0.25 px jump threshold.

Each range covers one stretch where the orders did not move by more than 0.25 px
(centre shift, or tilt from the left to the right of the CCD). Red and Blue move on
different dates, so each arm has its own list. A range may run over 1 January: its
flats from both years go into one Super Flat, which is then copied into the
Super_Flats folder of each later year as <mode>_Super_Flat_<arm>YYYY0101.fits, so it
can be found for the data of either year.

Ranges with more than MAX_FLATS (30) HR flat frames are split, in time order and at
whole nights, into equal parts of at most 30 frames, and each part is its own Super
Flat. Splitting into equal parts (rather than 30 + 30 + a few left over) keeps every
part close to equal (up to 30 frames), and gives more frequent Super Flats where there are
many flats. Ranges with 30 frames or fewer are made as one Super Flat. A range with fewer
than 10 frames was merged into the neighbouring range with the smaller jump (noted
under that range's heading), accepting that jump inside the merged Super Flat. 

Each entry is (start, end, copies, nights to leave out):
- start: first night, and the date in the file name. It is the day before the first
  flat of the part where possible, because reduce.py only uses a Super Flat dated
  strictly before the night it reduces.
- end: last night; the next entry starts the day after.
- copies: dates of the copies (1 January of each later year the entry covers).
- nights to leave out: their flats did not look like a normal flat (another lamp or
  fibre set-up, dark, or a one-off order position).
The comment on each entry gives its number of flat frames and nights; the line above
each group gives the range of unchanged order positions and the jump at its start.

This uses hrsreduce/master_flat/super_flat.py, which takes
each night's biases from its own year and data folder, accepts the wider older Blue
frames, leaves out the listed nights, and combines at most MAX_FLATS frames (spread
through the entry, should an entry hold more than its catalogue count).

Before running
- Old Super Flats are never overwritten. Where a new file or a copy has the name of an
  old one, the old one is moved into Super_Flats/old/ first. Old files with other
  dates would still be picked by reduce.py (it takes the closest earlier date); set
  MOVE_OLD = True to move all old Super Flats of these modes into Super_Flats/old/
  (reduce.py does not look in sub-folders). Nothing is deleted.
- Files made by this script carry SF_START and SF_END in the header, so a rerun
  replaces them instead of moving them aside.
- The dates were measured on HR flats only. The instrument moves as a whole, so the
  same dates very likely apply to MR, LR and HS, but that has not been checked.
- Blue ranges before 2015-04-26 are commented out: L0 and SubtractBias cannot yet
  process those frames (see the note under each).
"""

arms = ['Red', 'Blu']
modes = ['MR','LR', 'HS']
years = ['2016','2017','2018',2019','2020','2021','2022','2023','2024','2025','2026']          # e.g. ['2018', '2019'] to make only the entries starting in those years
MOVE_OLD = False      # True: move ALL old Super Flats of these modes and arms into Super_Flats/old/ first,
                      # whatever `years` is set to. On first run set this to True.
MAX_FLATS = 30        # largest number of flat frames combined into one Super Flat

ROOTS = ['/Volumes/SALT_DATA/', '/Users/daniel/Desktop/SALT_HRS_DATA/']


def base_dir(yyyymmdd):
    """Raw data up to 2022 are on the external drive, from 2023 on the internal disk."""
    return ROOTS[0] if yyyymmdd[:4] <= '2022' else ROOTS[1]


RANGES = {
    'Red': [
        # -- 20150101 to 20150623: start of the data; split into 2 parts
        ('20150101', '20150321', [], ['20150109', '20150112', '20150204', '20150209']),   # 26 frames, 9 nights
        ('20150322', '20150623', [], []),   # 29 frames, 8 nights
        # -- 20150624 to 20160313: after a jump of -3.75 px shift, -2.69 px tilt
        ('20150624', '20160313', ['20160101'], ['20150809', '20150921', '20160107']),   # 30 frames, 10 nights
        # -- 20160314 to 20160919: after a jump of -0.53 px shift, -0.03 px tilt; split into 2 parts
        ('20160314', '20160520', [], ['20160329']),   # 21 frames, 7 nights
        ('20160521', '20160919', [], ['20160604', '20160911']),   # 24 frames, 8 nights
        # -- 20160920 to 20171211: after a jump of -0.09 px shift, +0.47 px tilt; split into 10 parts
        ('20160920', '20161119', [], []),   # 27 frames, 9 nights
        ('20161120', '20161129', [], ['20161127']),   # 27 frames, 8 nights
        ('20161130', '20161220', [], []),   # 28 frames, 19 nights
        ('20161221', '20170104', ['20170101'], []),   # 28 frames, 14 nights
        ('20170105', '20170113', [], []),   # 27 frames, 9 nights
        ('20170114', '20170304', [], []),   # 28 frames, 10 nights
        ('20170305', '20170514', [], []),   # 27 frames, 8 nights
        ('20170515', '20170728', [], []),   # 29 frames, 10 nights
        ('20170729', '20171010', [], []),   # 27 frames, 9 nights
        ('20171011', '20171211', [], []),   # 27 frames, 9 nights
        # -- 20171212 to 20180825: after a jump of +0.29 px shift, -0.03 px tilt; split into 4 parts
        ('20171212', '20180207', ['20180101'], []),   # 21 frames, 7 nights
        ('20180208', '20180520', [], []),   # 22 frames, 7 nights
        ('20180521', '20180628', [], ['20180521']),   # 21 frames, 7 nights
        ('20180629', '20180825', [], []),   # 21 frames, 7 nights
        # -- 20180826 to 20181111: after a jump of +1.43 px shift, +15.46 px tilt
        ('20180826', '20181111', [], ['20181015', '20181018', '20181028']),   # 18 frames, 6 nights
        # -- 20181112 to 20190731: after a jump of -5.28 px shift, -3.06 px tilt; split into 4 parts
        ('20181112', '20190107', ['20190101'], []),   # 27 frames, 9 nights
        ('20190108', '20190305', [], []),   # 24 frames, 8 nights
        ('20190306', '20190511', [], []),   # 28 frames, 9 nights
        ('20190512', '20190731', [], ['20190621', '20190627', '20190701', '20190704']),   # 25 frames, 9 nights
        # -- 20190801 to 20200316: after a jump of +0.25 px shift, +0.30 px tilt; split into 3 parts
        ('20190801', '20190928', [], []),   # 27 frames, 9 nights
        ('20190929', '20191202', [], ['20191005']),   # 30 frames, 10 nights
        ('20191203', '20200316', ['20200101'], ['20200115']),   # 27 frames, 9 nights
        # -- 20200317 to 20200617: after a jump of -1.77 px shift, -0.92 px tilt
        ('20200317', '20200617', [], ['20200509']),   # 18 frames, 6 nights
        # -- 20200618 to 20210219: after a jump of +0.41 px shift, +0.23 px tilt; split into 4 parts
        ('20200618', '20200816', [], []),   # 27 frames, 9 nights
        ('20200817', '20201014', [], []),   # 26 frames, 8 nights
        ('20201015', '20201217', [], []),   # 27 frames, 9 nights
        ('20201218', '20210219', ['20210101'], []),   # 27 frames, 9 nights
        # -- 20210220 to 20210622: after a jump of +0.08 px shift, +0.28 px tilt; split into 2 parts
        ('20210220', '20210413', [], []),   # 24 frames, 8 nights
        ('20210414', '20210622', [], []),   # 24 frames, 8 nights
        # -- 20210623 to 20211018: after a jump of +0.06 px shift, -0.30 px tilt; split into 2 parts
        ('20210623', '20210812', [], []),   # 21 frames, 7 nights
        ('20210813', '20211018', [], ['20211002']),   # 24 frames, 8 nights
        # -- 20211019 to 20211118: after a jump of -0.65 px shift, -0.50 px tilt
        ('20211019', '20211118', [], []),   # 12 frames, 4 nights
        # -- 20211119 to 20220208: after a jump of +0.30 px shift, +0.09 px tilt; split into 2 parts
        ('20211119', '20211224', [], []),   # 15 frames, 5 nights
        ('20211225', '20220208', ['20220101'], []),   # 16 frames, 5 nights
        # -- 20220209 to 20220330: after a jump of -1.38 px shift, +0.97 px tilt
        ('20220209', '20220330', [], []),   # 21 frames, 7 nights
        # -- 20220331 to 20220612: after a jump of +0.27 px shift, +0.06 px tilt; split into 2 parts
        ('20220331', '20220506', [], []),   # 18 frames, 6 nights
        ('20220507', '20220612', [], []),   # 18 frames, 5 nights
        # -- 20220613 to 20230423: after a jump of +0.66 px shift, +0.24 px tilt; split into 5 parts
        ('20220613', '20220824', [], []),   # 29 frames, 9 nights
        ('20220825', '20221010', [], []),   # 27 frames, 7 nights
        ('20221011', '20221224', [], ['20221030']),   # 30 frames, 10 nights
        ('20221225', '20230216', ['20230101'], []),   # 27 frames, 9 nights
        ('20230217', '20230423', [], []),   # 30 frames, 10 nights
        # -- 20230424 to 20230720: after a jump of -1.06 px shift, -0.60 px tilt
        ('20230424', '20230720', [], []),   # 27 frames, 9 nights
        # -- 20230721 to 20231221: after a jump of +0.43 px shift, +0.22 px tilt; split into 3 parts
        ('20230721', '20230908', [], []),   # 21 frames, 7 nights
        ('20230909', '20231103', [], []),   # 24 frames, 8 nights
        ('20231104', '20231221', [], []),   # 21 frames, 7 nights
        # -- 20231222 to 20240821: after a jump of +0.38 px shift, +0.11 px tilt; split into 4 parts
        ('20231222', '20240213', ['20240101'], []),   # 24 frames, 8 nights
        ('20240214', '20240424', [], []),   # 24 frames, 8 nights
        ('20240425', '20240624', [], []),   # 24 frames, 8 nights
        ('20240625', '20240821', [], []),   # 24 frames, 8 nights
        # -- 20240822 to 20250624: after a jump of +2.08 px shift, +1.39 px tilt; split into 4 parts
        ('20240822', '20241025', [], []),   # 27 frames, 9 nights
        ('20241026', '20241227', [], []),   # 24 frames, 8 nights
        ('20241228', '20250227', ['20250101'], []),   # 27 frames, 9 nights
        ('20250228', '20250624', [], ['20250513']),   # 27 frames, 9 nights
        # -- 20250625 to 20260427: after a jump of -0.75 px shift, -0.37 px tilt; split into 5 parts
        #    merged in, as it had fewer than 10 frames: includes 20250625-20250708 (6 frames), whose orders differ from the rest by 0.40 px
        ('20250625', '20250825', [], []),   # 27 frames, 9 nights
        ('20250826', '20250909', [], []),   # 26 frames, 3 nights
        ('20250910', '20251112', [], []),   # 27 frames, 9 nights
        ('20251113', '20260113', ['20260101'], ['20251220']),   # 27 frames, 9 nights
        ('20260114', '20260427', [], ['20260323', '20260404', '20260409', '20260413', '20260417', '20260421', '20260423']),   # 27 frames, 9 nights
        # -- 20260428 to 20260617: after a jump of +0.37 px shift, +0.03 px tilt
        ('20260428', '20260617', [], ['20260508']),   # 27 frames, 8 nights
    ],
    'Blu': [
        # -- 20140101 to 20140818: start of the data
        # ('20140101', '20140818', [], []),   # 3 frames, 1 night. Not made: Blue headers before December 2014 have no NAMPS, so L0 cannot read them
        # -- 20140819 to 20141020: after a jump of -0.11 px shift, +0.31 px tilt
        # ('20140819', '20141020', [], ['20140922']),   # 10 frames, 2 nights. Not made: Blue headers before December 2014 have no NAMPS, so L0 cannot read them
        # -- 20141021 to 20150202: after a jump of -6.95 px shift, -3.04 px tilt
        # ('20141021', '20150202', ['20150101'], ['20141029', '20141216', '20141218', '20141222', '20150109']),   # 18 frames, 8 nights. Not made: two-amplifier frames: SubtractBias cannot read their two OS_MEAN values
        # -- 20150203 to 20150219: after a jump of -0.21 px shift, -0.29 px tilt
        # ('20150203', '20150219', [], []),   # 15 frames, 4 nights. Not made: two-amplifier frames: SubtractBias cannot read their two OS_MEAN values
        # -- 20150220 to 20150425: after a jump of +0.19 px shift, +0.28 px tilt
        # ('20150220', '20150425', [], []),   # 29 frames, 9 nights. Not made: two-amplifier frames: SubtractBias cannot read their two OS_MEAN values
        # -- 20150426 to 20150623: readout changes from two amplifiers (mirrored) to one
        ('20150426', '20150623', [], []),   # 15 frames, 4 nights
        # -- 20150624 to 20160313: after a jump of +0.39 px shift, +0.00 px tilt; split into 2 parts
        ('20150624', '20151210', [], ['20150809', '20150921']),   # 29 frames, 9 nights
        ('20151211', '20160313', ['20160101'], ['20160107']),   # 29 frames, 9 nights
        # -- 20160314 to 20161012: after a jump of -0.41 px shift, -0.09 px tilt; split into 2 parts
        ('20160314', '20160618', [], ['20160329', '20160604']),   # 26 frames, 9 nights
        ('20160619', '20161012', [], []),   # 27 frames, 9 nights
        # -- 20161013 to 20170312: after a jump of +0.29 px shift, -0.01 px tilt; split into 6 parts
        ('20161013', '20161121', [], []),   # 27 frames, 9 nights
        ('20161122', '20161203', [], []),   # 28 frames, 9 nights
        ('20161204', '20161223', [], []),   # 27 frames, 20 nights
        ('20161224', '20170105', ['20170101'], []),   # 28 frames, 12 nights
        ('20170106', '20170114', [], []),   # 27 frames, 9 nights
        ('20170115', '20170312', [], []),   # 28 frames, 10 nights
        # -- 20170313 to 20180706: after a jump of -0.54 px shift, -0.09 px tilt; split into 6 parts
        ('20170313', '20170529', [], []),   # 30 frames, 9 nights
        ('20170530', '20170821', [], []),   # 29 frames, 10 nights
        ('20170822', '20171030', [], []),   # 27 frames, 9 nights
        ('20171031', '20180116', ['20180101'], []),   # 30 frames, 10 nights
        ('20180117', '20180512', [], []),   # 28 frames, 9 nights
        ('20180513', '20180706', [], []),   # 30 frames, 10 nights
        # -- 20180707 to 20180825: after a jump of +0.28 px shift, -0.13 px tilt
        ('20180707', '20180825', [], []),   # 17 frames, 6 nights
        # -- 20180826 to 20181111: after a jump of -0.13 px shift, +6.10 px tilt
        ('20180826', '20181111', [], []),   # 27 frames, 9 nights
        # -- 20181112 to 20200316: after a jump of -24.24 px shift, +1.26 px tilt; split into 7 parts
        ('20181112', '20190116', ['20190101'], []),   # 30 frames, 10 nights
        ('20190117', '20190319', [], []),   # 28 frames, 9 nights
        ('20190320', '20190523', [], []),   # 27 frames, 9 nights
        ('20190524', '20190725', [], []),   # 29 frames, 10 nights
        ('20190726', '20190928', [], []),   # 30 frames, 10 nights
        ('20190929', '20191202', [], ['20191005', '20191123']),   # 27 frames, 9 nights
        ('20191203', '20200316', ['20200101'], []),   # 30 frames, 10 nights
        # -- 20200317 to 20200617: after a jump of +1.67 px shift, +3.00 px tilt
        ('20200317', '20200617', [], []),   # 21 frames, 7 nights
        # -- 20200618 to 20200830: after a jump of -0.42 px shift, +0.16 px tilt
        ('20200618', '20200830', [], []),   # 30 frames, 10 nights
        # -- 20200831 to 20201209: after a jump of -1.77 px shift, -3.06 px tilt; split into 2 parts
        ('20200831', '20201020', [], []),   # 23 frames, 7 nights
        ('20201021', '20201209', [], []),   # 21 frames, 7 nights
        # -- 20201210 to 20210108: after a jump of +0.34 px shift, +0.05 px tilt
        ('20201210', '20210108', ['20210101'], []),   # 12 frames, 4 nights
        # -- 20210109 to 20210211: after a jump of -0.31 px shift, -0.01 px tilt
        ('20210109', '20210211', [], []),   # 15 frames, 5 nights
        # -- 20210212 to 20211018: after a jump of +0.67 px shift, -0.21 px tilt; split into 4 parts
        ('20210212', '20210409', [], []),   # 24 frames, 8 nights
        ('20210410', '20210619', [], []),   # 24 frames, 8 nights
        ('20210620', '20210812', [], []),   # 24 frames, 8 nights
        ('20210813', '20211018', [], ['20211002']),   # 24 frames, 8 nights
        # -- 20211019 to 20220208: after a jump of +0.26 px shift, -0.15 px tilt; split into 2 parts
        ('20211019', '20211209', [], []),   # 21 frames, 7 nights
        ('20211210', '20220208', ['20220101'], []),   # 22 frames, 7 nights
        # -- 20220209 to 20230121: after a jump of +1.32 px shift, -7.64 px tilt; split into 5 parts
        ('20220209', '20220421', [], []),   # 30 frames, 10 nights
        ('20220422', '20220618', [], []),   # 30 frames, 9 nights
        ('20220619', '20220916', [], ['20220729', '20220904']),   # 27 frames, 9 nights
        ('20220917', '20221119', [], []),   # 30 frames, 10 nights
        ('20221120', '20230121', ['20230101'], []),   # 30 frames, 10 nights
        # -- 20230122 to 20230423: after a jump of +0.27 px shift, -0.03 px tilt; split into 2 parts
        ('20230122', '20230226', [], []),   # 18 frames, 6 nights
        ('20230227', '20230423', [], ['20230326']),   # 21 frames, 7 nights
        # -- 20230424 to 20240821: after a jump of +0.33 px shift, -0.21 px tilt; split into 7 parts
        ('20230424', '20230720', [], []),   # 27 frames, 9 nights
        ('20230721', '20230924', [], []),   # 27 frames, 9 nights
        ('20230925', '20231121', [], []),   # 27 frames, 9 nights
        ('20231122', '20240125', ['20240101'], []),   # 27 frames, 9 nights
        ('20240126', '20240407', [], []),   # 27 frames, 9 nights
        ('20240408', '20240615', [], []),   # 27 frames, 9 nights
        ('20240616', '20240821', [], []),   # 27 frames, 9 nights
        # -- 20240822 to 20250825: after a jump of -0.39 px shift, -0.27 px tilt; split into 5 parts
        ('20240822', '20241025', [], []),   # 27 frames, 9 nights
        ('20241026', '20250102', ['20250101'], []),   # 27 frames, 9 nights
        ('20250103', '20250227', [], []),   # 24 frames, 8 nights
        ('20250228', '20250624', [], ['20250513']),   # 27 frames, 9 nights
        ('20250625', '20250825', [], []),   # 27 frames, 9 nights
        # -- 20250826 to 20251002: after a jump of -0.08 px shift, -0.45 px tilt; split into 2 parts
        ('20250826', '20250902', [], []),   # 23 frames, 2 nights
        ('20250903', '20251002', [], []),   # 12 frames, 4 nights
        # -- 20251003 to 20251218: after a jump of +0.28 px shift, +0.26 px tilt; split into 2 parts
        ('20251003', '20251112', [], []),   # 18 frames, 6 nights
        ('20251113', '20251218', [], []),   # 18 frames, 6 nights
        # -- 20251219 to 20260218: after a jump of -0.40 px shift, -0.08 px tilt
        #    merged in, as it had fewer than 10 frames: includes 20260127-20260218 (9 frames) after a jump of +0.28 px shift, +0.01 px tilt
        ('20251219', '20260218', ['20260101'], []),   # 27 frames, 9 nights
        # -- 20260219 to 20260402: after a jump of -0.94 px shift, -0.09 px tilt
        ('20260219', '20260402', [], []),   # 15 frames, 5 nights
        # -- 20260403 to 20260617: after a jump of +0.35 px shift, +0.01 px tilt; split into 2 parts
        #    merged in, as it had fewer than 10 frames: includes 20260613-20260617 (3 frames) after a jump of +0.31 px shift, -0.04 px tilt
        ('20260403', '20260427', [], []),   # 30 frames, 6 nights
        ('20260428', '20260617', [], []),   # 30 frames, 9 nights
    ],
}


def sf_name(arm, mode, yyyymmdd):
    return os.path.join(base_dir(yyyymmdd), arm, yyyymmdd[:4], 'Super_Flats',
                        f'{mode}_Super_Flat_{"R" if arm == "Red" else "H"}{yyyymmdd}.fits')


def made_here(path):
    try:
        return 'SF_START' in fits.getheader(path)
    except Exception:
        return False


def move_aside(path):
    """Move an old Super Flat into Super_Flats/old/ (never deletes)."""
    old_dir = os.path.join(os.path.dirname(path), 'old')
    os.makedirs(old_dir, exist_ok=True)
    dest = os.path.join(old_dir, os.path.basename(path))
    n = 1
    while os.path.exists(dest):
        dest = os.path.join(old_dir, os.path.basename(path).replace('.fits', f'_{n}.fits'))
        n += 1
    shutil.move(path, dest)
    print(f'Moved old {path} -> {dest}')


def clear_name(path):
    """Make room for a new file: an old file of that name is moved aside, one made here is replaced."""
    if os.path.exists(path) and not made_here(path):
        move_aside(path)


failed = []
for mode in modes:
    for arm in arms:
        if MOVE_OLD:
            for root in ROOTS:
                for f in sorted(glob.glob(f'{root}{arm}/????/Super_Flats/{mode}_Super_Flat_*.fits')):
                    if not made_here(f):
                        move_aside(f)
        for start, end, copies, exclude in RANGES[arm]:
            if years and start[:4] not in years:
                continue
            try:
                clear_name(sf_name(arm, mode, start))
                master = SuperFlat(start[:4], base_dir(start), arm, mode, start, end, roots=ROOTS,
                                   exclude_nights=exclude, max_flats=MAX_FLATS).create_superflat()
                if master is None:
                    failed.append((mode, arm, start, end, 'no usable flats'))
                    continue
                for c in copies:
                    dest = sf_name(arm, mode, c)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    clear_name(dest)
                    shutil.copy2(master, dest)
                    print(f'Copied {master} -> {dest}')
            except Exception:
                # Keep going: one range that fails should not stop the others
                traceback.print_exc()
                failed.append((mode, arm, start, end, 'error'))

if failed:
    print('\nRanges with no Super Flat made:')
    for f in failed:
        print('  ', *f)
