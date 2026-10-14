"""
Remove the slash from FITS header comments.

Some FITS viewers cut a header card at every slash, so a comment such as "CCD gain (photons/ADU)" is shown
truncated or garbled.  The raw SALT headers carry three such comments (GAIN, NODCOUNT, PRESCAN) and they are
propagated into every file the pipeline writes.  ``clean_header_comments`` rewords the known cases and
replaces any other slash in a comment by a hyphen; values are never touched.
"""
from astropy.io import fits

# known phrases, replaced before the generic hyphen
REPLACEMENTS = (
    ('photons/ADU', 'photons per ADU'),
    ('e/ADU', 'e per ADU'),
    ('Nod/Shuffles', 'Nod and Shuffles'),
    ('start/end', 'start and end'),
    ('km/s', 'km s^-1'),
    ('m/s', 'm s^-1'),
)


def clean_comment(comment):
    """Return the comment with every slash removed (known phrases reworded, the rest hyphenated)."""
    for old, new in REPLACEMENTS:
        comment = comment.replace(old, new)
    return comment.replace('/', '-')


def clean_header_comments(header):
    """
    Reword, in place, every comment of a header that contains a slash.

    Parameters
    ----------
    header : astropy.io.fits.Header
        The header to clean; its values are left as they are.

    Returns
    -------
    int
        Number of comments changed.
    """
    changed = 0
    for card in header.cards:
        if '/' in card.comment:
            card.comment = clean_comment(card.comment)
            changed += 1
    return changed


def clean_hdulist_comments(hdul):
    """Apply ``clean_header_comments`` to every HDU of an HDUList; returns the number of comments changed."""
    return sum(clean_header_comments(hdu.header) for hdu in hdul)


def clean_comments(obj):
    """Clean the comments of a Header, of a single HDU or of every HDU of an HDUList; returns the number changed."""
    if isinstance(obj, fits.Header):
        return clean_header_comments(obj)
    if hasattr(obj, 'header'):
        return clean_header_comments(obj.header)
    return sum(clean_header_comments(hdu.header) for hdu in obj)
