"""Date input handling."""

from __future__ import annotations

import datetime as dt


def to_date(value) -> dt.date:
    """Accept datetime.date, datetime.datetime, 'YYYY-MM-DD' or 'YYYYMMDD'."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        s = value.strip()
        for fmt in ('%Y-%m-%d', '%Y%m%d'):
            try:
                return dt.datetime.strptime(s, fmt).date()
            except ValueError:
                pass
    raise ValueError(f'Unrecognized date {value!r}; use YYYY-MM-DD, YYYYMMDD '
                     'or datetime.date')


def normalize_dates(dates=None, start=None, end=None) -> list[dt.date]:
    """Sorted, de-duplicated list of dates.

    Either `dates` (one date or an iterable of dates), or `start` and `end`
    (both INCLUSIVE). Not both.
    """
    if dates is not None and (start is not None or end is not None):
        raise ValueError('Pass either dates or start/end, not both')
    if dates is not None:
        items = [dates] if isinstance(dates, (str, dt.date)) else list(dates)
        out = sorted({to_date(d) for d in items})
    elif start is not None and end is not None:
        s, e = to_date(start), to_date(end)
        if e < s:
            raise ValueError(f'end {e} is before start {s}')
        out = [s + dt.timedelta(days=i) for i in range((e - s).days + 1)]
    else:
        raise ValueError('Pass dates, or both start and end')
    if not out:
        raise ValueError('No dates given')
    return out


def label(d: dt.date) -> str:
    """Date label used by every collection's system:index: 'YYYYMMDD'."""
    return d.strftime('%Y%m%d')
