"""
Generic utility functions
"""

from datetime import timedelta


def totim_to_datetime(totim, start='1-1-1970', timeunit='D'):
    """

    Parameters
    ----------
    totim : list or numpy array

    start : str
        Starting date for simulation. (default is 1-1-1970).
    timeunit : string
        time unit of the simulation time. Valid values are 'S'econds,
        'M'inutes, 'H'ours, 'D'ays, 'Y'ears. (default is 'D').

    Returns
    -------
    out : list
        datetime object calculated from start and totim values

    """
    key = None
    fact = 1.
    if timeunit.upper() == 'S':
        key = 'seconds'
    elif timeunit.upper() == 'M':
        key = 'minutes'
    elif timeunit.upper() == 'H':
        key = 'hours'
    elif timeunit.upper() == 'D':
        key = 'days'
    elif timeunit.upper() == 'Y':
        key = 'days'
        fact = 365.25
    else:
        err = "'S'econds, 'M'inutes, 'H'ours, 'D'ays, 'Y'ears are the " + \
              "only timeunit values that can be passed to totim_" + \
              "to_datetime() function"
        raise Exception(err)
    out = []
    kwargs = {}
    for to in totim:
        kwargs[key] = to * fact
        t = timedelta(**kwargs)
        out.append(start + t)
    return out
