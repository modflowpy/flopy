import numpy as np

def get_transmissivities(heads, m,
                         r=None, c=None, x=None, y=None,
                         sctop=None, scbot=None, nodata=-999):
    """Computes a transmissivities in each model layer at specified locations and open intervals.
    A saturated thickness is determined for each row, column or x, y location supplied,
    based on the open interval (sctop, scbot), if supplied, otherwise the layer tops and bottoms
    and the water table are used.

    Parameters
    ----------
    heads : 2D array
        numpy array of shape nlay by n locations
    m : flopy.modflow.Modflow object
        Must have dis, sr, and lpf or upw packages.
    r : 1D array-like of ints, of length n locations
        row indices (optional; alternately specify x, y)
    c : 1D array-like of ints, of length n locations
        column indicies (optional; alternately specify x, y)
    x : 1D array-like of floats, of length n locations
        x locations in real world coordinates (optional)
    y : 1D array-like of floats, of length n locations
        y locations in real world coordinates (optional)
    sctop : 1D array-like of floats, of length n locations
        open interval tops (optional; default is model top)
    scbot : 1D array-like of floats, of length n locations
        open interval bottoms (optional; default is model bottom)
    nodata : numeric
        optional; locations where heads=nodata will be assigned T=0

    Returns
    -------
    T : 2D array of same shape as heads (nlay x n locations)
        Transmissivities in each layer at each location
    """
    if r is not None and c is not None:
        pass
    elif x is not None and y is not None:
        # get row, col for observation locations
        r, c = m.sr.get_rc(x, y)
    else:
        raise ValueError('Must specify row, column or x, y locations.')

    # get k-values and botms at those locations
    paklist = m.get_package_list()
    if 'LPF' in paklist:
        hk = m.lpf.hk.array[:, r, c]
    elif 'UPW' in paklist:
        hk = m.upw.hk.array[:, r, c]
    else:
        raise ValueError('No LPF or UPW package.')

    botm = m.dis.botm.array[:, r, c]
    assert heads.shape == botm.shape, 'Shape of heads array must be nlay x nhyd'

    # set open interval tops/bottoms to model top/bottom if None
    if sctop is None:
        sctop = m.dis.top.array[r, c]
    if scbot is None:
        scbot = m.dis.botm.array[-1, r, c]

    # make an array of layer tops
    tops = np.empty_like(botm, dtype=float)
    tops[0, :] = m.dis.top.array[r, c]
    tops[1:, :] = botm[:-1]

    # expand top and bottom arrays to be same shape as botm, thickness, etc.
    # (so we have an open interval value for each layer)
    sctoparr = np.zeros(botm.shape)
    sctoparr[:] = sctop
    scbotarr = np.zeros(botm.shape)
    scbotarr[:] = scbot

    # start with layer tops
    # set tops above heads to heads
    # set tops above screen top to screen top
    # (we only care about the saturated open interval)
    openinvtop = tops.copy()
    openinvtop[openinvtop > heads] = heads[openinvtop > heads]
    openinvtop[openinvtop > sctoparr] = sctoparr[openinvtop > sctop]

    # start with layer bottoms
    # set bottoms below screened interval to screened interval bottom
    # set screen bottoms below bottoms to layer bottoms
    openinvbotm = botm.copy()
    openinvbotm[openinvbotm < scbotarr] = scbotarr[openinvbotm < scbot]
    openinvbotm[scbotarr < botm] = botm[scbotarr < botm]

    # compute thickness of open interval in each layer
    thick = openinvtop - openinvbotm
    thick[thick < 0] = 0
    thick[heads == nodata] = 0  # exclude nodata cells

    # compute transmissivities
    T = thick * hk
    return T