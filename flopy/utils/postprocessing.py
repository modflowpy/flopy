import numpy as np


def get_transmissivities(heads, m,
                         r=None, c=None, x=None, y=None,
                         sctop=None, scbot=None, nodata=-999):
    """
    Computes transmissivity in each model layer at specified locations and
    open intervals. A saturated thickness is determined for each row, column
    or x, y location supplied, based on the open interval (sctop, scbot),
    if supplied, otherwise the layer tops and bottoms and the water table
    are used.

    Parameters
    ----------
    heads : 2D array OR 3D array
        numpy array of shape nlay by n locations (2D) OR complete heads array
        of the model for one time (3D)
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

    if heads.shape == (m.nlay, m.nrow, m.ncol):
        heads = heads[:, r, c]

    msg = 'Shape of heads array must be nlay x nhyd'
    assert heads.shape == botm.shape, msg

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

    # assign open intervals above or below model to closest cell in column
    not_in_layer = np.sum(thick < 0, axis=0)
    not_in_any_layer = not_in_layer == thick.shape[0]
    for i, n in enumerate(not_in_any_layer):
        if n:
            closest = np.argmax(thick[:, i])
            thick[closest, i] = 1.
    thick[thick < 0] = 0
    thick[heads == nodata] = 0  # exclude nodata cells

    # compute transmissivities
    T = thick * hk
    return T


def get_water_table(heads, nodata, per_idx=None):
    """
    Get a 2D array representing the water table elevation for each
    stress period in heads array.
    
    Parameters
    ----------
    heads : 3 or 4-D np.ndarray
        Heads array.
    nodata : real
        HDRY value indicating dry cells.
    per_idx : int or sequence of ints
        stress periods to return. If None,
        returns all stress periods (default is None).
    Returns
    -------
    wt : 2 or 3-D np.ndarray of water table elevations
        for each stress period.
    """
    heads = np.array(heads, ndmin=4)
    nper, nlay, nrow, ncol = heads.shape
    if per_idx is None:
        per_idx = list(range(nper))
    elif np.isscalar(per_idx):
        per_idx = [per_idx]
    wt = []
    for per in per_idx:
        wt_per = []
        for i in range(nrow):
            for j in range(ncol):
                for k in range(nlay):
                    if heads[per, k, i, j] != nodata:
                        wt_per.append(heads[per, k, i, j])
                        break
                    elif k == nlay - 1:
                        wt_per.append(nodata)
        assert len(wt_per) == nrow * ncol
        wt.append(np.reshape(wt_per, (nrow, ncol)))
    return np.squeeze(wt)


def get_saturated_thickness(heads, m, nodata, per_idx=None):
    """
    Calculates the saturated thickness for each cell from the heads
    array for each stress period.

    Parameters
    ----------
    heads : 3 or 4-D np.ndarray
        Heads array.
    m : flopy.modflow.Modflow object
        Must have a flopy.modflow.ModflowDis object attached.
    nodata : real
        HDRY value indicating dry cells.
    per_idx : int or sequence of ints
        stress periods to return. If None,
        returns all stress periods (default).
    Returns
    -------
    sat_thickness : 3 or 4-D np.ndarray
        Array of saturated thickness
    """
    # internal calculations done on a masked array
    heads = np.ma.array(heads, ndmin=4, mask=heads == nodata)
    botm = m.dis.botm.array
    thickness = m.dis.thickness.array
    nper, nlay, nrow, ncol = heads.shape
    if per_idx is None:
        per_idx = list(range(nper))
    elif np.isscalar(per_idx):
        per_idx = [per_idx]

    sat_thickness = []
    for per in per_idx:
        hds = heads[per]
        perthickness = hds - botm
        conf = perthickness > thickness
        perthickness[conf] = thickness[conf]
        # convert to nan-filled array, as is expected(!?)
        sat_thickness.append(perthickness.filled(np.nan))
    return np.squeeze(sat_thickness)


def get_gradients(heads, m, nodata, per_idx=None):
    """
    Calculates the hydraulic gradients from the heads
    array for each stress period.

    Parameters
    ----------
    heads : 3 or 4-D np.ndarray
        Heads array.
    m : flopy.modflow.Modflow object
        Must have a flopy.modflow.ModflowDis object attached.
    nodata : real
        HDRY value indicating dry cells.
    per_idx : int or sequence of ints
        stress periods to return. If None,
        returns all stress periods (default).
    Returns
    -------
    grad : 3 or 4-D np.ndarray
        Array of hydraulic gradients
    """
    # internal calculations done on a masked array
    heads = np.ma.array(heads, ndmin=4, mask=heads == nodata)
    nper, nlay, nrow, ncol = heads.shape
    if per_idx is None:
        per_idx = list(range(nper))
    elif np.isscalar(per_idx):
        per_idx = [per_idx]

    grad = []
    for per in per_idx:
        hds = heads[per]
        zcnt_per = np.ma.array(m.dis.zcentroids, mask=hds.mask)
        unsat = zcnt_per > hds
        zcnt_per[unsat] = hds[unsat]

        # apply .diff on data and mask components separately
        diff_mask = np.diff(hds.mask, axis=0)
        dz = np.ma.array(np.diff(zcnt_per.data, axis=0), mask=diff_mask)
        dh = np.ma.array(np.diff(hds.data, axis=0), mask=diff_mask)
        # convert to nan-filled array, as is expected(!?)
        grad.append((dh / dz).filled(np.nan))
    return np.squeeze(grad)
