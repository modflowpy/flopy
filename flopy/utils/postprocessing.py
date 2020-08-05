import numpy as np


def get_transmissivities(
    heads,
    m,
    r=None,
    c=None,
    x=None,
    y=None,
    sctop=None,
    scbot=None,
    nodata=-999,
):
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
        column indices (optional; alternately specify x, y)
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
        r, c = m.sr.get_ij(x, y)
    else:
        raise ValueError("Must specify row, column or x, y locations.")

    # get k-values and botms at those locations
    paklist = m.get_package_list()
    if "LPF" in paklist:
        hk = m.lpf.hk.array[:, r, c]
    elif "UPW" in paklist:
        hk = m.upw.hk.array[:, r, c]
    else:
        raise ValueError("No LPF or UPW package.")

    botm = m.dis.botm.array[:, r, c]

    if heads.shape == (m.nlay, m.nrow, m.ncol):
        heads = heads[:, r, c]

    msg = "Shape of heads array must be nlay x nhyd"
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
            thick[closest, i] = 1.0
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

    # get confined or unconfined/convertible info
    if m.has_package("BCF6") or m.has_package("LPF") or m.has_package("UPW"):
        if m.has_package("BCF6"):
            laytyp = m.lpf.laycon.array
        elif m.has_package("LPF"):
            laytyp = m.lpf.laytyp.array
        else:
            laytyp = m.upw.laytyp.array
        if len(laytyp) == 1:
            is_conf = np.full(m.modelgrid.shape, laytyp == 0)
        else:
            laytyp = laytyp.reshape(m.modelgrid.nlay, 1, 1)
            is_conf = np.logical_and(
                (laytyp == 0), np.full(m.modelgrid.shape, True)
            )
    elif m.has_package("NPF"):
        is_conf = m.npf.icelltype.array == 0
    else:
        raise ValueError(
            "No flow package was found when trying to determine "
            "the layer type."
        )

    # calculate saturated thickness
    sat_thickness = []
    for per in per_idx:
        hds = heads[per]
        perthickness = hds - botm
        conf = np.logical_or(perthickness > thickness, is_conf)
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


def get_extended_budget(
    cbcfile,
    precision="single",
    idx=None,
    kstpkper=None,
    totim=None,
    boundary_ifaces=None,
    hdsfile=None,
    model=None,
):
    """
    Get the flow rate across cell faces including potential stresses applied
    along boundaries at a given time. Only implemented for "classical" MODFLOW
    versions where the budget is recorded as FLOW RIGHT FACE, FLOW FRONT FACE
    and FLOW LOWER FACE arrays.

    Parameters
    ----------
    cbcfile : str
        Cell by cell file produced by Modflow.
    precision : str
        Binary file precision, default is 'single'.
    idx : int or list
            The zero-based record number.
    kstpkper : tuple of ints
        A tuple containing the time step and stress period (kstp, kper).
        The kstp and kper values are zero based.
    totim : float
        The simulation time.
    boundary_ifaces : dictionary {str: int or list}
        A dictionary defining how to treat stress flows at boundary cells.
        The keys are budget terms corresponding to stress packages (same term
        as in the overall volumetric budget printed in the listing file).
        The values are either a single iface number to be applied to all cells
        for the stress package, or a list of lists describing individual
        boundary cells in the same way as in the package input plus the iface
        number appended. The iface number indicates the face to which the
        stress flow is assigned, following the MODPATH convention (see MODPATH
        user guide).
        Example:
        boundary_ifaces = {
        'RECHARGE': 6,
        'RIVER LEAKAGE': 6,
        'CONSTANT HEAD': [[lay, row, col, iface], ...],
        'WELLS': [[lay, row, col, flux, iface], ...],
        'HEAD DEP BOUNDS': [[lay, row, col, head, cond, iface], ...]}.
        Note: stresses that are not informed in boundary_ifaces are implicitly
        treated as internally-distributed sinks/sources.
    hdsfile : str
        Head file produced by MODFLOW (only required if boundary_ifaces is
        used).
    model : flopy.modflow.Modflow object
        Modflow model instance (only required if boundary_ifaces is used).

    Returns
    -------
    (Qx_ext, Qy_ext, Qz_ext) : tuple
        Flow rates across cell faces.
        Qx_ext is a ndarray of size (nlay, nrow, ncol + 1).
        Qy_ext is a ndarray of size (nlay, nrow + 1, ncol). The sign is such
        that the y axis is considered to increase in the north direction.
        Qz_ext is a ndarray of size (nlay + 1, nrow, ncol). The sign is such
        that the z axis is considered to increase in the upward direction.
    """
    import flopy.utils.binaryfile as bf

    # define useful stuff
    cbf = bf.CellBudgetFile(cbcfile, precision=precision)
    nlay, nrow, ncol = cbf.nlay, cbf.nrow, cbf.ncol
    rec_names = cbf.get_unique_record_names(decode=True)
    err_msg = " not found in the budget file."

    # get flow across right face
    Qx_ext = np.zeros((nlay, nrow, ncol + 1), dtype=np.float32)
    if ncol > 1:
        budget_term = "FLOW RIGHT FACE"
        matched_name = [s for s in rec_names if budget_term in s]
        if not matched_name:
            raise RuntimeError(budget_term + err_msg)
        frf = cbf.get_data(
            idx=idx, kstpkper=kstpkper, totim=totim, text=budget_term
        )
        Qx_ext[:, :, 1:] = frf[0]
        # SWI2 package
        budget_term_swi = "SWIADDTOFRF"
        matched_name_swi = [s for s in rec_names if budget_term_swi in s]
        if matched_name_swi:
            frf_swi = cbf.get_data(
                idx=idx, kstpkper=kstpkper, totim=totim, text=budget_term_swi
            )
            Qx_ext[:, :, 1:] += frf_swi[0]

    # get flow across front face
    Qy_ext = np.zeros((nlay, nrow + 1, ncol), dtype=np.float32)
    if nrow > 1:
        budget_term = "FLOW FRONT FACE"
        matched_name = [s for s in rec_names if budget_term in s]
        if not matched_name:
            raise RuntimeError(budget_term + err_msg)
        fff = cbf.get_data(
            idx=idx, kstpkper=kstpkper, totim=totim, text=budget_term
        )
        Qy_ext[:, 1:, :] = -fff[0]
        # SWI2 package
        budget_term_swi = "SWIADDTOFFF"
        matched_name_swi = [s for s in rec_names if budget_term_swi in s]
        if matched_name_swi:
            fff_swi = cbf.get_data(
                idx=idx, kstpkper=kstpkper, totim=totim, text=budget_term_swi
            )
            Qy_ext[:, 1:, :] -= fff_swi[0]

    # get flow across lower face
    Qz_ext = np.zeros((nlay + 1, nrow, ncol), dtype=np.float32)
    if nlay > 1:
        budget_term = "FLOW LOWER FACE"
        matched_name = [s for s in rec_names if budget_term in s]
        if not matched_name:
            raise RuntimeError(budget_term + err_msg)
        flf = cbf.get_data(
            idx=idx, kstpkper=kstpkper, totim=totim, text=budget_term
        )
        Qz_ext[1:, :, :] = -flf[0]
        # SWI2 package
        budget_term_swi = "SWIADDTOFLF"
        matched_name_swi = [s for s in rec_names if budget_term_swi in s]
        if matched_name_swi:
            flf_swi = cbf.get_data(
                idx=idx, kstpkper=kstpkper, totim=totim, text=budget_term_swi
            )
            Qz_ext[1:, :, :] -= flf_swi[0]

    # deal with boundary cells
    if boundary_ifaces is not None:
        # need calculated heads for some stresses and to check hnoflo and hdry
        if hdsfile is None:
            raise ValueError(
                "hdsfile must be provided when using " "boundary_ifaces"
            )
        hds = bf.HeadFile(hdsfile, precision=precision)
        head = hds.get_data(idx=idx, kstpkper=kstpkper, totim=totim)

        # get hnoflo and hdry values
        if model is None:
            raise ValueError(
                "model must be provided when using " "boundary_ifaces"
            )
        noflo_or_dry = np.logical_or(head == model.hnoflo, head == model.hdry)

        for budget_term, iface_info in boundary_ifaces.items():
            # look for budget term in budget file
            matched_name = [s for s in rec_names if budget_term in s]
            if not matched_name:
                raise RuntimeError(
                    "Budget term " + budget_term + " not found"
                    ' in "' + cbcfile + '" file.'
                )
            if len(matched_name) > 1:
                raise RuntimeError(
                    "Budget term " + budget_term + " found"
                    " in several record names. Use a more "
                    " precise name."
                )
            Q_stress = cbf.get_data(
                idx=idx,
                kstpkper=kstpkper,
                totim=totim,
                text=matched_name[0],
                full3D=True,
            )[0]

            # remove potential leading and trailing spaces
            budget_term = budget_term.strip()

            # weirdly, MODFLOW puts recharge in all potential recharge cells
            # and not only the actual cells; thus, correct this by putting 0
            # away from water table cells
            if budget_term == "RECHARGE":
                # find the water table as the first active cell in each column
                water_table = np.full((nlay, nrow, ncol), False)
                water_table[0, :, :] = np.logical_not(noflo_or_dry[0, :, :])
                already_found = water_table[0, :, :]
                for lay in range(1, nlay):
                    if np.sum(already_found) == nrow * ncol:
                        break
                    water_table[lay, :, :] = np.logical_and(
                        np.logical_not(noflo_or_dry[lay, :, :]),
                        np.logical_not(already_found),
                    )
                    already_found = np.logical_or(
                        already_found, water_table[lay, :, :]
                    )
                Q_stress[np.logical_not(water_table)] = 0.0

            # case where the same iface is assigned to all cells
            if isinstance(iface_info, int):
                if iface_info == 1:
                    Qx_ext[:, :, :-1] += Q_stress
                elif iface_info == 2:
                    Qx_ext[:, :, 1:] -= Q_stress
                elif iface_info == 3:
                    Qy_ext[:, 1:, :] += Q_stress
                elif iface_info == 4:
                    Qy_ext[:, :-1, :] -= Q_stress
                elif iface_info == 5:
                    Qz_ext[1:, :, :] += Q_stress
                elif iface_info == 6:
                    Qz_ext[:-1, :, :] -= Q_stress

            # case where iface is assigned individually per cell
            elif isinstance(iface_info, list):
                # impose a unique iface (normally = 6) for some stresses
                # (note: UZF RECHARGE, GW ET and SURFACE LEAKAGE are all
                # related to the UZF package)
                if (
                    budget_term == "RECHARGE"
                    or budget_term == "ET"
                    or budget_term == "UZF RECHARGE"
                    or budget_term == "GW ET"
                    or budget_term == "SURFACE LEAKAGE"
                ):
                    raise ValueError(
                        "This function imposes the use of a "
                        "unique iface (normally = 6) for the "
                        + budget_term
                        + " budget term."
                    )

                # loop through boundary cells
                for cell_info in iface_info:
                    lay, row, col = cell_info[0], cell_info[1], cell_info[2]
                    if noflo_or_dry[lay, row, col]:
                        continue
                    iface = cell_info[-1]
                    # Here, where appropriate, we recalculate Q_stress_cell
                    # using package input. This gives more flexibility than
                    # directly taking the value saved by MODFLOW. Indeed, it
                    # allows for a same type of stress to be applied several
                    # times to the same cell but to different faces
                    # (whereas MODFLOW only saves one lumped  value per
                    # stress type per cell).
                    # Note: this flexibility is not supported for:
                    # - FHB package (we would need to interpolate inputs
                    #   across time steps as done in the package; complicated)
                    # - RES package (we would need to interpolate inputs
                    #   across time steps as done in the package; complicated)
                    # - STR package (we would need first to retrieve river
                    #   stage from model outputs; complicated)
                    # - SFR1 package (we would need to retrieve river
                    #   stage and conductance from model outputs; complicated)
                    # - SFR2 package (even more complicated than SFR1)
                    # - LAK3 package (we would need to retrieve lake
                    #   stage and conductance from model outputs; complicated)
                    # - MNW1 package (we would need to retrieve well head and
                    #   conductance from model outputs; complicated)
                    # - MNW2 package (even more complicated than MNW1)
                    if budget_term == "WELLS":
                        Q_stress_cell = cell_info[3]
                    elif budget_term == "HEAD DEP BOUNDS":
                        ghb_head = cell_info[3]
                        ghb_cond = cell_info[4]
                        model_head = head[lay, row, col]
                        Q_stress_cell = ghb_cond * (ghb_head - model_head)
                    elif budget_term == "RIVER LEAKAGE":
                        riv_stage = cell_info[3]
                        riv_cond = cell_info[4]
                        riv_rbot = cell_info[5]
                        model_head = head[lay, row, col]
                        if model_head > riv_rbot:
                            Q_stress_cell = riv_cond * (riv_stage - model_head)
                        else:
                            Q_stress_cell = riv_cond * (riv_stage - riv_rbot)
                    elif budget_term == "DRAINS":
                        drn_stage = cell_info[3]
                        drn_cond = cell_info[4]
                        model_head = head[lay, row, col]
                        if model_head > drn_stage:
                            Q_stress_cell = drn_cond * (drn_stage - model_head)
                        else:
                            continue
                    # Else, take the value saved by MODFLOW.
                    # This includes the budget terms:
                    # - 'CONSTANT HEAD' for:
                    #      * head specified through -1 in IBOUND
                    #      * head specified in CHD package
                    #      * head specified in FHB package
                    # - 'SPECIFIED FLOWS' for flow specified in FHB package
                    # - 'RESERV. LEAKAGE' for RES package
                    # - 'STREAM LEAKAGE' for:
                    #      * STR package
                    #      * SFR1 package
                    #      * SFR2 package
                    #  - 'LAKE SEEPAGE' for LAK3 package
                    #  - 'MNW' for MNW1 package
                    #  - 'MNW2' for MNW2 package
                    #  - 'SWIADDTOCH' for SWI2 package
                    else:
                        Q_stress_cell = Q_stress[lay, row, col]

                    if iface == 1:
                        Qx_ext[lay, row, col] += Q_stress_cell
                    elif iface == 2:
                        Qx_ext[lay, row, col + 1] -= Q_stress_cell
                    elif iface == 3:
                        Qy_ext[lay, row + 1, col] += Q_stress_cell
                    elif iface == 4:
                        Qy_ext[lay, row, col] -= Q_stress_cell
                    elif iface == 5:
                        Qz_ext[lay + 1, row, col] += Q_stress_cell
                    elif iface == 6:
                        Qz_ext[lay, row, col] -= Q_stress_cell
            else:
                raise TypeError(
                    "boundary_ifaces value must be either " "int or list."
                )

    return Qx_ext, Qy_ext, Qz_ext


def get_specific_discharge(
    model,
    cbcfile,
    precision="single",
    idx=None,
    kstpkper=None,
    totim=None,
    boundary_ifaces=None,
    hdsfile=None,
    position="centers",
):
    """
    Get the discharge vector at cell centers at a given time. For "classical"
    MODFLOW versions, we calculate it from the flow rate across cell faces.
    For MODFLOW 6, we directly take it from MODFLOW output (this requires
    setting the option "save_specific_discharge" in the NPF package).

    Parameters
    ----------
    model : flopy.modflow.Modflow object
        Modflow model instance.
    cbcfile : str
        Cell by cell file produced by Modflow.
    precision : str
        Binary file precision, default is 'single'.
    idx : int or list
            The zero-based record number.
    kstpkper : tuple of ints
        A tuple containing the time step and stress period (kstp, kper).
        The kstp and kper values are zero based.
    totim : float
        The simulation time.
    boundary_ifaces : dictionary {str: int or list}
        A dictionary defining how to treat stress flows at boundary cells.
        Only implemented for "classical" MODFLOW versions where the budget is
        recorded as FLOW RIGHT FACE, FLOW FRONT FACE and FLOW LOWER FACE
        arrays.
        The keys are budget terms corresponding to stress packages (same term
        as in the overall volumetric budget printed in the listing file).
        The values are either a single iface number to be applied to all cells
        for the stress package, or a list of lists describing individual
        boundary cells in the same way as in the package input plus the iface
        number appended. The iface number indicates the face to which the
        stress flow is assigned, following the MODPATH convention (see MODPATH
        user guide).
        Example:
        boundary_ifaces = {
        'RECHARGE': 6,
        'RIVER LEAKAGE': 6,
        'WELLS': [[lay, row, col, flux, iface], ...],
        'HEAD DEP BOUNDS': [[lay, row, col, head, cond, iface], ...]}.
        Note: stresses that are not informed in boundary_ifaces are implicitly
        treated as internally-distributed sinks/sources.
    hdsfile : str
        Head file produced by MODFLOW. Head is used to calculate saturated
        thickness and to determine if a cell is inactive or dry. If not
        provided, all cells are considered fully saturated.
        hdsfile is also required if the budget term 'HEAD DEP BOUNDS',
        'RIVER LEAKAGE' or 'DRAINS' is present in boundary_ifaces and that the
        corresponding value is a list.
    position : str
        Position at which the specific discharge will be calculated. Possible
        values are "centers" (default), "faces" and "vertices".

    Returns
    -------
    (qx, qy, qz) : tuple
        Discharge vector.
        qx, qy, qz are ndarrays of size (nlay, nrow, ncol) for a structured
        grid or size (nlay, ncpl) for an unstructured grid.
        The sign of qy is such that the y axis is considered to increase
        in the north direction.
        The sign of qz is such that the z axis is considered to increase
        in the upward direction.
        Note: if hdsfile is provided, inactive and dry cells are set to NaN.
    """
    import flopy.utils.binaryfile as bf

    # check if budget file has classical budget terms
    cbf = bf.CellBudgetFile(cbcfile, precision=precision)
    rec_names = cbf.get_unique_record_names(decode=True)
    classical_budget_terms = [
        "FLOW RIGHT FACE",
        "FLOW FRONT FACE",
        "FLOW RIGHT FACE",
    ]
    classical_budget = False
    for budget_term in classical_budget_terms:
        matched_name = [s for s in rec_names if budget_term in s]
        if matched_name:
            classical_budget = True
            break

    if hdsfile is not None:
        hds = bf.HeadFile(hdsfile, precision=precision)
        head = hds.get_data(idx=idx, kstpkper=kstpkper, totim=totim)

    if classical_budget:
        # get extended budget
        Qx_ext, Qy_ext, Qz_ext = get_extended_budget(
            cbcfile,
            precision=precision,
            idx=idx,
            kstpkper=kstpkper,
            totim=totim,
            boundary_ifaces=boundary_ifaces,
            hdsfile=hdsfile,
            model=model,
        )

        # get saturated thickness (head - bottom elev for unconfined layer)
        if hdsfile is None:
            sat_thk = model.dis.thickness.array
        else:
            sat_thk = get_saturated_thickness(head, model, model.hdry)
            sat_thk = sat_thk.reshape(model.modelgrid.shape)

        # inform modelgrid of no-flow and dry cells
        modelgrid = model.modelgrid
        if modelgrid._idomain is None:
            modelgrid._idomain = model.dis.ibound
        if hdsfile is not None:
            noflo_or_dry = np.logical_or(
                head == model.hnoflo, head == model.hdry
            )
            modelgrid._idomain[noflo_or_dry] = 0

        # get cross section areas along x
        delc = np.reshape(modelgrid.delc, (1, modelgrid.nrow, 1))
        cross_area_x = np.empty(modelgrid.shape, dtype=float)
        cross_area_x = delc * sat_thk

        # get cross section areas along y
        delr = np.reshape(modelgrid.delr, (1, 1, modelgrid.ncol))
        cross_area_y = np.empty(modelgrid.shape, dtype=float)
        cross_area_y = delr * sat_thk

        # get cross section areas along z
        cross_area_z = np.ones(modelgrid.shape) * delc * delr

        # calculate qx, qy, qz
        if position == "centers":
            qx = 0.5 * (Qx_ext[:, :, 1:] + Qx_ext[:, :, :-1]) / cross_area_x
            qy = 0.5 * (Qy_ext[:, 1:, :] + Qy_ext[:, :-1, :]) / cross_area_y
            qz = 0.5 * (Qz_ext[1:, :, :] + Qz_ext[:-1, :, :]) / cross_area_z
        elif position == "faces" or position == "vertices":
            cross_area_x = modelgrid.array_at_faces(cross_area_x, "x")
            cross_area_y = modelgrid.array_at_faces(cross_area_y, "y")
            cross_area_z = modelgrid.array_at_faces(cross_area_z, "z")
            qx = Qx_ext / cross_area_x
            qy = Qy_ext / cross_area_y
            qz = Qz_ext / cross_area_z
        else:
            raise ValueError(
                '"' + position + '" is not a valid value for ' "position"
            )
        if position == "vertices":
            qx = modelgrid.array_at_verts(qx)
            qy = modelgrid.array_at_verts(qy)
            qz = modelgrid.array_at_verts(qz)

    else:
        # check valid options
        if boundary_ifaces is not None:
            import warnings

            warnings.warn(
                "the boundary_ifaces option is not implemented "
                'for "non-classical" MODFLOW versions where the '
                "budget is not recorded as FLOW RIGHT FACE, "
                "FLOW FRONT FACE and FLOW LOWER FACE; it will be "
                "ignored",
                UserWarning,
            )
        if position != "centers":
            raise NotImplementedError(
                'position can only be "centers" for '
                '"non-classical" MODFLOW versions where '
                "the budget is not recorded as FLOW "
                "RIGHT FACE, FLOW FRONT FACE and FLOW "
                "LOWER FACE"
            )

        is_spdis = [s for s in rec_names if "DATA-SPDIS" in s]
        if not is_spdis:
            err_msg = (
                "Could not find suitable records in the budget file "
                "to construct the discharge vector."
            )
            raise RuntimeError(err_msg)
        spdis = cbf.get_data(
            text="DATA-SPDIS", idx=idx, kstpkper=kstpkper, totim=totim
        )[0]
        nnodes = model.modelgrid.nnodes
        qx = np.full((nnodes), np.nan)
        qy = np.full((nnodes), np.nan)
        qz = np.full((nnodes), np.nan)
        idx = np.array(spdis["node"]) - 1
        qx[idx] = spdis["qx"]
        qy[idx] = spdis["qy"]
        qz[idx] = spdis["qz"]
        shape = model.modelgrid.shape
        qx.shape = shape
        qy.shape = shape
        qz.shape = shape

    # set no-flow and dry cells to NaN
    if hdsfile is not None and position == "centers":
        noflo_or_dry = np.logical_or(head == model.hnoflo, head == model.hdry)
        qx[noflo_or_dry] = np.nan
        qy[noflo_or_dry] = np.nan
        qz[noflo_or_dry] = np.nan

    return qx, qy, qz
