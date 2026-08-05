"""
Ghost node correction (GNC) data for quadtree-like grids.

Ghost node data can be computed from a grid, a grid conforming array of
refinement levels, and the grid connectivity, or read from gridgen output
using :meth:`flopy.utils.gridgen.Gridgen.get_gnc`.  The record arrays produced
by either route are converted to MODFLOW 6 and MODFLOW-USG package input by
:func:`get_gridprops_gnc6` and :func:`get_gridprops_gnc5`.

"""

import numpy as np


def get_gnc_dtype(numalphaj):
    """
    Get the record dtype for ghost node data with numalphaj contributing
    cells

    Parameters
    ----------
    numalphaj : int
        Number of contributing cells per ghost node

    Returns
    -------
    dtype : np.dtype

    """
    dtype = [("n", int), ("m", int)]
    dtype += [(f"j{i}", int) for i in range(numalphaj)]
    dtype += [(f"alpha{i}", float) for i in range(numalphaj)]
    return np.dtype(dtype)


def get_numalphaj(gnc):
    """
    Get the number of contributing cells in a ghost node record array

    Parameters
    ----------
    gnc : np.recarray
        Ghost node data

    Returns
    -------
    numalphaj : int

    """
    return len([name for name in gnc.dtype.names if name.startswith("j")])


def _gnc_nodes(gnc):
    """Return the cellid column names of a ghost node record array"""
    numalphaj = get_numalphaj(gnc)
    return ["n", "m"] + [f"j{i}" for i in range(numalphaj)]


def _get_ia(ia=None, iac=None):
    """Build the zero-based ia array from ia or iac"""
    if ia is not None:
        return np.asarray(ia, dtype=int)
    if iac is None:
        return None
    return np.concatenate(([0], np.cumsum(np.asarray(iac, dtype=int))))


def _check_gnc(gnc, ia=None, ja=None, iac=None):
    """
    Raise if ghost node records are not valid MODFLOW input

    Parameters
    ----------
    gnc : np.recarray
        Ghost node data
    ia : array_like
        Zero-based CRS row pointer.  Connectivity is not checked if ja is
        None or if both ia and iac are None.
    ja : array_like
        Zero-based CRS column indices
    iac : array_like
        Number of connections per cell, used if ia is None

    """
    ia = _get_ia(ia, iac)
    numalphaj = get_numalphaj(gnc)
    alpha = np.zeros(len(gnc))
    for i in range(numalphaj):
        alpha += gnc[f"alpha{i}"]
    for irec, total in enumerate(alpha):
        if total >= 1.0:
            raise ValueError(
                f"gnc record {irec}: contributing factors sum to {total}, "
                "which must be less than one"
            )

    if ia is None or ja is None:
        return
    ia = np.asarray(ia, dtype=int)
    ja = np.asarray(ja, dtype=int)
    nodes = ia.shape[0] - 1
    for irec, rec in enumerate(gnc):
        n, m = rec["n"], rec["m"]
        # a node number outside the grid would either index past the end of ia
        # or, when it is negative, silently wrap and check the wrong cell
        for name, node in (("n", n), ("m", m)):
            if not 0 <= node < nodes:
                raise ValueError(
                    f"gnc record {irec}: cell {name} is {node}, which is not a "
                    f"cell of a grid with {nodes} cells"
                )
        # MODFLOW 6 rejects a ghost node whose n-m connection is absent
        if m not in ja[ia[n] : ia[n + 1]]:
            raise ValueError(
                f"gnc record {irec}: cell {n} is not connected to cell {m}"
            )


def _node_centers(modelgrid):
    """Return cell center arrays with one value per node"""
    xc = np.asarray(modelgrid.xcellcenters).ravel()
    yc = np.asarray(modelgrid.ycellcenters).ravel()
    nnodes = modelgrid.nnodes
    if xc.shape[0] == nnodes:
        return xc, yc
    # a vertex grid stores one value per cell2d, repeated for every layer
    nlay = nnodes // xc.shape[0]
    return np.tile(xc, nlay), np.tile(yc, nlay)


def _node_layers(modelgrid):
    """Return the layer index of every node"""
    ncpl = modelgrid.ncpl
    if np.isscalar(ncpl):
        ncpl = np.full(modelgrid.nlay, ncpl, dtype=int)
    return np.repeat(np.arange(len(ncpl)), ncpl)


def _shared_edge_connectivity(modelgrid):
    """Build ia and ja from the cells that share an edge in every layer"""
    neighbors = modelgrid.neighbors(method="rook")
    ncpl = modelgrid.ncpl
    if not np.isscalar(ncpl):
        if ncpl.min() != ncpl.max():
            raise ValueError(
                "Connectivity cannot be built for a grid with a different "
                "number of cells in each layer, supply ia or iac and ja"
            )
        ncpl = int(ncpl.min())
    nlay = modelgrid.nnodes // ncpl

    # ghost nodes are horizontal, so only the connections within a layer
    # are needed and the same layout is repeated for every layer
    iac, ja = [], []
    for k in range(nlay):
        for icpl in range(ncpl):
            conn = sorted(neighbors.get(icpl, []))
            iac.append(len(conn) + 1)
            ja.append(k * ncpl + icpl)
            ja.extend(k * ncpl + j for j in conn)
    return _get_ia(iac=iac), np.array(ja, dtype=int)


def _contributing_cells(dn, naxis, d_nm, rtol):
    """
    Select the cells that contribute to the ghost node on one connection

    Parameters
    ----------
    dn : np.ndarray
        Offset from cell n to each of its neighbors, one row per neighbor
    naxis : np.ndarray
        Connection axis of each neighbor of cell n
    d_nm : np.ndarray
        Offset from cell n to the connected cell m
    rtol : float
        Relative tolerance on the transverse offset

    Returns
    -------
    sel : np.ndarray or None
        Mask of the contributing neighbors, None when the connection needs no
        ghost node
    alpha : float
        Total contributing factor

    """
    axis = int(np.argmax(np.abs(d_nm)))
    trans = 1 - axis
    offset = d_nm[trans]
    # the cell centers line up across the face, so there is nothing to correct
    if abs(offset) <= rtol * abs(d_nm[axis]):
        return None, 0.0

    # the contributing cells are the neighbors of n on the side of the offset
    sel = (naxis == trans) & (np.sign(dn[:, trans]) == np.sign(offset))
    if not sel.any():
        return None, 0.0
    return sel, abs(offset) / np.abs(dn[sel, trans]).mean()


def _cell_areas(modelgrid):
    """Return the area of every node from the cell vertices"""
    verts = np.asarray(modelgrid.verts)
    iverts = modelgrid.iverts
    areas = np.empty(len(iverts))
    for i, iv in enumerate(iverts):
        iv = [j for j in iv if j is not None]
        x, y = verts[iv, 0], verts[iv, 1]
        areas[i] = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if areas.shape[0] != modelgrid.nnodes:
        nlay = modelgrid.nnodes // areas.shape[0]
        areas = np.tile(areas, nlay)
    return areas


def get_gnc(
    modelgrid,
    level=None,
    ia=None,
    ja=None,
    iac=None,
    ihc=None,
    numalphaj=None,
    rtol=1.0e-6,
):
    """
    Compute ghost node correction data for a quadtree-like grid

    A ghost node is added in cell n for every horizontal connection to a
    finer cell m whose center is offset from the center of n transverse to
    the connection.  The head at the ghost node is interpolated between cell
    n and the neighbors of n on the side of the offset.

    Parameters
    ----------
    modelgrid : flopy.discretization.UnstructuredGrid or VertexGrid
        Grid the ghost nodes are computed for.  Connectivity is taken from
        the grid when ia, ja, and iac are None, either from the iac and ja
        the grid carries or from the cells that share an edge.
    level : array_like
        Grid conforming array of refinement levels, where a larger value is a
        finer cell.  Cell areas are used if None.
    ia : array_like
        Zero-based CRS row pointer
    ja : array_like
        Zero-based CRS column indices, with the diagonal first in each row
    iac : array_like
        Number of connections per cell, used if ia is None
    ihc : array_like
        Connection type for each entry in ja, where 0 is a vertical
        connection.  Connections between cells in different layers are
        treated as vertical if None.
    numalphaj : int
        Number of contributing cells written per ghost node.  The largest
        number found is used if None.  Records with fewer contributing cells
        repeat a cell rather than pad with zeros, which MODFLOW-USG requires.
    rtol : float
        Relative tolerance used to decide whether the center of cell m is
        offset from the center of cell n (default is 1.0e-6).

    Returns
    -------
    gnc : np.recarray
        Record array with fields n, m, j0 to j[numalphaj-1], and alpha0 to
        alpha[numalphaj-1].  Node numbers are zero-based.

    Notes
    -----
    Only horizontal corrections are computed, matching gridgen.

    """
    ia = _get_ia(ia, iac)
    if ia is None:
        ia = _get_ia(iac=getattr(modelgrid, "iac", None))
    if ja is None:
        ja = getattr(modelgrid, "ja", None)
    if ia is None or ja is None:
        # a vertex grid does not carry connectivity, so build it from the
        # cells that share an edge
        ia, ja = _shared_edge_connectivity(modelgrid)
    ja = np.asarray(ja, dtype=int)

    nnodes = modelgrid.nnodes
    xc, yc = _node_centers(modelgrid)

    # a larger size is a coarser cell
    if level is None:
        size = _cell_areas(modelgrid)
    else:
        size = -np.asarray(level, dtype=float).ravel()
        if size.shape[0] != nnodes:
            size = np.tile(size, nnodes // size.shape[0])

    if ihc is None:
        layer = _node_layers(modelgrid)
        horizontal = layer[ja] == layer[np.repeat(np.arange(nnodes), np.diff(ia))]
    else:
        horizontal = np.asarray(ihc) != 0

    records = []
    for n in range(nnodes):
        ipos = np.arange(ia[n] + 1, ia[n + 1])
        ipos = ipos[horizontal[ipos]]
        conn = ja[ipos]
        if conn.size == 0:
            continue

        # column 0 is the x offset and column 1 the y offset to each neighbor
        d = np.column_stack((xc[conn] - xc[n], yc[conn] - yc[n]))
        axis = np.argmax(np.abs(d), axis=1)

        for k, m in enumerate(conn):
            if size[m] >= size[n]:
                continue
            sel, alpha = _contributing_cells(d, axis, d[k], rtol)
            if sel is None:
                continue
            js = conn[sel]
            records.append((n, m, js, alpha / js.size))

    if numalphaj is None:
        numalphaj = max((len(rec[2]) for rec in records), default=1)
    dtype = get_gnc_dtype(numalphaj)

    gnc = np.recarray((len(records),), dtype=dtype)
    for irec, (n, m, js, alpha) in enumerate(records):
        if js.size > numalphaj:
            raise ValueError(
                f"gnc record {irec}: cell {n} has {js.size} contributing cells, "
                f"which is more than numalphaj of {numalphaj}"
            )
        # pad by repeating the first contributing cell and splitting its
        # factor, which both MODFLOW 6 and MODFLOW-USG accumulate
        nrepeat = numalphaj - js.size + 1
        nodes = np.concatenate((np.repeat(js[:1], nrepeat), js[1:]))
        alphas = np.concatenate(
            (np.full(nrepeat, alpha / nrepeat), np.full(js.size - 1, alpha))
        )
        gnc["n"][irec] = n
        gnc["m"][irec] = m
        for i in range(numalphaj):
            gnc[f"j{i}"][irec] = nodes[i]
            gnc[f"alpha{i}"][irec] = alphas[i]

    return gnc


def get_gridprops_gnc6(
    gnc, dis_type="disv", ncpl=None, ia=None, ja=None, iac=None, check=True
):
    """
    Get a dictionary of information needed to create a MODFLOW 6 GNC
    Package.  The returned dictionary can be unpacked directly into the
    ModflowGwfgnc constructor.

    Parameters
    ----------
    gnc : np.recarray
        Ghost node data with zero-based node numbers
    dis_type : str
        Discretization the cellids are built for.  Valid options are 'disv'
        (default) and 'disu'.
    ncpl : int
        Number of cells per layer, required for 'disv'
    ia : array_like
        Zero-based CRS row pointer, used to check connectivity
    ja : array_like
        Zero-based CRS column indices, used to check connectivity
    iac : array_like
        Number of connections per cell, used if ia is None
    check : bool
        Verify that each n-m pair is connected and that the contributing
        factors sum to less than one (default is True).

    Returns
    -------
    gridprops : dict

    Notes
    -----
    The correction is applied implicitly unless the explicit option is set,
    so the BICGSTAB linear acceleration option should be specified in the IMS
    Package.  numgnc is zero for a grid without ghost nodes, in which case
    the package should not be created.

    """
    if check:
        _check_gnc(gnc, ia=ia, ja=ja, iac=iac)

    dis_type = dis_type.lower()
    if dis_type == "disv":
        if ncpl is None:
            raise ValueError("ncpl is required to build disv cellids")

        def cellid(node):
            return (node // ncpl, node % ncpl)
    elif dis_type == "disu":

        def cellid(node):
            return (node,)
    else:
        raise ValueError(f"Unknown dis_type {dis_type}, expected 'disv' or 'disu'")

    numalphaj = get_numalphaj(gnc)
    names = _gnc_nodes(gnc)
    gncdata = [
        tuple(cellid(rec[name]) for name in names)
        + tuple(rec[f"alpha{i}"] for i in range(numalphaj))
        for rec in gnc
    ]

    return {
        "numgnc": len(gncdata),
        "numalphaj": numalphaj,
        "gncdata": gncdata,
    }


def get_gridprops_gnc5(gnc, i2kn=0, isymgncn=0, ia=None, ja=None, iac=None, check=True):
    """
    Get a dictionary of information needed to create a MODFLOW-USG GNC
    Package.  The returned dictionary can be unpacked directly into the
    MfUsgGnc constructor.

    Parameters
    ----------
    gnc : np.recarray
        Ghost node data with zero-based node numbers
    i2kn : int
        Apply the second-order correction to unconfined transmissivity
        (default is 0).
    isymgncn : int
        Update the right-hand side vector for symmetric systems instead of
        the left-hand side matrix (default is 0).
    ia : array_like
        Zero-based CRS row pointer, used to check connectivity
    ja : array_like
        Zero-based CRS column indices, used to check connectivity
    iac : array_like
        Number of connections per cell, used if ia is None
    check : bool
        Verify that each n-m pair is connected and that the contributing
        factors sum to less than one (default is True).

    Returns
    -------
    gridprops : dict

    Notes
    -----
    Contributing factors are always written, so iflalphan is 0.  The default
    asymmetric implementation requires an asymmetric solver.  numgnc is zero
    for a grid without ghost nodes, in which case the package should not be
    created.

    """
    # imported here because flopy.mfusg imports flopy.utils
    from ..mfusg.mfusggnc import MfUsgGnc

    if check:
        _check_gnc(gnc, ia=ia, ja=ja, iac=iac)

    numalphaj = get_numalphaj(gnc)
    iflalphan = 0
    gncdata = MfUsgGnc.get_empty(
        numgnc=len(gnc), numalphaj=numalphaj, iflalphan=iflalphan
    )
    # MfUsgGnc.write_file converts to one-based node numbers
    gncdata["NodeN"] = gnc["n"]
    gncdata["NodeM"] = gnc["m"]
    for i in range(numalphaj):
        gncdata[f"Node{i}"] = gnc[f"j{i}"]
        gncdata[f"Alpha{i}"] = gnc[f"alpha{i}"]

    return {
        "numgnc": len(gncdata),
        "numalphaj": numalphaj,
        "i2kn": i2kn,
        "isymgncn": isymgncn,
        "iflalphan": iflalphan,
        "gncdata": gncdata,
    }


def get_gnc_exchange(
    modelgrid1,
    modelgrid2,
    exchangedata,
    idomain1=None,
    numalphaj=None,
    rtol=1.0e-6,
):
    """
    Compute ghost node correction data for a GWF-GWF exchange

    A ghost node is added in the cell of model 1 for every horizontal exchange
    connection to a finer cell of model 2 whose center is offset transverse to
    the connection.  The head at the ghost node is interpolated between the
    cell of model 1 and its own neighbors, so model 1 must be the coarser of
    the two models.

    Parameters
    ----------
    modelgrid1 : flopy.discretization.Grid
        Grid of the first model of the exchange, which holds the ghost nodes
        and the contributing cells
    modelgrid2 : flopy.discretization.Grid
        Grid of the second model of the exchange
    exchangedata : list
        Exchange records, each starting with cellidm1, cellidm2, and ihc, as
        returned by :meth:`flopy.utils.lgrutil.Lgr.get_exchange_data`
    idomain1 : array_like
        Idomain of the first model, used to skip inactive contributing cells.
        Taken from modelgrid1 if None.
    numalphaj : int
        Number of contributing cells written per ghost node.  The largest
        number found is used if None.
    rtol : float
        Relative tolerance used to decide whether the center of the cell of
        model 2 is offset from the center of the cell of model 1 (default is
        1.0e-6).

    Returns
    -------
    gridprops : dict
        Dictionary with numgnc, numalphaj, and gncdata, which can be unpacked
        into the gnc package of a ModflowGwfgwf exchange.

    Notes
    -----
    MODFLOW 6 requires one ghost node record for every exchange record, in the
    same order, so connections that need no correction are written with a
    cellid of zero and a contributing factor of zero.  Only horizontal
    corrections are computed.

    """
    if idomain1 is None:
        idomain1 = getattr(modelgrid1, "idomain", None)

    xc1, yc1 = _node_centers(modelgrid1)
    xc2, yc2 = _node_centers(modelgrid2)
    area1, area2 = _cell_areas(modelgrid1), _cell_areas(modelgrid2)
    active1 = None if idomain1 is None else np.asarray(idomain1).ravel() > 0
    neighbors = modelgrid1.neighbors(method="rook")

    # the contributing cells of a cell are the same for every exchange record
    # that uses it, so the offsets to its neighbors are built once
    cache = {}

    def offsets(node):
        if node not in cache:
            conn = np.array(sorted(neighbors.get(node, [])), dtype=int)
            if active1 is not None and conn.size:
                conn = conn[active1[conn]]
            if conn.size == 0:
                cache[node] = (conn, None, None)
            else:
                dn = np.column_stack((xc1[conn] - xc1[node], yc1[conn] - yc1[node]))
                cache[node] = (conn, dn, np.argmax(np.abs(dn), axis=1))
        return cache[node]

    records = []
    for rec in exchangedata:
        cellidn, cellidm, ihc = rec[0], rec[1], rec[2]
        n = modelgrid1.get_node([tuple(cellidn)])[0]
        m = modelgrid2.get_node([tuple(cellidm)])[0]

        js, alpha = np.array([], dtype=int), 0.0
        if ihc != 0 and area2[m] < area1[n]:
            conn, dn, naxis = offsets(n)
            if dn is not None:
                d_nm = np.array([xc2[m] - xc1[n], yc2[m] - yc1[n]])
                sel, total = _contributing_cells(dn, naxis, d_nm, rtol)
                if sel is not None:
                    js, alpha = conn[sel], total / sel.sum()
        records.append((cellidn, cellidm, js, alpha))

    if numalphaj is None:
        # every exchange gets a record, so a connection that needs no
        # correction contributes a length of zero; MODFLOW 6 still needs at
        # least one contributing cell per record
        numalphaj = max(max((len(rec[2]) for rec in records), default=1), 1)

    ndim1 = len(records[0][0]) if records else 3
    none_cellid = tuple([-1] * ndim1)  # written as a cellid of zero

    gncdata = []
    for irec, (cellidn, cellidm, js, alpha) in enumerate(records):
        if len(js) > numalphaj:
            raise ValueError(
                f"gnc record {irec}: cell {cellidn} has {len(js)} contributing "
                f"cells, which is more than numalphaj of {numalphaj}"
            )
        cellids = [tuple(int(v) for v in modelgrid1.get_lrc([int(j)])[0]) for j in js]
        alphas = [float(alpha)] * len(js)
        # pad with a cellid of zero, which MODFLOW 6 skips
        cellids += [none_cellid] * (numalphaj - len(js))
        alphas += [0.0] * (numalphaj - len(js))
        gncdata.append(
            tuple(
                [tuple(int(v) for v in cellidn), tuple(int(v) for v in cellidm)]
                + cellids
                + alphas
            )
        )

    return {
        "numgnc": len(gncdata),
        "numalphaj": numalphaj,
        "gncdata": gncdata,
    }
