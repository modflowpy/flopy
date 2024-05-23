"""
Grid utilities
"""

from math import floor
from typing import Collection, Iterable, List, Sequence, Tuple, Union

import numpy as np

from .cvfdutil import centroid_of_polygon, get_disv_gridprops


def get_lni(ncpl, nodes) -> List[Tuple[int, int]]:
    """
    Get layer index and within-layer node index (both 0-based).

     | Node count per layer may be an int or array-like of integers.
    An integer ncpl indicates all layers have the same node count.
    If an integer ncpl is less than any specified node numbers, the
    grid is understood to have at least enough layers to contain them.

     | If ncpl is array-like it is understood to describe node count
    per zero-indexed layer.

    Parameters
    ----------
    ncpl: node count per layer (int or array-like of ints)
    nodes : node numbers (array-like of nodes)

    Returns
    -------
        A list of tuples (layer index, node index)
    """

    if not isinstance(ncpl, (int, list, tuple, np.ndarray)):
        raise ValueError("ncpl must be int or array-like")
    if not isinstance(nodes, (list, tuple, np.ndarray)):
        raise ValueError("nodes must be array-like")

    if len(nodes) == 0:
        return []

    if isinstance(ncpl, int):
        # infer min number of layers to hold given node numbers
        layers = range(floor(np.max(nodes) / ncpl) if len(nodes) > 0 else 1)
        counts = [ncpl for _ in layers]
    else:
        counts = list(ncpl)

    tuples = []
    for nn in nodes if nodes else range(sum(counts)):
        csum = np.cumsum([0] + counts)
        layer = max(0, np.searchsorted(csum, nn) - 1)
        nidx = nn - sum([counts[l] for l in range(0, layer)])

        # np.searchsorted assigns the first index of each layer
        # to the previous layer in layer 2+, so correct for it
        correct = layer + 1 < len(csum) and nidx == counts[layer]
        tuples.append((layer + 1, 0) if correct else (layer, nidx))

    return tuples


def get_disu_kwargs(
    nlay,
    nrow,
    ncol,
    delr,
    delc,
    tp,
    botm,
    return_vertices=False,
):
    """
    Create args needed to construct a DISU package for a regular
    MODFLOW grid.

    Parameters
    ----------
    nlay : int
        Number of layers
    nrow : int
        Number of rows
    ncol : int
        Number of columns
    delr : numpy.ndarray
        Column spacing along a row
    delc : numpy.ndarray
        Row spacing along a column
    tp : int or numpy.ndarray
        Top elevation(s) of cells in the model's top layer
    botm : numpy.ndarray
        Bottom elevation(s) for each layer
    return_vertices: bool
        If true, then include vertices and cell2d in kwargs
    """

    def get_nn(k, i, j):
        return k * nrow * ncol + i * ncol + j

    if not isinstance(delr, np.ndarray):
        delr = np.array(delr)
    if not isinstance(delc, np.ndarray):
        delc = np.array(delc)
    assert delr.shape == (ncol,)
    assert delc.shape == (nrow,)

    nodes = nlay * nrow * ncol
    iac = np.zeros((nodes), dtype=int)
    ja = []
    area = np.zeros((nodes), dtype=float)
    top = np.zeros((nodes), dtype=float)
    bot = np.zeros((nodes), dtype=float)
    ihc = []
    cl12 = []
    hwva = []
    for k in range(nlay):
        for i in range(nrow):
            for j in range(ncol):
                # diagonal
                n = get_nn(k, i, j)
                ja.append(n)
                iac[n] += 1
                area[n] = delr[j] * delc[i]
                ihc.append(k + 1)  # put layer in diagonal for flopy plotting
                cl12.append(n + 1)
                hwva.append(n + 1)
                if k == 0:
                    top[n] = tp.item() if isinstance(tp, np.ndarray) else tp
                else:
                    top[n] = botm[k - 1]
                bot[n] = botm[k]
                # up
                if k > 0:
                    ja.append(get_nn(k - 1, i, j))
                    iac[n] += 1
                    ihc.append(0)
                    dz = botm[k - 1] - botm[k]
                    cl12.append(0.5 * dz)
                    hwva.append(delr[j] * delc[i])
                # back
                if i > 0:
                    ja.append(get_nn(k, i - 1, j))
                    iac[n] += 1
                    ihc.append(1)
                    cl12.append(0.5 * delc[i])
                    hwva.append(delr[j])
                # left
                if j > 0:
                    ja.append(get_nn(k, i, j - 1))
                    iac[n] += 1
                    ihc.append(1)
                    cl12.append(0.5 * delr[j])
                    hwva.append(delc[i])
                # right
                if j < ncol - 1:
                    ja.append(get_nn(k, i, j + 1))
                    iac[n] += 1
                    ihc.append(1)
                    cl12.append(0.5 * delr[j])
                    hwva.append(delc[i])
                # front
                if i < nrow - 1:
                    ja.append(get_nn(k, i + 1, j))
                    iac[n] += 1
                    ihc.append(1)
                    cl12.append(0.5 * delc[i])
                    hwva.append(delr[j])
                # bottom
                if k < nlay - 1:
                    ja.append(get_nn(k + 1, i, j))
                    iac[n] += 1
                    ihc.append(0)
                    if k == 0:
                        dz = tp - botm[k]
                    else:
                        dz = botm[k - 1] - botm[k]
                    cl12.append(0.5 * dz)
                    hwva.append(delr[j] * delc[i])
    ja = np.array(ja, dtype=int)
    nja = ja.shape[0]
    hwva = np.array(hwva, dtype=float)

    # build vertices
    nvert = None
    if return_vertices:
        xv = np.cumsum(delr)
        xv = np.array([0] + list(xv))
        ymax = delc.sum()
        yv = np.cumsum(delc)
        yv = ymax - np.array([0] + list(yv))
        xmg, ymg = np.meshgrid(xv, yv)
        nvert = xv.shape[0] * yv.shape[0]
        verts = np.array(list(zip(xmg.flatten(), ymg.flatten())))
        vertices = []
        for i in range(nvert):
            vertices.append((i, verts[i, 0], verts[i, 1]))

        cell2d = []
        icell = 0
        for k in range(nlay):
            for i in range(nrow):
                for j in range(ncol):
                    iv0 = j + i * (ncol + 1)  # upper left vertex
                    iv1 = iv0 + 1  # upper right vertex
                    iv3 = iv0 + ncol + 1  # lower left vertex
                    iv2 = iv3 + 1  # lower right vertex
                    iverts = [iv0, iv1, iv2, iv3]
                    vlist = [(verts[iv, 0], verts[iv, 1]) for iv in iverts]
                    xc, yc = centroid_of_polygon(vlist)
                    cell2d.append([icell, xc, yc, len(iverts)] + iverts)
                    icell += 1

    kw = {}
    kw["nodes"] = nodes
    kw["nja"] = nja
    kw["nvert"] = nvert
    kw["top"] = top
    kw["bot"] = bot
    kw["area"] = area
    kw["iac"] = iac
    kw["ja"] = ja
    kw["ihc"] = ihc
    kw["cl12"] = cl12
    kw["hwva"] = hwva
    if return_vertices:
        kw["vertices"] = vertices
        kw["cell2d"] = cell2d
    return kw


def get_disv_kwargs(
    nlay,
    nrow,
    ncol,
    delr,
    delc,
    tp,
    botm,
    xoff=0.0,
    yoff=0.0,
):
    """
    Create args needed to construct a DISV package.

    Parameters
    ----------
    nlay : int
        Number of layers
    nrow : int
        Number of rows
    ncol : int
        Number of columns
    delr : float or numpy.ndarray
        Column spacing along a row with shape (ncol)
    delc : float or numpy.ndarray
        Row spacing along a column with shape (nrow)
    tp : float or numpy.ndarray
        Top elevation(s) of cells in the model's top layer with shape (nrow, ncol)
    botm : list of floats or numpy.ndarray
        Bottom elevation(s) of all cells in the model with shape (nlay, nrow, ncol)
    xoff : float
        Value to add to all x coordinates.  Optional (default = 0.)
    yoff : float
        Value to add to all y coordinates.  Optional (default = 0.)
    """

    # validate input
    ncpl = nrow * ncol

    # delr check
    if np.isscalar(delr):
        delr = delr * np.ones(ncol, dtype=float)
    else:
        assert delr.shape == (ncol,), "delr must be array with shape (ncol,)"

    # delc check
    if np.isscalar(delc):
        delc = delc * np.ones(nrow, dtype=float)
    else:
        assert delc.shape == (nrow,), "delc must be array with shape (nrow,)"

    # tp check
    if np.isscalar(tp):
        tp = tp * np.ones((nrow, ncol), dtype=float)
    else:
        assert tp.shape == (
            nrow,
            ncol,
        ), "tp must be scalar or array with shape (nrow, ncol)"

    # botm check
    if np.isscalar(botm):
        botm = botm * np.ones((nlay, nrow, ncol), dtype=float)
    elif isinstance(botm, List):
        assert (
            len(botm) == nlay
        ), "if botm provided as a list it must have length nlay"
        b = np.empty((nlay, nrow, ncol), dtype=float)
        for k in range(nlay):
            b[k] = botm[k]
        botm = b
    else:
        assert botm.shape == (
            nlay,
            nrow,
            ncol,
        ), "botm must be array with shape (nlay, nrow, ncol)"

    # build vertices
    xv = np.cumsum(delr)
    xv = np.array([0] + list(xv))
    ymax = delc.sum()
    yv = np.cumsum(delc)
    yv = ymax - np.array([0] + list(yv))
    xmg, ymg = np.meshgrid(xv, yv)
    verts = np.array(list(zip(xmg.flatten(), ymg.flatten())))
    verts[:, 0] += xoff
    verts[:, 1] += yoff

    # build iverts (list of vertices for each cell)
    iverts = []
    for i in range(nrow):
        for j in range(ncol):
            # number vertices in clockwise order
            iv0 = j + i * (ncol + 1)  # upper left vertex
            iv1 = iv0 + 1  # upper right vertex
            iv3 = iv0 + ncol + 1  # lower left vertex
            iv2 = iv3 + 1  # lower right vertex
            iverts.append([iv0, iv1, iv2, iv3])
    kw = get_disv_gridprops(verts, iverts)

    # reshape and add top and bottom
    kw["top"] = tp.reshape(ncpl)
    kw["botm"] = botm.reshape(nlay, ncpl)
    kw["nlay"] = nlay
    return kw


def uniform_flow_field(qx, qy, qz, shape, delr=None, delc=None, delv=None):
    nlay, nrow, ncol = shape

    # create spdis array for the uniform flow field
    dt = np.dtype(
        [
            ("ID1", np.int32),
            ("ID2", np.int32),
            ("FLOW", np.float64),
            ("QX", np.float64),
            ("QY", np.float64),
            ("QZ", np.float64),
        ]
    )
    spdis = np.array(
        [(id1, id1, 0.0, qx, qy, qz) for id1 in range(nlay * nrow * ncol)],
        dtype=dt,
    )

    # create the flowja array for the uniform flow field (assume top-bot = 1)
    flowja = []
    if delr is None:
        delr = 1.0
    if delc is None:
        delc = 1.0
    if delv is None:
        delv = 1.0
    for k in range(nlay):
        for i in range(nrow):
            for j in range(ncol):
                # diagonal
                flowja.append(0.0)
                # up
                if k > 0:
                    flowja.append(-qz * delr * delc)
                # back
                if i > 0:
                    flowja.append(-qy * delr * delv)
                # left
                if j > 0:
                    flowja.append(qx * delc * delv)
                # right
                if j < ncol - 1:
                    flowja.append(-qx * delc * delv)
                # front
                if i < nrow - 1:
                    flowja.append(qy * delr * delv)
                # bottom
                if k < nlay - 1:
                    flowja.append(qz * delr * delc)
    flowja = np.array(flowja, dtype=np.float64)
    return spdis, flowja
