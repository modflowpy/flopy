"""
Grid utilities
"""
from math import floor
from typing import Collection, Iterable, List, Sequence, Tuple, Union

import numpy as np


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
        raise ValueError(f"ncpl must be int or array-like")
    if not isinstance(nodes, (list, tuple, np.ndarray)):
        raise ValueError(f"nodes must be array-like")

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


def get_disu_kwargs(nlay, nrow, ncol, delr, delc, tp, botm):
    """
    Simple utility for creating args needed to construct
    a disu package
    """

    def get_nn(k, i, j):
        return k * nrow * ncol + i * ncol + j

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
                area[n] = delr[i] * delc[j]
                ihc.append(n + 1)
                cl12.append(n + 1)
                hwva.append(n + 1)
                if k == 0:
                    top[n] = tp
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
                    hwva.append(delr[i] * delc[j])
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
                    hwva.append(delr[i] * delc[j])
    ja = np.array(ja, dtype=int)
    nja = ja.shape[0]
    hwva = np.array(hwva, dtype=float)
    kw = {}
    kw["nodes"] = nodes
    kw["nja"] = nja
    kw["nvert"] = None
    kw["top"] = top
    kw["bot"] = bot
    kw["area"] = area
    kw["iac"] = iac
    kw["ja"] = ja
    kw["ihc"] = ihc
    kw["cl12"] = cl12
    kw["hwva"] = hwva
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
