from itertools import product

import numpy as np
import pytest
from modflow_devtools.markers import requires_pkg

from flopy.utils.gridutil import (
    get_disu_kwargs,
    get_disv_kwargs,
    get_lni,
    uniform_flow_field,
)


@pytest.mark.parametrize(
    "ncpl, nn, expected_layer, expected_ni",
    [
        (10, 0, 0, 0),
        ([10, 10], 0, 0, 0),
        ([10, 10], 10, 1, 0),
        ([10, 10], 9, 0, 9),
        ([10, 10], 15, 1, 5),
        ([10, 20], 29, 1, 19),
    ],
)
def test_get_lni(ncpl, nn, expected_layer, expected_ni):
    # pair with next neighbor unless last in layer,
    # in which case pair with previous neighbor
    t = 1
    if nn == 9 or nn == 29:
        t = -1

    nodes = [nn, nn + t]
    lni = get_lni(ncpl, nodes)
    assert isinstance(lni, list)
    i = 0
    for actual_layer, actual_ni in lni:
        assert actual_layer == expected_layer
        assert actual_ni == expected_ni + (i * t)
        i += 1


def test_get_lni_no_nodes():
    lni = get_lni(10, [])
    assert isinstance(lni, list)
    assert len(lni) == 0


@pytest.mark.parametrize(
    "ncpl, nodes, expected",
    [
        (5, [14], [(2, 4)]),
        (10, [14], [(1, 4)]),
        (20, [14], [(0, 14)]),
        (20, [14, 24], [(0, 14), (1, 4)]),
    ],
)
def test_get_lni_infers_layer_count_when_int_ncpl(ncpl, nodes, expected):
    lni = get_lni(ncpl, nodes)
    assert isinstance(lni, list)
    for i, ln in enumerate(lni):
        assert ln == expected[i]


@requires_pkg("shapely")
@pytest.mark.parametrize(
    "nlay, nrow, ncol, delr, delc, tp, botm",
    [
        (
            1,
            61,
            61,
            np.array(61 * [50]),
            np.array(61 * [50]),
            np.array([-10]),
            np.array([-30.0, -50.0]),
        ),
        (
            2,
            61,
            61,
            np.array(61 * [50]),
            np.array(61 * [50]),
            np.array([-10]),
            np.array([-30.0, -50.0]),
        ),
        (
            1,  # nlay
            3,  # nrow
            4,  # ncol
            np.array(4 * [4.0]),  # delr
            np.array(3 * [3.0]),  # delc
            np.array([-10]),  # top
            np.array([-30.0]),  # botm
        ),
    ],
)
def test_get_disu_kwargs(nlay, nrow, ncol, delr, delc, tp, botm):
    kwargs = get_disu_kwargs(
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        tp=tp,
        botm=botm,
        return_vertices=True,
    )

    from pprint import pprint

    pprint(kwargs["area"])

    assert kwargs["nodes"] == nlay * nrow * ncol
    assert kwargs["nvert"] == (nrow + 1) * (ncol + 1)

    area = np.array([dr * dc for (dr, dc) in product(delr, delc)], dtype=float)
    area = np.array(nlay * [area]).flatten()
    assert np.array_equal(kwargs["area"], area)

    # TODO: test other properties
    # print(kwargs["iac"])
    # print(kwargs["ihc"])
    # print(kwargs["ja"])
    # print(kwargs["nja"])


@requires_pkg("shapely")
@pytest.mark.parametrize(
    "nlay, nrow, ncol, delr, delc, tp, botm",
    [
        (
            1,
            61,
            61,
            np.array(61 * [50.0]),
            np.array(61 * [50.0]),
            -10.0,
            -50.0,
        ),
        (
            2,
            61,
            61,
            np.array(61 * [50.0]),
            np.array(61 * [50.0]),
            -10.0,
            [-30.0, -50.0],
        ),
    ],
)
def test_get_disv_kwargs(nlay, nrow, ncol, delr, delc, tp, botm):
    kwargs = get_disv_kwargs(
        nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc, tp=tp, botm=botm
    )

    assert kwargs["nlay"] == nlay
    assert kwargs["ncpl"] == nrow * ncol
    assert kwargs["nvert"] == (nrow + 1) * (ncol + 1)

    # TODO: test other properties
    # print(kwargs["vertices"])
    # print(kwargs["cell2d"])


@requires_pkg("shapely")
@pytest.mark.parametrize(
    "qx, qy, qz, nlay, nrow, ncol",
    [
        (1, 0, 0, 1, 1, 10),
        (0, 1, 0, 1, 1, 10),
        (0, 0, 1, 1, 1, 10),
        (1, 0, 0, 1, 10, 10),
        (1, 0, 0, 2, 10, 10),
        (1, 1, 0, 2, 10, 10),
        (1, 1, 1, 2, 10, 10),
        (2, 1, 1, 2, 10, 10),
    ],
)
def test_uniform_flow_field(qx, qy, qz, nlay, nrow, ncol):
    shape = nlay, nrow, ncol
    spdis, flowja = uniform_flow_field(qx, qy, qz, shape)

    assert spdis.shape == (nlay * nrow * ncol,)
    for i, t in enumerate(spdis.flatten()):
        assert t[0] == t[1] == i
        assert t[3] == qx
        assert t[4] == qy
        assert t[5] == qz

    # TODO: check flowja
    # print(flowja.shape)
