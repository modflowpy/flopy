import numpy as np
import pandas as pd
import pytest

from flopy.plot.plotutil import (
    MP7_ENDPOINT_DTYPE,
    MP7_PATHLINE_DTYPE,
    PRT_PATHLINE_DTYPE,
    to_mp7_endpoints,
    to_mp7_pathlines,
    to_prt_pathlines,
)

PRT_TEST_PATHLINES = pd.DataFrame.from_records(
    [
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0.0,
            0.000000,
            0.100000,
            9.1,
            0.5,
            "PRP000000001",
        ],
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            0.0,
            0.063460,
            0.111111,
            9.0,
            0.5,
            "PRP000000001",
        ],
        [
            1,
            1,
            1,
            1,
            1,
            1,
            11,
            0,
            1,
            1,
            0.0,
            0.830431,
            0.184020,
            8.0,
            0.5,
            "PRP000000001",
        ],
        [
            1,
            1,
            1,
            1,
            1,
            1,
            21,
            0,
            1,
            1,
            0.0,
            2.026390,
            0.267596,
            7.0,
            0.5,
            "PRP000000001",
        ],
        [
            1,
            1,
            1,
            1,
            1,
            1,
            31,
            0,
            1,
            1,
            0.0,
            3.704265,
            0.360604,
            6.0,
            0.5,
            "PRP000000001",
        ],
        [
            1,
            1,
            1,
            1,
            1,
            1,
            60,
            0,
            1,
            1,
            0.0,
            39.087992,
            9.639587,
            4.0,
            0.5,
            "PRP000000001",
        ],
        [
            1,
            1,
            1,
            1,
            1,
            1,
            70,
            0,
            1,
            1,
            0.0,
            40.765791,
            9.732597,
            3.0,
            0.5,
            "PRP000000001",
        ],
        [
            1,
            1,
            1,
            1,
            1,
            1,
            80,
            0,
            1,
            1,
            0.0,
            41.961755,
            9.816110,
            2.0,
            0.5,
            "PRP000000001",
        ],
        [
            1,
            1,
            1,
            1,
            1,
            1,
            90,
            0,
            1,
            1,
            0.0,
            42.728752,
            9.888968,
            1.0,
            0.5,
            "PRP000000001",
        ],
        [
            1,  # kper
            1,  # kstp
            1,  # imdl
            1,  # iprp
            1,  # irpt
            1,  # ilay
            100,  # icell
            0,  # izone
            5,  # istatus
            3,  # ireason
            0.0,  # trelease
            42.728752,  # t
            9.888968,  # x
            1.0,  # y
            0.5,  # z
            "PRP000000001",  # name
        ],
    ],
    columns=PRT_PATHLINE_DTYPE.names,
)
MP7_TEST_PATHLINES = pd.DataFrame.from_records(
    [
        [
            1,  # particleid
            1,  # particlegroup
            1,  # sequencenumber
            1,  # particleidloc
            0.0,  # time
            1.0,  # x
            2.0,  # y
            3.0,  # z
            1,  # k
            1,  # node
            0.1,  # xloc
            0.1,  # yloc
            0.1,  # zloc
            1,  # stressperiod
            1,  # timestep
        ],
        [
            1,
            1,
            1,
            1,
            1.0,  # time
            2.0,  # x
            3.0,  # y
            4.0,  # z
            2,  # k
            2,  # node
            0.9,  # xloc
            0.9,  # yloc
            0.9,  # zloc
            1,  # stressperiod
            1,  # timestep
        ],
    ],
    columns=MP7_PATHLINE_DTYPE.names,
)
MP7_TEST_ENDPOINTS = pd.DataFrame.from_records(
    [
        [
            1,  # particleid
            1,  # particlegroup
            1,  # particleidloc
            2,  # status (terminated at boundary face)
            0.0,  # time0
            1.0,  # time
            1,  # node0
            1,  # k0
            0.1,  # xloc0
            0.1,  # yloc0
            0.1,  # zloc0
            0.0,  # x0
            1.0,  # y0
            2.0,  # z0
            1,  # zone0
            1,  # initialcellface
            5,  # node
            2,  # k
            0.9,  # xloc
            0.9,  # yloc
            0.9,  # zloc
            10.0,  # x
            11.0,  # y
            12.0,  # z
            2,  # zone
            2,  # cellface
        ],
        [
            2,  # particleid
            1,  # particlegroup
            2,  # particleidloc
            2,  # status (terminated at boundary face)
            0.0,  # time0
            2.0,  # time
            1,  # node0
            1,  # k0
            0.1,  # xloc0
            0.1,  # yloc0
            0.1,  # zloc0
            0.0,  # x0
            1.0,  # y0
            2.0,  # z0
            1,  # zone0
            1,  # initialcellface
            5,  # node
            2,  # k
            0.9,  # xloc
            0.9,  # yloc
            0.9,  # zloc
            10.0,  # x
            11.0,  # y
            12.0,  # z
            2,  # zone
            2,  # cellface
        ],
        [
            3,  # particleid
            1,  # particlegroup
            3,  # particleidloc
            2,  # status (terminated at boundary face)
            0.0,  # time0
            3.0,  # time
            1,  # node0
            1,  # k0
            0.1,  # xloc0
            0.1,  # yloc0
            0.1,  # zloc0
            0.0,  # x0
            1.0,  # y0
            2.0,  # z0
            1,  # zone0
            1,  # initialcellface
            5,  # node
            2,  # k
            0.9,  # xloc
            0.9,  # yloc
            0.9,  # zloc
            10.0,  # x
            11.0,  # y
            12.0,  # z
            2,  # zone
            2,  # cellface
        ],
    ],
    columns=MP7_ENDPOINT_DTYPE.names,
)


@pytest.mark.parametrize("dataframe", [True, False])
def test_to_mp7_pathlines(dataframe):
    prt_pls = (
        PRT_TEST_PATHLINES if dataframe else PRT_TEST_PATHLINES.to_records(index=False)
    )
    mp7_pls = to_mp7_pathlines(prt_pls)
    assert (
        type(prt_pls) == type(mp7_pls) == (pd.DataFrame if dataframe else np.recarray)
    )
    assert len(mp7_pls) == 10
    assert set(
        dict(mp7_pls.dtypes).keys() if dataframe else mp7_pls.dtype.names
    ) == set(MP7_PATHLINE_DTYPE.names)


@pytest.mark.parametrize("dataframe", [True, False])
def test_to_mp7_pathlines_empty(dataframe):
    mp7_pls = to_mp7_pathlines(
        pd.DataFrame.from_records([], columns=PRT_PATHLINE_DTYPE.names)
        if dataframe
        else np.recarray((0,), dtype=PRT_PATHLINE_DTYPE)
    )
    assert mp7_pls.empty if dataframe else mp7_pls.size == 0
    if dataframe:
        mp7_pls = mp7_pls.to_records(index=False)
    assert mp7_pls.dtype == MP7_PATHLINE_DTYPE


@pytest.mark.parametrize("dataframe", [True, False])
def test_to_mp7_pathlines_noop(dataframe):
    prt_pls = (
        MP7_TEST_PATHLINES if dataframe else MP7_TEST_PATHLINES.to_records(index=False)
    )
    mp7_pls = to_mp7_pathlines(prt_pls)
    assert (
        type(prt_pls) == type(mp7_pls) == (pd.DataFrame if dataframe else np.recarray)
    )
    assert len(mp7_pls) == 2
    assert set(
        dict(mp7_pls.dtypes).keys() if dataframe else mp7_pls.dtype.names
    ) == set(MP7_PATHLINE_DTYPE.names)
    assert np.array_equal(
        mp7_pls if dataframe else pd.DataFrame(mp7_pls), MP7_TEST_PATHLINES
    )


@pytest.mark.parametrize("dataframe", [True, False])
def test_to_mp7_endpoints(dataframe):
    mp7_eps = to_mp7_endpoints(
        PRT_TEST_PATHLINES if dataframe else PRT_TEST_PATHLINES.to_records(index=False)
    )
    assert len(mp7_eps) == 1
    assert np.isclose(mp7_eps.time[0], PRT_TEST_PATHLINES.t.max())
    assert set(
        dict(mp7_eps.dtypes).keys() if dataframe else mp7_eps.dtype.names
    ) == set(MP7_ENDPOINT_DTYPE.names)


@pytest.mark.parametrize("dataframe", [True, False])
def test_to_mp7_endpoints_empty(dataframe):
    mp7_eps = to_mp7_endpoints(
        pd.DataFrame.from_records([], columns=PRT_PATHLINE_DTYPE.names)
        if dataframe
        else np.recarray((0,), dtype=PRT_PATHLINE_DTYPE)
    )
    assert mp7_eps.empty if dataframe else mp7_eps.size == 0
    if dataframe:
        mp7_eps = mp7_eps.to_records(index=False)
    assert mp7_eps.dtype == MP7_ENDPOINT_DTYPE


@pytest.mark.parametrize("dataframe", [True, False])
def test_to_mp7_endpoints_noop(dataframe):
    """Test a recarray or dataframe which already contains MP7 endpoint data"""
    mp7_eps = to_mp7_endpoints(
        MP7_TEST_ENDPOINTS if dataframe else MP7_TEST_ENDPOINTS.to_records(index=False)
    )
    assert np.array_equal(
        mp7_eps if dataframe else pd.DataFrame(mp7_eps), MP7_TEST_ENDPOINTS
    )


@pytest.mark.parametrize("dataframe", [True, False])
def test_to_prt_pathlines_roundtrip(dataframe):
    mp7_pls = to_mp7_pathlines(
        PRT_TEST_PATHLINES if dataframe else PRT_TEST_PATHLINES.to_records(index=False)
    )
    prt_pls = to_prt_pathlines(mp7_pls)
    if not dataframe:
        prt_pls = pd.DataFrame(prt_pls)
    assert np.allclose(
        PRT_TEST_PATHLINES.drop(
            ["imdl", "iprp", "irpt", "name", "istatus", "ireason"],
            axis=1,
        ),
        prt_pls.drop(
            ["imdl", "iprp", "irpt", "name", "istatus", "ireason"],
            axis=1,
        ),
    )


@pytest.mark.parametrize("dataframe", [True, False])
def test_to_prt_pathlines_roundtrip_empty(dataframe):
    mp7_pls = to_mp7_pathlines(
        pd.DataFrame.from_records([], columns=PRT_PATHLINE_DTYPE.names)
        if dataframe
        else np.recarray((0,), dtype=PRT_PATHLINE_DTYPE)
    )
    prt_pls = to_prt_pathlines(mp7_pls)
    assert mp7_pls.empty if dataframe else mp7_pls.size == 0
    assert prt_pls.empty if dataframe else mp7_pls.size == 0
    assert set(
        dict(mp7_pls.dtypes).keys() if dataframe else mp7_pls.dtype.names
    ) == set(MP7_PATHLINE_DTYPE.names)
