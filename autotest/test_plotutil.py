import numpy as np
import pandas as pd
import pytest

from flopy.plot.plotutil import (
    to_mp7_endpoints,
    to_mp7_pathlines,
    to_prt_pathlines,
)
from flopy.utils.modpathfile import (
    EndpointFile as MpEndpointFile,
)
from flopy.utils.modpathfile import (
    PathlineFile as MpPathlineFile,
)
from flopy.utils.prtfile import PathlineFile as PrtPathlineFile

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
    columns=PrtPathlineFile.dtypes["base"].names,
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
    columns=MpPathlineFile.dtypes[7].names,
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
    columns=MpEndpointFile.dtypes[7].names,
)


@pytest.mark.parametrize("dataframe", [True, False])
@pytest.mark.parametrize("source", ["prt"])  # , "mp3", "mp5", "mp6"])
def test_to_mp7_pathlines(dataframe, source):
    if source == "prt":
        pls = (
            PRT_TEST_PATHLINES
            if dataframe
            else PRT_TEST_PATHLINES.to_records(index=False)
        )
    elif source == "mp3":
        pass
    elif source == "mp5":
        pass
    elif source == "mp7":
        pass
    mp7_pls = to_mp7_pathlines(pls)
    assert (
        type(pls)
        == type(mp7_pls)
        == (pd.DataFrame if dataframe else np.recarray)
    )
    assert len(mp7_pls) == 10
    assert set(
        dict(mp7_pls.dtypes).keys() if dataframe else mp7_pls.dtype.names
    ) == set(MpPathlineFile.dtypes[7].names)


@pytest.mark.parametrize("dataframe", [True, False])
@pytest.mark.parametrize("source", ["prt"])  # , "mp3", "mp5", "mp6"])
def test_to_mp7_pathlines_empty(dataframe, source):
    if source == "prt":
        pls = to_mp7_pathlines(
            pd.DataFrame.from_records(
                [], columns=PrtPathlineFile.dtypes["base"].names
            )
            if dataframe
            else np.recarray((0,), dtype=PrtPathlineFile.dtypes["base"])
        )
    elif source == "mp3":
        pass
    elif source == "mp5":
        pass
    elif source == "mp7":
        pass
    assert pls.empty if dataframe else pls.size == 0
    if dataframe:
        pls = pls.to_records(index=False)
    assert pls.dtype == MpPathlineFile.dtypes[7]


@pytest.mark.parametrize("dataframe", [True, False])
def test_to_mp7_pathlines_noop(dataframe):
    pls = (
        MP7_TEST_PATHLINES
        if dataframe
        else MP7_TEST_PATHLINES.to_records(index=False)
    )
    mp7_pls = to_mp7_pathlines(pls)
    assert (
        type(pls)
        == type(mp7_pls)
        == (pd.DataFrame if dataframe else np.recarray)
    )
    assert len(mp7_pls) == 2
    assert set(
        dict(mp7_pls.dtypes).keys() if dataframe else mp7_pls.dtype.names
    ) == set(MpPathlineFile.dtypes[7].names)
    assert np.array_equal(
        mp7_pls if dataframe else pd.DataFrame(mp7_pls), MP7_TEST_PATHLINES
    )


@pytest.mark.parametrize("dataframe", [True, False])
@pytest.mark.parametrize("source", ["prt"])  # , "mp3", "mp5", "mp6"])
def test_to_mp7_endpoints(dataframe, source):
    if source == "prt":
        eps = to_mp7_endpoints(
            PRT_TEST_PATHLINES
            if dataframe
            else PRT_TEST_PATHLINES.to_records(index=False)
        )
    elif source == "mp3":
        pass
    elif source == "mp5":
        pass
    elif source == "mp6":
        pass
    assert len(eps) == 1
    assert np.isclose(eps.time[0], PRT_TEST_PATHLINES.t.max())
    assert set(
        dict(eps.dtypes).keys() if dataframe else eps.dtype.names
    ) == set(MpEndpointFile.dtypes[7].names)


@pytest.mark.parametrize("dataframe", [True, False])
@pytest.mark.parametrize("source", ["prt"])  # , "mp3", "mp5", "mp6"])
def test_to_mp7_endpoints_empty(dataframe, source):
    eps = to_mp7_endpoints(
        pd.DataFrame.from_records(
            [], columns=PrtPathlineFile.dtypes["base"].names
        )
        if dataframe
        else np.recarray((0,), dtype=PrtPathlineFile.dtypes["base"])
    )
    assert eps.empty if dataframe else eps.size == 0
    if dataframe:
        eps = eps.to_records(index=False)
    assert eps.dtype == MpEndpointFile.dtypes[7]


@pytest.mark.parametrize("dataframe", [True, False])
def test_to_mp7_endpoints_noop(dataframe):
    """Test a recarray or dataframe which already contains MP7 endpoint data"""
    eps = to_mp7_endpoints(
        MP7_TEST_ENDPOINTS
        if dataframe
        else MP7_TEST_ENDPOINTS.to_records(index=False)
    )
    assert np.array_equal(
        eps if dataframe else pd.DataFrame(eps), MP7_TEST_ENDPOINTS
    )


@pytest.mark.parametrize("dataframe", [True, False])
@pytest.mark.parametrize("source", ["prt"])  # , "mp3", "mp5", "mp6"])
def test_to_prt_pathlines_roundtrip(dataframe, source):
    if source == "prt":
        pls = to_mp7_pathlines(
            PRT_TEST_PATHLINES
            if dataframe
            else PRT_TEST_PATHLINES.to_records(index=False)
        )
    elif source == "mp3":
        pass
    elif source == "mp5":
        pass
    elif source == "mp6":
        pass
    prt_pls = to_prt_pathlines(pls)
    if not dataframe:
        prt_pls = pd.DataFrame(prt_pls)
    # import pdb; pdb.set_trace()
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
@pytest.mark.parametrize("source", ["prt"])  # , "mp3", "mp5", "mp6"])
def test_to_prt_pathlines_roundtrip_empty(dataframe, source):
    if source == "prt":
        pls = to_mp7_pathlines(
            pd.DataFrame.from_records(
                [], columns=PrtPathlineFile.dtypes["base"].names
            )
            if dataframe
            else np.recarray((0,), dtype=PrtPathlineFile.dtypes["base"])
        )
    elif source == "mp3":
        pass
    elif source == "mp5":
        pass
    elif source == "mp6":
        pass
    prt_pls = to_prt_pathlines(pls)
    assert pls.empty if dataframe else pls.size == 0
    assert prt_pls.empty if dataframe else pls.size == 0
    assert set(
        dict(pls.dtypes).keys() if dataframe else pls.dtype.names
    ) == set(MpPathlineFile.dtypes[7].names)
