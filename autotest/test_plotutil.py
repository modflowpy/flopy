import numpy as np
import pandas as pd
import pytest

from flopy.plot.plotutil import (
    to_mp7_endpoints,
    to_mp7_pathlines,
    to_prt_pathlines,
)

# test PRT-MP7 pathline conversion functions
# todo: define fields in a single location and reference from here
# todo: support optional grid parameter to conversion functions


prt_pl_cols = []


mp7_pl_cols = [
    "particleid",
    "particlegroup",
    "sequencenumber",
    "particleidloc",
    "time",
    "xloc",
    "yloc",
    "zloc",
    "x",
    "y",
    "z",
    "node",
    "k",
    "stressperiod",
    "timestep",
]


mp7_ep_cols = [
    "particleid",
    "particlegroup",
    "particleidloc",
    "time",
    "time0",
    "xloc",
    "xloc0",
    "yloc",
    "yloc0",
    "zloc",
    "zloc0",
    "x0",
    "y0",
    "z0",
    "x",
    "y",
    "z",
    "node",
    "node0",
    "k",
    "k0",
    "zone",
    "zone0",
    "initialcellface",
    "cellface",
    "status",
]


pls = pd.DataFrame.from_records(
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
            1,
            1,
            1,
            1,
            1,
            1,
            100,
            0,
            5,
            3,
            0.0,
            42.728752,
            9.888968,
            1.0,
            0.5,
            "PRP000000001",
        ],
    ],
    columns=[
        "kper",
        "kstp",
        "imdl",
        "iprp",
        "irpt",
        "ilay",
        "icell",
        "izone",
        "istatus",
        "ireason",
        "trelease",
        "t",
        "x",
        "y",
        "z",
        "name",
    ],
)


@pytest.mark.parametrize("dataframe", [True, False])
def test_to_mp7_pathlines(dataframe):
    inp_pls = pls if dataframe else pls.to_records(index=False)
    mp7_pls = to_mp7_pathlines(inp_pls)
    assert (
        type(inp_pls)
        == type(mp7_pls)
        == (pd.DataFrame if dataframe else np.recarray)
    )
    assert len(mp7_pls) == 10
    assert set(
        dict(mp7_pls.dtypes).keys() if dataframe else mp7_pls.dtype.names
    ) == set(mp7_pl_cols)


@pytest.mark.parametrize("dataframe", [True, False])
def test_to_mp7_endpoints(dataframe):
    inp_pls = pls if dataframe else pls.to_records(index=False)
    mp7_eps = to_mp7_endpoints(inp_pls)
    assert len(mp7_eps) == 1
    assert set(
        dict(mp7_eps.dtypes).keys() if dataframe else mp7_eps.dtype.names
    ) == set(mp7_ep_cols)


def test_to_prt_pathlines_roundtrip():
    inp_pls = pls
    mp7_pls = to_mp7_pathlines(inp_pls)
    prt_pls = to_prt_pathlines(mp7_pls)
    inp_pls.drop(
        ["imdl", "iprp", "irpt", "name", "istatus", "ireason"],
        axis=1,
        inplace=True,
    )
    prt_pls.drop(
        ["imdl", "iprp", "irpt", "name", "istatus", "ireason"],
        axis=1,
        inplace=True,
    )
    assert np.allclose(inp_pls, prt_pls)
