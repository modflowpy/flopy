"""Tests for the SWR1 Process package (flopy.modflow.ModflowSwr1)."""

import filecmp
import shutil

import numpy as np
import pytest
from modflow_devtools.markers import requires_exe

from flopy.mbase import run_model
from flopy.modflow import (
    Modflow,
    ModflowBas,
    ModflowDis,
    ModflowGhb,
    ModflowNwt,
    ModflowOc,
    ModflowSfr2,
    ModflowSwr1,
    ModflowUpw,
)
from flopy.modflow.mfswr1 import SwrExternal
from flopy.utils import SwrBudget, SwrExchange, SwrStage

SAMPLE_OUTPUTS = (
    "SWRSample05.hds",
    "SWRSample05_QAQ.bin",
    "SWRSample05_GroupFlow.flow",
    "SWRSample05_Stage.csv",
    "SWRSample05.riv",
)


@pytest.fixture
def sample_ws(function_tmpdir, example_data_path):
    ws = function_tmpdir / "swr_sample05"
    shutil.copytree(example_data_path / "swr_sample05", ws)
    return ws


def load_sample(ws, nam="SWRSample05.nam"):
    return Modflow.load(
        nam,
        model_ws=ws,
        version="mfnwt",
        check=False,
        load_only=["dis", "bas6", "swr"],
    )


# --------------------------------------------------------------------------
# USGS SWR test problem 5
# --------------------------------------------------------------------------
def test_load_usgs_sample(sample_ws):
    m = load_sample(sample_ws)
    swr = m.get_package("SWR")
    assert isinstance(swr, ModflowSwr1)
    assert swr.nreaches == 19
    assert len(swr.stress_period_data) == m.nper
    # OPEN/CLOSE references and the STAGE time series are kept
    rain0 = swr.stress_period_data[0]["rain"]
    assert isinstance(rain0, SwrExternal)
    assert rain0.fname.endswith("Rainfall_001.ref")
    assert swr.tabfiles[0]["data"].shape[1] == 2

    swr.fn_path = str(sample_ws / "roundtrip.swr")
    swr.write_file()
    text = (sample_ws / "roundtrip.swr").read_text()
    assert text.count("OPEN/CLOSE") >= m.nper

    # loading the written file and writing it again gives the same text
    swr2 = ModflowSwr1.load(str(sample_ws / "roundtrip.swr"), m)
    swr2.fn_path = str(sample_ws / "again.swr")
    swr2.write_file()
    again = (sample_ws / "again.swr").read_text()
    assert again.splitlines()[1:] == text.splitlines()[1:]


@pytest.mark.slow
@requires_exe("mfnwt")
def test_usgs_sample_roundtrip_identical_results(sample_ws):
    """load -> write -> run gives byte-identical results to the original."""
    success, buff = run_model("mfnwt", "SWRSample05.nam", model_ws=sample_ws)
    assert success, buff[-20:]
    ref = sample_ws / "ref_results"
    ref.mkdir()
    for f in SAMPLE_OUTPUTS:
        shutil.move(sample_ws / f, ref / f)

    m = load_sample(sample_ws)
    swr = m.get_package("SWR")
    swr.fn_path = str(sample_ws / "roundtrip.swr")
    swr.write_file()
    nam = (sample_ws / "SWRSample05.nam").read_text()
    (sample_ws / "roundtrip.nam").write_text(
        nam.replace("SWRSample05.swr", "roundtrip.swr")
    )
    success, buff = run_model("mfnwt", "roundtrip.nam", model_ws=sample_ws)
    assert success, buff[-20:]
    for f in SAMPLE_OUTPUTS:
        assert filecmp.cmp(ref / f, sample_ws / f, shallow=False), f


# --------------------------------------------------------------------------
# model built from scratch: diffusive-wave channel over an aquifer
# --------------------------------------------------------------------------
NCOL, NREACH = 20, 20
DX, WIDTH = 100.0, 10.0
Q_BASE, Q_PEAK = 5.0, 50.0


def hydrograph():
    t = np.arange(0.0, 3 * 86400.0 + 1, 1800.0)
    q = Q_BASE + (Q_PEAK - Q_BASE) * np.exp(-0.5 * ((t - 43200.0) / 10800.0) ** 2)
    return np.column_stack([t, q])


def build_channel_model(ws, binary_tab=False):
    m = Modflow("chan", version="mfnwt", exe_name="mfnwt", model_ws=ws)
    zbed = np.linspace(10.0, 8.1, NCOL)  # slope ~1e-3
    ModflowDis(
        m,
        nlay=1,
        nrow=1,
        ncol=NCOL,
        delr=DX,
        delc=500.0,
        top=zbed[None, :] + 3.0,
        botm=-20.0,
        nper=2,
        perlen=[86400.0, 86400.0],
        nstp=[24, 24],
        steady=[False, False],
        itmuni=1,
        lenuni=2,
    )
    ModflowBas(m, ibound=np.ones((1, 1, NCOL), int), strt=zbed[None, None, :] - 0.5)
    ModflowUpw(m, hk=1e-4, vka=1e-4, sy=0.2, ss=1e-5, laytyp=1)
    ModflowGhb(
        m,
        stress_period_data={0: [[0, 0, 0, 9.0, 1.0], [0, 0, NCOL - 1, 7.5, 1.0]]},
    )
    ModflowNwt(m, headtol=1e-4, fluxtol=10, maxiterout=200)
    ModflowOc(
        m,
        stress_period_data={
            (k, s): ["save head", "save budget"] for k in range(2) for s in range(24)
        },
    )

    rd = ModflowSwr1.get_empty_reach_data(NREACH)
    rd["iroutetype"] = 3  # diffusive wave
    rd["irgnum"] = rd["reach"]  # one reach per group
    rd["irch"] = 1
    rd["jrch"] = np.arange(1, NCOL + 1)
    rd["rlen"] = DX
    conn = {
        r: [c for c in (r - 1, r + 1) if 1 <= c <= NREACH] for r in range(1, NREACH + 1)
    }
    geometries = {
        r: {
            "igeotype": 1,
            "igcndop": 1,
            "gmanning": 0.035,
            "gwidth": WIDTH,
            "gbelev": float(zbed[r - 1]),
            "glk": 1e-6,
        }
        for r in range(1, NREACH + 1)
    }
    sp0 = ModflowSwr1.get_empty_spd()
    sp0.update(
        boundary=[(r, 1) for r in range(1, NREACH + 1)],
        geometry_assignment=[(r, r, 0.0) for r in range(1, NREACH + 1)],
        geometries=geometries,
        # critical-depth outflow boundary at the last reach
        structures={NREACH: [{"istrconn": 0, "istrtype": 2, "strlen": 0.0}]},
        stage=[(r, float(zbed[r - 1]) + 0.5) for r in range(1, NREACH + 1)],
        aux=[(r, float(r)) for r in range(1, NREACH + 1)],
        evap=[(r, 1e-8) for r in range(1, NREACH + 1)],
    )
    dro_map = np.arange(1, NCOL + 1, dtype=np.int32)[None, :]
    swr = ModflowSwr1(
        m,
        iswrpstg=-110,
        iswrpqaq=-111,
        iswrprgf=-112,
        iswrpqm=-113,
        auxiliary=["RKM"],
        options={
            "USE_TABFILES": None,
            "SAVE_SWROBSERVATIONS": -115,
            "SAVE_SWRDT": 116,
            "USE_DIRECT_RUNOFF": 117,
            "USE_UPSTREAM_WEIGHTING": None,
        },
        dlenconv=1.0,
        timeconv=1.0,
        rtini=30.0,
        rtmin=1.0,
        rtmax=600.0,
        rtprn=0.0,
        rtmult=1.2,
        ntmult=3,
        dmingrad=1e-12,
        dmindpth=1e-3,
        dmaxrai=0.0,
        dmaxstg=0.25,
        dmaxinf=0.0,
        isolver=2,
        nouter=50,
        ninner=30,
        ibt=1,
        tols=1e-6,
        tolr=1e-3,
        tola=0.0,
        dampss=1.0,
        damptr=1.0,
        iprswr=0,
        mutswr=2,
        ipc=2,
        reach_data=rd,
        connectivity=conn,
        tabfiles=[
            {
                "itab": 1,
                "ctabtype": "LATFLOW",
                "itabunit": -118 if binary_tab else 118,
                "cintp": "INTERPOLATE",
                "reaches": [1],
                "data": hydrograph(),
            }
        ],
        observations=[
            ("stg10", "STAGE", 10),
            ("q10", "FLOW", 10, 11),
            ("bf10", "BASEFLOW", 10, 0),
            ("dep20", "DEPTH", 20),
        ],
        stress_period_data={0: sp0},
        direct_runoff={
            0: {
                "map": dro_map,
                "mult": np.ones((1, NCOL)),
                "val": np.full((1, NCOL), 1e-9),
            }
        },
    )
    return m, swr


def test_stub_signature_still_works(function_tmpdir):
    m = Modflow(version="mfnwt", model_ws=function_tmpdir)
    swr = ModflowSwr1(m)
    assert m.get_package("SWR") is swr
    assert swr.nreaches == 0


def test_binary_tabfile_written_as_stream(function_tmpdir):
    """Binary tab files are raw float32 pairs (ACCESS='STREAM'), and a
    warning is issued because SWR 1.05 crashes on them (BACKSPACE on a
    negative unit)."""
    with pytest.warns(UserWarning, match="BACKSPACE"):
        m, swr = build_channel_model(function_tmpdir, binary_tab=True)
    m.write_input()
    raw = np.fromfile(function_tmpdir / "chan.swr.tab1", "<f4").reshape(-1, 2)
    assert np.allclose(raw, hydrograph(), rtol=1e-6)
    assert "DATA(BINARY)" in (function_tmpdir / "chan.nam").read_text()


@requires_exe("mfnwt")
def test_channel_model_runs(function_tmpdir):
    m, swr = build_channel_model(function_tmpdir)
    m.write_input()
    success, buff = m.run_model(silent=True)
    assert success, buff

    stg = SwrStage(function_tmpdir / "chan.swr.stage.bin")
    assert stg.get_ntimes() == 48
    qaq = SwrExchange(function_tmpdir / "chan.swr.qaq.bin")
    assert np.all(np.isfinite(qaq.get_data(idx=-1)["exchange"]))
    budget = SwrBudget(function_tmpdir / "chan.swr.rgflow.bin")
    assert budget.get_ntimes() == 48
    # flood wave passes: stage at reach 10 rises above its initial value
    ts = stg.get_ts(irec=9)
    assert ts["stage"].max() > ts["stage"][0] + 0.1

    # load the written file back and write it again: text must be identical
    m2 = Modflow.load(
        "chan.nam",
        model_ws=function_tmpdir,
        version="mfnwt",
        check=False,
        load_only=["dis", "bas6", "swr"],
    )
    swr2 = m2.get_package("SWR")
    assert isinstance(swr2, ModflowSwr1)
    assert swr2.tabfiles[0]["itabunit"] == 118
    assert swr2.observations == swr.observations
    assert set(swr2.direct_runoff) == {0}
    first = open(swr.fn_path).read()
    swr2.fn_path = str(function_tmpdir / "again.swr")
    swr2.write_file()
    assert open(swr2.fn_path).read().splitlines()[1:] == first.splitlines()[1:]


# --------------------------------------------------------------------------
# I/O roundtrip of every structure type and geometry option
# --------------------------------------------------------------------------
def build_all_structures_model(ws):
    m, swr = build_channel_model(ws)
    # SFR present -> ISFRSEG/ISFRRCH are written for unconnected structures
    # (the SFR2 constructor fails with a single reach, so use two)
    rd = ModflowSfr2.get_empty_reach_data(2)
    rd["j"], rd["iseg"], rd["ireach"], rd["rchlen"] = [0, 1], 1, [1, 2], 100.0
    rd["strtop"], rd["slope"], rd["strthick"], rd["strhc1"] = 9.0, 1e-3, 1.0, 1e-6
    sd = ModflowSfr2.get_empty_segment_data(1)
    sd["nseg"], sd["icalc"], sd["roughch"] = 1, 1, 0.03
    sd["width1"], sd["width2"], sd["flow"] = 5.0, 5.0, 0.1
    ModflowSfr2(m, nstrm=-2, nss=1, isfropt=1, reach_data=rd, segment_data={0: sd})
    sp1 = ModflowSwr1.get_empty_spd()
    sp1["geometry_assignment"] = [
        (1, 101, 0.0),
        (2, 102, 0.1),
        (3, 103, 0.0),
        (4, 104, -0.2),
        (5, 105, 0.0),
    ]
    sp1["geometries"] = {
        101: {
            "igeotype": 2,
            "igcndop": 0,
            "gmanning": 0.03,
            "gwidth": 5.0,
            "gbelev": 9.0,
            "gsslope": 2.0,
            "gcnd": 12.5,
        },
        102: {
            "igeotype": 3,
            "igcndop": 2,
            "gmanning": 0.04,
            "gcndln": 50.0,
            "xsec": [(0, 12), (5, 9), (10, 9), (15, 12)],
        },
        103: {
            "igeotype": 4,
            "igcndop": 3,
            "gmanning": 0.0,
            "glk": 1e-5,
            "gcndln": 25.0,
            "table": [
                (9, 0, 0, 0, 0),
                (10, 500, 12, 1000, 10),
                (11, 1500, 14, 1100, 22),
            ],
        },
        104: {
            "igeotype": 5,
            "igcndop": 1,
            "gmanning": 0.1,
            "glk": 1e-6,
            "getextd": 1.5,
        },
        105: {
            "igeotype": 1,
            "igcndop": 1,
            "gmanning": 0.03,
            "gwidth": 8.0,
            "gbelev": 8.5,
            "glk": 2e-6,
        },
    }
    sp1["structures"] = {
        1: [
            {
                "istrconn": 2,
                "istrtype": -2,
                "strinv": 9.1,
                "strlen": 40.0,
                "strlen2": 60.0,
            }
        ],
        2: [
            {"istrconn": 3, "istrtype": 4, "rating": [(9, 0), (10, 5), (11, 20)]},
            {
                "istrconn": 3,
                "istrtype": 5,
                "strcd": 0.6,
                "strcd2": 0.8,
                "strinv": 9.0,
                "strinv2": 8.9,
                "strwid": -2.0,
                "strwid2": 1.5,
                "strlen": 20.0,
                "strman": 0.013,
                "istrdir": 0,
            },
        ],
        3: [
            {
                "istrconn": 4,
                "istrtype": 7,
                "strcd": 0.6,
                "strcd2": 0.7,
                "strcd3": 0.5,
                "strinv": 8.8,
                "strwid": 4.0,
                "strval": 1.0,
                "istrdir": 1,
            },
            {
                "istrconn": 4,
                "istrtype": 10,
                "strcd": 0.6,
                "strcd2": 0.7,
                "strcd3": 0.5,
                "strinv": 8.8,
                "strwid": 4.0,
                "strval": 0.5,
                "istrdir": 0,
                "cstrotyp": "FLOW",
                "istrorch": 3,
                "istroqcon": 4,
                "cstrlo": "GE",
                "cstrcrit": 10.0,
                "strcritc": 0.5,
                "strrt": 0.01,
                "strmax": 3.0,
                "cstrval": "TABDATA2",
            },
        ],
        4: [
            {
                "istrconn": 5,
                "istrtype": 13,
                "strcd": 0.6,
                "strcd3": 0.5,
                "strwsmo": -0.1,
                "strval": 0.0,
                "istrdir": 0,
            }
        ],
        5: [
            {
                "istrconn": 0,
                "istrtype": 12,
                "strlen": 100.0,
                "strval": "TABDATA3",
                "isfrseg": 1,
                "isfrrch": 1,
            },
            {"istrconn": 0, "istrtype": 11, "isfrseg": 1, "isfrrch": 1},
            {
                "istrconn": 0,
                "istrtype": 1,
                "cstrotyp": "STAGE",
                "cstrcrit": "REACH6",
                "strmax": 2.0,
                "isfrseg": 0,
            },
            {
                "istrconn": 0,
                "istrtype": 3,
                "strval": 0.2,
                "isfrseg": 0,
                "cstrotyp": "STAGE",
                "istrorch": 5,
                "cstrlo": "LT",
                "cstrcrit": 9.5,
                "strcritc": 0.1,
                "strrt": 0.001,
                "strmax": 1.0,
            },
        ],
        6: [],
        7: None,
    }
    sp1["stage"] = np.full((1, NCOL), 9.5)  # item 14b grid
    sp1["rain"] = np.full((1, NCOL), 1e-8)  # item 7b grid
    sp1["latflow"] = [(3, 0.5), (4, -0.1)]  # item 9a list
    swr.stress_period_data[1] = sp1
    swr.tabfiles += [
        {"itab": 2, "ctabtype": "STRUCTURE", "itabunit": 119, "cintp": "NONE"},
        {"itab": 3, "ctabtype": "STRUCTURE", "itabunit": 120, "cintp": "AVERAGE"},
        {"itab": 4, "ctabtype": "TIME", "itabunit": 121, "cintp": "NONE"},
        {
            "itab": 5,
            "ctabtype": "RAIN",
            "itabunit": 122,
            "cintp": "NONE",
            "reaches": "ALL",
        },
    ]
    for iu in (119, 120, 121, 122):
        m.add_external(f"tab{iu}.dat", iu)
    swr.isolver, swr.ipc, swr.nlevels, swr.droptol = 3, 4, 7, 1e-4
    swr.ibt, swr.ibtprt = 5, 1
    swr.options["USE_FRACTIONAL_TOLR"] = None
    swr.ptolr = 0.01
    m.write_input()
    for iu in (119, 120, 121, 122):  # minimal tab files so units resolve
        (ws / f"tab{iu}.dat").write_text("0 1\n1e9 1\n")
    return m, swr


def test_all_structure_types_io_roundtrip(function_tmpdir):
    ws = function_tmpdir
    m, swr = build_all_structures_model(ws)
    m2 = Modflow.load(
        "chan.nam",
        model_ws=ws,
        version="mfnwt",
        check=False,
        load_only=["dis", "bas6", "sfr", "swr"],
    )
    swr2 = m2.get_package("SWR")
    s1 = swr2.stress_period_data[1]
    assert s1["structures"][2][1]["strwid2"] == 1.5
    assert s1["structures"][3][1]["cstrval"] == "TABDATA2"
    assert s1["structures"][5][0]["strval"] == "TABDATA3"
    assert s1["structures"][5][2]["cstrcrit"] == "REACH6"
    assert s1["structures"][7] is None and s1["structures"][6] == []
    assert s1["geometries"][103]["table"].shape == (3, 5)
    assert np.allclose(s1["stage"], 9.5) and s1["stage"].shape == (1, NCOL)
    assert swr2.ipc == 4 and swr2.ibtprt == 1
    assert swr2.ptolr == pytest.approx(0.01)
    first = open(swr.fn_path).read().splitlines()[1:]
    swr2.fn_path = str(ws / "again.swr")
    swr2.write_file()
    assert open(swr2.fn_path).read().splitlines()[1:] == first


@requires_exe("mfnwt")
def test_all_structure_types_accepted_by_mfnwt(function_tmpdir):
    """MODFLOW's own reader must accept every line written above."""
    ws = function_tmpdir
    m, swr = build_all_structures_model(ws)
    # only reading the input matters: run one short time step per period
    m.dis.perlen = [60.0, 60.0]
    m.dis.nstp = [1, 1]
    m.oc.stress_period_data = {(0, 0): ["save head"], (1, 0): ["save head"]}
    m.write_input()
    success, buff = m.run_model(silent=True)
    assert success, buff[-40:]


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def test_option_errors(function_tmpdir):
    m, swr = build_channel_model(function_tmpdir)
    with pytest.raises(ValueError):
        ModflowSwr1._normalise_options(
            {"USE_RCMREORDERING": None, "USE_RCMREORDERING_IF_IMPROVEMENT": None}
        )
    with pytest.raises(ValueError):
        ModflowSwr1._normalise_options({"SAVE_CONVERGENCE_HISTORY": -5})
    with pytest.raises(ValueError):
        ModflowSwr1._normalise_options({"NOT_AN_OPTION": None})
    swr.connectivity[1] = list(range(2, 80))  # line > 199 characters
    with pytest.raises(ValueError):
        swr.write_file()


def test_validate_hydraulic_consistency(function_tmpdir):
    m, swr = build_channel_model(function_tmpdir)
    assert swr.validate() == []
    swr.connectivity[5] = [4]  # 5 no longer lists 6, but 6 still lists 5
    sp0 = swr.stress_period_data[0]
    sp0["structures"][3] = [
        {
            "istrconn": 9,
            "istrtype": 6,
            "strcd": 0.6,
            "strcd3": 0.5,
            "strinv": 9,
            "strwid": 3,
            "strval": 0,
            "istrdir": 0,
        }
    ]
    sp0["structures"][4] = [
        {"istrconn": 0, "istrtype": 12, "strlen": 10, "strval": "TABDATA9"}
    ]
    sp0["geometries"][2] = {
        "igeotype": 2,
        "igcndop": 1,
        "gmanning": 0.03,
        "gwidth": 5.0,
        "gbelev": 9.0,
        "glk": 1e-6,
    }
    sp0["stage"] = [(1, 10.0)]
    errs = "\n".join(swr.validate())
    assert "not symmetric: 6 -> 5" in errs
    assert "ISTRCONN 9 is not connected to reach 3" in errs
    assert "TABDATA9 refers to an undefined table" in errs
    assert "geometry 2: missing gsslope" in errs
    assert "stage must be given for every reach" in errs
    with pytest.raises(ValueError, match="SWR1 input errors"):
        swr.write_file()


def test_zero_based_kij(function_tmpdir):
    """zero-based k, i, j give exactly the same file as one-based."""
    m1, s1 = build_channel_model(function_tmpdir / "a")
    m1.write_input()
    m2, s2 = build_channel_model(function_tmpdir / "b")
    s2.zero_based = True
    s2.reach_data["irch"] -= 1
    s2.reach_data["jrch"] -= 1
    s2.reach_data["krch"] -= 1
    m2.write_input()
    a = open(s1.fn_path).read().splitlines()[1:]
    b = open(s2.fn_path).read().splitlines()[1:]
    assert a == b
    s3 = ModflowSwr1.load(s1.fn_path, m1, zero_based=True)
    assert s3.reach_data["jrch"].min() == 0 and s3.zero_based


@requires_exe("mfnwt")
def test_external_files_run(function_tmpdir):
    """SwrExternal writes OPEN/CLOSE files that MODFLOW reads."""
    m, swr = build_channel_model(function_tmpdir)
    sp0 = swr.stress_period_data[0]
    sp0["stage"] = SwrExternal(np.array(sp0["stage"]), "ext/stage0.txt")
    sp0["evap"] = SwrExternal(np.full((1, NCOL), 1e-8), "ext/evap0.txt")
    m.write_input()
    swr_text = open(swr.fn_path).read()
    assert "OPEN/CLOSE ext/stage0.txt" in swr_text
    assert "OPEN/CLOSE ext/evap0.txt 1 (FREE) -1" in swr_text
    success, buff = m.run_model(silent=True)
    assert success, buff
    m2 = Modflow.load(
        "chan.nam",
        model_ws=function_tmpdir,
        version="mfnwt",
        check=False,
        load_only=["dis", "bas6", "swr"],
    )
    sp = m2.get_package("SWR").stress_period_data[0]
    assert sp["stage"].fname == "ext/stage0.txt"
    assert sp["evap"].fname == "ext/evap0.txt"
