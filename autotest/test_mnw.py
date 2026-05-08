"""
test MNW1 and MNW2 packages
"""

import os
import shutil

import numpy as np
import pandas as pd
import pytest
from modflow_devtools.markers import requires_pkg

from flopy.modflow import Mnw, Modflow, ModflowDis, ModflowMnw2


@pytest.fixture
def mnw2_examples_path(example_data_path):
    return example_data_path / "mnw2_examples"


@pytest.fixture
def mnw1_path(example_data_path):
    return example_data_path / "mf2005_test"


def test_load(function_tmpdir, mnw2_examples_path):
    """t027 test load of MNW2 Package"""
    # load in the test problem (1 well, 3 stress periods)
    m = Modflow.load(
        "MNW2-Fig28.nam",
        model_ws=mnw2_examples_path,
        verbose=True,
        forgive=False,
    )
    ws = function_tmpdir
    m.change_model_ws(ws)
    assert m.has_package("MNW2")
    assert m.has_package("MNWI")

    # load a real mnw2 package from a steady state model (multiple wells)
    m2 = Modflow("br", model_ws=ws)
    path = mnw2_examples_path
    mnw2_2 = ModflowMnw2.load(path / "BadRiver_cal.mnw2", m2)
    mnw2_2.write_file(ws / "brtest.mnw2")

    m3 = Modflow("br", model_ws=ws)
    mnw2_3 = ModflowMnw2.load(ws / "brtest.mnw2", m3)
    mnw2_2.node_data.sort(order="wellid")
    mnw2_3.node_data.sort(order="wellid")
    assert np.array_equal(mnw2_2.node_data, mnw2_3.node_data)
    assert (
        mnw2_2.stress_period_data[0].qdes - mnw2_3.stress_period_data[0].qdes
    ).max() < 0.01
    assert (
        np.abs(
            mnw2_2.stress_period_data[0].qdes - mnw2_3.stress_period_data[0].qdes
        ).min()
        < 0.01
    )


def test_mnw1_load_write(function_tmpdir, mnw1_path):
    m = Modflow.load(
        "mnw1.nam",
        model_ws=mnw1_path,
        load_only=["mnw1"],
        verbose=True,
        forgive=False,
    )
    ws = function_tmpdir
    assert m.has_package("MNW1")
    assert m.mnw1.mxmnw == 120
    for i in range(3):
        assert len(m.mnw1.stress_period_data[i]) == 17
        assert len(np.unique(m.mnw1.stress_period_data[i]["mnw_no"])) == 15
        assert len(set(m.mnw1.stress_period_data[i]["label"])) == 4

    shutil.copy(mnw1_path / "mnw1.nam", ws)
    shutil.copy(mnw1_path / "mnw1.dis", ws)
    shutil.copy(mnw1_path / "mnw1.bas", ws)

    m.mnw1.fn_path = ws / "mnw1.mnw"
    m.mnw1.write_file()

    m2 = Modflow.load(
        "mnw1.nam",
        model_ws=ws,
        load_only=["mnw1"],
        verbose=True,
        forgive=False,
    )
    for k, v in m.mnw1.stress_period_data.data.items():
        assert np.array_equal(v, m2.mnw1.stress_period_data[k])


@pytest.mark.parametrize("dataframe", [True, False])
def test_make_package(function_tmpdir, dataframe):
    """t027 test make MNW2 Package"""
    ws = function_tmpdir
    m4 = Modflow("mnw2example", model_ws=ws)
    dis = ModflowDis(nrow=5, ncol=5, nlay=3, nper=3, top=10, botm=0, model=m4)

    # make the package from the tables (ztop, zbotm format)
    node_data = np.array(
        [
            (0, 1, 1, 9.5, 7.1, "well1", "skin", -1, 0, 0, 0, 1.0, 2.0, 5.0, 6.2),
            (1, 1, 1, 7.1, 5.1, "well1", "skin", -1, 0, 0, 0, 0.5, 2.0, 5.0, 6.2),
            (2, 3, 3, 9.1, 3.7, "well2", "skin", -1, 0, 0, 0, 1.0, 2.0, 5.0, 4.1),
        ],
        dtype=[
            ("index", "<i8"),
            ("i", "<i8"),
            ("j", "<i8"),
            ("ztop", "<f8"),
            ("zbotm", "<f8"),
            ("wellid", "O"),
            ("losstype", "O"),
            ("pumploc", "<i8"),
            ("qlimit", "<i8"),
            ("ppflag", "<i8"),
            ("pumpcap", "<i8"),
            ("rw", "<f8"),
            ("rskin", "<f8"),
            ("kskin", "<f8"),
            ("zpump", "<f8"),
        ],
    ).view(np.recarray)

    stress_period_data = {
        0: np.array(
            [(0, 0, "well1", 0), (1, 0, "well2", 0)],
            dtype=[("index", "<i8"), ("per", "<i8"), ("wellid", "O"), ("qdes", "<i8")],
        ).view(np.recarray),
        1: np.array(
            [(2, 1, "well1", 100), (3, 1, "well2", 1000)],
            dtype=[("index", "<i8"), ("per", "<i8"), ("wellid", "O"), ("qdes", "<i8")],
        ).view(np.recarray),
    }

    if dataframe:
        node_data = pd.DataFrame(node_data)

    mnw2_4 = ModflowMnw2(
        model=m4,
        mnwmax=2,
        nodtot=3,
        node_data=node_data,
        stress_period_data=stress_period_data,
        itmp=[2, 2, -1],
        # reuse second per pumping for last stress period
    )
    m4.write_input()

    well1 = mnw2_4.node_data[mnw2_4.node_data["wellid"] == "well1"]
    assert isinstance(well1, np.recarray)
    assert np.all(np.diff(well1["ztop"]) < 0)
    assert np.all(np.diff(well1["zbotm"]) < 0)

    # make the package from the tables (k, i, j format)
    node_data = np.array(
        [
            (0, 3, 1, 1, "well1", "skin", -1, 0, 0, 0, 1.0, 2.0, 5.0, 6.2),
            (1, 2, 1, 1, "well1", "skin", -1, 0, 0, 0, 0.5, 2.0, 5.0, 6.2),
            (2, 1, 3, 3, "well2", "skin", -1, 0, 0, 0, 1.0, 2.0, 5.0, 4.1),
        ],
        dtype=[
            ("index", "<i8"),
            ("k", "<i8"),
            ("i", "<i8"),
            ("j", "<i8"),
            ("wellid", "O"),
            ("losstype", "O"),
            ("pumploc", "<i8"),
            ("qlimit", "<i8"),
            ("ppflag", "<i8"),
            ("pumpcap", "<i8"),
            ("rw", "<f8"),
            ("rskin", "<f8"),
            ("kskin", "<f8"),
            ("zpump", "<f8"),
        ],
    ).view(np.recarray)

    stress_period_data = {
        0: np.array(
            [(0, 0, "well1", 0), (1, 0, "well2", 0)],
            dtype=[("index", "<i8"), ("per", "<i8"), ("wellid", "O"), ("qdes", "<i8")],
        ).view(np.recarray),
        1: np.array(
            [(2, 1, "well1", 100), (3, 1, "well2", 1000)],
            dtype=[("index", "<i8"), ("per", "<i8"), ("wellid", "O"), ("qdes", "<i8")],
        ).view(np.recarray),
    }

    if dataframe:
        node_data = pd.DataFrame(node_data)

    mnw2_4 = ModflowMnw2(
        model=m4,
        mnwmax=2,
        nodtot=3,
        node_data=node_data,
        stress_period_data=stress_period_data,
        itmp=[2, 2, -1],
        # reuse second per pumping for last stress period
    )
    well1 = mnw2_4.node_data[mnw2_4.node_data["wellid"] == "well1"]
    assert isinstance(well1, np.recarray)
    assert np.all(np.diff(well1["k"]) > 0)
    spd = m4.mnw2.stress_period_data[0]
    inds = spd.k, spd.i, spd.j
    assert np.array_equal(np.array(inds).transpose(), np.array([(2, 1, 1), (1, 3, 3)]))
    m4.write_input()

    # make the package from the objects
    # reuse second per pumping for last stress period
    mnw2fromobj = ModflowMnw2(model=m4, mnwmax=2, mnw=mnw2_4.mnw, itmp=[2, 2, -1])
    # verify that the two input methods produce the same results
    assert np.array_equal(
        mnw2_4.stress_period_data[1], mnw2fromobj.stress_period_data[1]
    )
    # verify that loaded package is the same
    m5 = Modflow("mnw2example", model_ws=ws)
    dis = ModflowDis(nrow=5, ncol=5, nlay=3, nper=3, top=10, botm=0, model=m5)
    mnw2_5 = ModflowMnw2.load(mnw2_4.fn_path, m5)
    assert np.array_equal(mnw2_4.stress_period_data[1], mnw2_5.stress_period_data[1])


@pytest.mark.parametrize("dataframe", [True, False])
def test_mnw2_create_file(function_tmpdir, dataframe):
    """
    Test for issue #556, Mnw2 crashed if wells have
    multiple node lengths
    """

    mf = Modflow("test_mfmnw2", exe_name="mf2005")
    ws = function_tmpdir
    wellids = [1, 2]
    nlayers = [2, 4]
    stress_period_data = pd.DataFrame([[0, 1]], columns=["per", "qdes"])
    wells = []
    for i in range(len(wellids)):
        node_data = pd.DataFrame(
            columns=[
                "k",
                "i",
                "j",
                "ztop",
                "zbotm",
                "wellid",
                "losstype",
                "ppflag",
                "pumploc",
                "qlimit",
                "pumpcap",
                "qcut",
            ]
        )
        for j in range(nlayers[i]):
            node_data.loc[j, "k"] = j
            node_data.loc[j, "i"] = 1
            node_data.loc[j, "j"] = 1
            node_data.loc[j, "ztop"] = 0
            node_data.loc[j, "zbotm"] = 0
            node_data.loc[j, "wellid"] = wellids[i]
            node_data.loc[j, "losstype"] = "SKIN"
            node_data.loc[j, "ppflag"] = 0
            node_data.loc[j, "pumploc"] = 0
            node_data.loc[j, "qlimit"] = 0
            node_data.loc[j, "pumpcap"] = 0
            node_data.loc[j, "qcut"] = 0

        wl = Mnw(
            wellids[i],
            nnodes=nlayers[i],
            nper=len(stress_period_data.index),
            node_data=(
                node_data.to_records(index=False) if not dataframe else node_data
            ),
            stress_period_data=(
                stress_period_data.to_records(index=False)
                if not dataframe
                else stress_period_data
            ),
        )

        wells.append(wl)

    mnw2 = ModflowMnw2(
        model=mf,
        mnwmax=len(wells),
        mnw=wells,
        itmp=list((np.ones(len(stress_period_data.index)) * len(wellids)).astype(int)),
    )

    if len(mnw2.node_data) != 6:
        raise AssertionError("Node data not properly set")

    mnw2.write_file(ws / "ndata.mnw2")


@requires_pkg("netCDF4")
@pytest.mark.slow
def test_export(function_tmpdir, mnw2_examples_path):
    """t027 test export of MNW2 Package to netcdf files"""
    import netCDF4

    ws = function_tmpdir
    m = Modflow.load(
        "MNW2-Fig28.nam",
        model_ws=mnw2_examples_path,
        load_only=["dis", "bas6", "mnwi", "mnw2", "wel"],
        verbose=True,
        check=False,
    )

    # netDF4 tests
    # first with path as string
    fcw = m.wel.export(os.path.join(ws, "MNW2-Fig28_well.nc"))
    fcw.write()

    # with Path
    fcw = m.wel.export(ws / "MNW2-Fig28_well.nc")
    fcw.write()

    fpth = ws / "MNW2-Fig28.nc"
    # test context statement
    with m.mnw2.export(fpth):
        pass
    fpth = ws / "MNW2-Fig28.nc"
    nc = netCDF4.Dataset(fpth)
    assert np.array_equal(
        nc.variables["mnw2_qdes"][:, 0, 29, 40],
        np.array([0.0, -10000.0, -10000.0], dtype="float32"),
    )
    assert np.sum(nc.variables["mnw2_rw"][:, :, 29, 40]) - 5.1987 < 1e-4

    # TODO need to add shapefile test


def test_blank_lines(function_tmpdir):
    mnw2str = """3 50 0
EB-33 -3
SKIN -1 0 0 0
0.375 0.667 50
 0 -1 1 1
-1 -2 1 1
-2 -3 1 1
-3
eb-35 3
SKIN 0 0 0 0
0.375 1.0 50
1 1 1 83 24
2 1 1
3 1 1

EB-36 -2
SKIN -1 0 0 0
.375 1.0 50
 0 -1 1 1
-1 -2 1 1
-2

3
EB-33 -11229.2
EB-35 -534.72

eb-36 -534.72

"""
    ws = function_tmpdir
    fpth = ws / "mymnw2.mnw2"
    f = open(fpth, "w")
    f.write(mnw2str)
    f.close()

    disstr = """ 3 1 1 1 4 1
 0 0 0
constant 1
constant 1
constant  0 top of model
constant -1 bottom of layer 1
constant -2 bottom of layer 2
constant -3 bottom of layer 3
 1.  1 1. Tr    PERLEN NSTP TSMULT Ss/tr
"""

    fpth = ws / "mymnw2.dis"
    f = open(fpth, "w")
    f.write(disstr)
    f.close()

    namstr = """lst  101 mymnw2.lst
dis  102 mymnw2.dis
mnw2 103 mymnw2.mnw2"""

    fpth = ws / "mymnw2.nam"
    f = open(fpth, "w")
    f.write(namstr)
    f.close()

    m = Modflow.load("mymnw2.nam", model_ws=ws, verbose=True, check=False)
    mnw2 = m.mnw2
    wellids = ["eb-33", "eb-35", "eb-36"]
    rates = [np.float32(-11229.2), np.float32(-534.72), np.float32(-534.72)]

    wellids2 = sorted(mnw2.mnw.keys())
    emsg = "incorrect keys returned from load mnw2"
    assert wellids2 == wellids, emsg

    spd = mnw2.stress_period_data[0]

    wellids2 = list(spd["wellid"])
    emsg = "incorrect keys returned from load mnw2 stress period data"
    for wellid, wellid2 in zip(wellids, wellids2):
        emsg += f"\n    {wellid} -- {wellid2}"
    assert wellids2 == wellids, emsg

    rates2 = list(spd["qdes"])
    emsg = "incorrect qdes rates returned from load mnw2 stress period data"
    for rate, rate2 in zip(rates, rates2):
        emsg += f"\n    {rate} -- {rate2}"
    assert rates2 == rates, emsg


def test_make_well():
    w1 = Mnw(wellid="Case-1")
    assert w1.wellid == "case-1", "did not correctly convert well id to lower case"


def test_checks(mnw2_examples_path):
    """t027 test MNW2 Package checks in FloPy"""
    m = Modflow.load(
        "MNW2-Fig28.nam",
        model_ws=mnw2_examples_path,
        load_only=["dis", "bas6", "mnwi", "wel"],
        verbose=True,
        check=False,
    )
    chk = m.check()
    assert "MNWI package present without MNW2 package." in ".".join(
        chk.summary_array.desc
    )


def test_mnw1_add_flag(function_tmpdir):
    """Test MNW1 ADD flag parsing and writing"""
    mnw1_str = """       120       -90       0     REFERENCE SP = 2
Well model will use SKIN
#
        2                                                 SP 1
        1         1         1      0      100    0.5        1        SITE:Well-A
        1         2         2      0      200    0.5        1
#                                Multi-node switch    Switch to specify Hlim         Auxiliary
        2 ADD
        1         1         1  -100.0     100    0.5     1   DD 50   1.e16   1         SITE:Well-A
        1         2         2  -200.0     200    0.5     1   DD 50   1.e16   1
#
        1
        1         1         1  -100.0     100    0.5     1   DD 50   1.e16   1         SITE:Well-A
"""  # noqa: E501
    ws = function_tmpdir
    fpth = ws / "test_add.mnw"
    with open(fpth, "w") as f:
        f.write(mnw1_str)

    # Create minimal model files
    disstr = """1 1 1 3 1 1
 0 0 0
constant 1
constant 1
constant  0 top of model
constant -1 bottom of layer 1
 1.  1 1. Tr
 1.  1 1. Tr
 1.  1 1. Tr
"""  # noqa: E501
    dis_path = ws / "test_add.dis"
    with open(dis_path, "w") as f:
        f.write(disstr)

    basstr = """         0         0     1     1     1     1     0     0
     1     1     1
        0.
"""  # noqa: E501
    bas_path = ws / "test_add.bas"
    with open(bas_path, "w") as f:
        f.write(basstr)

    namstr = """lst  101 test_add.lst
dis  102 test_add.dis
bas6 103 test_add.bas
mnw1 104 test_add.mnw"""
    nam_path = ws / "test_add.nam"
    with open(nam_path, "w") as f:
        f.write(namstr)

    # Load and test ADD flag parsing
    m = Modflow.load(
        "test_add.nam",
        model_ws=ws,
        load_only=["mnw1"],
        verbose=True,
        forgive=False,
    )

    # Verify add parameter was loaded correctly
    assert hasattr(m.mnw1, "add"), "MNW1 package missing 'add' attribute"
    assert len(m.mnw1.add) == 3, (
        f"Expected 3 stress periods in add, got {len(m.mnw1.add)}"
    )
    assert m.mnw1.add[0] is False, "SP 1: add should be False"
    assert m.mnw1.add[1] is True, "SP 2: add should be True"
    assert m.mnw1.add[2] is False, "SP 3: add should be False"

    # Write to new file and verify round-trip
    m.mnw1.fn_path = ws / "test_add_write.mnw"
    m.mnw1.write_file()

    # Create new model pointing to written file
    m2 = Modflow("test_add_write", model_ws=ws)
    m2.read_dis(ws / "test_add.dis", check=False)
    m2.read_bas6(ws / "test_add.bas", check=False)

    from flopy.modflow import ModflowMnw1

    mnw1_2 = ModflowMnw1.load(ws / "test_add_write.mnw", m2)

    # Verify add flag was preserved after write and reload
    assert hasattr(mnw1_2, "add"), "Reloaded MNW1 package missing 'add' attribute"
    assert mnw1_2.add[0] is False, "Reloaded SP 1: add should be False"
    assert mnw1_2.add[1] is True, "Reloaded SP 2: add should be True"
    assert mnw1_2.add[2] is False, "Reloaded SP 3: add should be False"


def test_mnw1_stress_period_no_wells(function_tmpdir):
    """Test MNW1 loading with stress periods that have no wells (itmp <= 0)"""
    # This tests the fix for handling empty stress periods in the load routine
    # File format: initial conditions, then for each SP: itmp, then well data (only if itmp > 0)
    mnw1_str = """       120       -90       0     REFERENCE SP = 1
Well model will use SKIN
#
        1                                                 Initial well
        1         1         1      0      100    0.5        1        SITE:Well-A
#                                Multi-node switch    Switch to specify Hlim         Auxiliary
        1
        1         1         1  -100.0     100    0.5     1   DD 50   1.e16   1         SITE:Well-A
#
        0
#
        1
        1         1         1  -100.0     100    0.5     1   DD 50   1.e16   1         SITE:Well-A
"""  # noqa: E501
    ws = function_tmpdir
    fpth = ws / "test_no_wells.mnw"
    with open(fpth, "w") as f:
        f.write(mnw1_str)

    # Create minimal model files
    disstr = """1 1 1 3 1 1
 0 0 0
constant 1
constant 1
constant  0 top of model
constant -1 bottom of layer 1
 1.  1 1. Tr
 1.  1 1. Tr
 1.  1 1. Tr
"""  # noqa: E501
    dis_path = ws / "test_no_wells.dis"
    with open(dis_path, "w") as f:
        f.write(disstr)

    basstr = """         0         0     1     1     1     1     0     0
     1     1     1
        0.
"""  # noqa: E501
    bas_path = ws / "test_no_wells.bas"
    with open(bas_path, "w") as f:
        f.write(basstr)

    namstr = """lst  101 test_no_wells.lst
dis  102 test_no_wells.dis
bas6 103 test_no_wells.bas
mnw1 104 test_no_wells.mnw"""
    nam_path = ws / "test_no_wells.nam"
    with open(nam_path, "w") as f:
        f.write(namstr)

    # Load the file - should not fail when SP 2 has no wells
    m = Modflow.load(
        "test_no_wells.nam",
        model_ws=ws,
        load_only=["mnw1"],
        verbose=True,
        forgive=False,
    )

    # Verify stress_period_data was loaded correctly
    assert m.mnw1.stress_period_data is not None, "stress_period_data is None"
    assert len(m.mnw1.stress_period_data) == 3, (
        f"Expected 3 stress periods, got {len(m.mnw1.stress_period_data)}"
    )

    # SP 1 should have wells
    assert len(m.mnw1.stress_period_data[0]) == 1, "SP 1 should have 1 well"

    # SP 2 should have no wells (empty record array)
    assert len(m.mnw1.stress_period_data[1]) == 0, "SP 2 should have 0 wells"

    # SP 3 should have wells
    assert len(m.mnw1.stress_period_data[2]) == 1, "SP 3 should have 1 well"

    # Write and reload to verify empty stress period is preserved
    m.mnw1.fn_path = ws / "test_no_wells_write.mnw"
    m.mnw1.write_file()

    # Create new model and reload
    m2 = Modflow("test_no_wells_write", model_ws=ws)
    m2.read_dis(ws / "test_no_wells.dis", check=False)
    m2.read_bas6(ws / "test_no_wells.bas", check=False)

    from flopy.modflow import ModflowMnw1

    mnw1_2 = ModflowMnw1.load(ws / "test_no_wells_write.mnw", m2)

    # Verify empty stress period is still empty after round-trip
    assert len(mnw1_2.stress_period_data[0]) == 1, "Reloaded SP 1 should have 1 well"
    assert len(mnw1_2.stress_period_data[1]) == 0, "Reloaded SP 2 should have 0 wells"
    assert len(mnw1_2.stress_period_data[2]) == 1, "Reloaded SP 3 should have 1 well"
