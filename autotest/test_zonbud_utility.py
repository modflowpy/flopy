import os

import numpy as np
import pandas as pd
import pytest
from modflow_devtools.markers import requires_exe, requires_pkg

from flopy.mf6 import MFSimulation
from flopy.utils import ZoneBudget, ZoneBudget6, ZoneFile6


@pytest.fixture
def loadpth(example_data_path):
    return example_data_path / "zonbud_examples"


@pytest.fixture
def cbc_f(loadpth):
    return loadpth / "freyberg.gitcbc"


@pytest.fixture
def zon_f(loadpth):
    return loadpth / "zonef_mlt.zbr"


@pytest.fixture
def zbud_f(loadpth):
    return loadpth / "freyberg_mlt.csv"


def read_zonebudget_file(fname):
    with open(fname) as f:
        lines = f.readlines()

    rows = []
    for line in lines:
        items = line.split(",")

        # Read time step information for this block
        if "Time Step" in line:
            kstp, kper, totim = (
                int(items[1]) - 1,
                int(items[3]) - 1,
                float(items[5]),
            )
            continue

        # Get names of zones
        elif "ZONE" in items[1]:
            zonenames = [i.strip() for i in items[1:-1]]
            zonenames = ["_".join(z.split()) for z in zonenames]
            continue

        # Set flow direction flag--inflow
        elif "IN" in items[1]:
            flow_dir = "FROM"
            continue

        # Set flow direction flag--outflow
        elif "OUT" in items[1]:
            flow_dir = "TO"
            continue

        # Get mass-balance information for this block
        elif (
            "Total" in items[0]
            or "IN-OUT" in items[0]
            or "Percent Error" in items[0]
        ):
            continue

        # End of block
        elif items[0] == "" and items[1] == "\n":
            continue

        record = f"{flow_dir}_" + "_".join(items[0].strip().split())
        if record.startswith(("FROM_", "TO_")):
            record = "_".join(record.split("_")[1:])
        vals = [float(i) for i in items[1:-1]]
        row = (
            totim,
            kstp,
            kper,
            record,
        ) + tuple(v for v in vals)
        rows.append(row)
    dtype_list = [
        ("totim", float),
        ("time_step", int),
        ("stress_period", int),
        ("name", "<U50"),
    ] + [(z, "<f8") for z in zonenames]
    dtype = np.dtype(dtype_list)
    return np.array(rows, dtype=dtype)


@pytest.mark.parametrize("rtol", [1e-2])
def test_compare2zonebudget(cbc_f, zon_f, zbud_f, rtol):
    """
    t039 Compare output from zonbud.exe to the budget calculated by zonbud
    utility using the multilayer transient freyberg model.
    """
    zba = read_zonebudget_file(zbud_f)
    zonenames = [n for n in zba.dtype.names if "ZONE" in n]
    times = np.unique(zba["totim"])

    zon = ZoneBudget.read_zone_file(zon_f)
    zb = ZoneBudget(cbc_f, zon, totim=times, verbose=False)
    fpa = zb.get_budget()

    for time in times:
        zb_arr = zba[zba["totim"] == time]
        fp_arr = fpa[fpa["totim"] == time]
        for name in fp_arr["name"]:
            r1 = np.where(zb_arr["name"] == name)
            r2 = np.where(fp_arr["name"] == name)
            if r1[0].shape[0] < 1 or r2[0].shape[0] < 1:
                continue
            if r1[0].shape[0] != r2[0].shape[0]:
                continue
            a1 = np.array([v for v in zb_arr[zonenames][r1[0]][0]])
            a2 = np.array([v for v in fp_arr[zonenames][r2[0]][0]])
            allclose = np.allclose(a1, a2, rtol)

            mxdiff = np.abs(a1 - a2).max()
            idxloc = np.argmax(np.abs(a1 - a2))
            # txt = '{}: {} - Max: {}  a1: {}  a2: {}'.format(time,
            #                                                 name,
            #                                                 mxdiff,
            #                                                 a1[idxloc],
            #                                                 a2[idxloc])
            # print(txt)
            s = f"Zonebudget arrays do not match at time {time} ({name}): {mxdiff}."
            assert allclose, s


def test_zonbud_get_record_names(cbc_f, zon_f):
    """
    t039 Test zonbud get_record_names method
    """
    zon = ZoneBudget.read_zone_file(zon_f)
    zb = ZoneBudget(cbc_f, zon, kstpkper=(0, 0))
    recnames = zb.get_record_names()
    assert len(recnames) > 0, "No record names returned."
    recnames = zb.get_record_names(stripped=True)
    assert len(recnames) > 0, "No record names returned."


def test_zonbud_aliases(cbc_f, zon_f):
    """
    t039 Test zonbud aliases
    """
    zon = ZoneBudget.read_zone_file(zon_f)
    aliases = {1: "Trey", 2: "Mike", 4: "Wilson", 0: "Carini"}
    zb = ZoneBudget(
        cbc_f, zon, kstpkper=(0, 1096), aliases=aliases, verbose=True
    )
    bud = zb.get_budget()
    assert bud[bud["name"] == "FROM_Mike"].shape[0] > 0, "No records returned."


def test_zonbud_to_csv(function_tmpdir, cbc_f, zon_f):
    """
    t039 Test zonbud export to csv file method
    """
    zon = ZoneBudget.read_zone_file(zon_f)
    zb = ZoneBudget(cbc_f, zon, kstpkper=[(0, 1094), (0, 1096)])
    f_out = function_tmpdir / "test.csv"
    zb.to_csv(f_out)
    with open(f_out) as f:
        lines = f.readlines()
    assert len(lines) > 0, "No data written to csv file."


def test_zonbud_math(cbc_f, zon_f):
    """
    t039 Test zonbud math methods
    """
    zon = ZoneBudget.read_zone_file(zon_f)
    cmd = ZoneBudget(cbc_f, zon, kstpkper=(0, 1096))
    cmd / 35.3147
    cmd * 12.0
    cmd + 1e6
    cmd - 1e6


def test_zonbud_copy(cbc_f, zon_f):
    """
    t039 Test zonbud copy
    """
    zon = ZoneBudget.read_zone_file(zon_f)
    cfd = ZoneBudget(cbc_f, zon, kstpkper=(0, 1096))
    cfd2 = cfd.copy()
    assert cfd is not cfd2, "Copied object is a shallow copy."


def test_zonbud_readwrite_zbarray(function_tmpdir):
    """
    t039 Test zonbud read write
    """
    x = np.random.randint(100, 200, size=(5, 150, 200))
    ZoneBudget.write_zone_file(function_tmpdir / "randint", x)
    ZoneBudget.write_zone_file(
        function_tmpdir / "randint", x, fmtin=35, iprn=2
    )
    z = ZoneBudget.read_zone_file(function_tmpdir / "randint")
    assert np.array_equal(x, z), "Input and output arrays do not match."


def test_dataframes(cbc_f, zon_f):
    zon = ZoneBudget.read_zone_file(zon_f)
    cmd = ZoneBudget(cbc_f, zon, totim=1095.0)
    df = cmd.get_dataframes()
    assert len(df) > 0, "Output DataFrames empty."


def test_get_budget(cbc_f, zon_f):
    zon = ZoneBudget.read_zone_file(zon_f)
    aliases = {1: "Trey", 2: "Mike", 4: "Wilson", 0: "Carini"}
    zb = ZoneBudget(cbc_f, zon, kstpkper=(0, 0), aliases=aliases)
    zb.get_budget(names="FROM_CONSTANT_HEAD", zones=1)
    zb.get_budget(names=["FROM_CONSTANT_HEAD"], zones=[1, 2])
    zb.get_budget(net=True)


def test_get_model_shape(cbc_f, zon_f):
    ZoneBudget(
        cbc_f,
        ZoneBudget.read_zone_file(zon_f),
        kstpkper=(0, 0),
        verbose=True,
    ).get_model_shape()


@pytest.mark.parametrize("rtol", [1e-2])
def test_zonbud_active_areas_zone_zero(loadpth, cbc_f, rtol):
    # Read ZoneBudget executable output and reformat
    zbud_f = loadpth / "zonef_mlt_active_zone_0.2.csv"
    zbud = pd.read_csv(zbud_f)
    zbud.columns = [c.strip() for c in zbud.columns]
    zbud.columns = ["_".join(c.split()) for c in zbud.columns]
    zbud.index = pd.Index([f"ZONE_{z}" for z in zbud.ZONE.values], name="name")
    cols = [c for c in zbud.columns if "ZONE_" in c]
    zbud = zbud[cols]

    # Run ZoneBudget utility and reformat output
    zon_f = loadpth / "zonef_mlt_active_zone_0.zbr"
    zon = ZoneBudget.read_zone_file(zon_f)
    zb = ZoneBudget(cbc_f, zon, kstpkper=(0, 1096))
    fpbud = zb.get_dataframes().reset_index()
    fpbud = fpbud[["name"] + [c for c in fpbud.columns if "ZONE" in c]]
    fpbud = fpbud.set_index("name").T
    fpbud = fpbud[[c for c in fpbud.columns if "ZONE" in c]]
    fpbud = fpbud.loc[[f"ZONE_{z}" for z in range(1, 4)]]

    # Test for equality
    allclose = np.allclose(zbud, fpbud, rtol)
    s = "Zonebudget arrays do not match."
    assert allclose, s


def test_read_zone_file(function_tmpdir):
    zf = (
        "2    2    4\n"
        "INTERNAL     (4I3)\n"
        "  1  1  1  0\n"
        "  0  1  1  1\n"
        "INTERNAL     (4I3)\n"
        "  1  1  1  0\n"
        "  0  1  1  1\n"
        "  0"
    )
    name = function_tmpdir / "zonefiletest.txt"
    with open(name, "w") as foo:
        foo.write(zf)
    zones = ZoneBudget.read_zone_file(name)
    if zones.shape != (2, 2, 4):
        raise AssertionError("zone file read failed")


@pytest.mark.mf6
@requires_exe("mf6")
def test_zonebudget_6(function_tmpdir, example_data_path):
    exe_name = "mf6"
    zb_exe_name = "zbud6"

    sim_ws = example_data_path / "mf6" / "test001e_UZF_3lay"
    sim = MFSimulation.load(sim_ws=sim_ws, exe_name=exe_name)
    sim.simulation_data.mfpath.set_sim_path(function_tmpdir)
    sim.write_simulation()
    success, _ = sim.run_simulation()

    grb_file = function_tmpdir / "test001e_UZF_3lay.dis.grb"
    cbc_file = function_tmpdir / "test001e_UZF_3lay.cbc"

    ml = sim.get_model("gwf_1")
    idomain = np.ones(ml.modelgrid.shape, dtype=int)

    zb = ZoneBudget6(model_ws=function_tmpdir, exe_name=zb_exe_name)
    zf = ZoneFile6(zb, idomain)
    zb.grb = str(grb_file)
    zb.cbc = str(cbc_file)
    zb.write_input(line_length=21)
    success, _ = zb.run_model()

    assert success, "Zonebudget run failed"

    df = zb.get_dataframes()

    assert isinstance(df, pd.DataFrame)

    zb_pkg = ml.uzf.output.zonebudget(idomain)
    zb_pkg.change_model_ws(function_tmpdir)
    zb_pkg.name = "uzf_zonebud"
    zb_pkg.write_input()
    success, _ = zb_pkg.run_model(exe_name=zb_exe_name)

    assert success, "UZF package zonebudget run failed"

    df = zb_pkg.get_dataframes()

    assert isinstance(df, pd.DataFrame)

    # test aliases
    zb = ZoneBudget6(model_ws=function_tmpdir, exe_name=zb_exe_name)
    zf = ZoneFile6(zb, idomain, aliases={1: "test alias", 2: "test pop"})
    zb.grb = str(grb_file)
    zb.cbc = str(cbc_file)
    zb.write_input(line_length=5)
    success, _ = zb.run_model()

    assert success, "UZF package zonebudget run failed"

    df = zb.get_dataframes()

    assert list(df)[0] == "test_alias", "Alias testing failed"


@pytest.mark.mf6
@requires_exe("mf6")
def test_zonebudget6_from_output_method(function_tmpdir, example_data_path):
    exe_name = "mf6"
    zb_exe_name = "zbud6"

    sim_ws = example_data_path / "mf6" / "test001e_UZF_3lay"
    sim = MFSimulation.load(sim_ws=sim_ws, exe_name=exe_name)
    sim.simulation_data.mfpath.set_sim_path(function_tmpdir)
    sim.write_simulation()
    success, _ = sim.run_simulation()

    gwf = sim.get_model("gwf_1")

    idomain = np.ones(gwf.modelgrid.shape, dtype=int)
    zonbud = gwf.output.zonebudget(idomain)
    zonbud.write_input()
    success, buff = zonbud.run_model(exe_name=zb_exe_name)

    assert success, "zonebudget6 model run failed"
